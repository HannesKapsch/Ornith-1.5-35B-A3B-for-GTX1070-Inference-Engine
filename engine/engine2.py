"""Ornith engine v2: three-tier expert storage (VRAM -> pinned RAM -> SSD) with next-layer prediction.

Per token the GPU runs 41 recorded CUDA-graph segments: seg_l = MoE(l-1) + attention(l) + router(l).
Between segments the CPU reads which experts layer l picked and resolves each one:
  VRAM slot   -> pointer into the VRAM pool
  RAM slot    -> pointer into pinned RAM (the GEMV reads it directly over PCIe, zero-copy)
  SSD         -> read into a RAM slot now (blocking), then as above
Meanwhile the router of layer l+1 applied to layer l's hidden state predicts the next experts;
the top `prefetch` of them are read from the SSD into RAM slots by background threads.
Between tokens, experts that keep coming from RAM are promoted into VRAM (async DMA).
RAM use is bounded by `ram_gb` (pinned); the OS file cache may additionally hold parts of experts.bin
(that memory stays reclaimable by Windows).
"""
import os, time, ctypes
from collections import OrderedDict, defaultdict
import numpy as np
import cupy as cp
import torch
from engine import Ornith, dev
from kern import fn, rgemv, stream, i32, f32, i64
from pack import BLOB, OFF, OUT as EXPERTS_BIN
from diskio import Reader, submit_many


_PINNED = []
PRIO_READ = __import__('os').environ.get('PRIO_READ', '0') == '1'   # PriorityReader (18.10.)
ATTN6 = __import__('os').environ.get('ATTN6', '0') == '1'   # long-context decode attention attn_partial6 (night 2)
ATTN8 = __import__('os').environ.get('ATTN8', '1') == '1'   # attn_partial8: 4 heads per warp share K/V conversions (45k x1.39, 150k x1.27 per layer)
ATTN6_NS = int(__import__('os').environ.get('ATTN6_NS', 64))   # split-tile kernel (v8) for every long context (> 4k); v6 was only worth it from 32k


def pinned(nbytes):
    ptr = cp.cuda.runtime.hostAlloc(nbytes, 0)
    _PINNED.append(ptr)
    return torch.from_numpy(np.ctypeslib.as_array((ctypes.c_uint8 * nbytes).from_address(ptr)))


def free_pinned():
    """release every pinned RAM-tier chunk (call after the engine is gone)"""
    while _PINNED:
        cp.cuda.runtime.freeHost(_PINNED.pop())


class Ornith2(Ornith):
    def __init__(self, vram_gb=3.0, ram_gb=4.0, prefetch=12, admit=1.5, promote=64, buffered=True, free_gb=None, cor=True, **kw):
        """free_gb=None: fixed expert pool of vram_gb + KV cache for max_ctx.
        free_gb=x (arena mode, needs kv8): one VRAM block of all memory except x GB; the KV cache grows into it
        in chunks of CK tokens as the context grows, the expert cache gets whatever is left."""
        self.arena_mode = free_gb is not None
        super().__init__(cache_gb=0 if self.arena_mode else vram_gb, kv_alloc=not self.arena_mode, **kw)
        self._rope_init()
        self.prefetch, self.admit, self.promote_cap = prefetch, admit, promote
        self.split_reads = 1                  # >1: blocking misses read in parallel pieces (measured: no gain)
        self.cor = cor                        # copy-on-read: RAM experts are written into a VRAM slot by the GEMV that reads them
        self.cor_pend, self.tok_vs = [], set()  # (key, vslot) inserted during the current token; VRAM slots touched
        # RAM tier
        self.R = max(64, int(ram_gb * 2 ** 30 // BLOB))
        chunk = 256  # slots per pinned chunk (~430 MB), Windows-friendly
        self.ram = []
        c, tries = 0, 0
        while c < self.R:
            try:
                self.ram.append(pinned(min(chunk, self.R - c) * BLOB))
                c += chunk
            except Exception as ex:          # e.g. cudaErrorAlreadyMapped: a previous process's pinned RAM is not freed yet
                tries += 1
                if tries <= 10:
                    time.sleep(1.0)
                    continue
                self.R = max(64, c)          # give up on the rest: smaller RAM tier
                (kw.get('log') or print)(f'RAM tier reduced to {self.R * BLOB / 2 ** 30:.1f} GB ({ex})')
                break
        self.R = min(self.R, len(self.ram) * chunk) if self.ram else self.R
        self.ram_addr = [self.ram[s // chunk].data_ptr() + (s % chunk) * BLOB for s in range(self.R)]
        self.rlru = OrderedDict()             # (l,e) -> ram slot
        self.rfree = list(range(self.R))
        self.rpending = {}                    # (l,e) -> (future, slot) in-flight SSD reads
        self.rbusy = set()                    # RAM slots being copied to VRAM
        if PRIO_READ:                         # background reads by priority (next layer first), demand waits steal queued jobs
            from diskio import PriorityReader
            self.reader = PriorityReader(EXPERTS_BIN, nobuf=not buffered, threads=8)
        else:
            self.reader = Reader(EXPERTS_BIN, nobuf=not buffered, threads=8)
        # VRAM tier (dynamic, filled by promotion)
        self.vmap, self.vowner = {}, OrderedDict()   # (l,e) -> vslot, vslot -> (l,e) (LRU order: oldest first)
        self.vfree = list(range(self.S_slots))
        self.vpend = []                       # (event, [(key, vslot)])
        self.cstream = torch.cuda.Stream()
        self.freq = np.zeros((self.L, self.E))
        # global scales of every expert (3 floats), from the blobs
        g = np.zeros((self.L, self.E, 3), np.float32)
        for l in range(self.L):
            g[l] = np.frombuffer(np.asarray(self.host[l, :, OFF['g2']:OFF['g2'] + 12]).tobytes(), np.float32).reshape(self.E, 3)
        self.g2 = g
        # per-layer pinned pointer staging + device copies
        k1 = self.K + 1
        self.ptr_host = torch.zeros(self.L, 5, 2 * k1, dtype=torch.int64).pin_memory()   # gu_w, gu_s, dn_w, dn_s, (counts)
        self.g_host = torch.zeros(self.L, 3, 2 * k1, dtype=torch.float32).pin_memory()     # gu_g, dn_g
        self.routes_host = torch.zeros(self.L + 1, 8, dtype=torch.int64).pin_memory()
        self.wts_host = torch.zeros(self.L, k1).pin_memory()
        self.ptr_np, self.g_np = self.ptr_host.numpy(), self.g_host.numpy()
        K1 = self.K + 1
        lay = [('gu_w', torch.int64, 2 * K1), ('gu_s', torch.int64, 2 * K1), ('dn_w', torch.int64, K1), ('dn_s', torch.int64, K1),
               ('wb', torch.float32, K1), ('gu_wd', torch.int64, 2 * K1), ('gu_sd', torch.int64, 2 * K1), ('dn_wd', torch.int64, K1), ('dn_sd', torch.int64, K1),
               ('gu_g', torch.float32, 2 * K1), ('dn_g', torch.float32, K1), ('counts', torch.int32, 2)]
        off, self.stage_off = 0, {}
        for name, dt, n in lay:
            self.stage_off[name] = (off, dt, n)
            off += n * torch.tensor([], dtype=dt).element_size()
            off = (off + 7) // 8 * 8
        self.stage_bytes = off
        self.stage_host = torch.zeros(self.L, off, dtype=torch.uint8).pin_memory()
        self.stage_dev = torch.zeros(off, dtype=torch.uint8, device=dev)
        for name, (o, dt, n) in self.stage_off.items():     # kernel arguments now live inside stage_dev
            setattr(self, name, self.stage_dev[o:o + n * torch.tensor([], dtype=dt).element_size()].view(dt))
        self.stage_np = {name: self.stage_host[:, o:o + n * torch.tensor([], dtype=dt).element_size()].view(dt).numpy()
                         for name, (o, dt, n) in self.stage_off.items()}
        self.shared_g2 = [np.frombuffer(d['shared'][OFF['g2']:OFF['g2'] + 12].cpu().numpy().tobytes(), np.float32).copy()
                          for d in self.layers]
        self.offs = np.array([OFF['gw'], OFF['uw'], OFF['gs'], OFF['us'], OFF['dw'], OFF['ds']], np.int64)
        self.pred_host = torch.full((self.L + 1, 16), -1, dtype=torch.int64).pin_memory()   # router writes here directly (mapped)
        self.pred_dev = torch.zeros(self.L + 1, 16, dtype=torch.int64, device=dev)
        self.dummy_w = torch.zeros(32, device=dev)
        self.dummy_c = torch.zeros(2, dtype=torch.int32, device=dev)
        self.counts_dev = torch.zeros(2, dtype=torch.int32, device=dev)
        self.rlog = torch.zeros(self.E + 1, device=dev)
        self.rlog2 = torch.zeros(self.E + 1, device=dev)
        self.routes_all = torch.zeros(self.L, 8, dtype=torch.int64, device=dev)
        self.graphs = {}
        self.stats = defaultdict(int)
        self.feat = torch.zeros(len(self.FEAT_LAYERS), self.H, device=dev)   # draft-model features of the last token
        if os.environ.get('ATT4') == '1':      # lossy: attention / DeltaNet projections FP8 -> NVFP4 (decode path)
            self._att4()
        if self.arena_mode:
            self._arena_init(free_gb)

    def _w(self, d, k):
        """(format, weight ptrs, block-scale ptrs, global scale) of an attention projection: NVFP4 copy if made, else FP8"""
        return d.get(k + '_4') or ('fp8', d[k + '_p'], None, d[k + '_s'])

    def _att4(self):
        """requantize the FP8 projections (per-tensor scale) to NVFP4 (fp8 e4m3 scale per 16 values + global scale), like the
        experts. Lossy (measure the quality before using it). The FP8 tensors stay for the prefill."""
        mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)
        P = lambda *ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
        def q4(w8, s):
            N, K = w8.shape
            Wq = torch.empty(N, K // 2, dtype=torch.uint8, device=dev); Sq = torch.empty(N, K // 16, dtype=torch.uint8, device=dev)
            amax_all = 0.0
            for r in range(0, N, 1024):
                amax_all = max(amax_all, (w8[r:r + 1024].view(torch.float8_e4m3fn).float() * s).abs().max().item())
            g = max(amax_all / (448 * 6), 1e-30)
            for r in range(0, N, 1024):
                b = (w8[r:r + 1024].view(torch.float8_e4m3fn).float() * s).view(-1, K // 16, 16)
                bs = (b.abs().amax(-1) / 6 / g).clamp(max=448).to(torch.float8_e4m3fn)
                sc = (bs.float() * g).unsqueeze(-1)
                q = torch.where(sc > 0, b / sc.clamp(min=1e-30), torch.zeros_like(b))
                idx = torch.bucketize(q.abs(), mids)
                code = (idx | (((q < 0) & (idx > 0)).long() << 3)).view(b.shape[0], K).to(torch.uint8)
                Wq[r:r + 1024] = code[:, 0::2] | (code[:, 1::2] << 4)
                Sq[r:r + 1024] = bs.view(-1, K // 16).view(torch.uint8)
            return Wq, Sq, g
        for d in self.layers:
            names = [('qkv', 'qkv'), ('z', 'z'), ('out', 'out')] if d['kind'] == 'lin' else [('q', 'q'), ('o', 'o')]
            for k, t in names:
                Wq, Sq, g = q4(d[t], d[k + '_s'].item())
                d[k + '_4t'] = (Wq, Sq)
                d[k + '_4'] = ('nvfp4', P(Wq), P(Sq), torch.tensor([g], device=dev))
            if d['kind'] == 'full':
                (Wk, Sk, gk), (Wv, Sv, gv) = q4(d['k'], d['kv_s'][0].item()), q4(d['v'], d['kv_s'][1].item())
                d['kv_4t'] = (Wk, Sk, Wv, Sv)
                d['kv_4'] = ('nvfp4', P(Wk, Wv), P(Sk, Sv), torch.tensor([gk, gv], device=dev))
        torch.cuda.synchronize()

    # ------------------------------------------------------------------ KV arena
    CK = 4096                                 # tokens per KV chunk

    def _arena_init(self, free_gb):
        assert self.kv8, 'arena mode needs the int8 KV cache'
        nf, CK = self.nfull, self.CK
        kb, sb = 2 * CK * 256, 2 * CK * 16    # per layer and chunk: int8 rows / fp16 scale rows of both kv heads
        self.CSB = nf * 2 * (kb + sb)         # bytes per chunk (K, V, K scales, V scales of every full layer)
        self.kvdefs = dict(CK=CK, KCH=self.CSB // 256, SCH=self.CSB // 16)
        self.pool = None
        torch.cuda.synchronize(); torch.cuda.empty_cache()
        free = torch.cuda.mem_get_info()[0]
        A = int(free - free_gb * 2 ** 30 - 0.85 * 2 ** 30) // BLOB * BLOB   # 0.85 GB: prefill scratch, graphs, workspaces (measured)
        need = self.kv_bytes(self.max_ctx) + 2 * 256 * BLOB + 512 * 180 * 1024
        assert A >= need, f'arena {A / 2 ** 30:.2f} GB < {need / 2 ** 30:.2f} GB needed for max_ctx={self.max_ctx}'
        self.arena = torch.empty(A, dtype=torch.uint8, device=dev)
        self.A = A
        b = self.arena.data_ptr()
        self.kvp = [(b + i * kb, b + nf * kb + i * kb, b + 2 * nf * kb + i * sb, b + 2 * nf * kb + nf * sb + i * sb)
                    for i in range(nf)]
        self.kv_chunks = 0
        self.S_slots = A // BLOB
        self.vfree = list(range(self.S_slots))

    def kv_bytes(self, ntok):
        return -(-ntok // self.CK) * self.CSB

    def vaddr(self, vs):
        """VRAM address of expert slot vs (arena: slots count down from the top, the KV cache grows up from the bottom)"""
        if self.arena_mode:
            return self.arena.data_ptr() + self.A - (vs + 1) * BLOB
        return self.pool.data_ptr() + vs * BLOB

    def prefill_mem(self, ntok=None):
        """VRAM the prefill may use for its buffers: everything above the KV cache of ntok tokens (default max_ctx)"""
        return self.arena[self.kv_bytes(ntok or self.max_ctx):] if self.arena_mode else self.pool

    def ensure_kv(self, ntok):
        """make room for ntok tokens of KV cache; evicts the expert slots the new chunks overlap"""
        if not self.arena_mode:
            return
        need = -(-ntok // self.CK)
        if need <= self.kv_chunks:
            return
        assert ntok <= self.max_ctx, f'context {ntok} > max_ctx {self.max_ctx}'
        self._settle_promotions()
        usable = (self.A - need * self.CSB) // BLOB
        for vs in range(usable, self.S_slots):
            key = self.vowner.pop(vs, None)
            if key is not None:
                self.vmap.pop(key, None)
        self.vfree = [v for v in self.vfree if v < usable or v >= 1 << 20]   # >= 1 << 20: slots in the output head block
        self.S_slots, self.kv_chunks = usable, need
        self.stats['kv_grow'] += 1

    def _settle_promotions(self):
        for ev, items in self.vpend:
            ev.synchronize()
            for key, vs, rs in items:
                self.vmap[key] = vs
                self.rbusy.discard(rs)
        self.vpend = []

    # ------------------------------------------------------------------ RoPE rows (3D positions of image tokens)
    XR = 16384                                # extra cos/sin rows for image tokens (per conversation)

    def _rope_init(self):
        self.cos = torch.cat([self.cos, torch.zeros(self.XR, self.cos.shape[1], device=dev)]).contiguous()
        self.sin = torch.cat([self.sin, torch.zeros(self.XR, self.sin.shape[1], device=dev)]).contiguous()
        self.rp = torch.arange(self.max_ctx, dtype=torch.int32, device=dev)      # cache slot -> RoPE table row
        self.rope_delta, self.rope_next = 0, 0
        th = self.cfg['rope_parameters']['rope_theta']
        self.inv_freq = 1.0 / (th ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))            # 32 frequencies
        i = torch.arange(32)
        self.rope_axis = torch.where((i % 3 == 1) & (i < 33), 1, torch.where((i % 3 == 2) & (i < 30), 2, 0))  # interleaved mrope [11,11,10]

    def rope_reset(self):
        if self.rope_delta or self.rope_next:
            self.rp.copy_(torch.arange(self.max_ctx, dtype=torch.int32, device=dev))
        self.rope_delta, self.rope_next = 0, 0

    def add_image_rope(self, p0, gh, gw):
        """image tokens at cache slots p0 .. p0+gh*gw-1 (row-major, merged grid gh x gw): 3D positions (s, s+r, s+c) with
        s = the RoPE position the next text token would get; text after the image continues at s + max(gh, gw)"""
        n = gh * gw
        assert self.rope_next + n <= self.XR, 'too many image tokens in this conversation'
        s = p0 + self.rope_delta
        r, c = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing='ij')
        pos3 = torch.stack([torch.full((n,), s), s + r.reshape(-1), s + c.reshape(-1)])          # [3, n]
        ang = pos3[self.rope_axis].T.double() * self.inv_freq[None]                               # [n, 32]
        row0 = self.max_ctx + self.rope_next
        self.cos[row0:row0 + n] = ang.cos().float().to(dev)
        self.sin[row0:row0 + n] = ang.sin().float().to(dev)
        self.rp[p0:p0 + n] = torch.arange(row0, row0 + n, dtype=torch.int32, device=dev)
        self.rope_next += n
        self.rope_delta = s + max(gh, gw) - (p0 + n)
        tail = torch.arange(p0 + n, self.max_ctx, dtype=torch.int32, device=dev) + self.rope_delta
        self.rp[p0 + n:] = tail.clamp(0, self.max_ctx - 1)

    def rope_state(self, ntok):
        k = self.rope_next
        return {'rp': self.rp[:ntok].cpu(), 'cos': self.cos[self.max_ctx:self.max_ctx + k].cpu(),
                'sin': self.sin[self.max_ctx:self.max_ctx + k].cpu(), 'delta': self.rope_delta, 'next': k}

    def set_rope_state(self, st):
        self.rope_reset()
        k, n = st['next'], st['rp'].numel()
        self.cos[self.max_ctx:self.max_ctx + k] = st['cos'].to(dev); self.sin[self.max_ctx:self.max_ctx + k] = st['sin'].to(dev)
        self.rp[:n] = st['rp'].to(dev)
        self.rope_delta, self.rope_next = st['delta'], k
        if self.rope_delta:
            self.rp[n:] = (torch.arange(n, self.max_ctx, dtype=torch.int32, device=dev) + self.rope_delta).clamp(0, self.max_ctx - 1)

    def reset(self):
        super().reset()
        self.rope_reset()
        if self.arena_mode and self.kv_chunks:   # new conversation: the KV chunks go back to the expert cache
            self._settle_promotions()
            old, self.S_slots, self.kv_chunks = self.S_slots, self.A // BLOB, 0
            self.vfree += list(range(old, self.S_slots))

    def vram_split(self):
        """(KV cache GB, expert cache GB) currently in use"""
        kv = self.kv_chunks * self.CSB if self.arena_mode else 0
        return kv / 2 ** 30, len(self.vmap) * BLOB / 2 ** 30

    # ------------------------------------------------------------------ tiers
    def _ram_slot(self, protect):
        if self.rfree:
            return self.rfree.pop()
        for key in self.rlru:
            s = self.rlru[key]
            if s not in protect and s not in self.rbusy:
                del self.rlru[key]
                return s
        raise RuntimeError('RAM tier too small')

    def _prefetch(self, l, experts, protect):
        jobs = []
        for e in experts:
            key = (l, e)
            if key in self.vmap or key in self.rlru or key in self.rpending:
                continue
            s = self._ram_slot(protect)
            protect = protect | {s}
            self.rpending[key] = (None, s)
            jobs.append((key, s))
        if jobs:
            rq = [((k[0] * self.E + k[1]) * BLOB, BLOB, self.ram_addr[s]) for k, s in jobs]
            futs = self.reader.psubmit(rq, l) if hasattr(self.reader, 'psubmit') else submit_many(self.reader, rq)
            for (k, s), fu in zip(jobs, futs):
                self.rpending[k] = (fu, s)
            self.stats['prefetched'] += len(jobs)

    def _resolve(self, l, e, protect):
        key = (l, e)
        if key in self.vmap:
            self.stats['vram'] += 1
            vs = self.vmap[key]
            if vs in self.vowner:
                self.vowner.move_to_end(vs)
            self.tok_vs.add(vs)
            return self.vaddr(vs), None
        if key in self.rpending:
            fu, s = self.rpending.pop(key)
            fu.result()
            self.rlru[key] = s
            self.stats['ram_prefetched'] += 1
        elif key in self.rlru:
            self.stats['ram'] += 1
        else:
            s = self._ram_slot(protect)
            P = self.split_reads                # blocking miss: read in P parallel pieces (one memcpy thread is ~5 GB/s)
            if P > 1:
                c = (BLOB // P + 4095) // 4096 * 4096
                base = (l * self.E + e) * BLOB
                for f in submit_many(getattr(self, 'reader_fast', self.reader), [(base + o, min(c, BLOB - o), self.ram_addr[s] + o) for o in range(0, BLOB, c)]):
                    f.result()
            else:
                self.reader._read((l * self.E + e) * BLOB, BLOB, self.ram_addr[s])
            self.rlru[key] = s
            self.stats['ssd'] += 1
        self.rlru.move_to_end(key)
        s = self.rlru[key]
        return self.ram_addr[s], s

    def _cor_slot(self):
        """VRAM slot for a copy-on-read insert: a free one, else the least recently used one not touched this token"""
        if self.vfree:
            return self.vfree.pop()
        for vs in self.vowner:
            if vs not in self.tok_vs:
                old = self.vowner.pop(vs)
                self.vmap.pop(old, None)
                self.stats['cor_evict'] += 1
                return vs
        return None

    def _set_ptrs(self, l, experts, k):
        """fill pointer arrays for layer l: experts[0..k) + shared expert at index k"""
        protect = set()
        addrs, dsts = [], []
        for e in experts[:k]:
            a, s = self._resolve(l, e, protect)
            d = 0
            if s is not None:
                protect.add(s)
                if self.cor:
                    vs = self._cor_slot()
                    if vs is not None:
                        self.tok_vs.add(vs)
                        self.cor_pend.append(((l, e), vs))
                        d = self.vaddr(vs)
            addrs.append(a); dsts.append(d)
        self.freq[l, experts[:k]] += 1.0
        addrs.append(self.layers[l]['shared'].data_ptr()); dsts.append(0)
        k1 = len(addrs)
        a = np.array(addrs, np.int64)[:, None] + self.offs[None]          # [k1, 6]
        dd = np.array(dsts, np.int64)[:, None]
        dd = np.where(dd != 0, dd + self.offs[None], 0)
        S = self.stage_np
        S['gu_wd'][l, 0:2 * k1:2], S['gu_wd'][l, 1:2 * k1:2] = dd[:, 0], dd[:, 1]
        S['gu_sd'][l, 0:2 * k1:2], S['gu_sd'][l, 1:2 * k1:2] = dd[:, 2], dd[:, 3]
        S['dn_wd'][l, :k1], S['dn_sd'][l, :k1] = dd[:, 4], dd[:, 5]
        S['gu_w'][l, 0:2 * k1:2], S['gu_w'][l, 1:2 * k1:2] = a[:, 0], a[:, 1]
        S['gu_s'][l, 0:2 * k1:2], S['gu_s'][l, 1:2 * k1:2] = a[:, 2], a[:, 3]
        S['dn_w'][l, :k1], S['dn_s'][l, :k1] = a[:, 4], a[:, 5]
        g = np.concatenate([self.g2[l, experts[:k]], self.shared_g2[l][None]])  # [k1, 3]
        S['gu_g'][l, 0:2 * k1:2], S['gu_g'][l, 1:2 * k1:2], S['dn_g'][l, :k1] = g[:, 0], g[:, 1], g[:, 2]
        S['counts'][l, 0], S['counts'][l, 1] = 2 * k1, k1
        return protect

    # ------------------------------------------------------------------ graph segments
    def _attn(self, l, ns):
        d, H, s = self.layers[l], self.H, stream()
        fn('rmsnorm1p')((1,), (256,), (self.x.data_ptr(), d['ln1'].data_ptr(), self.h.data_ptr(), i32(H), f32(self.eps)), stream=s)
        i = d['idx']
        if d['kind'] == 'lin':
            rgemv(*self._w(d, 'qkv'), self.h, self.qkv, 8192, H, 1, assign=True)
            rgemv(*self._w(d, 'z'), self.h, self.zb, 4096, H, 1, assign=True)
            rgemv('f32', d['ab_p'], None, None, self.h, self.ab, 64, H, 1, assign=True)
            fn('dn_prep')((64,), (128,), (self.qkv.data_ptr(), self.conv_state[i].data_ptr(), d['conv'].data_ptr(),
                                           self.ab.data_ptr(), d['A_log'].data_ptr(), d['dt_bias'].data_ptr(),
                                           self.qkv2.data_ptr(), self.gb.data_ptr()), stream=s)
            fn('dn_recur')((32,), (128,), (self.qkv2.data_ptr(), self.gb.data_ptr(), self.S[i].data_ptr(),
                                            self.zb.data_ptr(), d['normw'].data_ptr(), self.olin.data_ptr(), f32(self.eps)), stream=s)
            rgemv(*self._w(d, 'out'), self.olin, self.x, H, 4096, 1)
        else:
            rgemv(*self._w(d, 'q'), self.h, self.qf, 8192, H, 1, assign=True)
            rgemv(*self._w(d, 'kv'), self.h, self.kvb, 512, H, 2, xdiv=2, ystride=512, assign=True)
            kv8 = int(self.kv8)
            kc, vc, ks, vs = self.kvp[i]
            fn('attn_prep', KV8=kv8, WIN=self.WIN, RING=self.RING, **self.kvdefs)((18,), (256,), (self.qf.data_ptr(), self.kvb.data_ptr(), d['qn'].data_ptr(), d['kn'].data_ptr(),
                                                    self.cos.data_ptr(), self.sin.data_ptr(), self.pos_t.data_ptr(), self.qr.data_ptr(),
                                                    self.gate.data_ptr(), kc, vc, ks, vs, i32(self.max_ctx), f32(self.eps),
                                                    self.Kr[i].data_ptr(), self.Vr[i].data_ptr(), self.rp.data_ptr()), stream=s)
            long_ctx = self.kv8 and ns * 64 > 2 * self.WIN
            a6 = long_ctx and (ATTN6 or ns >= ATTN6_NS)   # long context: 4 tiles per block, online softmax, double buffering (150k: -24 % attention)
            pargs = (self.qr.data_ptr(), kc, vc, ks, vs, self.pos_t.data_ptr(), self.pm.data_ptr(), self.pl.data_ptr(), self.pacc.data_ptr(),
                     i32(self.max_ctx), i32(self.nsplit), self.Kr[i].data_ptr(), self.Vr[i].data_ptr())
            if a6:
                fn('attn_partial8' if ATTN8 else 'attn_partial6', KV8=kv8, WIN=self.WIN, RING=self.RING, SUB=4, **self.kvdefs)((2, (ns + 3) // 4), (256,), pargs, stream=s)
            else:
                fn('attn_partial4' if long_ctx else 'attn_partial', KV8=kv8, WIN=self.WIN, RING=self.RING, **self.kvdefs)((2, ns), (256,), pargs, stream=s)
            fn('attn_combine5' if a6 else 'attn_combine', KV8=kv8, WIN=self.WIN, RING=self.RING, **(dict(SUB=4) if a6 else {}))((16,), (256,), (self.pos_t.data_ptr(), self.pm.data_ptr(), self.pl.data_ptr(),
                                                       self.pacc.data_ptr(), self.gate.data_ptr(), self.oatt.data_ptr(), i32(self.nsplit)), stream=s)
            rgemv(*self._w(d, 'o'), self.oatt, self.x, H, 4096, 1)
        fn('rmsnorm1p')((1,), (256,), (self.x.data_ptr(), d['ln2'].data_ptr(), self.h.data_ptr(), i32(H), f32(self.eps)), stream=s)
        o = OFF
        common = (i64(o['gw']), i64(o['uw']), i64(o['dw']), i64(o['gs']), i64(o['us']), i64(o['ds']), i64(o['g2']), i32(0))
        rgemv('bf16', d['rw_p'], None, None, self.h, self.rlog, self.E + 1, H, 1, assign=True)
        if getattr(self, 'ov', False):          # Ornith3 (engine3.py): router2 splits VRAM / non-VRAM experts
            self._router2(l, d, common, s)
        else:
            fn('router')((1,), (256,), (self.rlog.data_ptr(), d['sgate'].data_ptr(), self.h.data_ptr(), self.loc.data_ptr(),
                                        i64(0), i32(l), i32(self.E), i32(H), i32(self.K), f32(self.adaptive),
                                        self.wout.data_ptr(), self.routes_host[l].data_ptr(), self.counts_dev.data_ptr(),
                                        i64(0), i64(0), i64(0), i64(0), i64(0), i64(0)) + common, stream=s)
        if self.prefetch and l + 1 < self.L and not getattr(self, 'ov', False):   # predict layer l+1 from this hidden state
            dn = self.layers[l + 1]
            rgemv('bf16', dn['rw_p'], None, None, self.h, self.rlog2, self.E + 1, H, 1, assign=True)
            fn('router')((1,), (256,), (self.rlog2.data_ptr(), dn['sgate'].data_ptr(), self.h.data_ptr(), self.loc.data_ptr(),
                                        i64(0), i32(l + 1), i32(self.E), i32(H), i32(self.prefetch), f32(0.0),
                                        self.dummy_w.data_ptr(), self.pred_host[l + 1].data_ptr(), self.dummy_c.data_ptr(),
                                        i64(0), i64(0), i64(0), i64(0), i64(0), i64(0)) + common, stream=s)

    def _moe(self, l):
        H, I, k1, s = self.H, self.I, self.K + 1, stream()
        fn('cpyn')((1,), (128,), (self.stage_host[l].data_ptr(), self.stage_dev.data_ptr(), i32(self.stage_bytes // 8)), stream=s)
        self.gu.zero_()
        cw = dict(wdst=self.gu_wd, sdst=self.gu_sd) if self.cor else {}
        rgemv('nvfp4', self.gu_w, self.gu_s, self.gu_g, self.h, self.gu, I, H, 2 * k1, xdiv=2 * k1, ystride=I, mcount=self.counts[0:1], **cw)
        fn('silu_mul')(((k1 * I + 255) // 256,), (256,), (self.gu.data_ptr(), self.hb.data_ptr(), self.counts.data_ptr(), i32(I)), stream=s)
        rgemv('nvfp4', self.dn_w, self.dn_s, self.dn_g, self.hb, self.x, H, I, k1, xdiv=1, xstride=I, ydiv=k1,
              wts=self.wout, mcount=self.counts[1:2], **(dict(wdst=self.dn_wd, sdst=self.dn_sd) if self.cor else {}))

    def _final(self):
        fn('rmsnorm1p')((1,), (256,), (self.x.data_ptr(), self.norm.data_ptr(), self.h.data_ptr(), i32(self.H), f32(self.eps)), stream=stream())
        rgemv('nvfp4', self.lm_p, self.lm_sp, self.lm_g, self.h, self.logits, self.V, self.H, 1, assign=True)
        self.next_tok = getattr(self, 'next_tok', torch.zeros(1, dtype=torch.int64, device=dev))
        self.next_tok.copy_(self.logits.argmax().view(1))

    FEAT_LAYERS = tuple(int(x) for x in __import__('os').environ.get('FEAT_LAYERS', '1,6,11,16,22,27,32,37').split(','))  # DFlash: residual after these layers

    def _segment(self, i, ns):
        if i > 0:
            self._moe(i - 1)
            if i - 1 in self.FEAT_LAYERS:         # residual after layer i-1 (before attention i adds to x)
                self.feat[self.FEAT_LAYERS.index(i - 1)].copy_(self.x)
        if i < self.L:
            self._attn(i, ns)
        else:
            self._final()

    def _replay(self, i, ns):
        key = (i, ns)
        g = self.graphs.get(key)
        if g is None:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            snap = [t.clone() for t in self._state()]
            with torch.cuda.stream(st):
                self._segment(i, ns)            # warm-up (compiles kernels)
            torch.cuda.current_stream().wait_stream(st)
            for t, v in zip(self._state(), snap):
                t.copy_(v)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._segment(i, ns)
            self.graphs[key] = g
        g.replay()

    def _state(self):
        p, r = self._pos, self._pos % self.RING
        if self.arena_mode:                   # the whole KV chunk holding position p
            c = p // self.CK
            kv = [self.arena[c * self.CSB:(c + 1) * self.CSB]]
        else:
            kv = [self.Kc[:, :, p], self.Vc[:, :, p], self.Ks[:, :, p], self.Vs[:, :, p]]
        return [self.x, self.h, self.wout, self.S, self.conv_state] + kv + [self.Kr[:, :, r], self.Vr[:, :, r]]

    # ------------------------------------------------------------------ promotion RAM -> VRAM (between tokens)
    def _promote(self, used):
        still = []
        for ev, items in self.vpend:
            if ev.query():
                for key, vs, rs in items:
                    self.vmap[key] = vs
                    self.rbusy.discard(rs)
            else:
                still.append((ev, items))
        self.vpend = still
        self.freq *= 0.9
        if self.cor:
            return
        cand = sorted((k for k in used if k not in self.vmap and k in self.rlru and self.freq[k] >= self.admit),
                      key=lambda k: -self.freq[k])[:self.promote_cap]
        if not cand:
            return
        pend_v = {vs for _, items in self.vpend for _, vs, _ in items}
        victims = sorted((self.freq[k], vs) for vs, k in self.vowner.items() if k not in used and vs not in pend_v) \
            if len(self.vfree) < len(cand) else []
        jobs = []
        for key in cand:
            if self.vfree:
                vs = self.vfree.pop()
            elif victims and victims[0][0] < self.freq[key]:
                _, vs = victims.pop(0)
                old = self.vowner.pop(vs)
                self.vmap.pop(old, None)
            else:
                break
            self.vowner[vs] = key
            jobs.append((key, vs))
        if not jobs:
            return
        self.cstream.wait_stream(torch.cuda.current_stream())
        items = []
        for key, vs in jobs:
            rs = self.rlru[key]
            self.rbusy.add(rs)
            cp.cuda.runtime.memcpyAsync(self.vaddr(vs), self.ram_addr[rs], BLOB, 1, self.cstream.cuda_stream)
            items.append((key, vs, rs))
        ev = torch.cuda.Event()
        ev.record(self.cstream)
        self.vpend.append((ev, items))
        self.stats['promoted'] += len(jobs)

    # ------------------------------------------------------------------ one token
    @torch.no_grad()
    def step(self, tok, pos, capture=None):
        self.ensure_kv(pos + 1)
        self.pos_t.fill_(pos)
        self._pos = pos
        self.x.copy_(self.embed[tok].float())
        need = (pos + 64) // 64
        ns = min(self.nsplit, max(4, 1 << (need - 1).bit_length()))
        ev = torch.cuda.Event()
        used = set()
        self.tok_vs = set()
        # promotions of pending RAM slots must not be evicted while copying
        for i in range(self.L + 1):
            self._replay(i, ns)
            if i == self.L:
                break
            ev.record()
            ev.synchronize()
            ex = [e for e in self.routes_host[i].tolist() if e >= 0]
            protect = self._set_ptrs(i, ex, len(ex))
            for key in [k for k in self.rpending if k[0] == i]:   # unused predictions: keep data, evict first
                fu, rs = self.rpending.pop(key)
                fu.result()
                self.rlru[key] = rs
                self.rlru.move_to_end(key, last=False)
                self.stats['mispredicted'] += 1
            used.update((i, e) for e in ex)
            if self.prefetch and i + 1 < self.L:
                self._prefetch(i + 1, [e for e in self.pred_host[i + 1].tolist() if e >= 0], protect)
        torch.cuda.current_stream().synchronize()
        for key, vs in self.cor_pend:          # copy-on-read inserts are complete now
            self.vmap[key] = vs
            self.vowner[vs] = key
            self.stats['cor_insert'] += 1
        self.cor_pend = []
        self._promote(used)
        return self.logits

    def generate(self, ids, n_new, stop_ids=()):
        self.reset()
        for pos, t in enumerate(ids[:-1]):
            self.step(t, pos)
        cur, out = ids[-1], []
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(n_new):
            self.step(cur, len(ids) - 1 + i)
            cur = int(self.next_tok.item())
            out.append(cur)
            if cur in stop_ids:
                break
        return out, len(out) / (time.time() - t0)
