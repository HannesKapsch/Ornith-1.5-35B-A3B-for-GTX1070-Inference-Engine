"""Chunked prefill for Ornith2: many prompt tokens per pass.

While prefilling, the VRAM expert-cache pool (arena mode: the part above the KV cache of max_ctx) is borrowed: the first 433 MB hold all 256 experts of the
current layer (copied from a pinned double buffer that background threads fill from experts.bin),
the rest holds the chunk activations. The decode cache is cleared and refills itself afterwards.
"""
import os, time, ctypes
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cupy as cp
import torch
from kern import fn as kfn, rgemv, stream, i32, f32
from kern_prefill import fn as pfn, bgemm
from pack import BLOB, OFF, OUT as EXPERTS_BIN
from engine2 import pinned, _PINNED

dev = 'cuda'
LAYER_BYTES = 256 * BLOB
ATTN12 = os.environ.get('ATTN12', '1') == '1'   # attn_prefill12 (internal benchmark: 1.68 vs 1.11 TFLOPS)
PER_TOK = 4 * (2048 * 2 + 8192 * 2 + 4096 * 3 + 1024 + 64 * 2 + 257 + 9 + 8 + 512 * 8 * 3 + 512 * 3)   # activation bytes per chunk token
READ_THREADS = int(__import__('os').environ.get('READ_THREADS', '6'))   # parallel expert reads per layer
CAPTURE = None                         # dict -> collect predictor training data (internal benchmark)
PF_VRAM = __import__('os').environ.get('PF_VRAM', '0') == '1'   # prefill uses experts still in the VRAM cache in place
KEEP_CACHE = __import__('os').environ.get('KEEP_CACHE', '1') == '1'   # prefill keeps the VRAM expert cache it does not need


class Prefill:
    def __init__(self, eng, int8_attn=False):
        self.e = eng
        self.int8_attn = int8_attn      # DP4A Q.K for contexts beyond the exact window (slightly lossy)
        e = eng
        # pinned double buffer for whole expert layers (2 x 433 MB), freed by close()
        self.host = [pinned(LAYER_BYTES), pinned(LAYER_BYTES)]
        self.io = ThreadPoolExecutor(1)
        self.fd = open(EXPERTS_BIN, 'rb', buffering=0)
        self.xmap = torch.zeros(4096 * 8, dtype=torch.int32, device=dev)
        self.rsc = torch.zeros(4096 * 8, device=dev)
        self.roff = torch.zeros(257, dtype=torch.int32, device=dev)
        self.roff1 = torch.zeros(2, dtype=torch.int32, device=dev)
        self.wp = torch.zeros(2, 3, 256, dtype=torch.int64, device=dev)  # [region][gate/up/down] weight pointers
        self.sp = torch.zeros(2, 3, 256, dtype=torch.int64, device=dev)  # scale pointers
        self.gs = torch.zeros(3, 256, device=dev)                       # global scales
        self.cstream = torch.cuda.Stream()
        self.P = lambda *ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
        self.mem = None
        self._layout(e.prefill_mem())

    def _layout(self, mem):
        """place the prefill buffers in mem (arena mode: everything above the KV cache the prompt needs)"""
        if self.mem is not None and mem.data_ptr() == self.mem.data_ptr() and mem.numel() == self.mem.numel():
            return
        self.mem = mem
        # chunk size from the budget: 2 x 433 MB experts + ~176 KB per token of activations
        per_tok = PER_TOK
        avail = mem.numel() - 2 * LAYER_BYTES - 64 * 2 ** 20
        self.CH = int(min(4096, max(256, avail // per_tok)) // 64 * 64)
        T = self.CH
        off = [2 * LAYER_BYTES]

        def take(n, dt=torch.float32):
            sz = n * torch.tensor([], dtype=dt).element_size()
            o = (off[0] + 255) // 256 * 256
            off[0] = o + sz
            assert off[0] <= mem.numel(), 'expert cache pool too small for prefill buffers'
            return mem[o:o + sz].view(dt)

        self.X, self.HN = take(T * 2048).view(T, 2048), take(T * 2048).view(T, 2048)
        self.QKV, self.QKV2 = take(T * 8192).view(T, 8192), take(T * 8192).view(T, 8192)
        self.Z, self.OL, self.QR = take(T * 4096).view(T, 4096), take(T * 4096).view(T, 4096), take(T * 4096).view(T, 4096)
        self.KVb = take(T * 1024).view(T, 1024)
        self.AB, self.GB = take(T * 64).view(T, 64), take(T * 64).view(T, 64)
        self.RL, self.RW = take(T * 257).view(T, 257), take(T * 9).view(T, 9)
        self.ROUT = take(T * 8, torch.int32).view(T, 8)
        self.G1, self.U1, self.H1 = (take(T * 8 * 512).view(T * 8, 512) for _ in range(3))
        self.SG, self.SU, self.SH = (take(T * 512).view(T, 512) for _ in range(3))
        self.ptab_base = np.zeros((2, 3, 2, 256), np.int64)
        for r in range(2):
            b0 = mem.data_ptr() + r * LAYER_BYTES
            for k, (wo, so) in enumerate((('gw', 'gs'), ('uw', 'us'), ('dw', 'ds'))):
                self.ptab_base[r, k, 0] = [b0 + x * BLOB + OFF[wo] for x in range(256)]
                self.ptab_base[r, k, 1] = [b0 + x * BLOB + OFF[so] for x in range(256)]
        if not hasattr(self, 'ptab_t'):
            self.ptab_t = torch.zeros(self.e.L, 3, 2, 256, dtype=torch.int64).pin_memory()
            self.ptab = self.ptab_t.numpy()
            self.wpl = [torch.zeros(3, 2, 256, dtype=torch.int64, device=dev) for _ in range(2)]
        for r in range(2):
            base = mem.data_ptr() + r * LAYER_BYTES
            for k, (wo, so) in enumerate((('gw', 'gs'), ('uw', 'us'), ('dw', 'ds'))):
                self.wp[r, k] = torch.tensor([base + x * BLOB + OFF[wo] for x in range(256)], dtype=torch.int64)
                self.sp[r, k] = torch.tensor([base + x * BLOB + OFF[so] for x in range(256)], dtype=torch.int64)

    def _read_layer_par(self, l, buf):
        """same bytes as _read_layer, but RAM-tier copies and file ranges (<= 16 experts each) run on READ_THREADS threads"""
        if not hasattr(self, 'rpool'):
            self.rpool = ThreadPoolExecutor(READ_THREADS)
            from diskio import Reader
            self.rreader = Reader(EXPERTS_BIN, nobuf=False, threads=1)
        base = self.host[buf].data_ptr()
        inram = getattr(self, 'inram', {}).get(l, {})
        iv = getattr(self, 'invram', {}).get(l, {})
        jobs, e = [], 0
        while e < 256:
            if e in iv:                        # still in the VRAM cache: used in place
                e += 1
                continue
            if e in inram:
                jobs.append((ctypes.memmove, (base + e * BLOB, inram[e], BLOB)))
                e += 1
                continue
            f = e
            while f < 256 and f not in inram and f not in iv and f - e < 16:
                f += 1
            jobs.append((self.rreader._read, (l * LAYER_BYTES + e * BLOB, (f - e) * BLOB, base + e * BLOB)))
            e = f
        for fu in [self.rpool.submit(fn_, *a) for fn_, a in jobs]:
            fu.result()
        return l

    def close(self):
        self.fd.close()

    def _read_layer(self, l, buf):
        if READ_THREADS > 1:
            return self._read_layer_par(l, buf)
        mv = memoryview(self.host[buf].numpy())
        inram = getattr(self, 'inram', {}).get(l, {})
        if not inram:
            self.fd.seek(l * LAYER_BYTES)
            n = 0
            while n < LAYER_BYTES:
                n += self.fd.readinto(mv[n:])
            return l
        # experts already in the engine's pinned RAM tier: copy from there; read only the runs of the others from the file
        base = self.host[buf].data_ptr()
        e = 0
        while e < 256:
            if e in inram:
                ctypes.memmove(base + e * BLOB, inram[e], BLOB)
                e += 1
                continue
            f = e
            while f < 256 and f not in inram:
                f += 1
            self.fd.seek(l * LAYER_BYTES + e * BLOB)
            n, end = e * BLOB, f * BLOB
            while n < end:
                n += self.fd.readinto(mv[n:end])
            e = f
        return l

    @torch.no_grad()
    def run(self, ids, p0=0, log=None, emb=None):
        """emb: optional (index LongTensor, rows FloatTensor [n, H]) replacing the input embeddings of ids[index]
        (image tokens get the vision encoder output instead of the token embedding)"""
        e = self.e
        self.emb = emb
        # borrow the pool: drop the decode expert cache
        for ev, items in e.vpend:
            ev.synchronize()
        e.vpend = []
        e.ensure_kv(p0 + len(ids))
        mem = e.prefill_mem(p0 + len(ids))
        if getattr(e, 'arena_mode', False) and KEEP_CACHE:
            # take only what this prompt needs (2 expert layers + activations of one chunk) right above the KV cache;
            # expert slots above that (they count down from the top of the arena) keep their experts for the decode
            n = len(ids)
            ch = min(4096, max(256, -(-n // 64) * 64))
            need = 2 * LAYER_BYTES + 64 * 2 ** 20 + PER_TOK * ch + 4 * 2 ** 20
            if need < mem.numel():
                mem = mem[:need]
            kv_end = e.kv_bytes(p0 + n)
            usable = (e.A - kv_end - mem.numel()) // BLOB          # slots vs < usable lie above the prefill buffers
            for vs in [v for v in e.vowner if usable <= v < 1 << 20]:     # (>= 1 << 20: slots in the output head block)
                key = e.vowner.pop(vs)
                e.vmap.pop(key, None)
                if hasattr(e, 'vloc_np'):                      # Ornith3: this expert is no longer in VRAM
                    e.vloc_np[key[0] * e.E + key[1]] = 0
            kept = set(e.vowner)
            e.vfree = [v for v in range(e.S_slots) if v not in kept] + [v for v in e.vfree if v >= 1 << 20]
            e.stats['prefill_kept'] = len(kept)
        else:
            e.vmap, e.vowner = {}, __import__('collections').OrderedDict()
            if hasattr(e, 'vloc_np'): e.vloc_np[:] = 0
            e.vfree = list(range(e.S_slots)) + (e.xslots() if hasattr(e, 'xslots') else [])
        self._layout(mem)
        t0 = time.time()
        for c in range(0, len(ids), self.CH):
            self._chunk(ids[c:c + self.CH], p0 + c, c)
            if log:
                log(f'prefill {min(c + self.CH, len(ids))}/{len(ids)} tokens ({time.time() - t0:.0f}s)')
        # leave the last hidden state in e.x and compute its logits
        T = len(ids) - (len(ids) - 1) // self.CH * self.CH
        e.x.copy_(self.X[T - 1])
        kfn('rmsnorm1p')((1,), (256,), (e.x.data_ptr(), e.norm.data_ptr(), e.h.data_ptr(), i32(e.H), f32(e.eps)), stream=stream())
        if hasattr(e, 'head_logits'):           # Ornith3: honours the language filter
            e.head_logits(assign=False)
        else:
            e.logits.zero_()
            rgemv('nvfp4', e.lm_p, e.lm_sp, e.lm_g, e.h, e.logits, e.V, e.H, 1)
        torch.cuda.synchronize()
        return e.logits

    def _chunk(self, ids, p0, c0=0):
        e, T, H, s = self.e, len(ids), self.e.H, stream()
        self.roff1[0], self.roff1[1] = 0, T
        X = self.X[:T]
        X.copy_(e.embed[torch.tensor(ids)].float())
        if getattr(self, 'emb', None) is not None:              # image tokens of this chunk: vision embeddings
            idx, rows = self.emb
            m = (idx >= c0) & (idx < c0 + T)
            if m.any():
                X[(idx[m] - c0).to(X.device)] = rows[m].to(X.device, torch.float32)
        main = torch.cuda.current_stream()
        self.inram = {}                        # layer -> {expert: pinned RAM address} (RAM tier is not touched during prefill)
        if getattr(e, 'rlru', None) and getattr(e, 'prefill_from_ram', True):
            for (ll, ee), rs in e.rlru.items():
                self.inram.setdefault(ll, {})[ee] = e.ram_addr[rs]
        self.invram = {}                       # layer -> {expert: VRAM cache address}: used in place, not read/copied
        if PF_VRAM and getattr(e, 'arena_mode', False):
            for (ll, ee), vs in e.vmap.items():
                self.invram.setdefault(ll, {})[ee] = e.vaddr(vs)
        h2d_ev = [None] * e.L                  # copy of layer l's experts into VRAM region l%2 finished
        moe_ev = [None] * e.L                  # layer l's expert GEMMs finished (region l%2 free again)

        def h2d(l):                            # pinned host buffer l%2 -> VRAM region l%2, on the copy stream
            if l >= 2:
                self.cstream.wait_event(moe_ev[l - 2])
            iv = self.invram.get(l)
            dst, src = self.mem.data_ptr() + (l % 2) * LAYER_BYTES, self.host[l % 2].data_ptr()
            if not iv:
                cp.cuda.runtime.memcpyAsync(dst, src, LAYER_BYTES, 1, self.cstream.cuda_stream)
            else:                              # only the runs of experts that are not in the VRAM cache
                x = 0
                while x < 256:
                    if x in iv:
                        x += 1; continue
                    y = x
                    while y < 256 and y not in iv:
                        y += 1
                    cp.cuda.runtime.memcpyAsync(dst + x * BLOB, src + x * BLOB, (y - x) * BLOB, 1, self.cstream.cuda_stream)
                    x = y
            ev = torch.cuda.Event(); ev.record(self.cstream); h2d_ev[l] = ev

        reads = {0: self.io.submit(self._read_layer, 0, 0)}
        reads[0].result(); h2d(0)
        if e.L > 1:
            reads[1] = self.io.submit(self._read_layer, 1, 1)
        for l, d in enumerate(e.layers):
            if l + 1 < e.L:                    # bring layer l+1 into VRAM while this layer computes
                reads[l + 1].result()
                h2d(l + 1)
            if l + 2 < e.L:                    # host buffer l%2 is free once the copy of layer l is done
                h2d_ev[l].synchronize()
                reads[l + 2] = self.io.submit(self._read_layer, l + 2, l % 2)
            pfn('rmsnorm1p_rows')((T,), (256,), (X.data_ptr(), d['ln1'].data_ptr(), self.HN.data_ptr(), i32(H), f32(e.eps)), stream=s)
            if d['kind'] == 'lin':
                i = d['idx']
                bgemm('fp8', d['qkv_p'], None, d['qkv_s'], self.roff1, self.HN, H, self.QKV, 8192, 8192, H, 1, T)
                bgemm('fp8', d['z_p'], None, d['z_s'], self.roff1, self.HN, H, self.Z, 4096, 4096, H, 1, T)
                bgemm('f32', d['ab_p'], None, None, self.roff1, self.HN, H, self.AB, 64, 64, H, 1, T)
                pfn('conv_seq')(((8192 + 127) // 128,), (128,), (self.QKV.data_ptr(), self.QKV2.data_ptr(), e.conv_state[i].data_ptr(),
                                                                   d['conv'].data_ptr(), i32(T), i32(8192)), stream=s)
                pfn('dn_qknorm')((T, 32), (128,), (self.QKV2.data_ptr(), i32(8192)), stream=s)
                pfn('dn_gates')((T,), (32,), (self.AB.data_ptr(), d['A_log'].data_ptr(), d['dt_bias'].data_ptr(), self.GB.data_ptr()), stream=s)
                pfn('dn_recur_seq')((32,), (128,), (self.QKV2.data_ptr(), self.GB.data_ptr(), e.S[i].data_ptr(), self.Z.data_ptr(),
                                                     d['normw'].data_ptr(), self.OL.data_ptr(), i32(T), f32(e.eps)), stream=s)
                bgemm('fp8', d['out_p'], None, d['out_s'], self.roff1, self.OL, 4096, X, H, H, 4096, 1, T, acc=1)
            else:
                i = d['idx']
                kv8 = int(e.kv8)
                bgemm('fp8', d['q_p'], None, d['q_s'], self.roff1, self.HN, H, self.QKV, 8192, 8192, H, 1, T)
                kvp = self.P(d['k'], d['v'])
                # k and v as two groups over the same rows would need two outputs; run them separately
                bgemm('fp8', kvp[0:1], None, d['kv_s'][0:1], self.roff1, self.HN, H, self.KVb, 1024, 512, H, 1, T)
                bgemm('fp8', kvp[1:2], None, d['kv_s'][1:2], self.roff1, self.HN, H, self.KVb[:, 512:], 1024, 512, H, 1, T)
                kc, vc, ks, vs = e.kvp[i]
                pfn('attn_prep_rows', KV8=kv8, **e.kvdefs)((T, 18), (256,), (self.QKV.data_ptr(), self.KVb.data_ptr(), d['qn'].data_ptr(), d['kn'].data_ptr(),
                                                                e.cos.data_ptr(), e.sin.data_ptr(), i32(p0), self.QR.data_ptr(), self.Z.data_ptr(),
                                                                kc, vc, ks, vs, i32(e.max_ctx), f32(e.eps), e.Kr[i].data_ptr(), e.Vr[i].data_ptr(), i32(e.RING),
                                                                e.rp.data_ptr()), stream=s)
                use_i8 = self.int8_attn and kv8 and p0 + T > e.WIN
                v12 = kv8 and not use_i8 and ATTN12           # 1.5x faster (pipelined K/V loads, 2 blocks per SM), same math
                pfn('attn_prefill_i8' if use_i8 else 'attn_prefill12' if v12 else 'attn_prefill7', KV8=kv8, **e.kvdefs)(
                    (((T + 3) // 4) if v12 else ((T + 7) // 8), 2), (256,), (self.QR.data_ptr(), self.Z.data_ptr(), i32(T), i32(p0),
                                                                     kc, vc, ks, vs,
                                                                     e.Kr[i].data_ptr(), e.Vr[i].data_ptr(), i32(e.max_ctx), i32(e.RING),
                                                                     i32(e.WIN if e.kv8 else 1 << 30), self.OL.data_ptr()), stream=s)
                bgemm('fp8', d['o_p'], None, d['o_s'], self.roff1, self.OL, 4096, X, H, H, 4096, 1, T, acc=1)
            # ---- MoE
            pfn('rmsnorm1p_rows')((T,), (256,), (X.data_ptr(), d['ln2'].data_ptr(), self.HN.data_ptr(), i32(H), f32(e.eps)), stream=s)
            bgemm('bf16', d['rw_p'], None, None, self.roff1, self.HN, H, self.RL, 257, 257, H, 1, T)
            pfn('router_rows')((T,), (256,), (self.RL.data_ptr(), i32(e.E), i32(e.K), f32(e.adaptive), self.ROUT.data_ptr(), self.RW.data_ptr()), stream=s)
            if CAPTURE is not None:            # training data for an expert predictor: routes of every layer, hidden state of some
                CAPTURE.setdefault(('r', l), []).append(self.ROUT[:T].to(torch.int16).cpu())
                if l in CAPTURE.get('layers', ()):
                    CAPTURE.setdefault(('h', l), []).append(self.HN[:T].half().cpu())
            # shared expert (all tokens), weighted by sigmoid(shared gate) = RW[:, 8]
            sb = d['shared'].data_ptr()
            sw = lambda key: torch.tensor([sb + OFF[key]], dtype=torch.int64, device=dev)
            g2 = e.shared_g2[l]
            bgemm('nvfp4', sw('gw'), sw('gs'), torch.tensor([g2[0]], device=dev), self.roff1, self.HN, H, self.SG, 512, 512, H, 1, T)
            bgemm('nvfp4', sw('uw'), sw('us'), torch.tensor([g2[1]], device=dev), self.roff1, self.HN, H, self.SU, 512, 512, H, 1, T)
            pfn('silu_mul_rows')(((T * 512 + 255) // 256,), (256,), (self.SG.data_ptr(), self.SU.data_ptr(), self.SH.data_ptr(), i32(T * 512)), stream=s)
            rs = self.RW[:T, 8].contiguous()
            bgemm('nvfp4', sw('dw'), sw('ds'), torch.tensor([g2[2]], device=dev), self.roff1, self.SH, 512, X, H, H, 512, 1, T, rscale=rs, acc=1)
            # routed experts: sort token slots by expert
            routes = self.ROUT[:T].cpu().numpy().ravel()
            wts = self.RW[:T, :8].cpu().numpy().ravel()
            valid = routes >= 0
            slot_tok = np.repeat(np.arange(T, dtype=np.int32), 8)[valid]
            ex, w = routes[valid], wts[valid]
            order = np.argsort(ex, kind='stable')
            ex, slot_tok, w = ex[order], slot_tok[order], w[order]
            counts = np.bincount(ex, minlength=256)
            roff = np.zeros(257, np.int32); roff[1:] = np.cumsum(counts)
            n = len(ex)
            self.xmap[:n].copy_(torch.from_numpy(slot_tok))
            self.rsc[:n].copy_(torch.from_numpy(w.astype(np.float32)))
            self.roff.copy_(torch.from_numpy(roff))
            maxc = int(counts.max())
            main.wait_event(h2d_ev[l])
            self.gs.copy_(torch.from_numpy(e.g2[l].T.copy()))
            r = l % 2
            iv = self.invram.get(l)
            if iv:                             # experts still in the VRAM cache: point the GEMMs at them
                tab = self.ptab[l]
                tab[:] = self.ptab_base[r]
                for x, a in iv.items():
                    for k, (wo, so) in enumerate((('gw', 'gs'), ('uw', 'us'), ('dw', 'ds'))):
                        tab[k, 0, x] = a + OFF[wo]; tab[k, 1, x] = a + OFF[so]
                self.wpl[r].copy_(self.ptab_t[l], non_blocking=True)
                wp = [self.wpl[r][k, 0] for k in range(3)]; sp_ = [self.wpl[r][k, 1] for k in range(3)]
            else:
                wp = [self.wp[r, k] for k in range(3)]; sp_ = [self.sp[r, k] for k in range(3)]
            bgemm('nvfp4', wp[0], sp_[0], self.gs[0], self.roff, self.HN, H, self.G1, 512, 512, H, 256, maxc, xmap=self.xmap)
            bgemm('nvfp4', wp[1], sp_[1], self.gs[1], self.roff, self.HN, H, self.U1, 512, 512, H, 256, maxc, xmap=self.xmap)
            pfn('silu_mul_rows')(((n * 512 + 255) // 256,), (256,), (self.G1.data_ptr(), self.U1.data_ptr(), self.H1.data_ptr(), i32(n * 512)), stream=s)
            bgemm('nvfp4', wp[2], sp_[2], self.gs[2], self.roff, self.H1, 512, X, H, H, 512, 256, maxc,
                  cmap=self.xmap, rscale=self.rsc, acc=2)
            ev = torch.cuda.Event(); ev.record(main); moe_ev[l] = ev
