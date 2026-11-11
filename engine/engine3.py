"""Ornith3 = Ornith2 + overlap (night 16./17.10.).

After router(l) the GPU immediately computes the experts that already sit in VRAM (router2 takes their pointers from
`vloc`, a pinned GPU-visible table [layer*E + expert] -> VRAM blob address or 0) plus the shared expert (MoE_A),
while the CPU resolves the remaining experts (RAM tier / SSD) for MoE_B, which runs at the start of the next segment.
Needs copy-on-read (no DMA promotion), arena mode.
"""
import time, os, os as _os
import numpy as np
import torch
from engine2 import Ornith2, dev
from kern import fn, rgemv, stream, i32, f32, i64


DMA2 = _os.environ.get('DMA2', '0') == '1'
HITSTAT = _os.environ.get('HITSTAT', '0') == '1'


import cupy as _cp
MH8 = _cp.RawKernel(r'''extern "C" __global__ void mh8(const long long* r, float* out) {
  int t = threadIdx.x; out[t] = 0.f; __syncthreads();
  if (t < 8) { long long e = r[t]; if (e >= 0 && e < 256) out[e] = 1.f; } }''', 'mh8')


def _load_pred(path):
    """optional learned 2-layer-ahead predictors (only with env PRED; off by default): {input layer: (W1 [2816,128], b1, W2 [128,256], b2, a)} on the GPU"""
    import glob, re
    nets = {}
    for f in sorted(glob.glob(os.path.join(path, 'pred_S_l*_t*.pt'))):
        ck = torch.load(f)
        l = ck['layer']
        nets[l] = (ck['f']['0.weight'].T.contiguous().to(dev), ck['f']['0.bias'].to(dev), ck['f']['2.weight'].T.contiguous().to(dev),
                   ck['f']['2.bias'].to(dev), float(ck['a']))
    return nets


class Ornith3(Ornith2):
    def __init__(self, **kw):
        kw['cor'] = True
        self.pred_nets = _load_pred(_os.environ['PRED']) if _os.environ.get('PRED') else {}
        self.pred_lam = float(_os.environ.get('PRED_LAM', 0.25))
        self.skip_b = float(_os.environ.get('SKIPB', 0.0))   # lossy: skip non-VRAM experts below this routed weight (set_quality)
        self.mh_cur, self.mh_prev = torch.zeros(256, device=dev), torch.zeros(256, device=dev)
        warm_cache = kw.pop('warm_cache', None)
        super().__init__(**kw)
        self.ov = True
        self.exclusive = __import__('os').environ.get('EXCL', '1' if kw.get('ram_gb', 4.0) < 10 else '0') == '1'   # measured: +5 % at 3-8 GB pin, slightly worse at 12
        L, E, K1 = self.L, self.E, self.K + 1
        self.vloc = torch.zeros(L * E, dtype=torch.int64).pin_memory()
        self.vloc_np = self.vloc.numpy()
        self.g2tab = torch.from_numpy(self.g2.reshape(-1).copy()).to(dev)
        self.g2sh = torch.from_numpy(np.stack(self.shared_g2)).to(dev)
        z = lambda n, dt=torch.int64: torch.zeros(n, dtype=dt, device=dev)
        self.A_gu_w, self.A_gu_s, self.A_gu_g = z(2 * K1), z(2 * K1), z(2 * K1, torch.float32)
        self.A_dn_w, self.A_dn_s, self.A_dn_g = z(K1), z(K1), z(K1, torch.float32)
        self.A_w, self.A_cnt = z(K1, torch.float32), z(2, torch.int32)
        self.routesB = torch.full((L, 8), -1, dtype=torch.int64).pin_memory()
        self.wB = torch.zeros(L, 8).pin_memory()
        import os
        self.onegraph = os.environ.get('ONEGRAPH', '0') == '1'   # measured: no gain (26.6 vs 26.8)   # one graph per layer, CPU polls a flag written by router2
        self.tokseq = torch.zeros(1, dtype=torch.int64).pin_memory(); self.tokseq_np = self.tokseq.numpy()
        self.flag = torch.zeros(L, dtype=torch.int64).pin_memory(); self.flag_np = self.flag.numpy()
        self.seq = 0
        self.evg_on = os.environ.get('EVG', '0') == '1'   # measured: no gain (29.71-29.75 both ways)   # one graph per layer with an event node after the router (41 launches/token)
        self.evg = [torch.cuda.Event(external=True) for _ in range(L)]
        self.pf2 = int(os.environ.get('PF2', '8' if kw.get('ram_gb', 4.0) >= 10 else '0'))   # 2 layers ahead (router l+2 on h_l): +1 % at 12 GB, -3 % at 3 GB
        self.pred2_host = torch.full((L + 2, 16), -1, dtype=torch.int64).pin_memory()
        self.rlog3 = torch.zeros(E + 1, device=dev)
        if warm_cache if warm_cache is not None else os.environ.get('WARM', '0') == '1':   # read experts.bin once in the background (cold OS cache)
            import threading
            def warm():
                from pack import OUT as path
                with open(path, 'rb', buffering=0) as f:
                    while f.read(64 << 20):
                        pass
            threading.Thread(target=warm, daemon=True).start()
        self.defer_unused = os.environ.get('DEFER', '1' if kw.get('ram_gb', 4.0) >= 6 else '0') == '1'   # +2 % at 12 GB, neutral at 3 GB
        self.late = []
        from diskio import Reader
        from pack import OUT as EXPERTS_BIN
        self.split_reads = int(os.environ.get('SPLIT', '1'))   # measured: no gain
        if self.split_reads > 1:                  # own pool: blocking misses must not queue behind prefetch reads
            self.reader_fast = Reader(EXPERTS_BIN, nobuf=False, threads=self.split_reads)
        import os
        self.dmapf = int(os.environ.get('DMAPF', '5' if kw.get('ram_gb', 4.0) >= 10 else '0'))   # DMA-ahead 1 layer: +5-7 % at 13.5 GB, +-0 at 3 GB (18.10.)
        self.dma_ev = [None] * (L + 1)
        self.dma_busy = []
        self._dma_keys = []
        self.dmaph = os.environ.get('DMAPH', '1') == '1'     # copies start when the line is idle (after the zero-copy MoE)
        self.dma2k = int(os.environ.get('DMA2K', '0'))
        self.cstream2 = torch.cuda.Stream(priority=0)
        self.dma_ev2 = [None] * (L + 1)
        self.evph = [torch.cuda.Event(external=True) for _ in range(L + 1)]
        self.tl = os.environ.get('TL', '0') == '1'
        self._tl = []
        self.ram_gb = kw.get('ram_gb', 4.0)
        S = self.stage_np                         # per-layer row views + python copies for the per-layer CPU work
        self._rows = [tuple(S[n][l] for n in ('gu_w', 'gu_s', 'dn_w', 'dn_s', 'gu_wd', 'gu_sd', 'dn_wd', 'dn_sd', 'gu_g', 'dn_g', 'wb', 'counts'))
                      for l in range(L)]
        self._offs = tuple(int(v) for v in self.offs)
        self._g2l = self.g2.tolist()
        self._wB_np = self.wB.numpy()
        self._routes_np, self._routesB_np, self._pred_np, self._pred2_np = (self.routes_host.numpy(), self.routesB.numpy(),
                                                                             self.pred_host.numpy(), self.pred2_host.numpy())
        from pack import BLOB
        c, self.ram_chunks = 0, []                # (first slot, slots, pinned tensor) of every RAM-tier chunk
        for t in self.ram:
            n = t.numel() // BLOB
            self.ram_chunks.append((c, n, t)); c += n

    def _ram_defaults(self, gb):
        """measured best settings per pin size (env variables override): measured on a GTX 1070"""
        import os
        self.exclusive = os.environ.get('EXCL', '1' if gb < 10 else '0') == '1'
        self.defer_unused = os.environ.get('DEFER', '1' if gb >= 6 else '0') == '1'
        if hasattr(self, 'dmapf'):                # DMA-ahead follows the pin size too (the RAM button switches at runtime)
            self.dmapf = int(os.environ.get('DMAPF', '5' if gb >= 10 else '0'))
        pf2 = int(os.environ.get('PF2', '8' if gb >= 10 else '0'))
        if pf2 != self.pf2:                       # pf2 is part of the recorded graphs
            self.pf2 = pf2
            self.graphs.clear()

    def set_ram_gb(self, gb):
        """Resize the pinned RAM tier at runtime (call between answers, from the CUDA thread). Returns the new size in GB.
        Shrinking frees whole chunks (~430 MB) above the new size; growing pins new chunks (retries, stops early if Windows
        refuses). Experts that lose their RAM slot simply come from the file again when needed."""
        import cupy as cp
        from pack import BLOB
        from engine2 import pinned, _PINNED
        torch.cuda.synchronize()
        self.cstream.synchronize()
        for key, (fu, rs) in list(self.rpending.items()):   # finish background reads
            fu.result()
            self.rlru[key] = rs
        self.rpending.clear()
        self._settle_promotions()
        self.rbusy.clear()
        newR = max(64, int(gb * 2 ** 30 // BLOB))
        if newR < self.R:
            for key in [k for k, rs in self.rlru.items() if rs >= newR]:
                del self.rlru[key]
            self.rfree = [rs for rs in self.rfree if rs < newR]
            keep = []
            for st, n, t in self.ram_chunks:
                if st >= newR:                    # chunk entirely above the new size: unpin it
                    p = t.data_ptr()
                    del t
                    if p in _PINNED:
                        _PINNED.remove(p)
                    cp.cuda.runtime.freeHost(p)
                else:
                    keep.append((st, n, t))
            self.ram_chunks = keep
            self.ram = [t for _, _, t in keep]
            self.ram_addr = self.ram_addr[:newR]
            self.R = newR
        elif newR > self.R:
            c, tries = self.R, 0
            while c < newR:
                n = min(256, newR - c)
                try:
                    t = pinned(n * BLOB)
                except Exception:
                    tries += 1
                    if tries <= 5:
                        time.sleep(1.0)
                        continue
                    break                         # Windows refuses more pinned memory: keep what we have
                self.ram_chunks.append((c, n, t)); self.ram.append(t)
                self.ram_addr += [t.data_ptr() + i * BLOB for i in range(n)]
                self.rfree += list(range(c, c + n))
                c += n
            self.R = c
        self.ram_gb = self.R * BLOB / 2 ** 30
        self._ram_defaults(self.ram_gb)
        return self.ram_gb

    # ------------------------------------------------------------------ hot set (fast start)
    def save_hot(self, path):
        """remember which experts sit in VRAM and in the RAM tier (LRU order, oldest first)"""
        import json
        json.dump({'vram': [list(k) for k in self.vowner.values()], 'ram': [list(k) for k in self.rlru]}, open(path, 'w'))

    def load_hot(self, path):
        """refill RAM tier and VRAM cache from a saved hot set (reads sorted by file offset, in parallel). Returns seconds."""
        import json, cupy as cp
        from pack import BLOB
        from diskio import submit_many
        t0 = time.time()
        try:
            d = json.load(open(path))
        except Exception:
            return 0.0
        E = self.E
        self._settle_promotions()
        want_ram = [tuple(k) for k in d.get('ram', [])][-max(0, self.R - 64):]     # most recent ones, keep a few free slots
        want_v = [tuple(k) for k in d.get('vram', [])][-len(self.vfree):]
        need = [k for k in dict.fromkeys(want_ram + want_v) if k not in self.rlru and k not in self.rpending]
        jobs = []
        for k in sorted(need, key=lambda k: k[0] * E + k[1]):
            if not self.rfree:
                break
            jobs.append((k, self.rfree.pop()))
        rq = [((k[0] * E + k[1]) * BLOB, BLOB, self.ram_addr[s]) for k, s in jobs]
        for f in (self.reader.psubmit(rq, 0) if hasattr(self.reader, 'psubmit') else submit_many(self.reader, rq)):
            f.result()
        order = {k: i for i, k in enumerate(want_ram)}
        for k, s in sorted(jobs, key=lambda j: order.get(j[0], -1)):
            self.rlru[k] = s
            self.rlru.move_to_end(k)
        n = 0
        for k in want_v:                          # VRAM: copy from the RAM tier
            if k in self.vmap or k not in self.rlru or not self.vfree:
                continue
            vs = self.vfree.pop()
            cp.cuda.runtime.memcpyAsync(self.vaddr(vs), self.ram_addr[self.rlru[k]], BLOB, 1, self.cstream.cuda_stream)
            self.vmap[k] = vs; self.vowner[vs] = k
            self.vloc_np[k[0] * E + k[1]] = self.vaddr(vs)
            n += 1
        self.cstream.synchronize()
        self.stats['hot_ram'], self.stats['hot_vram'] = len(jobs), n
        return time.time() - t0

    def _vloc_rebuild(self):
        self.vloc_np[:] = 0
        for (l, e), vs in self.vmap.items():
            self.vloc_np[l * self.E + e] = self.vaddr(vs)

    def ensure_kv(self, ntok):
        g = self.stats['kv_grow']
        super().ensure_kv(ntok)
        if self.stats['kv_grow'] != g:
            self._vloc_rebuild()

    def _router2(self, l, d, common, s):
        if self.pf2 and l + 2 < self.L and l not in self.pred_nets:
            d2 = self.layers[l + 2]
            rgemv('bf16', d2['rw_p'], None, None, self.h, self.rlog3, self.E + 1, self.H, 1, assign=True)
            fn('router')((1,), (256,), (self.rlog3.data_ptr(), d2['sgate'].data_ptr(), self.h.data_ptr(), self.loc.data_ptr(),
                                        i64(0), i32(l + 2), i32(self.E), i32(self.H), i32(self.pf2), f32(0.0),
                                        self.dummy_w.data_ptr(), self.pred2_host[l + 2].data_ptr(), self.dummy_c.data_ptr(),
                                        i64(0), i64(0), i64(0), i64(0), i64(0), i64(0)) + common, stream=s)
        if self.prefetch and l + 1 < self.L:   # prediction for layer l+1 first, so the flag written by router2 comes last
            dn = self.layers[l + 1]
            rgemv('bf16', dn['rw_p'], None, None, self.h, self.rlog2, self.E + 1, self.H, 1, assign=True)
            fn('router')((1,), (256,), (self.rlog2.data_ptr(), dn['sgate'].data_ptr(), self.h.data_ptr(), self.loc.data_ptr(),
                                        i64(0), i32(l + 1), i32(self.E), i32(self.H), i32(self.prefetch), f32(0.0),
                                        self.dummy_w.data_ptr(), self.pred_host[l + 1].data_ptr(), self.dummy_c.data_ptr(),
                                        i64(0), i64(0), i64(0), i64(0), i64(0), i64(0)) + common, stream=s)
        fn('router2')((1,), (256,), (self.rlog.data_ptr(), d['sgate'].data_ptr(), self.h.data_ptr(), self.vloc.data_ptr(),
                                     i64(d['shared'].data_ptr()), i32(l), i32(self.E), i32(self.H), i32(self.K), f32(self.adaptive),
                                     self.A_w.data_ptr(), self.routes_host[l].data_ptr(), self.A_cnt.data_ptr(),
                                     self.A_gu_w.data_ptr(), self.A_gu_s.data_ptr(), self.A_gu_g.data_ptr(),
                                     self.A_dn_w.data_ptr(), self.A_dn_s.data_ptr(), self.A_dn_g.data_ptr()) + common
                      + (self.g2tab.data_ptr(), self.g2sh[l].data_ptr(), self.routesB[l].data_ptr(), self.wB[l].data_ptr(),
                         self.tokseq.data_ptr() if self.onegraph else 0, self.flag.data_ptr() if self.onegraph else 0,
                         f32(getattr(self, 'skip_b', 0.0))), stream=s)
        if self.pf2 and l + 2 < self.L and l in self.pred_nets:
            self._predict2(l, common, s)

    def _predict2(self, l, common, s):
        """learned prediction of layer l+2 (PREDICTOR.md): logits = a*router(l+2)(h) + lam * f([h, mh(routes l), mh(routes l-1), 0.1*router])"""
        d2 = self.layers[l + 2]
        rgemv('bf16', d2['rw_p'], None, None, self.h, self.rlog3, self.E + 1, self.H, 1, assign=True)
        MH8((1,), (256,), (self.routes_host[l].data_ptr(), self.mh_cur.data_ptr()), stream=s)
        MH8((1,), (256,), (self.routes_host[l - 1].data_ptr(), self.mh_prev.data_ptr()), stream=s)
        W1, b1, W2, b2, a = self.pred_nets[l]
        rl = self.rlog3[:256]
        x = torch.cat([self.h, self.mh_cur, self.mh_prev, rl * 0.1])
        f = torch.nn.functional.gelu(x @ W1 + b1) @ W2 + b2
        self.rlog3[:256] = a * rl + self.pred_lam * f
        fn('router')((1,), (256,), (self.rlog3.data_ptr(), d2['sgate'].data_ptr(), self.h.data_ptr(), self.loc.data_ptr(),
                                    i64(0), i32(l + 2), i32(self.E), i32(self.H), i32(self.pf2), f32(0.0),
                                    self.dummy_w.data_ptr(), self.pred2_host[l + 2].data_ptr(), self.dummy_c.data_ptr(),
                                    i64(0), i64(0), i64(0), i64(0), i64(0), i64(0)) + common, stream=s)

    def _cor_slot(self):
        if self.vfree:
            return self.vfree.pop()
        for vs in self.vowner:
            if vs not in self.tok_vs:
                old = self.vowner.pop(vs)
                self.vmap.pop(old, None)
                self.vloc_np[old[0] * self.E + old[1]] = 0
                self.stats['cor_evict'] += 1
                return vs
        return None

    def _dma_ahead(self, l, pred, k=None, stream=None, phase=None, tag='', phase_l=None):
        """copy predicted experts of layer l that sit only in pinned RAM into VRAM slots (DMA overlaps the kernels)"""
        import cupy as cp
        from pack import BLOB
        n = 0
        cs = stream or self.cstream
        ph = self.dmaph if phase is None else phase
        for e in pred[:(k or self.dmapf)]:
            key = (l, e)
            if key in self.vmap or key not in self.rlru:
                continue
            vs = self._cor_slot()
            if vs is None:
                break
            rs = self.rlru[key]
            self.rbusy.add(rs); self.dma_busy.append(rs)
            self.tok_vs.add(vs)
            if n == 0 and ph:                     # start the copies when layer l's (or phase_l's) attention starts (line idle)
                cs.wait_event(self.evph[l if phase_l is None else phase_l])
            cp.cuda.runtime.memcpyAsync(self.vaddr(vs), self.ram_addr[rs], BLOB, 1, cs.cuda_stream)
            self.vmap[key] = vs
            self.vowner[vs] = key
            self.vloc_np[l * self.E + e] = self.vaddr(vs)
            self._dma_keys.append(key)
            n += 1
        if n:
            ev = torch.cuda.Event()
            ev.record(cs)
            if self.dma_ev[l] is not None and cs is not self.cstream:   # layer l already has copies on the other stream: wait for both
                self.dma_ev2[l] = ev
            else:
                self.dma_ev[l] = ev
            self.stats['dma_ahead' + tag] += n

    def _moe_A(self, l):
        H, I, k1, s = self.H, self.I, self.K + 1, stream()
        self.gu.zero_()
        rgemv('nvfp4', self.A_gu_w, self.A_gu_s, self.A_gu_g, self.h, self.gu, I, H, 2 * k1, xdiv=2 * k1, ystride=I, mcount=self.A_cnt[0:1])
        fn('silu_mul')(((k1 * I + 255) // 256,), (256,), (self.gu.data_ptr(), self.hb.data_ptr(), self.A_cnt.data_ptr(), i32(I)), stream=s)
        rgemv('nvfp4', self.A_dn_w, self.A_dn_s, self.A_dn_g, self.hb, self.x, H, I, k1, xdiv=1, xstride=I, ydiv=k1,
              wts=self.A_w, mcount=self.A_cnt[1:2])

    def _moe_B(self, l):
        H, I, k1, s = self.H, self.I, self.K + 1, stream()
        fn('cpyn')((1,), (128,), (self.stage_host[l].data_ptr(), self.stage_dev.data_ptr(), i32(self.stage_bytes // 8)), stream=s)
        self.gu.zero_()
        rgemv('nvfp4', self.gu_w, self.gu_s, self.gu_g, self.h, self.gu, I, H, 2 * k1, xdiv=2 * k1, ystride=I, mcount=self.counts[0:1],
              wdst=self.gu_wd, sdst=self.gu_sd)
        fn('silu_mul')(((k1 * I + 255) // 256,), (256,), (self.gu.data_ptr(), self.hb.data_ptr(), self.counts.data_ptr(), i32(I)), stream=s)
        rgemv('nvfp4', self.dn_w, self.dn_s, self.dn_g, self.hb, self.x, H, I, k1, xdiv=1, xstride=I, ydiv=k1,
              wts=self.wb, mcount=self.counts[1:2], wdst=self.dn_wd, sdst=self.dn_sd)

    def _segment(self, i, ns, part=1):
        if part == 2:
            self._moe_A(i)
            return
        if part == 7:                             # evg: one graph per layer, event node right after the router
            if i > 0:
                self._moe_B(i - 1)
                if i - 1 in self.FEAT_LAYERS:
                    self.feat[self.FEAT_LAYERS.index(i - 1)].copy_(self.x)
            if i < self.L:
                self._attn(i, ns)
                self.evg[i].record()
                self._moe_A(i)
            else:
                self._final()
            return
        if part == 3:                             # onegraph: everything of layer i in one graph
            self._segment(i, ns, 1)
            if i < self.L:
                self._moe_A(i)
            return
        if i > 0:
            self._moe_B(i - 1)
            if i - 1 in self.FEAT_LAYERS:
                self.feat[self.FEAT_LAYERS.index(i - 1)].copy_(self.x)
        if i < self.L:
            if self.dmaph:                        # phase marker: the zero-copy MoE of layer i-1 is over, the PCIe line is idle now
                self.evph[i].record()
            self._attn(i, ns)
        else:
            self._final()

    def _replay(self, i, ns, part=1):
        key = (i, ns, part)
        g = self.graphs.get(key)
        if g is None:
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            snap = [t.clone() for t in self._state()]
            with torch.cuda.stream(st):
                self._segment(i, ns, part)
            torch.cuda.current_stream().wait_stream(st)
            for t, v in zip(self._state(), snap):
                t.copy_(v)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._segment(i, ns, part)
            self.graphs[key] = g
        g.replay()

    def _set_ptrs_B(self, l, ex, exB):
        """VRAM experts (done by MoE_A) only get their LRU touch; the rest are resolved and staged for MoE_B"""
        sb = set(exB)
        for e in ex:                              # first: mark every VRAM slot this layer uses (never a cor victim)
            if e not in sb:
                vs = self.vmap.get((l, e))
                if vs is not None:
                    self.tok_vs.add(vs)
                    if vs in self.vowner:
                        self.vowner.move_to_end(vs)
                self.stats['vram'] += 1
        self.freq[l, ex] += 1.0
        protect, addrs, dsts = set(), [], []
        for e in exB:
            a, sl = self._resolve(l, e, protect)
            d = 0
            if sl is not None:
                protect.add(sl)
                vs = self._cor_slot()
                if vs is not None:
                    self.tok_vs.add(vs)
                    self.cor_pend.append(((l, e), vs))
                    d = self.vaddr(vs)
            addrs.append(a); dsts.append(d)
        k1 = len(exB)
        rows = self._rows[l]
        if k1:                                    # plain element writes: for 1-3 experts much cheaper than numpy array ops
            gw, gs, dw, dsr, gwd, gsd, dwd, dsd, gg, dg, wb, _ = rows
            ogw, ouw, ogs, ous, odw, ods = self._offs
            g2l, wBl = self._g2l[l], self._wB_np[l]
            for j in range(k1):
                a, d, (g0, g1, g2_) = addrs[j], dsts[j], g2l[exB[j]]
                gw[2 * j] = a + ogw; gw[2 * j + 1] = a + ouw; gs[2 * j] = a + ogs; gs[2 * j + 1] = a + ous
                dw[j] = a + odw; dsr[j] = a + ods
                if d:
                    gwd[2 * j] = d + ogw; gwd[2 * j + 1] = d + ouw; gsd[2 * j] = d + ogs; gsd[2 * j + 1] = d + ous
                    dwd[j] = d + odw; dsd[j] = d + ods
                else:
                    gwd[2 * j] = gwd[2 * j + 1] = gsd[2 * j] = gsd[2 * j + 1] = dwd[j] = dsd[j] = 0
                gg[2 * j] = g0; gg[2 * j + 1] = g1; dg[j] = g2_
                wb[j] = wBl[j]
        c = rows[11]
        c[0] = 2 * k1; c[1] = k1
        return protect

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
        st = torch.cuda.current_stream()          # once per token (each current_stream() call costs ~25 us)
        if self.onegraph:
            self.seq += 1
            self.tokseq_np[0] = self.seq
        st = torch.cuda.current_stream()
        if self.evg_on and not self.onegraph and not self.dmapf:
            self._replay(0, ns, 7)
            for i in range(self.L):
                self.evg[i].synchronize()         # router(i) done; MoE_A(i) is running
                ex, protect = self._after_router_a(i)
                self._replay(i + 1, ns, 7)
                self._after_router_b(i, ex, protect, used)
            self._finish_token(used)
            return self.logits
        for i in range(self.L + 1):
            if self.onegraph:
                self._replay(i, ns, 3)
                if i == self.L:
                    break
                st.query()                        # WDDM: make sure the launch is submitted before spinning
                fl, sq, t0 = self.flag_np, self.seq, time.perf_counter()
                while fl[i] != sq:
                    time.sleep(0)                 # give the GIL to the reader threads
                    if time.perf_counter() - t0 > 5:
                        raise RuntimeError(f'router flag of layer {i} never arrived')
                self._after_router(i, used)
                continue
            if i == 0:
                self._replay(0, ns, 1)
            if i == self.L:
                break
            ev.record(st)
            if self.dma_ev2[i] is not None:       # second (speculative, 2-ahead) copy stream
                torch.cuda.current_stream().wait_event(self.dma_ev2[i]); self.dma_ev2[i] = None
            if self.dma_ev[i] is not None:        # experts of layer i copied ahead must have landed before MoE_A(i)
                self.stats['dma_ready' if self.dma_ev[i].query() else 'dma_late'] += 1   # diagnose: copy already done?
                if self.tl:
                    ta, tb = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    ta.record(st)
                torch.cuda.current_stream().wait_event(self.dma_ev[i])
                if self.tl:
                    tb.record(st); self._tl.append((ta, tb))
                self.dma_ev[i] = None
            self._replay(i, ns, 2)                # MoE_A(i) runs while the CPU resolves MoE_B(i)
            ev.synchronize()
            ex, protect = self._after_router_a(i)
            self._replay(i + 1, ns, 1)            # launch the next segment first, bookkeeping/prefetch while it runs
            self._after_router_b(i, ex, protect, used)
        self._finish_token(used)
        return self.logits

    def _after_router(self, i, used):
        ex, protect = self._after_router_a(i)
        self._after_router_b(i, ex, protect, used)

    def _after_router_a(self, i):
        ex = [e for e in self._routes_np[i].tolist() if e >= 0]
        exB = [e for e in self._routesB_np[i].tolist() if e >= 0]
        return ex, self._set_ptrs_B(i, ex, exB)

    def _after_router_b(self, i, ex, protect, used):
        if True:
            for key in [k for k in self.rpending if k[0] == i]:
                fu, rs = self.rpending.pop(key)
                if self.defer_unused:             # unused prediction: don't wait for its read now, settle after the token
                    self.late.append((key, fu, rs))
                    self.rbusy.add(rs)
                else:
                    fu.result()
                    self.rlru[key] = rs
                    self.rlru.move_to_end(key, last=False)
                self.stats['mispredicted'] += 1
            used.update((i, e) for e in ex)
            if self.prefetch and i + 1 < self.L:
                pred = [e for e in self._pred_np[i + 1].tolist() if e >= 0]
                if self.dmapf and not DMA2:
                    self._dma_ahead(i + 1, pred)
                self._prefetch(i + 1, pred, protect)
            if self.pf2 and i + 2 < self.L:
                p2 = [e for e in self._pred2_np[i + 2].tolist() if e >= 0]
                if self.dmapf and DMA2:           # copy 2 layers ahead: start when layer i+1's attention starts (~1.5 ms head start)
                    self._dma_ahead(i + 2, p2, phase_l=i + 1)
                elif self.dma2k:                  # extra speculative copies 2 layers ahead, own low-priority stream, no phase wait
                    self._dma_ahead(i + 2, p2, k=self.dma2k, stream=self.cstream2, phase=False, tag='2')
                self._prefetch(i + 2, p2, protect)

    def _finish_token(self, used):
        torch.cuda.current_stream().synchronize()
        if HITSTAT:                               # diagnose: live hit rate of the 2-ahead prediction (top-8) for layers 3..39
            for l in range(3, self.L):
                self.stats['hit2'] += len(set(self._pred2_np[l][:8].tolist()) & set(self._routes_np[l].tolist()))
                self.stats['hit2_n'] += 8
        if self._tl:                              # timeline: GPU time spent waiting for pre-copies
            self.stats['dma_wait_ms'] += sum(a.elapsed_time(b) for a, b in self._tl)
            self._tl = []
        if getattr(self, '_dma_keys', None):      # diagnose: how many pre-copied experts were really used
            self.stats['dma_used'] += sum(1 for k in self._dma_keys if k in used)
        self._dma_keys = []
        for key, fu, rs in self.late:
            fu.result()
            self.rbusy.discard(rs)
            if key not in self.rlru:
                self.rlru[key] = rs
                self.rlru.move_to_end(key, last=False)
            else:                                 # read again meanwhile (blocking miss): give the slot back
                self.rfree.append(rs)
        self.late = []
        for rs in self.dma_busy:
            self.rbusy.discard(rs)
        self.dma_busy = []
        for key, vs in self.cor_pend:             # copy-on-read inserts are complete now
            self.vmap[key] = vs
            self.vowner[vs] = key
            self.vloc_np[key[0] * self.E + key[1]] = self.vaddr(vs)
            self.stats['cor_insert'] += 1
            if self.exclusive and key in self.rlru:     # now in VRAM: its RAM copy is the first to be reused
                self.rlru.move_to_end(key, last=False)
        self.cor_pend = []
        self._promote(used)

    # ------------------------------------------------------------------ language filter (PLAN_SPRACHFILTER.md)
    # The output head only scores the tokens of the active script groups (latin always, cjk / other switchable).
    # The input side stays complete (pasted Chinese is still understood). The head lives in one VRAM block; the part a
    # smaller head leaves free holds extra expert slots (slot numbers >= XS). Switch between answers, on the CUDA thread.
    XS = 1 << 20

    def vaddr(self, vs):
        if vs >= self.XS:
            return self.lm_base + (vs - self.XS) * self._blob
        return super().vaddr(vs)

    def set_langs(self, cjk=True, other=True):
        """returns dict(tokens=active output tokens, slots=extra expert slots, mb=freed head MB)"""
        import json
        from pack import BLOB
        want = (bool(cjk), bool(other))
        if want == getattr(self, 'langs', (True, True)):
            return self.lang_info()
        torch.cuda.synchronize(); self.cstream.synchronize()
        self._settle_promotions()
        if not hasattr(self, 'lmbuf'):            # first switch: the full head goes to RAM, one VRAM block takes its place
            self._blob, self.lm_rb = BLOB, (self.lm_w.shape[1], self.lm_s.shape[1])
            self.lm_cpu = (self.lm_w.cpu(), self.lm_s.cpu())
            self.lm_w = self.lm_s = None
            torch.cuda.empty_cache()
            self.lmbuf = torch.empty(self.V * sum(self.lm_rb), dtype=torch.uint8, device=dev)
            g = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lang_groups.json')))['groups']
            self.lang_ids = {k: torch.tensor(v, dtype=torch.int64) for k, v in g.items()}
        rw, rs = self.lm_rb
        for vs in [v for v in self.vowner if v >= self.XS]:     # expert slots inside the head block go first
            key = self.vowner.pop(vs)
            self.vmap.pop(key, None)
            self.vloc_np[key[0] * self.E + key[1]] = 0
        self.vfree = [v for v in self.vfree if v < self.XS]
        w, s = self.lm_cpu
        if want == (True, True):
            ids, n = None, self.V
        else:
            ids = torch.cat([self.lang_ids['latin']] + [self.lang_ids[k] for k, on in zip(('cjk', 'other'), want) if on]).sort()[0]
            n = -(-ids.numel() // 256) * 256      # pad rows to a multiple of 256 (zero rows, never scattered back)
            w, s = torch.zeros(n, rw, dtype=torch.uint8), torch.zeros(n, rs, dtype=torch.uint8)
            w[:ids.numel()], s[:ids.numel()] = self.lm_cpu[0][ids], self.lm_cpu[1][ids]
        self.lmbuf[:n * rw].copy_(w.reshape(-1))
        self.lmbuf[n * rw:n * (rw + rs)].copy_(s.reshape(-1))
        P = lambda t: torch.tensor([t.data_ptr()], dtype=torch.int64, device=dev)
        self.lm_p, self.lm_sp = P(self.lmbuf[:n * rw]), P(self.lmbuf[n * rw:])
        self.lm_n = n
        self.lm_idx = None if ids is None else ids.to(dev)
        self.clog = None if ids is None else torch.zeros(n, device=dev)
        x0 = -(-n * (rw + rs) // 256) * 256
        self.lm_base = self.lmbuf.data_ptr() + x0
        nx = (self.lmbuf.numel() - x0) // BLOB
        self.vfree = [self.XS + i for i in range(nx)] + self.vfree
        self.graphs.clear()
        self.langs = want
        self.head_logits()                        # self.h still holds the last position: its logits under the new filter
        torch.cuda.synchronize()
        return self.lang_info()

    def xslots(self):
        """all expert slot numbers inside the head block"""
        if not hasattr(self, 'lmbuf'):
            return []
        return [self.XS + i for i in range((self.lmbuf.numel() - (self.lm_base - self.lmbuf.data_ptr())) // self._blob)]

    def lang_info(self):
        n = getattr(self, 'lm_idx', None)
        n = self.V if n is None else n.numel()
        nx = 0 if not hasattr(self, 'lmbuf') else (self.lmbuf.numel() - (self.lm_base - self.lmbuf.data_ptr())) // self._blob
        return dict(langs=dict(zip(('cjk', 'other'), getattr(self, 'langs', (True, True)))), tokens=n, slots=nx,
                    mb=round(nx * getattr(self, '_blob', 0) / 2 ** 20))

    def head_logits(self, assign=True):
        """output head on self.h -> self.logits (inactive tokens -inf)"""
        idx = getattr(self, 'lm_idx', None)
        if idx is None:
            if not assign:
                self.logits.zero_()
            rgemv('nvfp4', self.lm_p, self.lm_sp, self.lm_g, self.h, self.logits, getattr(self, 'lm_n', self.V), self.H, 1, assign=assign)
            return
        rgemv('nvfp4', self.lm_p, self.lm_sp, self.lm_g, self.h, self.clog, self.lm_n, self.H, 1, assign=True)
        self.logits.fill_(float('-inf'))
        self.logits.index_copy_(0, idx, self.clog[:idx.numel()])

    def _final(self):
        fn('rmsnorm1p')((1,), (256,), (self.x.data_ptr(), self.norm.data_ptr(), self.h.data_ptr(), i32(self.H), f32(self.eps)), stream=stream())
        self.head_logits()
        self.next_tok = getattr(self, 'next_tok', torch.zeros(1, dtype=torch.int64, device=dev))
        self.next_tok.copy_(self.logits.argmax().view(1))

    # ------------------------------------------------------------------ speed / quality trade-off (lossy, off by default)
    def set_quality(self, adaptive=0.0, skip_b=0.0):
        """adaptive: keep the fewest top experts whose renormalised mass >= adaptive (0 = all 8);
        skip_b: leave out experts that are not in VRAM and carry less than skip_b of the routed mass (0 = never)"""
        if (adaptive, skip_b) != (self.adaptive, self.skip_b):
            torch.cuda.synchronize()
            self.adaptive, self.skip_b = float(adaptive), float(skip_b)
            self.graphs.clear()

