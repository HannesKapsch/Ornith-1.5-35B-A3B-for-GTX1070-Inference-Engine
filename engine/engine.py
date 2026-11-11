"""Ornith-1.5-35B-A3B decode engine for an 8 GB Pascal GPU.

VRAM: attention / linear-attention weights (fp8), norms, routers, shared experts, lm_head (nvfp4),
linear-attention states, KV cache (fp16 or int8) and an expert cache. Experts live in experts.bin
(OS file cache / RAM / SSD) and are pulled into VRAM slots on demand.
"""
import json, os, time
from collections import OrderedDict
import numpy as np
import torch
import cupy as cp
from weights import Weights, LM, PATH
from kern import fn, rgemv, stream, i32, f32, i64
from pack import BLOB, OFF, OUT as EXPERTS_BIN, blob_of

dev = 'cuda'


class Ornith:
    def __init__(self, cache_gb=3.0, max_ctx=4096, kv8=False, adaptive=0.0, win=2048, log=print, kv_alloc=True):
        cfg = json.load(open(os.path.join(PATH, 'config.json')))['text_config']
        self.cfg = cfg
        self.L, self.H, self.E, self.K = cfg['num_hidden_layers'], cfg['hidden_size'], cfg['num_experts'], cfg['num_experts_per_tok']
        self.I = cfg['moe_intermediate_size']
        self.types = cfg['layer_types']
        self.eps = cfg['rms_norm_eps']
        self.max_ctx, self.kv8, self.adaptive = max_ctx, kv8, adaptive
        W = self.W = Weights()
        t0 = time.time()
        f = lambda k: W.raw(k).float().to(dev).contiguous()
        u8 = lambda k: W.raw(k).view(torch.uint8).to(dev).contiguous()
        sc = lambda k: torch.tensor([W.raw(k).float().item()], device=dev)
        P = lambda *ts: torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64, device=dev)
        self.layers = []
        nlin = nfull = 0
        for l in range(self.L):
            p = f'{LM}layers.{l}.'
            d = dict(ln1=f(p + 'input_layernorm.weight'), ln2=f(p + 'post_attention_layernorm.weight'),
                     rw=torch.cat([W.raw(p + 'mlp.gate.weight'), W.raw(p + 'mlp.shared_expert_gate.weight')]).to(dev).contiguous(),
                     sgate=torch.zeros(1, device=dev))
            d['rw_p'] = P(d['rw'])
            sb = torch.from_numpy(blob_of(W, p + 'mlp.shared_expert')).to(dev)
            d['shared'] = sb
            if self.types[l] == 'linear_attention':
                a = p + 'linear_attn.'
                d.update(kind='lin', idx=nlin,
                         qkv=u8(a + 'in_proj_qkv.weight'), qkv_s=sc(a + 'in_proj_qkv.weight_scale'),
                         z=u8(a + 'in_proj_z.weight'), z_s=sc(a + 'in_proj_z.weight_scale'),
                         ab=torch.cat([f(a + 'in_proj_a.weight'), f(a + 'in_proj_b.weight')]).contiguous(),
                         conv=f(a + 'conv1d.weight').reshape(-1, 4).contiguous(), A_log=f(a + 'A_log'),
                         dt_bias=f(a + 'dt_bias'), normw=f(a + 'norm.weight'),
                         out=u8(a + 'out_proj.weight'), out_s=sc(a + 'out_proj.weight_scale'))
                d.update(qkv_p=P(d['qkv']), z_p=P(d['z']), ab_p=P(d['ab']), out_p=P(d['out']))
                nlin += 1
            else:
                a = p + 'self_attn.'
                d.update(kind='full', idx=nfull,
                         q=u8(a + 'q_proj.weight'), q_s=sc(a + 'q_proj.weight_scale'),
                         k=u8(a + 'k_proj.weight'), v=u8(a + 'v_proj.weight'),
                         kv_s=torch.cat([sc(a + 'k_proj.weight_scale'), sc(a + 'v_proj.weight_scale')]),
                         qn=f(a + 'q_norm.weight'), kn=f(a + 'k_norm.weight'),
                         o=u8(a + 'o_proj.weight'), o_s=sc(a + 'o_proj.weight_scale'))
                d.update(q_p=P(d['q']), kv_p=P(d['k'], d['v']), o_p=P(d['o']))
                nfull += 1
            self.layers.append(d)
        self.nlin, self.nfull = nlin, nfull
        self.norm = f(LM + 'norm.weight')
        self.lm_w, self.lm_s = u8('lm_head.weight'), u8('lm_head.weight_scale')
        self.lm_g = sc('lm_head.weight_scale_2')
        self.lm_p, self.lm_sp = P(self.lm_w), P(self.lm_s)
        self.V = self.lm_w.shape[0]
        self.embed = W.raw(LM + 'embed_tokens.weight')               # bf16, stays on the CPU
        # rope (partial: 64 of 256 dims, theta 1e7)
        inv = 1.0 / (cfg['rope_parameters']['rope_theta'] ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
        ang = torch.arange(max_ctx, dtype=torch.float64)[:, None] * inv[None]
        self.cos, self.sin = ang.cos().float().to(dev).contiguous(), ang.sin().float().to(dev).contiguous()
        # recurrent / attention state
        self.S = torch.zeros(nlin, 32, 128, 128, device=dev)
        self.conv_state = torch.zeros(nlin, 8192, 3, device=dev)
        kvdt = torch.int8 if kv8 else torch.float16
        if kv_alloc:                                             # (Ornith2 arena mode places the KV cache itself)
            self.Kc = torch.zeros(nfull, 2, max_ctx, 256, dtype=kvdt, device=dev)
            self.Vc = torch.zeros(nfull, 2, max_ctx, 256, dtype=kvdt, device=dev)
            self.Ks = torch.ones(nfull, 2, max_ctx, 8, dtype=torch.float16, device=dev)   # int8 mode: fp16 scale per 32 dims
            self.Vs = torch.ones(nfull, 2, max_ctx, 8, dtype=torch.float16, device=dev)
            self.kvp = [(self.Kc[i].data_ptr(), self.Vc[i].data_ptr(), self.Ks[i].data_ptr(), self.Vs[i].data_ptr()) for i in range(nfull)]
        self.kvdefs = {}                                         # extra kernel defines for the KV layout
        self.WIN = win                                           # int8 mode: last WIN tokens stay exact (fp16 ring)
        self.RING = self.WIN + 4096                              # ring holds the window + one prefill chunk
        self.Kr = torch.zeros(nfull, 2, self.RING, 256, dtype=torch.float16, device=dev)
        self.Vr = torch.zeros(nfull, 2, self.RING, 256, dtype=torch.float16, device=dev)
        # scratch
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)
        self.x, self.h = z(self.H), z(self.H)
        self.qkv, self.qkv2, self.zb, self.ab, self.gb, self.olin = z(8192), z(8192), z(4096), z(64), z(64), z(4096)
        self.qf, self.kvb, self.qr, self.gate, self.oatt = z(8192), z(1024), z(4096), z(4096), z(4096)
        self.nsplit = (max_ctx + 63) // 64
        self.pm, self.pl, self.pacc = z(16 * self.nsplit), z(16 * self.nsplit), z(16 * self.nsplit * 256)
        k1 = self.K + 1
        self.wout, self.routes, self.counts = z(k1), z(8, dt=torch.int64), z(2, dt=torch.int32)
        self.gu_w, self.gu_s, self.gu_g = z(2 * k1, dt=torch.int64), z(2 * k1, dt=torch.int64), z(2 * k1)
        self.dn_w, self.dn_s, self.dn_g = z(k1, dt=torch.int64), z(k1, dt=torch.int64), z(k1)
        self.gu, self.hb = z(2 * k1 * self.I), z(k1 * self.I)
        self.logits = z(self.V)
        self.pos_t = z(1, dt=torch.int64)
        # experts: file-backed (OS cache), VRAM slot cache
        self.host = np.memmap(EXPERTS_BIN, dtype=np.uint8, mode='r', shape=(self.L, self.E, BLOB))
        self.set_cache(cache_gb)
        self.stats = dict(hits=0, misses=0)
        log(f'loaded in {time.time() - t0:.0f}s | VRAM {torch.cuda.memory_allocated() / 2 ** 30:.2f} GB')

    def set_cache(self, cache_gb):
        self.pool = None
        torch.cuda.empty_cache()
        self.S_slots = max(16, int(cache_gb * 2 ** 30 // BLOB))
        self.pool = torch.empty(self.S_slots * BLOB, dtype=torch.uint8, device=dev)
        self.lru = OrderedDict()
        self.free = list(range(self.S_slots))
        self.loc_host = torch.zeros(self.L * self.E, dtype=torch.int64)
        self.loc = torch.zeros(self.L * self.E, dtype=torch.int64, device=dev)

    def reset(self):
        self.S.zero_(); self.conv_state.zero_()

    # ------------------------------------------------------------------ experts
    def _ensure(self, l, experts):
        changed = False
        busy = set()
        for e in experts:
            key = (l, e)
            if key in self.lru:
                self.lru.move_to_end(key); self.stats['hits'] += 1
                busy.add(self.lru[key]); continue
            self.stats['misses'] += 1
            if self.free:
                s = self.free.pop()
            else:
                for k2 in self.lru:
                    if self.lru[k2] not in busy:
                        break
                s = self.lru.pop(k2)
                self.loc_host[k2[0] * self.E + k2[1]] = 0
            self.pool[s * BLOB:(s + 1) * BLOB].copy_(torch.from_numpy(np.asarray(self.host[l, e])), non_blocking=False)
            self.lru[key] = s
            busy.add(s)
            self.loc_host[l * self.E + e] = self.pool.data_ptr() + s * BLOB
            changed = True
        if changed:
            self.loc.copy_(self.loc_host)

    # ------------------------------------------------------------------ one token
    def _router(self, l, d, ptrs=1):
        o = OFF
        if not hasattr(self, 'rlog'):
            self.rlog = torch.zeros(self.E + 1, device=dev)
        self.rlog.zero_()
        rgemv('bf16', d['rw_p'], None, None, self.h, self.rlog, self.E + 1, self.H, 1)
        fn('router')((1,), (256,), (self.rlog.data_ptr(), d['sgate'].data_ptr(), self.h.data_ptr(), self.loc.data_ptr(),
                                    i64(d['shared'].data_ptr()), i32(l), i32(self.E), i32(self.H), i32(self.K), f32(self.adaptive),
                                    self.wout.data_ptr(), self.routes.data_ptr(), self.counts.data_ptr(),
                                    self.gu_w.data_ptr(), self.gu_s.data_ptr(), self.gu_g.data_ptr(),
                                    self.dn_w.data_ptr(), self.dn_s.data_ptr(), self.dn_g.data_ptr(),
                                    i64(o['gw']), i64(o['uw']), i64(o['dw']), i64(o['gs']), i64(o['us']), i64(o['ds']), i64(o['g2']), i32(ptrs)),
                     stream=stream())

    @torch.no_grad()
    def step(self, tok, pos, capture=None):
        H, I, k1 = self.H, self.I, self.K + 1
        s = stream()
        self.pos_t.fill_(pos)
        self.x.copy_(self.embed[tok].float())
        ns = min(self.nsplit, (pos + 64) // 64)
        for l, d in enumerate(self.layers):
            fn('rmsnorm1p')((1,), (256,), (self.x.data_ptr(), d['ln1'].data_ptr(), self.h.data_ptr(), i32(H), f32(self.eps)), stream=s)
            if d['kind'] == 'lin':
                i = d['idx']
                self.qkv.zero_(); self.zb.zero_(); self.ab.zero_()
                rgemv('fp8', d['qkv_p'], None, d['qkv_s'], self.h, self.qkv, 8192, H, 1)
                rgemv('fp8', d['z_p'], None, d['z_s'], self.h, self.zb, 4096, H, 1)
                rgemv('f32', d['ab_p'], None, None, self.h, self.ab, 64, H, 1)
                fn('dn_prep')((64,), (128,), (self.qkv.data_ptr(), self.conv_state[i].data_ptr(), d['conv'].data_ptr(),
                                               self.ab.data_ptr(), d['A_log'].data_ptr(), d['dt_bias'].data_ptr(),
                                               self.qkv2.data_ptr(), self.gb.data_ptr()), stream=s)
                fn('dn_recur')((32,), (128,), (self.qkv2.data_ptr(), self.gb.data_ptr(), self.S[i].data_ptr(),
                                                self.zb.data_ptr(), d['normw'].data_ptr(), self.olin.data_ptr(), f32(self.eps)), stream=s)
                rgemv('fp8', d['out_p'], None, d['out_s'], self.olin, self.x, H, 4096, 1)
            else:
                i = d['idx']
                self.qf.zero_(); self.kvb.zero_()
                rgemv('fp8', d['q_p'], None, d['q_s'], self.h, self.qf, 8192, H, 1)
                rgemv('fp8', d['kv_p'], None, d['kv_s'], self.h, self.kvb, 512, H, 2, xdiv=2, ystride=512)
                kv8 = int(self.kv8)
                fn('attn_prep', KV8=kv8, WIN=self.WIN, RING=self.RING)((18,), (256,), (self.qf.data_ptr(), self.kvb.data_ptr(), d['qn'].data_ptr(), d['kn'].data_ptr(),
                                                        self.cos.data_ptr(), self.sin.data_ptr(), self.pos_t.data_ptr(), self.qr.data_ptr(),
                                                        self.gate.data_ptr(), self.Kc[i].data_ptr(), self.Vc[i].data_ptr(),
                                                        self.Ks[i].data_ptr(), self.Vs[i].data_ptr(), i32(self.max_ctx), f32(self.eps),
                                                        self.Kr[i].data_ptr(), self.Vr[i].data_ptr(), i64(0)), stream=s)
                fn('attn_partial', KV8=kv8, WIN=self.WIN, RING=self.RING)((2, ns), (256,), (self.qr.data_ptr(), self.Kc[i].data_ptr(), self.Vc[i].data_ptr(),
                                                             self.Ks[i].data_ptr(), self.Vs[i].data_ptr(), self.pos_t.data_ptr(),
                                                             self.pm.data_ptr(), self.pl.data_ptr(), self.pacc.data_ptr(),
                                                             i32(self.max_ctx), i32(self.nsplit),
                                                             self.Kr[i].data_ptr(), self.Vr[i].data_ptr()), stream=s)
                fn('attn_combine', KV8=kv8)((16,), (256,), (self.pos_t.data_ptr(), self.pm.data_ptr(), self.pl.data_ptr(),
                                                           self.pacc.data_ptr(), self.gate.data_ptr(), self.oatt.data_ptr(), i32(self.nsplit)), stream=s)
                rgemv('fp8', d['o_p'], None, d['o_s'], self.oatt, self.x, H, 4096, 1)
            fn('rmsnorm1p')((1,), (256,), (self.x.data_ptr(), d['ln2'].data_ptr(), self.h.data_ptr(), i32(H), f32(self.eps)), stream=s)
            self._router(l, d, ptrs=0)
            ex = [e for e in self.routes.tolist() if e >= 0]       # sync: which experts are needed
            self._ensure(l, ex)
            self._router(l, d)                                     # recompute pointers with updated locations
            self.gu.zero_()
            rgemv('nvfp4', self.gu_w, self.gu_s, self.gu_g, self.h, self.gu, I, H, 2 * k1, xdiv=2 * k1, ystride=I, mcount=self.counts[0:1])
            fn('silu_mul')(((k1 * I + 255) // 256,), (256,), (self.gu.data_ptr(), self.hb.data_ptr(), self.counts.data_ptr(), i32(I)), stream=s)
            rgemv('nvfp4', self.dn_w, self.dn_s, self.dn_g, self.hb, self.x, H, I, k1, xdiv=1, xstride=I, ydiv=k1,
                  wts=self.wout, mcount=self.counts[1:2])
            if capture is not None:
                capture.append(self.x.clone())
        fn('rmsnorm1p')((1,), (256,), (self.x.data_ptr(), self.norm.data_ptr(), self.h.data_ptr(), i32(H), f32(self.eps)), stream=s)
        self.logits.zero_()
        rgemv('nvfp4', self.lm_p, self.lm_sp, self.lm_g, self.h, self.logits, self.V, H, 1)
        return self.logits

    def generate(self, ids, n_new, stop_ids=()):
        self.reset()
        for pos, t in enumerate(ids[:-1]):
            self.step(t, pos)
        cur, out = ids[-1], []
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(n_new):
            cur = int(self.step(cur, len(ids) - 1 + i).argmax())
            out.append(cur)
            if cur in stop_ids:
                break
        torch.cuda.synchronize()
        return out, len(out) / (time.time() - t0)
