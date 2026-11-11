"""Batched (prefill) kernels: many tokens per launch. fp32 math, weights decoded on the fly."""
import cupy as cp
import torch
from kern import stream, i32, f32, i64

_SRC = r'''
#include <cuda_fp16.h>
#define INF __int_as_float(0x7f800000)
// KV cache row addressing. Default: [kv head][maxT][...]. CK defined (arena mode): chunks of CK positions,
// chunk c starts KCH (int8 rows of 256) / SCH (scale rows of 8 halves) rows after chunk c-1, [kv head][CK][...] inside.
#ifdef CK
#define KROW(h, t) ((size_t)((t) / CK) * KCH + (size_t)(h) * CK + ((t) % CK))
#define SROW(h, t) ((size_t)((t) / CK) * SCH + (size_t)(h) * CK + ((t) % CK))
#else
#define KROW(h, t) ((size_t)(h) * maxT + (t))
#define SROW(h, t) ((size_t)(h) * maxT + (t))
#endif
__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); return v; }
__device__ float block_sum(float v, float* red) {
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v); __syncthreads(); if (lane == 0) red[w] = v; __syncthreads();
  v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.f; if (w == 0) v = warp_sum(v);
  if (threadIdx.x == 0) red[32] = v; __syncthreads(); return red[32];
}
__device__ __forceinline__ float fp8_e4m3(unsigned b) {
  unsigned s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
  float v = e ? __int_as_float(((e + 120) << 23) | (m << 20)) : (float)m * 0.001953125f;
  return s ? -v : v;
}

// ---------------------------------------------------------------- grouped tiled GEMM
// for group g (blockIdx.z): rows r in [roff[g], roff[g+1]):
//   C[cmap? cmap[r] : r][n] (op)= gsc[g] * rscale[r] * sum_k X[xmap? xmap[r] : r][k] * W_g[n][k]
// FMT 0 NVFP4, 1 FP8, 2 F32, 3 BF16.  ACC 0: store, 1: add, 2: atomicAdd
#define BM 64
#define BN 64
#define BK 16
extern "C" __global__ void bgemm(const long long* __restrict__ wptr, const long long* __restrict__ sptr, const float* __restrict__ gsc,
    const int* __restrict__ roff, const float* __restrict__ X, const int* __restrict__ xmap, int ldx,
    float* __restrict__ C, const int* __restrict__ cmap, int ldc, const float* __restrict__ rscale, int N, int K, int acc_mode)
{
  const int g = blockIdx.z;
  const int r0 = roff[g] + blockIdx.y * BM, r1 = roff[g + 1];
  if (r0 >= r1) return;
  const int n0 = blockIdx.x * BN;
  __shared__ float Xs[BK][BM + 4];
  __shared__ float Ws[BK][BN + 4];
  __shared__ float lut4[16], lut8[256];
  const float e2m1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  lut8[threadIdx.x & 255] = fp8_e4m3(threadIdx.x & 255);
  if (threadIdx.x < 16) lut4[threadIdx.x] = e2m1[threadIdx.x];
  const unsigned char* Wb = (const unsigned char*)wptr[g];
  const unsigned char* Sb = sptr ? (const unsigned char*)sptr[g] : 0;
  const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
  __syncthreads();
  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int i = threadIdx.x; i < BM * BK; i += 256) {
      int rr = i / BK, kk = i % BK, r = r0 + rr;
      float v = 0.f;
      if (r < r1) { int xr = xmap ? xmap[r] : r; v = X[(size_t)xr * ldx + k0 + kk]; }
      Xs[kk][rr] = v;
    }
    for (int i = threadIdx.x; i < BN * BK; i += 256) {
      int nn = i / BK, kk = i % BK, n = n0 + nn, k = k0 + kk;
      float w = 0.f;
      if (n < N) {
#if FMT == 0
        unsigned char b = Wb[(size_t)n * (K >> 1) + (k >> 1)];
        w = lut4[(k & 1) ? (b >> 4) : (b & 15)] * lut8[Sb[(size_t)n * (K >> 4) + (k >> 4)]];
#elif FMT == 1
        w = lut8[Wb[(size_t)n * K + k]];
#elif FMT == 2
        w = ((const float*)Wb)[(size_t)n * K + k];
#else
        w = __int_as_float(((unsigned)((const unsigned short*)Wb)[(size_t)n * K + k]) << 16);
#endif
      }
      Ws[kk][nn] = w;
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK; kk++) {
      float a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; i++) { a[i] = Xs[kk][ty * 4 + i]; b[i] = Ws[kk][tx * 4 + i]; }
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }
    __syncthreads();
  }
  const float gs = gsc ? gsc[g] : 1.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    int r = r0 + ty * 4 + i;
    if (r >= r1) continue;
    float sc = gs * (rscale ? rscale[r] : 1.f);
    size_t cr = (size_t)(cmap ? cmap[r] : r) * ldc;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      int n = n0 + tx * 4 + j;
      if (n >= N) continue;
      float v = acc[i][j] * sc;
      if (acc_mode == 2) atomicAdd(C + cr + n, v);
      else if (acc_mode == 1) C[cr + n] += v;
      else C[cr + n] = v;
    }
  }
}

// ---------------------------------------------------------------- row-wise helpers
extern "C" __global__ void rmsnorm1p_rows(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int n, float eps) {
  __shared__ float red[33];
  const float* xr = x + (size_t)blockIdx.x * n; float* o = out + (size_t)blockIdx.x * n;
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
  float r = rsqrtf(block_sum(s, red) / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = xr[i] * r * (1.f + w[i]);
}

// causal depthwise conv (kernel 4) over the chunk, continuing from conv_state; SiLU. one thread per channel
extern "C" __global__ void conv_seq(const float* __restrict__ in, float* __restrict__ out, float* __restrict__ st,
                                    const float* __restrict__ w, int T, int C) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  float s0 = st[c * 3], s1 = st[c * 3 + 1], s2 = st[c * 3 + 2];
  const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  for (int t = 0; t < T; t++) {
    float cur = in[(size_t)t * C + c];
    float o = w0 * s0 + w1 * s1 + w2 * s2 + w3 * cur;
    s0 = s1; s1 = s2; s2 = cur;
    out[(size_t)t * C + c] = o / (1.f + __expf(-o));
  }
  st[c * 3] = s0; st[c * 3 + 1] = s1; st[c * 3 + 2] = s2;
}

// per token: l2-normalise q/k heads (q also / sqrt(128)); grid (T, 32) x 128
extern "C" __global__ void dn_qknorm(float* __restrict__ qkv, int C) {
  __shared__ float red[33];
  float* p = qkv + (size_t)blockIdx.x * C + blockIdx.y * 128;
  float v = p[threadIdx.x];
  float inv = rsqrtf(block_sum(v * v, red) + 1e-6f);
  v *= inv;
  if (blockIdx.y < 16) v *= 0.08838834764831845f;
  p[threadIdx.x] = v;
}

// per token gates: grid T x 32
extern "C" __global__ void dn_gates(const float* __restrict__ ab, const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                    float* __restrict__ gb) {
  const int t = blockIdx.x, h = threadIdx.x;
  float a = ab[t * 64 + h] + dt_bias[h];
  float sp = a > 20.f ? a : log1pf(__expf(a));
  gb[t * 64 + h] = -__expf(A_log[h]) * sp;
  gb[t * 64 + 32 + h] = 1.f / (1.f + __expf(-ab[t * 64 + 32 + h]));
}

// sequential gated delta rule over the chunk, state kept in registers; gated RMSNorm. grid 32 x 128
extern "C" __global__ void dn_recur_seq(const float* __restrict__ qkv, const float* __restrict__ gb, float* __restrict__ S,
    const float* __restrict__ z, const float* __restrict__ normw, float* __restrict__ out, int T, float eps)
{
  __shared__ float qs[128], ks[128];
  __shared__ float red[33];
  const int h = blockIdx.x, j = threadIdx.x, kh = h >> 1;
  float* Sh = S + (size_t)h * 128 * 128;
  float col[128];
#pragma unroll
  for (int k = 0; k < 128; k++) col[k] = Sh[k * 128 + j];
  const float nw = normw[j];
  for (int t = 0; t < T; t++) {
    const float* row = qkv + (size_t)t * 8192;
    __syncthreads();
    qs[j] = row[kh * 128 + j];
    ks[j] = row[2048 + kh * 128 + j];
    __syncthreads();
    const float decay = __expf(gb[t * 64 + h]), beta = gb[t * 64 + 32 + h];
    float kv = 0.f;
#pragma unroll
    for (int k = 0; k < 128; k++) { col[k] *= decay; kv = fmaf(col[k], ks[k], kv); }
    const float delta = (row[4096 + h * 128 + j] - kv) * beta;
    float o = 0.f;
#pragma unroll
    for (int k = 0; k < 128; k++) { col[k] = fmaf(ks[k], delta, col[k]); o = fmaf(col[k], qs[k], o); }
    float r = rsqrtf(block_sum(o * o, red) / 128.f + eps);
    float zz = z[(size_t)t * 4096 + h * 128 + j];
    out[(size_t)t * 4096 + h * 128 + j] = o * r * nw * (zz / (1.f + __expf(-zz)));
  }
#pragma unroll
  for (int k = 0; k < 128; k++) Sh[k * 128 + j] = col[k];
}

extern "C" __global__ void silu_mul_rows(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float a = g[i];
  h[i] = a / (1.f + __expf(-a)) * u[i];
}

// router per token: logits [T][E+1] (last = shared gate logit) -> routes [T][8], weights [T][9] (index k = shared)
extern "C" __global__ void router_rows(const float* __restrict__ lgall, int E, int topk, float thr,
                                       int* __restrict__ routes, float* __restrict__ wout) {
  __shared__ float lg[256];
  __shared__ float red[33];
  __shared__ int sel[16];
  __shared__ float selw[16];
  const int t = blockIdx.x, lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  const float* L = lgall + (size_t)t * (E + 1);
  float v = threadIdx.x < E ? L[threadIdx.x] : -INF;
  float mx = v;
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
  if (lane == 0) red[w] = mx; __syncthreads();
  if (threadIdx.x == 0) { float m = red[0]; for (int q = 1; q < nw; q++) m = fmaxf(m, red[q]); red[32] = m; }
  __syncthreads(); mx = red[32]; __syncthreads();
  float ex = threadIdx.x < E ? __expf(v - mx) : 0.f;
  float sum = block_sum(ex, red);
  float p = threadIdx.x < E ? ex / sum : -1.f;
  for (int j = 0; j < topk; j++) {
    float bv = p; int bi = threadIdx.x;
    for (int o = 16; o; o >>= 1) {
      float ov = __shfl_xor_sync(0xffffffffu, bv, o); int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    __syncthreads();
    if (lane == 0) { lg[w] = bv; red[w] = __int_as_float(bi); }
    __syncthreads();
    if (threadIdx.x == 0) {
      float b = lg[0]; int bi2 = __float_as_int(red[0]);
      for (int q = 1; q < nw; q++) { float c = lg[q]; int ci = __float_as_int(red[q]); if (c > b || (c == b && ci < bi2)) { b = c; bi2 = ci; } }
      sel[j] = bi2; selw[j] = b;
    }
    __syncthreads();
    if (threadIdx.x == sel[j]) p = -1.f;
  }
  if (threadIdx.x == 0) {
    float s = 0.f;
    for (int j = 0; j < topk; j++) s += selw[j];
    int k = topk;
    if (thr > 0.f) {
      float c = 0.f;
      for (k = 0; k < topk; ) { c += selw[k] / s; k++; if (c >= thr) break; }
      s = 0.f; for (int j = 0; j < k; j++) s += selw[j];
    }
    for (int j = 0; j < topk; j++) { routes[t * topk + j] = j < k ? sel[j] : -1; wout[t * (topk + 1) + j] = j < k ? selw[j] / s : 0.f; }
    wout[t * (topk + 1) + topk] = 1.f / (1.f + __expf(-L[E]));
  }
}

// ---------------------------------------------------------------- full attention, batched
// prologue per token: grid (T, 18) x 256. pos = p0 + t
extern "C" __global__ void attn_prep_rows(const float* __restrict__ Q, const float* __restrict__ KVin,
    const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ cosT, const float* __restrict__ sinT,
    int p0, float* __restrict__ QR, float* __restrict__ GATE,
    void* __restrict__ Kc, void* __restrict__ Vc, __half* __restrict__ Ks, __half* __restrict__ Vs, int maxT, float eps,
    __half* __restrict__ Kr, __half* __restrict__ Vr, int ring, const int* __restrict__ rp)
{
  __shared__ float xs[256];
  __shared__ float red[33];
  const int tk = blockIdx.x, b = blockIdx.y, t = threadIdx.x, pos = p0 + tk;
  const int rpos = rp ? rp[pos] : pos;          // RoPE row (3D positions of image tokens live in extra table rows)
  const float* qf = Q + (size_t)tk * 8192;
  const float* kv = KVin + (size_t)tk * 1024;
  const bool isq = b < 16;
  const int h = isq ? b : b - 16;
  float x = isq ? qf[h * 512 + t] : kv[h * 256 + t];
  if (isq) GATE[(size_t)tk * 4096 + h * 256 + t] = qf[h * 512 + 256 + t];
  float r = rsqrtf(block_sum(x * x, red) / 256.f + eps);
  x = x * r * (1.f + (isq ? qn[t] : kn[t]));
  xs[t] = x;
  __syncthreads();
  if (t < 64) {
    const int i = t & 31;
    float c = cosT[rpos * 32 + i], s = sinT[rpos * 32 + i];
    x = t < 32 ? xs[t] * c - xs[t + 32] * s : xs[t] * c + xs[t - 32] * s;
  }
  if (isq) { QR[(size_t)tk * 4096 + h * 256 + t] = x; return; }
  float v = kv[512 + h * 256 + t];
  size_t base = KROW(h, pos) * 256 + t;
#if KV8
  float ka = warp_max(fabsf(x)), va = warp_max(fabsf(v));
  __half kh = __float2half(ka > 0.f ? ka / 127.f : 1.f), vh = __float2half(va > 0.f ? va / 127.f : 1.f);
  float ksc = __half2float(kh), vsc = __half2float(vh);
  ((signed char*)Kc)[base] = (signed char)max(-127, min(127, __float2int_rn(x / ksc)));
  ((signed char*)Vc)[base] = (signed char)max(-127, min(127, __float2int_rn(v / vsc)));
  if ((t & 31) == 0) { size_t sb = SROW(h, pos) * 8 + (t >> 5); Ks[sb] = kh; Vs[sb] = vh; }
  size_t rb = ((size_t)h * ring + (pos % ring)) * 256 + t;
  Kr[rb] = __float2half(x); Vr[rb] = __float2half(v);
#else
  ((__half*)Kc)[base] = __float2half(x);
  ((__half*)Vc)[base] = __float2half(v);
#endif
}

// causal flash attention for a block of 8 query tokens x 8 heads (one kv group) = 64 queries.
// thread = (query qi = tid/4, quarter = tid%4 owning 64 head dims). grid (ceil(T/8), 2) x 256.
#define KT 16
#define RS 288      // padded row stride (4 quarters x (64 + 8)) -> conflict-free shared reads
extern "C" __global__ void attn_prefill(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ float Kt[KT * RS], Vt[KT * RS];
  const int g = blockIdx.y, tid = threadIdx.x, qi = tid >> 2, qu = tid & 3;
  const int tb = blockIdx.x * 8, tq = tb + (qi >> 3), h = g * 8 + (qi & 7);
  const bool valid = tq < T;
  const int qpos = p0 + (valid ? tq : T - 1);
  float q[64], acc[64];
  const float* qp = QR + (size_t)(valid ? tq : T - 1) * 4096 + h * 256 + qu * 64;
#pragma unroll
  for (int d = 0; d < 64; d++) { q[d] = qp[d] * 0.0625f; acc[d] = 0.f; }
  float m = -INF, l = 0.f;
  const int kend = p0 + min(tb + 7, T - 1);          // last key any query of this block needs
  const int exact_from = p0 + tb - win + 1;           // keys >= this come from the exact fp16 ring
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();
#ifdef NOLOAD
    if (k0 == 0)
#endif
    for (int i = tid; i < KT * 256; i += 256) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kp <= kend) {
#if KV8
        if (kp >= exact_from) {
          size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + d;
          kv = __half2float(Kr[rb]); vv = __half2float(Vr[rb]);
        } else {
          size_t base = KROW(g, kp) * 256 + d, sb = SROW(g, kp) * 8 + (d >> 5);
          kv = (float)((const signed char*)Kc)[base] * __half2float(Ks[sb]);
          vv = (float)((const signed char*)Vc)[base] * __half2float(Vs[sb]);
        }
#else
        size_t base = KROW(g, kp) * 256 + d;
        kv = __half2float(((const __half*)Kc)[base]); vv = __half2float(((const __half*)Vc)[base]);
#endif
      }
      int sidx = j * RS + (d >> 6) * 72 + (d & 63);
      Kt[sidx] = kv; Vt[sidx] = vv;
    }
    __syncthreads();
    float s[KT];
    float tmax = -INF;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      const float* kr = Kt + j * RS + qu * 72;
      float p = 0.f;
#pragma unroll
      for (int d = 0; d < 64; d++) p = fmaf(q[d], kr[d], p);
      p += __shfl_xor_sync(0xffffffffu, p, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2);
      s[j] = (k0 + j <= qpos) ? p : -INF;
      tmax = fmaxf(tmax, s[j]);
    }
    float mn = fmaxf(m, tmax);
    if (mn == -INF) continue;
    float corr = __expf(m - mn);
    l *= corr;
#pragma unroll
    for (int d = 0; d < 64; d++) acc[d] *= corr;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      float p = __expf(s[j] - mn);
      l += p;
      const float* vr = Vt + j * RS + qu * 72;
#pragma unroll
      for (int d = 0; d < 64; d++) acc[d] = fmaf(p, vr[d], acc[d]);
    }
    m = mn;
  }
  if (!valid) return;
  const float inv = 1.f / l;
  const size_t ob = (size_t)tq * 4096 + h * 256 + qu * 64;
#pragma unroll
  for (int d = 0; d < 64; d++) {
    float gg = GATE[ob + d];
    O[ob + d] = acc[d] * inv * (1.f / (1.f + __expf(-gg)));
  }
}
// v2: thread = (query group qg = tid/16: 4 queries = 4 heads of one token, dim slice ds = tid%16: 16 dims)
// every K/V value read from shared memory feeds 4 queries. grid (ceil(T/8), 2) x 256.
#define KT2 16
#define RS2 272     // 16 slices x 17 (padding -> conflict-free)
extern "C" __global__ void attn_prefill2(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ float Kt[KT2 * RS2], Vt[KT2 * RS2];
  const int g = blockIdx.y, tid = threadIdx.x, qg = tid >> 4, ds = tid & 15;
  const int tb = blockIdx.x * 8;
  const int tok = tb + (qg >> 1);                 // 16 groups: 8 tokens x 2 halves of the 8 heads
  const int h0 = g * 8 + (qg & 1) * 4;
  const bool valid = tok < T;
  const int tq = valid ? tok : T - 1, qpos = p0 + tq;
  float q[4][16], acc[4][16], m[4], l[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float* qp = QR + (size_t)tq * 4096 + (h0 + i) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) { q[i][d] = qp[d] * 0.0625f; acc[i][d] = 0.f; }
    m[i] = -INF; l[i] = 0.f;
  }
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT2) {
    __syncthreads();
    for (int i = tid; i < KT2 * 256; i += 256) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kp <= kend) {
#if KV8
        if (kp >= exact_from) {
          size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + d;
          kv = __half2float(Kr[rb]); vv = __half2float(Vr[rb]);
        } else {
          size_t base = KROW(g, kp) * 256 + d, sb = SROW(g, kp) * 8 + (d >> 5);
          kv = (float)((const signed char*)Kc)[base] * __half2float(Ks[sb]);
          vv = (float)((const signed char*)Vc)[base] * __half2float(Vs[sb]);
        }
#else
        size_t base = KROW(g, kp) * 256 + d;
        kv = __half2float(((const __half*)Kc)[base]); vv = __half2float(((const __half*)Vc)[base]);
#endif
      }
      int sidx = j * RS2 + (d >> 4) * 17 + (d & 15);
      Kt[sidx] = kv; Vt[sidx] = vv;
    }
    __syncthreads();
    float s[4][KT2];
#pragma unroll
    for (int j = 0; j < KT2; j++) {
      const float* kr = Kt + j * RS2 + ds * 17;
      float p0_ = 0.f, p1 = 0.f, p2 = 0.f, p3 = 0.f;
#pragma unroll
      for (int d = 0; d < 16; d++) {
        float k = kr[d];
        p0_ = fmaf(q[0][d], k, p0_); p1 = fmaf(q[1][d], k, p1); p2 = fmaf(q[2][d], k, p2); p3 = fmaf(q[3][d], k, p3);
      }
#pragma unroll
      for (int o = 1; o < 16; o <<= 1) {
        p0_ += __shfl_xor_sync(0xffffffffu, p0_, o); p1 += __shfl_xor_sync(0xffffffffu, p1, o);
        p2 += __shfl_xor_sync(0xffffffffu, p2, o); p3 += __shfl_xor_sync(0xffffffffu, p3, o);
      }
      const bool ok = k0 + j <= qpos;
      s[0][j] = ok ? p0_ : -INF; s[1][j] = ok ? p1 : -INF; s[2][j] = ok ? p2 : -INF; s[3][j] = ok ? p3 : -INF;
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float tmax = -INF;
#pragma unroll
      for (int j = 0; j < KT2; j++) tmax = fmaxf(tmax, s[i][j]);
      float mn = fmaxf(m[i], tmax);
      if (mn == -INF) continue;
      float corr = __expf(m[i] - mn);
      l[i] *= corr;
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= corr;
      m[i] = mn;
#pragma unroll
      for (int j = 0; j < KT2; j++) { float p = __expf(s[i][j] - mn); s[i][j] = p; l[i] += p; }
    }
#pragma unroll
    for (int j = 0; j < KT2; j++) {
      const float* vr = Vt + j * RS2 + ds * 17;
#pragma unroll
      for (int d = 0; d < 16; d++) {
        float v = vr[d];
#pragma unroll
        for (int i = 0; i < 4; i++) acc[i][d] = fmaf(s[i][j] > -INF ? s[i][j] : 0.f, v, acc[i][d]);
      }
    }
  }
  if (!valid) return;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const float inv = 1.f / l[i];
    const size_t ob = (size_t)tok * 4096 + (h0 + i) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

// int8 variant: Q.K^T with __dp4a on int8 keys (q quantised per 32-dim group), softmax + P.V in fp32.
// QK in layout A (thread = query tid/4, quarter tid%4 = 64 dims), PV in layout B (thread = 4 queries x 16 dims).
#define KT3 16
extern "C" __global__ void attn_prefill_i8(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const signed char* __restrict__ Kc, const signed char* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ int Ki[KT3 * 96];            // per key: 4 quarters x 24 ints (16 used) -> conflict-free
  __shared__ float Ksc[KT3 * 8];
  __shared__ float Vt[KT3 * RS2];
  __shared__ float Sp[64 * (KT3 + 1)];
  const int g = blockIdx.y, tid = threadIdx.x;
  const int tb = blockIdx.x * 8;
  // ---- layout A (QK)
  const int qa = tid >> 2, qu = tid & 3;
  const int tokA = min(tb + (qa >> 3), T - 1), hA = g * 8 + (qa & 7), posA = p0 + tokA;
  int qi8[16]; float qsc[2];
  {
    const float* qp = QR + (size_t)tokA * 4096 + hA * 256 + qu * 64;
#pragma unroll
    for (int gq = 0; gq < 2; gq++) {
      float mx = 0.f;
#pragma unroll
      for (int d = 0; d < 32; d++) mx = fmaxf(mx, fabsf(qp[gq * 32 + d]));
      float sc = mx > 0.f ? mx / 127.f : 1.f;
      qsc[gq] = sc * 0.0625f;                                  // fold 1/sqrt(256)
#pragma unroll
      for (int w = 0; w < 8; w++) {
        int packed = 0;
#pragma unroll
        for (int b = 0; b < 4; b++) {
          int v = __float2int_rn(qp[gq * 32 + w * 4 + b] / sc);
          packed |= (v & 255) << (8 * b);
        }
        qi8[gq * 8 + w] = packed;
      }
    }
  }
  // ---- layout B (PV)
  const int qg = tid >> 4, ds = tid & 15;
  float acc[4][16], m[4], l[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
    m[i] = -INF; l[i] = 0.f;
  }
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT3) {
    __syncthreads();
    for (int i = tid; i < KT3 * 64; i += 256) {          // int8 keys, 4 bytes at a time
      int j = i >> 6, w = i & 63, kp = k0 + j;
      int v = 0;
      if (kp <= kend) v = *((const int*)(Kc + KROW(g, kp) * 256) + w);
      Ki[j * 96 + (w >> 4) * 24 + (w & 15)] = v;
    }
    if (tid < KT3 * 8) {
      int j = tid >> 3, kp = k0 + j;
      Ksc[tid] = kp <= kend ? __half2float(Ks[SROW(g, kp) * 8 + (tid & 7)]) : 0.f;
    }
    for (int i = tid; i < KT3 * 256; i += 256) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float vv = 0.f;
      if (kp <= kend) {
        if (kp >= exact_from) vv = __half2float(Vr[((size_t)g * ring + (kp % ring)) * 256 + d]);
        else vv = (float)Vc[KROW(g, kp) * 256 + d] * __half2float(Vs[SROW(g, kp) * 8 + (d >> 5)]);
      }
      Vt[j * RS2 + (d >> 4) * 17 + (d & 15)] = vv;
    }
    __syncthreads();
    // QK
#pragma unroll
    for (int j = 0; j < KT3; j++) {
      const int* kr = Ki + j * 96 + qu * 24;
      int d0 = 0, d1 = 0;
#pragma unroll
      for (int w = 0; w < 8; w++) { d0 = __dp4a(qi8[w], kr[w], d0); d1 = __dp4a(qi8[8 + w], kr[8 + w], d1); }
      float sc = (float)d0 * qsc[0] * Ksc[j * 8 + qu * 2] + (float)d1 * qsc[1] * Ksc[j * 8 + qu * 2 + 1];
      sc += __shfl_xor_sync(0xffffffffu, sc, 1);
      sc += __shfl_xor_sync(0xffffffffu, sc, 2);
      if (qu == 0) Sp[qa * (KT3 + 1) + j] = (k0 + j <= posA) ? sc : -INF;
    }
    __syncthreads();
    // softmax + PV
    float p[4][KT3];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float* sr = Sp + (qg * 4 + i) * (KT3 + 1);
      float tmax = -INF;
#pragma unroll
      for (int j = 0; j < KT3; j++) { p[i][j] = sr[j]; tmax = fmaxf(tmax, p[i][j]); }
      float mn = fmaxf(m[i], tmax);
      float corr = mn == -INF ? 1.f : __expf(m[i] - mn);
      l[i] *= corr;
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= corr;
#pragma unroll
      for (int j = 0; j < KT3; j++) { float e = mn == -INF ? 0.f : __expf(p[i][j] - mn); p[i][j] = e; l[i] += e; }
      m[i] = mn;
    }
#pragma unroll
    for (int j = 0; j < KT3; j++) {
      const float* vr = Vt + j * RS2 + ds * 17;
#pragma unroll
      for (int d = 0; d < 16; d++) {
        float v = vr[d];
#pragma unroll
        for (int i = 0; i < 4; i++) acc[i][d] = fmaf(p[i][j], v, acc[i][d]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int q = qg * 4 + i, tok = tb + (q >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / l[i];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (q & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

// exact fp32 v3: QK in layout A (query tid/4, quarter tid%4 = 64 dims, K tile padded per quarter),
// P.V in layout B (4 queries x 16 dims per thread, each V value feeds 4 queries). grid (ceil(T/8), 2) x 256
extern "C" __global__ void attn_prefill3(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ float Kt[KT * RS];
  __shared__ float Vt[KT * RS2];
  __shared__ float Sp[64 * (KT + 1)];
  const int g = blockIdx.y, tid = threadIdx.x, tb = blockIdx.x * 8;
  const int qa = tid >> 2, qu = tid & 3;
  const int tokA = min(tb + (qa >> 3), T - 1), posA = p0 + tokA;
  float q[64];
  {
    const float* qp = QR + (size_t)tokA * 4096 + (g * 8 + (qa & 7)) * 256 + qu * 64;
#pragma unroll
    for (int d = 0; d < 64; d++) q[d] = qp[d] * 0.0625f;
  }
  const int qg = tid >> 4, ds = tid & 15;
  float acc[4][16], m[4], l[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
    m[i] = -INF; l[i] = 0.f;
  }
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();
    for (int i = tid; i < KT * 256; i += 256) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kp <= kend) {
#if KV8
        if (kp >= exact_from) {
          size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + d;
          kv = __half2float(Kr[rb]); vv = __half2float(Vr[rb]);
        } else {
          size_t base = KROW(g, kp) * 256 + d, sb = SROW(g, kp) * 8 + (d >> 5);
          kv = (float)((const signed char*)Kc)[base] * __half2float(Ks[sb]);
          vv = (float)((const signed char*)Vc)[base] * __half2float(Vs[sb]);
        }
#else
        size_t base = KROW(g, kp) * 256 + d;
        kv = __half2float(((const __half*)Kc)[base]); vv = __half2float(((const __half*)Vc)[base]);
#endif
      }
      Kt[j * RS + (d >> 6) * 72 + (d & 63)] = kv;
      Vt[j * RS2 + (d >> 4) * 17 + (d & 15)] = vv;
    }
    __syncthreads();
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float* kr = Kt + j * RS + qu * 72;
      float p = 0.f;
#pragma unroll
      for (int d = 0; d < 64; d++) p = fmaf(q[d], kr[d], p);
      p += __shfl_xor_sync(0xffffffffu, p, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2);
      if (qu == 0) Sp[qa * (KT + 1) + j] = (k0 + j <= posA) ? p : -INF;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float* sr = Sp + (qg * 4 + i) * (KT + 1);
      float tmax = -INF;
      for (int j = 0; j < KT; j++) tmax = fmaxf(tmax, sr[j]);
      float mn = fmaxf(m[i], tmax);
      float corr = mn == -INF ? 1.f : __expf(m[i] - mn);
      l[i] *= corr;
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= corr;
      m[i] = mn;
    }
    __syncthreads();
    // turn scores into probabilities in place (each row once: threads with ds == 0)
    if (ds == 0) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float* sr = Sp + (qg * 4 + i) * (KT + 1);
        for (int j = 0; j < KT; j++) sr[j] = m[i] == -INF ? 0.f : __expf(sr[j] - m[i]);
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float* sr = Sp + (qg * 4 + i) * (KT + 1);
      float s = 0.f;
      for (int j = 0; j < KT; j++) s += sr[j];
      l[i] += s;
    }
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float* vr = Vt + j * RS2 + ds * 17;
      float p0_ = Sp[(qg * 4) * (KT + 1) + j], p1 = Sp[(qg * 4 + 1) * (KT + 1) + j];
      float p2 = Sp[(qg * 4 + 2) * (KT + 1) + j], p3 = Sp[(qg * 4 + 3) * (KT + 1) + j];
#pragma unroll
      for (int d = 0; d < 16; d++) {
        float v = vr[d];
        acc[0][d] = fmaf(p0_, v, acc[0][d]); acc[1][d] = fmaf(p1, v, acc[1][d]);
        acc[2][d] = fmaf(p2, v, acc[2][d]); acc[3][d] = fmaf(p3, v, acc[3][d]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int qq = qg * 4 + i, tok = tb + (qq >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / l[i];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (qq & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

extern "C" __global__ void attn_prefill4(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ __align__(16) float Kt[KT * RS];
  __shared__ __align__(16) float Vt[KT * 320];   // 16 slices x 20 (16B aligned, conflict-free LDS.128)
  __shared__ float Sp[64 * (KT + 1)];
  const int g = blockIdx.y, tid = threadIdx.x, tb = blockIdx.x * 8;
  const int qa = tid >> 2, qu = tid & 3;
  const int tokA = min(tb + (qa >> 3), T - 1), posA = p0 + tokA;
  float q[64];
  {
    const float* qp = QR + (size_t)tokA * 4096 + (g * 8 + (qa & 7)) * 256 + qu * 64;
#pragma unroll
    for (int d = 0; d < 64; d++) q[d] = qp[d] * 0.0625f;
  }
  const int qg = tid >> 4, ds = tid & 15;
  float acc[4][16], m[4], l[4];
#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
    m[i] = -INF; l[i] = 0.f;
  }
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();
    for (int i = tid; i < KT * 256; i += 256) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kp <= kend) {
#if KV8
        if (kp >= exact_from) {
          size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + d;
          kv = __half2float(Kr[rb]); vv = __half2float(Vr[rb]);
        } else {
          size_t base = KROW(g, kp) * 256 + d, sb = SROW(g, kp) * 8 + (d >> 5);
          kv = (float)((const signed char*)Kc)[base] * __half2float(Ks[sb]);
          vv = (float)((const signed char*)Vc)[base] * __half2float(Vs[sb]);
        }
#else
        size_t base = KROW(g, kp) * 256 + d;
        kv = __half2float(((const __half*)Kc)[base]); vv = __half2float(((const __half*)Vc)[base]);
#endif
      }
      Kt[j * RS + (d >> 6) * 72 + (d & 63)] = kv;
      Vt[j * 320 + (d >> 4) * 20 + (d & 15)] = vv;
    }
    __syncthreads();
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float4* kr = (const float4*)(Kt + j * RS + qu * 72);
      float p = 0.f;
#pragma unroll
      for (int d4 = 0; d4 < 16; d4++) {
        float4 k = kr[d4];
        p = fmaf(q[4 * d4], k.x, p); p = fmaf(q[4 * d4 + 1], k.y, p); p = fmaf(q[4 * d4 + 2], k.z, p); p = fmaf(q[4 * d4 + 3], k.w, p);
      }
      p += __shfl_xor_sync(0xffffffffu, p, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2);
      if (qu == 0) Sp[qa * (KT + 1) + j] = (k0 + j <= posA) ? p : -INF;
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float* sr = Sp + (qg * 4 + i) * (KT + 1);
      float tmax = -INF;
      for (int j = 0; j < KT; j++) tmax = fmaxf(tmax, sr[j]);
      float mn = fmaxf(m[i], tmax);
      float corr = mn == -INF ? 1.f : __expf(m[i] - mn);
      l[i] *= corr;
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= corr;
      m[i] = mn;
    }
    __syncthreads();
    // turn scores into probabilities in place (each row once: threads with ds == 0)
    if (ds == 0) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float* sr = Sp + (qg * 4 + i) * (KT + 1);
        for (int j = 0; j < KT; j++) sr[j] = m[i] == -INF ? 0.f : __expf(sr[j] - m[i]);
      }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float* sr = Sp + (qg * 4 + i) * (KT + 1);
      float s = 0.f;
      for (int j = 0; j < KT; j++) s += sr[j];
      l[i] += s;
    }
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float4* vr = (const float4*)(Vt + j * 320 + ds * 20);
      float p0_ = Sp[(qg * 4) * (KT + 1) + j], p1 = Sp[(qg * 4 + 1) * (KT + 1) + j];
      float p2 = Sp[(qg * 4 + 2) * (KT + 1) + j], p3 = Sp[(qg * 4 + 3) * (KT + 1) + j];
#pragma unroll
      for (int d4 = 0; d4 < 4; d4++) {
        float4 v4 = vr[d4];
        const float vv[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int e = 0; e < 4; e++) {
          const int d = 4 * d4 + e;
          acc[0][d] = fmaf(p0_, vv[e], acc[0][d]); acc[1][d] = fmaf(p1, vv[e], acc[1][d]);
          acc[2][d] = fmaf(p2, vv[e], acc[2][d]); acc[3][d] = fmaf(p3, vv[e], acc[3][d]);
        }
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int qq = qg * 4 + i, tok = tb + (qq >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / l[i];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (qq & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

// v5: like v3, but the online softmax runs in layout A right after the scores (all threads busy):
// every lane of a query quad has the full score row; lane qu exponentiates keys j%4 == qu.
extern "C" __global__ void attn_prefill5(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ float Kt[KT * RS];
  __shared__ float Vt[KT * RS2];
  __shared__ float Sp[64 * (KT + 1)];
  __shared__ float Corr[64], Lsum[64];
  const int g = blockIdx.y, tid = threadIdx.x, tb = blockIdx.x * 8;
  const int qa = tid >> 2, qu = tid & 3;
  const int tokA = min(tb + (qa >> 3), T - 1), posA = p0 + tokA;
  float q[64];
  {
    const float* qp = QR + (size_t)tokA * 4096 + (g * 8 + (qa & 7)) * 256 + qu * 64;
#pragma unroll
    for (int d = 0; d < 64; d++) q[d] = qp[d] * 0.0625f;
  }
  float mA = -INF, lA = 0.f;                     // softmax state of query qa (replicated in its 4 lanes)
  const int qg = tid >> 4, ds = tid & 15;
  float acc[4][16];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();
    for (int i = tid; i < KT * 256; i += 256) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kp <= kend) {
#if KV8
        if (kp >= exact_from) {
          size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + d;
          kv = __half2float(Kr[rb]); vv = __half2float(Vr[rb]);
        } else {
          size_t base = KROW(g, kp) * 256 + d, sb = SROW(g, kp) * 8 + (d >> 5);
          kv = (float)((const signed char*)Kc)[base] * __half2float(Ks[sb]);
          vv = (float)((const signed char*)Vc)[base] * __half2float(Vs[sb]);
        }
#else
        size_t base = KROW(g, kp) * 256 + d;
        kv = __half2float(((const __half*)Kc)[base]); vv = __half2float(((const __half*)Vc)[base]);
#endif
      }
      Kt[j * RS + (d >> 6) * 72 + (d & 63)] = kv;
      Vt[j * RS2 + (d >> 4) * 17 + (d & 15)] = vv;
    }
    __syncthreads();
    float sc[KT];
    float tmax = -INF;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      const float* kr = Kt + j * RS + qu * 72;
      float p = 0.f;
#pragma unroll
      for (int d = 0; d < 64; d++) p = fmaf(q[d], kr[d], p);
      p += __shfl_xor_sync(0xffffffffu, p, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2);
      p = (k0 + j <= posA) ? p : -INF;
      sc[j] = p;
      tmax = fmaxf(tmax, p);
    }
    const float mn = fmaxf(mA, tmax);
    const float corr = (mn == -INF) ? 1.f : __expf(mA - mn);
    float psum = 0.f;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      if ((j & 3) == qu) {
        float e = (mn == -INF) ? 0.f : __expf(sc[j] - mn);
        Sp[qa * (KT + 1) + j] = e;
        psum += e;
      }
    }
    psum += __shfl_xor_sync(0xffffffffu, psum, 1);
    psum += __shfl_xor_sync(0xffffffffu, psum, 2);
    lA = lA * corr + psum;
    mA = mn;
    if (qu == 0) Corr[qa] = corr;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float c = Corr[qg * 4 + i];
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= c;
    }
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float* vr = Vt + j * RS2 + ds * 17;
      float p0_ = Sp[(qg * 4) * (KT + 1) + j], p1 = Sp[(qg * 4 + 1) * (KT + 1) + j];
      float p2 = Sp[(qg * 4 + 2) * (KT + 1) + j], p3 = Sp[(qg * 4 + 3) * (KT + 1) + j];
#pragma unroll
      for (int d = 0; d < 16; d++) {
        float v = vr[d];
        acc[0][d] = fmaf(p0_, v, acc[0][d]); acc[1][d] = fmaf(p1, v, acc[1][d]);
        acc[2][d] = fmaf(p2, v, acc[2][d]); acc[3][d] = fmaf(p3, v, acc[3][d]);
      }
    }
  }
  if (qu == 0) Lsum[qa] = lA;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int qq = qg * 4 + i, tok = tb + (qq >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / Lsum[qq];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (qq & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

extern "C" __global__ void attn_prefill6(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ float Kt[KT * RS];
  __shared__ float Vt[KT * RS2];
  __shared__ float Sp[64 * (KT + 1)];
  __shared__ float Corr[64], Lsum[64];
  const int g = blockIdx.y, tid = threadIdx.x, tb = blockIdx.x * 8;
  const int qa = tid >> 2, qu = tid & 3;
  const int tokA = min(tb + (qa >> 3), T - 1), posA = p0 + tokA;
  float q[64];
  {
    const float* qp = QR + (size_t)tokA * 4096 + (g * 8 + (qa & 7)) * 256 + qu * 64;
#pragma unroll
    for (int d = 0; d < 64; d++) q[d] = qp[d] * 0.0625f;
  }
  float mA = -INF, lA = 0.f;                     // softmax state of query qa (replicated in its 4 lanes)
  const int qg = tid >> 4, ds = tid & 15;
  float acc[4][16];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();
    for (int i = tid; i < KT * 256; i += 256) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kp <= kend) {
#if KV8
        if (kp >= exact_from) {
          size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + d;
          kv = __half2float(Kr[rb]); vv = __half2float(Vr[rb]);
        } else {
          size_t base = KROW(g, kp) * 256 + d, sb = SROW(g, kp) * 8 + (d >> 5);
          kv = (float)((const signed char*)Kc)[base] * __half2float(Ks[sb]);
          vv = (float)((const signed char*)Vc)[base] * __half2float(Vs[sb]);
        }
#else
        size_t base = KROW(g, kp) * 256 + d;
        kv = __half2float(((const __half*)Kc)[base]); vv = __half2float(((const __half*)Vc)[base]);
#endif
      }
      Kt[j * RS + (d >> 6) * 72 + (d & 63)] = kv;
      Vt[j * RS2 + (d >> 4) * 17 + (d & 15)] = vv;
    }
    __syncthreads();
    float sc[KT];
    float tmax = -INF;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      const float* kr = Kt + j * RS + qu * 72;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;      // independent chains -> latency hiding
#pragma unroll
      for (int d = 0; d < 64; d += 4) {
        a0 = fmaf(q[d], kr[d], a0); a1 = fmaf(q[d + 1], kr[d + 1], a1);
        a2 = fmaf(q[d + 2], kr[d + 2], a2); a3 = fmaf(q[d + 3], kr[d + 3], a3);
      }
      float p = (a0 + a1) + (a2 + a3);
      p += __shfl_xor_sync(0xffffffffu, p, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2);
      p = (k0 + j <= posA) ? p : -INF;
      sc[j] = p;
      tmax = fmaxf(tmax, p);
    }
    const float mn = fmaxf(mA, tmax);
    const float corr = (mn == -INF) ? 1.f : __expf(mA - mn);
    float psum = 0.f;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      if ((j & 3) == qu) {
        float e = (mn == -INF) ? 0.f : __expf(sc[j] - mn);
        Sp[qa * (KT + 1) + j] = e;
        psum += e;
      }
    }
    psum += __shfl_xor_sync(0xffffffffu, psum, 1);
    psum += __shfl_xor_sync(0xffffffffu, psum, 2);
    lA = lA * corr + psum;
    mA = mn;
    if (qu == 0) Corr[qa] = corr;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float c = Corr[qg * 4 + i];
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= c;
    }
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float* vr = Vt + j * RS2 + ds * 17;
      float p0_ = Sp[(qg * 4) * (KT + 1) + j], p1 = Sp[(qg * 4 + 1) * (KT + 1) + j];
      float p2 = Sp[(qg * 4 + 2) * (KT + 1) + j], p3 = Sp[(qg * 4 + 3) * (KT + 1) + j];
#pragma unroll
      for (int d = 0; d < 16; d++) {
        float v = vr[d];
        acc[0][d] = fmaf(p0_, v, acc[0][d]); acc[1][d] = fmaf(p1, v, acc[1][d]);
        acc[2][d] = fmaf(p2, v, acc[2][d]); acc[3][d] = fmaf(p3, v, acc[3][d]);
      }
    }
  }
  if (qu == 0) Lsum[qa] = lA;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int qq = qg * 4 + i, tok = tb + (qq >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / Lsum[qq];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (qq & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

extern "C" __global__ void attn_prefill7(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ float Kt[KT * RS];
  __shared__ float Vt[KT * RS2];
  __shared__ float Sp[64 * (KT + 1)];
  __shared__ float Corr[64], Lsum[64];
  const int g = blockIdx.y, tid = threadIdx.x, tb = blockIdx.x * 8;
  const int qa = tid >> 2, qu = tid & 3;
  const int tokA = min(tb + (qa >> 3), T - 1), posA = p0 + tokA;
  float q[64];
  {
    const float* qp = QR + (size_t)tokA * 4096 + (g * 8 + (qa & 7)) * 256 + qu * 64;
#pragma unroll
    for (int d = 0; d < 64; d++) q[d] = qp[d] * 0.0625f;
  }
  float mA = -INF, lA = 0.f;                     // softmax state of query qa (replicated in its 4 lanes)
  const int qg = tid >> 4, ds = tid & 15;
  float acc[4][16];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();
    for (int i = tid; i < KT * 256; i += 256) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kp <= kend) {
#if KV8
        if (kp >= exact_from) {
          size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + d;
          kv = __half2float(Kr[rb]); vv = __half2float(Vr[rb]);
        } else {
          size_t base = KROW(g, kp) * 256 + d, sb = SROW(g, kp) * 8 + (d >> 5);
          kv = (float)((const signed char*)Kc)[base] * __half2float(Ks[sb]);
          vv = (float)((const signed char*)Vc)[base] * __half2float(Vs[sb]);
        }
#else
        size_t base = KROW(g, kp) * 256 + d;
        kv = __half2float(((const __half*)Kc)[base]); vv = __half2float(((const __half*)Vc)[base]);
#endif
      }
      Kt[j * RS + (d >> 6) * 72 + (d & 63)] = kv;
      Vt[j * RS2 + (d >> 4) * 17 + (d & 15)] = vv;
    }
    __syncthreads();
    float sc[KT];
    float tmax = -INF;
#pragma unroll
    for (int j = 0; j < KT; j += 2) {            // two keys at once: 8 independent FMA chains
      const float* kr = Kt + j * RS + qu * 72;
      const float* kr2 = kr + RS;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
#pragma unroll
      for (int d = 0; d < 64; d += 4) {
        a0 = fmaf(q[d], kr[d], a0); b0 = fmaf(q[d], kr2[d], b0);
        a1 = fmaf(q[d + 1], kr[d + 1], a1); b1 = fmaf(q[d + 1], kr2[d + 1], b1);
        a2 = fmaf(q[d + 2], kr[d + 2], a2); b2 = fmaf(q[d + 2], kr2[d + 2], b2);
        a3 = fmaf(q[d + 3], kr[d + 3], a3); b3 = fmaf(q[d + 3], kr2[d + 3], b3);
      }
      float p = (a0 + a1) + (a2 + a3), p2 = (b0 + b1) + (b2 + b3);
      p += __shfl_xor_sync(0xffffffffu, p, 1); p2 += __shfl_xor_sync(0xffffffffu, p2, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2); p2 += __shfl_xor_sync(0xffffffffu, p2, 2);
      p = (k0 + j <= posA) ? p : -INF;
      p2 = (k0 + j + 1 <= posA) ? p2 : -INF;
      sc[j] = p; sc[j + 1] = p2;
      tmax = fmaxf(tmax, fmaxf(p, p2));
    }
    const float mn = fmaxf(mA, tmax);
    const float corr = (mn == -INF) ? 1.f : __expf(mA - mn);
    float psum = 0.f;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      if ((j & 3) == qu) {
        float e = (mn == -INF) ? 0.f : __expf(sc[j] - mn);
        Sp[qa * (KT + 1) + j] = e;
        psum += e;
      }
    }
    psum += __shfl_xor_sync(0xffffffffu, psum, 1);
    psum += __shfl_xor_sync(0xffffffffu, psum, 2);
    lA = lA * corr + psum;
    mA = mn;
    if (qu == 0) Corr[qa] = corr;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float c = Corr[qg * 4 + i];
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= c;
    }
#ifndef NOPV
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float* vr = Vt + j * RS2 + ds * 17;
      float p0_ = Sp[(qg * 4) * (KT + 1) + j], p1 = Sp[(qg * 4 + 1) * (KT + 1) + j];
      float p2 = Sp[(qg * 4 + 2) * (KT + 1) + j], p3 = Sp[(qg * 4 + 3) * (KT + 1) + j];
#pragma unroll
      for (int d = 0; d < 16; d++) {
        float v = vr[d];
        acc[0][d] = fmaf(p0_, v, acc[0][d]); acc[1][d] = fmaf(p1, v, acc[1][d]);
        acc[2][d] = fmaf(p2, v, acc[2][d]); acc[3][d] = fmaf(p3, v, acc[3][d]);
      }
    }
#endif
  }
  if (qu == 0) Lsum[qa] = lA;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int qq = qg * 4 + i, tok = tb + (qq >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / Lsum[qq];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (qq & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

// v9: 512 threads (2x the warps of v7 at the same 64 queries per K/V tile); each thread holds half of v7's registers.
// v7 runs at 193 registers -> one 8-warp block per SM, idle in the load/QK/softmax/PV phases. Same math, other partial sums.
extern "C" __global__ void __launch_bounds__(512, 1) attn_prefill9(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ float Kt[KT * RS];
  __shared__ float Vt[KT * RS2];
  __shared__ float Sp[64 * (KT + 1)];
  __shared__ float Corr[64], Lsum[64];
  const int g = blockIdx.y, tid = threadIdx.x, tb = blockIdx.x * 8;
  const int qa = tid >> 3, qu = tid & 7;       // 512 threads: 64 queries x 8 parts of 32 dims
  const int tokA = min(tb + (qa >> 3), T - 1), posA = p0 + tokA;
  float q[32];
  {
    const float* qp = QR + (size_t)tokA * 4096 + (g * 8 + (qa & 7)) * 256 + qu * 32;
#pragma unroll
    for (int d = 0; d < 32; d++) q[d] = qp[d] * 0.0625f;
  }
  float mA = -INF, lA = 0.f;                     // softmax state of query qa (replicated in its 4 lanes)
  const int qg = tid >> 4, ds = tid & 15;      // PV: 32 groups of 2 queries x 16 slices of 16 dims
  float acc[2][16];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();
    for (int i = tid; i < KT * 256; i += 512) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kp <= kend) {
#if KV8
        if (kp >= exact_from) {
          size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + d;
          kv = __half2float(Kr[rb]); vv = __half2float(Vr[rb]);
        } else {
          size_t base = KROW(g, kp) * 256 + d, sb = SROW(g, kp) * 8 + (d >> 5);
          kv = (float)((const signed char*)Kc)[base] * __half2float(Ks[sb]);
          vv = (float)((const signed char*)Vc)[base] * __half2float(Vs[sb]);
        }
#else
        size_t base = KROW(g, kp) * 256 + d;
        kv = __half2float(((const __half*)Kc)[base]); vv = __half2float(((const __half*)Vc)[base]);
#endif
      }
      Kt[j * RS + (d >> 5) * 36 + (d & 31)] = kv;
      Vt[j * RS2 + (d >> 4) * 17 + (d & 15)] = vv;
    }
    __syncthreads();
    float sc[KT];
    float tmax = -INF;
#pragma unroll
    for (int j = 0; j < KT; j += 2) {            // two keys at once: 8 independent FMA chains
      const float* kr = Kt + j * RS + qu * 36;
      const float* kr2 = kr + RS;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
#pragma unroll
      for (int d = 0; d < 32; d += 4) {
        a0 = fmaf(q[d], kr[d], a0); b0 = fmaf(q[d], kr2[d], b0);
        a1 = fmaf(q[d + 1], kr[d + 1], a1); b1 = fmaf(q[d + 1], kr2[d + 1], b1);
        a2 = fmaf(q[d + 2], kr[d + 2], a2); b2 = fmaf(q[d + 2], kr2[d + 2], b2);
        a3 = fmaf(q[d + 3], kr[d + 3], a3); b3 = fmaf(q[d + 3], kr2[d + 3], b3);
      }
      float p = (a0 + a1) + (a2 + a3), p2 = (b0 + b1) + (b2 + b3);
      p += __shfl_xor_sync(0xffffffffu, p, 1); p2 += __shfl_xor_sync(0xffffffffu, p2, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2); p2 += __shfl_xor_sync(0xffffffffu, p2, 2);
      p += __shfl_xor_sync(0xffffffffu, p, 4); p2 += __shfl_xor_sync(0xffffffffu, p2, 4);
      p = (k0 + j <= posA) ? p : -INF;
      p2 = (k0 + j + 1 <= posA) ? p2 : -INF;
      sc[j] = p; sc[j + 1] = p2;
      tmax = fmaxf(tmax, fmaxf(p, p2));
    }
    const float mn = fmaxf(mA, tmax);
    const float corr = (mn == -INF) ? 1.f : __expf(mA - mn);
    float psum = 0.f;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      if ((j & 7) == qu) {
        float e = (mn == -INF) ? 0.f : __expf(sc[j] - mn);
        Sp[qa * (KT + 1) + j] = e;
        psum += e;
      }
    }
    psum += __shfl_xor_sync(0xffffffffu, psum, 1);
    psum += __shfl_xor_sync(0xffffffffu, psum, 2);
    psum += __shfl_xor_sync(0xffffffffu, psum, 4);
    lA = lA * corr + psum;
    mA = mn;
    if (qu == 0) Corr[qa] = corr;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const float c = Corr[qg * 2 + i];
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= c;
    }
#ifndef NOPV
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float* vr = Vt + j * RS2 + ds * 17;
      float p0_ = Sp[(qg * 2) * (KT + 1) + j], p1 = Sp[(qg * 2 + 1) * (KT + 1) + j];
#pragma unroll
      for (int d = 0; d < 16; d++) {
        float v = vr[d];
        acc[0][d] = fmaf(p0_, v, acc[0][d]); acc[1][d] = fmaf(p1, v, acc[1][d]);
      }
    }
#endif
  }
  if (qu == 0) Lsum[qa] = lA;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int qq = qg * 2 + i, tok = tb + (qq >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / Lsum[qq];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (qq & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

// v10 = v9 + 16-byte shared loads (float4) for K and V; V rows padded to 20 floats per 16 dims (aligned, conflict-free).
// v9: 512 threads (2x the warps of v7 at the same 64 queries per K/V tile); each thread holds half of v7's registers.
// v7 runs at 193 registers -> one 8-warp block per SM, idle in the load/QK/softmax/PV phases. Same math, other partial sums.
extern "C" __global__ void __launch_bounds__(512, 1) attn_prefill10(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ __align__(16) float Kt[KT * RS];
  __shared__ __align__(16) float Vt[KT * 320];
  __shared__ float Sp[64 * (KT + 1)];
  __shared__ float Corr[64], Lsum[64];
  const int g = blockIdx.y, tid = threadIdx.x, tb = blockIdx.x * 8;
  const int qa = tid >> 3, qu = tid & 7;       // 512 threads: 64 queries x 8 parts of 32 dims
  const int tokA = min(tb + (qa >> 3), T - 1), posA = p0 + tokA;
  float q[32];
  {
    const float* qp = QR + (size_t)tokA * 4096 + (g * 8 + (qa & 7)) * 256 + qu * 32;
#pragma unroll
    for (int d = 0; d < 32; d++) q[d] = qp[d] * 0.0625f;
  }
  float mA = -INF, lA = 0.f;                     // softmax state of query qa (replicated in its 4 lanes)
  const int qg = tid >> 4, ds = tid & 15;      // PV: 32 groups of 2 queries x 16 slices of 16 dims
  float acc[2][16];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();
    for (int i = tid; i < KT * 256; i += 512) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kp <= kend) {
#if KV8
        if (kp >= exact_from) {
          size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + d;
          kv = __half2float(Kr[rb]); vv = __half2float(Vr[rb]);
        } else {
          size_t base = KROW(g, kp) * 256 + d, sb = SROW(g, kp) * 8 + (d >> 5);
          kv = (float)((const signed char*)Kc)[base] * __half2float(Ks[sb]);
          vv = (float)((const signed char*)Vc)[base] * __half2float(Vs[sb]);
        }
#else
        size_t base = KROW(g, kp) * 256 + d;
        kv = __half2float(((const __half*)Kc)[base]); vv = __half2float(((const __half*)Vc)[base]);
#endif
      }
      Kt[j * RS + (d >> 5) * 36 + (d & 31)] = kv;
      Vt[j * 320 + (d >> 4) * 20 + (d & 15)] = vv;
    }
    __syncthreads();
    float sc[KT];
    float tmax = -INF;
#pragma unroll
    for (int j = 0; j < KT; j += 2) {            // two keys at once: 8 independent FMA chains
      const float* kr = Kt + j * RS + qu * 36;
      const float* kr2 = kr + RS;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
#pragma unroll
      for (int d = 0; d < 32; d += 4) {
        const float4 k1 = *(const float4*)(kr + d), k2 = *(const float4*)(kr2 + d);
        a0 = fmaf(q[d], k1.x, a0); b0 = fmaf(q[d], k2.x, b0);
        a1 = fmaf(q[d + 1], k1.y, a1); b1 = fmaf(q[d + 1], k2.y, b1);
        a2 = fmaf(q[d + 2], k1.z, a2); b2 = fmaf(q[d + 2], k2.z, b2);
        a3 = fmaf(q[d + 3], k1.w, a3); b3 = fmaf(q[d + 3], k2.w, b3);
      }
      float p = (a0 + a1) + (a2 + a3), p2 = (b0 + b1) + (b2 + b3);
      p += __shfl_xor_sync(0xffffffffu, p, 1); p2 += __shfl_xor_sync(0xffffffffu, p2, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2); p2 += __shfl_xor_sync(0xffffffffu, p2, 2);
      p += __shfl_xor_sync(0xffffffffu, p, 4); p2 += __shfl_xor_sync(0xffffffffu, p2, 4);
      p = (k0 + j <= posA) ? p : -INF;
      p2 = (k0 + j + 1 <= posA) ? p2 : -INF;
      sc[j] = p; sc[j + 1] = p2;
      tmax = fmaxf(tmax, fmaxf(p, p2));
    }
    const float mn = fmaxf(mA, tmax);
    const float corr = (mn == -INF) ? 1.f : __expf(mA - mn);
    float psum = 0.f;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      if ((j & 7) == qu) {
        float e = (mn == -INF) ? 0.f : __expf(sc[j] - mn);
        Sp[qa * (KT + 1) + j] = e;
        psum += e;
      }
    }
    psum += __shfl_xor_sync(0xffffffffu, psum, 1);
    psum += __shfl_xor_sync(0xffffffffu, psum, 2);
    psum += __shfl_xor_sync(0xffffffffu, psum, 4);
    lA = lA * corr + psum;
    mA = mn;
    if (qu == 0) Corr[qa] = corr;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const float c = Corr[qg * 2 + i];
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= c;
    }
#ifndef NOPV
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float* vr = Vt + j * 320 + ds * 20;
      float p0_ = Sp[(qg * 2) * (KT + 1) + j], p1 = Sp[(qg * 2 + 1) * (KT + 1) + j];
#pragma unroll
      for (int d4 = 0; d4 < 16; d4 += 4) {
        const float4 v4 = *(const float4*)(vr + d4);
        const float vv[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int t = 0; t < 4; t++) { acc[0][d4 + t] = fmaf(p0_, vv[t], acc[0][d4 + t]); acc[1][d4 + t] = fmaf(p1, vv[t], acc[1][d4 + t]); }
      }
    }
#endif
  }
  if (qu == 0) Lsum[qa] = lA;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int qq = qg * 2 + i, tok = tb + (qq >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / Lsum[qq];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (qq & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

// v11 = v10 + software pipelining (int8 KV only): the next K/V tile is fetched into registers (8 dims per thread,
// 8-byte int8 / 16-byte fp16 loads) while the current tile is computed, so the global latency hides behind the math.
// v10 = v9 + 16-byte shared loads (float4) for K and V; V rows padded to 20 floats per 16 dims (aligned, conflict-free).
// v9: 512 threads (2x the warps of v7 at the same 64 queries per K/V tile); each thread holds half of v7's registers.
// v7 runs at 193 registers -> one 8-warp block per SM, idle in the load/QK/softmax/PV phases. Same math, other partial sums.
extern "C" __global__ void __launch_bounds__(512, 1) attn_prefill11(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ __align__(16) float Kt[KT * RS];
  __shared__ __align__(16) float Vt[KT * 320];
  __shared__ float Sp[64 * (KT + 1)];
  __shared__ float Corr[64], Lsum[64];
  const int g = blockIdx.y, tid = threadIdx.x, tb = blockIdx.x * 8;
  const int qa = tid >> 3, qu = tid & 7;       // 512 threads: 64 queries x 8 parts of 32 dims
  const int tokA = min(tb + (qa >> 3), T - 1), posA = p0 + tokA;
  float q[32];
  {
    const float* qp = QR + (size_t)tokA * 4096 + (g * 8 + (qa & 7)) * 256 + qu * 32;
#pragma unroll
    for (int d = 0; d < 32; d++) q[d] = qp[d] * 0.0625f;
  }
  float mA = -INF, lA = 0.f;                     // softmax state of query qa (replicated in its 4 lanes)
  const int qg = tid >> 4, ds = tid & 15;      // PV: 32 groups of 2 queries x 16 slices of 16 dims
  float acc[2][16];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  const int lj = tid >> 5, ld0 = (tid & 31) * 8;           // fetch mapping: key lj of the tile, dims ld0 .. ld0+7
  uint4 rk = make_uint4(0, 0, 0, 0), rv = make_uint4(0, 0, 0, 0); float rks = 0.f, rvs = 0.f; int rmode = 0;   // 0 none, 1 int8, 2 fp16
#define FETCH(K0) { const int kp = (K0) + lj; rmode = 0;                                                       \
    if (kp <= kend) {                                                                                        \
      if (kp >= exact_from) { const size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + ld0;               \
        rk = *(const uint4*)(Kr + rb); rv = *(const uint4*)(Vr + rb); rmode = 2; }                           \
      else { const size_t base = KROW(g, kp) * 256 + ld0, sb = SROW(g, kp) * 8 + (ld0 >> 5);                \
        const uint2 k8 = *(const uint2*)((const signed char*)Kc + base), v8 = *(const uint2*)((const signed char*)Vc + base); \
        rk.x = k8.x; rk.y = k8.y; rv.x = v8.x; rv.y = v8.y; rks = __half2float(Ks[sb]); rvs = __half2float(Vs[sb]); rmode = 1; } } }
  FETCH(0)
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();                                          // everyone is done with the previous tile
    {
      float kf[8], vf[8];
      if (rmode == 1) {
        const unsigned kw[2] = {rk.x, rk.y}, vw[2] = {rv.x, rv.y};
#pragma unroll
        for (int t = 0; t < 8; t++) {
          kf[t] = (float)(signed char)(kw[t >> 2] >> (8 * (t & 3))) * rks;
          vf[t] = (float)(signed char)(vw[t >> 2] >> (8 * (t & 3))) * rvs;
        }
      } else if (rmode == 2) {
        const __half* kh = (const __half*)&rk; const __half* vh = (const __half*)&rv;
#pragma unroll
        for (int t = 0; t < 8; t++) { kf[t] = __half2float(kh[t]); vf[t] = __half2float(vh[t]); }
      } else {
#pragma unroll
        for (int t = 0; t < 8; t++) { kf[t] = 0.f; vf[t] = 0.f; }
      }
      float* kd = Kt + lj * RS + (ld0 >> 5) * 36 + (ld0 & 31);
      float* vd = Vt + lj * 320 + (ld0 >> 4) * 20 + (ld0 & 15);
      *(float4*)kd = make_float4(kf[0], kf[1], kf[2], kf[3]); *(float4*)(kd + 4) = make_float4(kf[4], kf[5], kf[6], kf[7]);
      *(float4*)vd = make_float4(vf[0], vf[1], vf[2], vf[3]); *(float4*)(vd + 4) = make_float4(vf[4], vf[5], vf[6], vf[7]);
    }
    __syncthreads();
    if (k0 + KT <= kend) FETCH(k0 + KT)                        // in flight while this tile is computed
#undef FETCH
    float sc[KT];
    float tmax = -INF;
#pragma unroll
    for (int j = 0; j < KT; j += 2) {            // two keys at once: 8 independent FMA chains
      const float* kr = Kt + j * RS + qu * 36;
      const float* kr2 = kr + RS;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
#pragma unroll
      for (int d = 0; d < 32; d += 4) {
        const float4 k1 = *(const float4*)(kr + d), k2 = *(const float4*)(kr2 + d);
        a0 = fmaf(q[d], k1.x, a0); b0 = fmaf(q[d], k2.x, b0);
        a1 = fmaf(q[d + 1], k1.y, a1); b1 = fmaf(q[d + 1], k2.y, b1);
        a2 = fmaf(q[d + 2], k1.z, a2); b2 = fmaf(q[d + 2], k2.z, b2);
        a3 = fmaf(q[d + 3], k1.w, a3); b3 = fmaf(q[d + 3], k2.w, b3);
      }
      float p = (a0 + a1) + (a2 + a3), p2 = (b0 + b1) + (b2 + b3);
      p += __shfl_xor_sync(0xffffffffu, p, 1); p2 += __shfl_xor_sync(0xffffffffu, p2, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2); p2 += __shfl_xor_sync(0xffffffffu, p2, 2);
      p += __shfl_xor_sync(0xffffffffu, p, 4); p2 += __shfl_xor_sync(0xffffffffu, p2, 4);
      p = (k0 + j <= posA) ? p : -INF;
      p2 = (k0 + j + 1 <= posA) ? p2 : -INF;
      sc[j] = p; sc[j + 1] = p2;
      tmax = fmaxf(tmax, fmaxf(p, p2));
    }
    const float mn = fmaxf(mA, tmax);
    const float corr = (mn == -INF) ? 1.f : __expf(mA - mn);
    float psum = 0.f;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      if ((j & 7) == qu) {
        float e = (mn == -INF) ? 0.f : __expf(sc[j] - mn);
        Sp[qa * (KT + 1) + j] = e;
        psum += e;
      }
    }
    psum += __shfl_xor_sync(0xffffffffu, psum, 1);
    psum += __shfl_xor_sync(0xffffffffu, psum, 2);
    psum += __shfl_xor_sync(0xffffffffu, psum, 4);
    lA = lA * corr + psum;
    mA = mn;
    if (qu == 0) Corr[qa] = corr;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const float c = Corr[qg * 2 + i];
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= c;
    }
#ifndef NOPV
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float* vr = Vt + j * 320 + ds * 20;
      float p0_ = Sp[(qg * 2) * (KT + 1) + j], p1 = Sp[(qg * 2 + 1) * (KT + 1) + j];
#pragma unroll
      for (int d4 = 0; d4 < 16; d4 += 4) {
        const float4 v4 = *(const float4*)(vr + d4);
        const float vv[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int t = 0; t < 4; t++) { acc[0][d4 + t] = fmaf(p0_, vv[t], acc[0][d4 + t]); acc[1][d4 + t] = fmaf(p1, vv[t], acc[1][d4 + t]); }
      }
    }
#endif
  }
  if (qu == 0) Lsum[qa] = lA;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int qq = qg * 2 + i, tok = tb + (qq >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / Lsum[qq];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (qq & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

// v12 = v11 with 256-thread blocks of 32 queries (4 tokens x 8 heads): 2 blocks per SM, one computes while the other syncs.
// v11 = v10 + software pipelining (int8 KV only): the next K/V tile is fetched into registers (8 dims per thread,
// 8-byte int8 / 16-byte fp16 loads) while the current tile is computed, so the global latency hides behind the math.
// v10 = v9 + 16-byte shared loads (float4) for K and V; V rows padded to 20 floats per 16 dims (aligned, conflict-free).
// v9: 512 threads (2x the warps of v7 at the same 64 queries per K/V tile); each thread holds half of v7's registers.
// v7 runs at 193 registers -> one 8-warp block per SM, idle in the load/QK/softmax/PV phases. Same math, other partial sums.
extern "C" __global__ void __launch_bounds__(256, 2) attn_prefill12(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ __align__(16) float Kt[KT * RS];
  __shared__ __align__(16) float Vt[KT * 320];
  __shared__ float Sp[64 * (KT + 1)];
  __shared__ float Corr[64], Lsum[64];
  const int g = blockIdx.y, tid = threadIdx.x, tb = blockIdx.x * 4;
  const int qa = tid >> 3, qu = tid & 7;       // 512 threads: 64 queries x 8 parts of 32 dims
  const int tokA = min(tb + (qa >> 3), T - 1), posA = p0 + tokA;
  float q[32];
  {
    const float* qp = QR + (size_t)tokA * 4096 + (g * 8 + (qa & 7)) * 256 + qu * 32;
#pragma unroll
    for (int d = 0; d < 32; d++) q[d] = qp[d] * 0.0625f;
  }
  float mA = -INF, lA = 0.f;                     // softmax state of query qa (replicated in its 4 lanes)
  const int qg = tid >> 4, ds = tid & 15;      // PV: 32 groups of 2 queries x 16 slices of 16 dims
  float acc[2][16];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
  const int kend = p0 + min(tb + 3, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  const int lj = tid >> 4, ld0 = (tid & 15) * 16;           // fetch mapping: key lj of the tile, dims ld0 .. ld0+7
  uint4 rk[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)}, rv[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)}; float rks = 0.f, rvs = 0.f; int rmode = 0;   // 0 none, 1 int8, 2 fp16
#define FETCH(K0) { const int kp = (K0) + lj; rmode = 0;                                                       \
    if (kp <= kend) {                                                                                        \
      if (kp >= exact_from) { const size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + ld0;               \
        rk[0] = *(const uint4*)(Kr + rb); rk[1] = *(const uint4*)(Kr + rb + 8); rv[0] = *(const uint4*)(Vr + rb); rv[1] = *(const uint4*)(Vr + rb + 8); rmode = 2; }                           \
      else { const size_t base = KROW(g, kp) * 256 + ld0, sb = SROW(g, kp) * 8 + (ld0 >> 5);                \
        rk[0] = *(const uint4*)((const signed char*)Kc + base); rv[0] = *(const uint4*)((const signed char*)Vc + base); \
        rks = __half2float(Ks[sb]); rvs = __half2float(Vs[sb]); rmode = 1; } } }
  FETCH(0)
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();                                          // everyone is done with the previous tile
    {
      float kf[16], vf[16];
      if (rmode == 1) {
        const unsigned kw[4] = {rk[0].x, rk[0].y, rk[0].z, rk[0].w}, vw[4] = {rv[0].x, rv[0].y, rv[0].z, rv[0].w};
#pragma unroll
        for (int t = 0; t < 16; t++) {
          kf[t] = (float)(signed char)(kw[t >> 2] >> (8 * (t & 3))) * rks;
          vf[t] = (float)(signed char)(vw[t >> 2] >> (8 * (t & 3))) * rvs;
        }
      } else if (rmode == 2) {
        const __half* kh = (const __half*)rk; const __half* vh = (const __half*)rv;
#pragma unroll
        for (int t = 0; t < 16; t++) { kf[t] = __half2float(kh[t]); vf[t] = __half2float(vh[t]); }
      } else {
#pragma unroll
        for (int t = 0; t < 16; t++) { kf[t] = 0.f; vf[t] = 0.f; }
      }
      float* kd = Kt + lj * RS + (ld0 >> 5) * 36 + (ld0 & 31);
      float* vd = Vt + lj * 320 + (ld0 >> 4) * 20;
#pragma unroll
      for (int t = 0; t < 16; t += 4) { *(float4*)(kd + t) = make_float4(kf[t], kf[t + 1], kf[t + 2], kf[t + 3]);
                                       *(float4*)(vd + t) = make_float4(vf[t], vf[t + 1], vf[t + 2], vf[t + 3]); }
    }
    __syncthreads();
    if (k0 + KT <= kend) FETCH(k0 + KT)                        // in flight while this tile is computed
#undef FETCH
    float sc[KT];
    float tmax = -INF;
#pragma unroll
    for (int j = 0; j < KT; j += 2) {            // two keys at once: 8 independent FMA chains
      const float* kr = Kt + j * RS + qu * 36;
      const float* kr2 = kr + RS;
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
#pragma unroll
      for (int d = 0; d < 32; d += 4) {
        const float4 k1 = *(const float4*)(kr + d), k2 = *(const float4*)(kr2 + d);
        a0 = fmaf(q[d], k1.x, a0); b0 = fmaf(q[d], k2.x, b0);
        a1 = fmaf(q[d + 1], k1.y, a1); b1 = fmaf(q[d + 1], k2.y, b1);
        a2 = fmaf(q[d + 2], k1.z, a2); b2 = fmaf(q[d + 2], k2.z, b2);
        a3 = fmaf(q[d + 3], k1.w, a3); b3 = fmaf(q[d + 3], k2.w, b3);
      }
      float p = (a0 + a1) + (a2 + a3), p2 = (b0 + b1) + (b2 + b3);
      p += __shfl_xor_sync(0xffffffffu, p, 1); p2 += __shfl_xor_sync(0xffffffffu, p2, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2); p2 += __shfl_xor_sync(0xffffffffu, p2, 2);
      p += __shfl_xor_sync(0xffffffffu, p, 4); p2 += __shfl_xor_sync(0xffffffffu, p2, 4);
      p = (k0 + j <= posA) ? p : -INF;
      p2 = (k0 + j + 1 <= posA) ? p2 : -INF;
      sc[j] = p; sc[j + 1] = p2;
      tmax = fmaxf(tmax, fmaxf(p, p2));
    }
    const float mn = fmaxf(mA, tmax);
    const float corr = (mn == -INF) ? 1.f : __expf(mA - mn);
    float psum = 0.f;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      if ((j & 7) == qu) {
        float e = (mn == -INF) ? 0.f : __expf(sc[j] - mn);
        Sp[qa * (KT + 1) + j] = e;
        psum += e;
      }
    }
    psum += __shfl_xor_sync(0xffffffffu, psum, 1);
    psum += __shfl_xor_sync(0xffffffffu, psum, 2);
    psum += __shfl_xor_sync(0xffffffffu, psum, 4);
    lA = lA * corr + psum;
    mA = mn;
    if (qu == 0) Corr[qa] = corr;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 2; i++) {
      const float c = Corr[qg * 2 + i];
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= c;
    }
#ifndef NOPV
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float* vr = Vt + j * 320 + ds * 20;
      float p0_ = Sp[(qg * 2) * (KT + 1) + j], p1 = Sp[(qg * 2 + 1) * (KT + 1) + j];
#pragma unroll
      for (int d4 = 0; d4 < 16; d4 += 4) {
        const float4 v4 = *(const float4*)(vr + d4);
        const float vv[4] = {v4.x, v4.y, v4.z, v4.w};
#pragma unroll
        for (int t = 0; t < 4; t++) { acc[0][d4 + t] = fmaf(p0_, vv[t], acc[0][d4 + t]); acc[1][d4 + t] = fmaf(p1, vv[t], acc[1][d4 + t]); }
      }
    }
#endif
  }
  if (qu == 0) Lsum[qa] = lA;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2; i++) {
    const int qq = qg * 2 + i, tok = tb + (qq >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / Lsum[qq];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (qq & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

extern "C" __global__ void attn_prefill8(const float* __restrict__ QR, const float* __restrict__ GATE, int T, int p0,
    const void* __restrict__ Kc, const void* __restrict__ Vc, const __half* __restrict__ Ks, const __half* __restrict__ Vs,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr, int maxT, int ring, int win, float* __restrict__ O)
{
  __shared__ __align__(16) float Kt[KT * RS];
  __shared__ float Vt[KT * RS2];
  __shared__ float Sp[64 * (KT + 1)];
  __shared__ float Corr[64], Lsum[64];
  const int g = blockIdx.y, tid = threadIdx.x, tb = blockIdx.x * 8;
  const int qa = tid >> 2, qu = tid & 3;
  const int tokA = min(tb + (qa >> 3), T - 1), posA = p0 + tokA;
  float q[64];
  {
    const float* qp = QR + (size_t)tokA * 4096 + (g * 8 + (qa & 7)) * 256 + qu * 64;
#pragma unroll
    for (int d = 0; d < 64; d++) q[d] = qp[d] * 0.0625f;
  }
  float mA = -INF, lA = 0.f;                     // softmax state of query qa (replicated in its 4 lanes)
  const int qg = tid >> 4, ds = tid & 15;
  float acc[4][16];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int d = 0; d < 16; d++) acc[i][d] = 0.f;
  const int kend = p0 + min(tb + 7, T - 1);
  const int exact_from = p0 + tb - win + 1;
  const size_t hb = (size_t)g * maxT;
  for (int k0 = 0; k0 <= kend; k0 += KT) {
    __syncthreads();
    for (int i = tid; i < KT * 256; i += 256) {
      int j = i >> 8, d = i & 255, kp = k0 + j;
      float kv = 0.f, vv = 0.f;
      if (kp <= kend) {
#if KV8
        if (kp >= exact_from) {
          size_t rb = ((size_t)g * ring + (kp % ring)) * 256 + d;
          kv = __half2float(Kr[rb]); vv = __half2float(Vr[rb]);
        } else {
          size_t base = KROW(g, kp) * 256 + d, sb = SROW(g, kp) * 8 + (d >> 5);
          kv = (float)((const signed char*)Kc)[base] * __half2float(Ks[sb]);
          vv = (float)((const signed char*)Vc)[base] * __half2float(Vs[sb]);
        }
#else
        size_t base = KROW(g, kp) * 256 + d;
        kv = __half2float(((const __half*)Kc)[base]); vv = __half2float(((const __half*)Vc)[base]);
#endif
      }
      Kt[j * RS + (d >> 6) * 72 + (d & 63)] = kv;
      Vt[j * RS2 + (d >> 4) * 17 + (d & 15)] = vv;
    }
    __syncthreads();
    float sc[KT];
    float tmax = -INF;
#pragma unroll
    for (int j = 0; j < KT; j += 2) {            // two keys at once: 8 independent FMA chains
      const float4* kr = (const float4*)(Kt + j * RS + qu * 72);
      const float4* kr2 = (const float4*)(Kt + (j + 1) * RS + qu * 72);
      float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f, b0 = 0.f, b1 = 0.f, b2 = 0.f, b3 = 0.f;
#pragma unroll
      for (int d4 = 0; d4 < 16; d4++) {
        const float4 k = kr[d4], k2 = kr2[d4];
        const int d = 4 * d4;
        a0 = fmaf(q[d], k.x, a0); b0 = fmaf(q[d], k2.x, b0);
        a1 = fmaf(q[d + 1], k.y, a1); b1 = fmaf(q[d + 1], k2.y, b1);
        a2 = fmaf(q[d + 2], k.z, a2); b2 = fmaf(q[d + 2], k2.z, b2);
        a3 = fmaf(q[d + 3], k.w, a3); b3 = fmaf(q[d + 3], k2.w, b3);
      }
      float p = (a0 + a1) + (a2 + a3), p2 = (b0 + b1) + (b2 + b3);
      p += __shfl_xor_sync(0xffffffffu, p, 1); p2 += __shfl_xor_sync(0xffffffffu, p2, 1);
      p += __shfl_xor_sync(0xffffffffu, p, 2); p2 += __shfl_xor_sync(0xffffffffu, p2, 2);
      p = (k0 + j <= posA) ? p : -INF;
      p2 = (k0 + j + 1 <= posA) ? p2 : -INF;
      sc[j] = p; sc[j + 1] = p2;
      tmax = fmaxf(tmax, fmaxf(p, p2));
    }
    const float mn = fmaxf(mA, tmax);
    const float corr = (mn == -INF) ? 1.f : __expf(mA - mn);
    float psum = 0.f;
#pragma unroll
    for (int j = 0; j < KT; j++) {
      if ((j & 3) == qu) {
        float e = (mn == -INF) ? 0.f : __expf(sc[j] - mn);
        Sp[qa * (KT + 1) + j] = e;
        psum += e;
      }
    }
    psum += __shfl_xor_sync(0xffffffffu, psum, 1);
    psum += __shfl_xor_sync(0xffffffffu, psum, 2);
    lA = lA * corr + psum;
    mA = mn;
    if (qu == 0) Corr[qa] = corr;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
      const float c = Corr[qg * 4 + i];
#pragma unroll
      for (int d = 0; d < 16; d++) acc[i][d] *= c;
    }
#ifndef NOPV
#pragma unroll 4
    for (int j = 0; j < KT; j++) {
      const float* vr = Vt + j * RS2 + ds * 17;
      float p0_ = Sp[(qg * 4) * (KT + 1) + j], p1 = Sp[(qg * 4 + 1) * (KT + 1) + j];
      float p2 = Sp[(qg * 4 + 2) * (KT + 1) + j], p3 = Sp[(qg * 4 + 3) * (KT + 1) + j];
#pragma unroll
      for (int d = 0; d < 16; d++) {
        float v = vr[d];
        acc[0][d] = fmaf(p0_, v, acc[0][d]); acc[1][d] = fmaf(p1, v, acc[1][d]);
        acc[2][d] = fmaf(p2, v, acc[2][d]); acc[3][d] = fmaf(p3, v, acc[3][d]);
      }
    }
#endif
  }
  if (qu == 0) Lsum[qa] = lA;
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int qq = qg * 4 + i, tok = tb + (qq >> 3);
    if (tok >= T) continue;
    const float inv = 1.f / Lsum[qq];
    const size_t ob = (size_t)tok * 4096 + (g * 8 + (qq & 7)) * 256 + ds * 16;
#pragma unroll
    for (int d = 0; d < 16; d++) {
      float gg = GATE[ob + d];
      O[ob + d] = acc[i][d] * inv * (1.f / (1.f + __expf(-gg)));
    }
  }
}

// ---------------------------------------------------------------- grouped GEMM v2: 64 rows x 128 outputs per block,
// each thread 4 rows x 8 outputs (two float4 column groups at tx*4 and 64+tx*4 -> conflict-free LDS.128).
// BK = 16 = one NVFP4 scale group. Same semantics as bgemm.
#define BM2 64
#define BN2 128
#define BK2 16
extern "C" __global__ void bgemm2(const long long* __restrict__ wptr, const long long* __restrict__ sptr, const float* __restrict__ gsc,
    const int* __restrict__ roff, const float* __restrict__ X, const int* __restrict__ xmap, int ldx,
    float* __restrict__ C, const int* __restrict__ cmap, int ldc, const float* __restrict__ rscale, int N, int K, int acc_mode)
{
  const int g = blockIdx.z;
  const int r0 = roff[g] + blockIdx.y * BM2, r1 = roff[g + 1];
  if (r0 >= r1) return;
  const int n0 = blockIdx.x * BN2;
  __shared__ __align__(16) float Xs[BK2][BM2];
  __shared__ __align__(16) float Ws[BK2][BN2];
  __shared__ float lut4[16], lut8[256];
  const float e2m1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  lut8[threadIdx.x & 255] = fp8_e4m3(threadIdx.x & 255);
  if (threadIdx.x < 16) lut4[threadIdx.x] = e2m1[threadIdx.x];
  const unsigned char* Wb = (const unsigned char*)wptr[g];
  const unsigned char* Sb = sptr ? (const unsigned char*)sptr[g] : 0;
  const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
  // X loader: row = tid / 4, k quad = tid % 4
  const int xr_l = tid >> 2, xk_l = (tid & 3) * 4;
  const int xr = r0 + xr_l;
  const float* xrow = (xr < r1) ? X + (size_t)(xmap ? xmap[xr] : xr) * ldx : 0;
  // W loader: thread < 128 decodes one output row's 16 k values
  const int wn = n0 + (tid & 127);
  float acc[4][8];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j] = 0.f;
  __syncthreads();
  for (int k0 = 0; k0 < K; k0 += BK2) {
    {
      float4 v = xrow ? *(const float4*)(xrow + k0 + xk_l) : make_float4(0.f, 0.f, 0.f, 0.f);
      Xs[xk_l][xr_l] = v.x; Xs[xk_l + 1][xr_l] = v.y; Xs[xk_l + 2][xr_l] = v.z; Xs[xk_l + 3][xr_l] = v.w;
    }
    if (tid < 128) {
      float w[16];
      if (wn < N) {
#if FMT == 0
        uint2 q = *(const uint2*)(Wb + (size_t)wn * (K >> 1) + (k0 >> 1));
        float sc = lut8[Sb[(size_t)wn * (K >> 4) + (k0 >> 4)]];
#pragma unroll
        for (int i = 0; i < 8; i++) { w[i] = lut4[(q.x >> (4 * i)) & 15] * sc; w[8 + i] = lut4[(q.y >> (4 * i)) & 15] * sc; }
#elif FMT == 1
        uint4 q = *(const uint4*)(Wb + (size_t)wn * K + k0);
        const unsigned v[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
        for (int c = 0; c < 4; c++)
#pragma unroll
          for (int i = 0; i < 4; i++) w[4 * c + i] = lut8[(v[c] >> (8 * i)) & 255];
#elif FMT == 2
        const float4* p = (const float4*)((const float*)Wb + (size_t)wn * K + k0);
#pragma unroll
        for (int c = 0; c < 4; c++) { float4 t = p[c]; w[4 * c] = t.x; w[4 * c + 1] = t.y; w[4 * c + 2] = t.z; w[4 * c + 3] = t.w; }
#else
        const uint4* p = (const uint4*)((const unsigned short*)Wb + (size_t)wn * K + k0);
#pragma unroll
        for (int c = 0; c < 2; c++) {
          uint4 t = p[c]; const unsigned v[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
          for (int i = 0; i < 4; i++) { w[8 * c + 2 * i] = __int_as_float(v[i] << 16); w[8 * c + 2 * i + 1] = __int_as_float(v[i] & 0xffff0000u); }
        }
#endif
      } else {
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = 0.f;
      }
#pragma unroll
      for (int i = 0; i < 16; i++) Ws[i][tid] = w[i];
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK2; kk++) {
      const float4 a = *(const float4*)&Xs[kk][ty * 4];
      const float4 b0 = *(const float4*)&Ws[kk][tx * 4];
      const float4 b1 = *(const float4*)&Ws[kk][64 + tx * 4];
      const float av[4] = {a.x, a.y, a.z, a.w};
      const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
    }
    __syncthreads();
  }
  const float gs = gsc ? gsc[g] : 1.f;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int r = r0 + ty * 4 + i;
    if (r >= r1) continue;
    const float sc = gs * (rscale ? rscale[r] : 1.f);
    const size_t cr = (size_t)(cmap ? cmap[r] : r) * ldc;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      const int n = n0 + (j < 4 ? tx * 4 + j : 64 + tx * 4 + j - 4);
      if (n >= N) continue;
      const float v = acc[i][j] * sc;
      if (acc_mode == 2) atomicAdd(C + cr + n, v);
      else if (acc_mode == 1) C[cr + n] += v;
      else C[cr + n] = v;
    }
  }
}

'''
_mods = {}


def fn(name, **defs):
    key = tuple(sorted(defs.items()))
    if key not in _mods:
        _mods[key] = cp.RawModule(code=_SRC, options=('-std=c++14',) + tuple(f'-D{k}={v}' for k, v in key))
    return _mods[key].get_function(name)


FMT = {'nvfp4': 0, 'fp8': 1, 'f32': 2, 'bf16': 3}


def bgemm(fmt, wptr, sptr, gsc, roff, X, ldx, C, ldc, N, K, groups, max_rows, xmap=None, cmap=None, rscale=None, acc=0, v=2):
    p = lambda t: 0 if t is None else t.data_ptr()
    if v == 2:
        fn('bgemm2', FMT=FMT[fmt])(((N + 127) // 128, (max_rows + 63) // 64, groups), (256,),
                                   (wptr.data_ptr(), p(sptr), p(gsc), roff.data_ptr(), X.data_ptr(), p(xmap), i32(ldx),
                                    C.data_ptr(), p(cmap), i32(ldc), p(rscale), i32(N), i32(K), i32(acc)), stream=stream())
        return
    fn('bgemm', FMT=FMT[fmt])(((N + 63) // 64, (max_rows + 63) // 64, groups), (256,),
                              (wptr.data_ptr(), p(sptr), p(gsc), roff.data_ptr(), X.data_ptr(), p(xmap), i32(ldx),
                               C.data_ptr(), p(cmap), i32(ldc), p(rscale), i32(N), i32(K), i32(acc)), stream=stream())
