"""CUDA kernels (CuPy/NVRTC) for Ornith-1.5-35B-A3B decode on Pascal (fp32 math, no fp8 hardware).

Weight formats are decoded in registers via shared-memory lookup tables:
  NVFP4: uint8 [N, K/2] (low nibble first) + fp8-e4m3 scale [N, K/16] + fp32 global scale
  FP8  : fp8-e4m3 [N, K] + fp32 per-tensor scale
  F32  : plain fp32 [N, K]
All GEMVs are 'row' kernels: one warp per output row, many matrices per launch (pointer arrays),
results are atomically ADDED to y (so y can be zero or a residual). Optional per-matrix weights and
a device-side count of active matrices (adaptive expert count without re-capturing graphs).
"""
import os
import cupy as cp
import torch

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
#ifndef WIN
#define WIN 2048
#endif
#ifndef RING
#define RING (WIN + 4096)
#endif

__device__ __forceinline__ float warp_sum(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
__device__ __forceinline__ float warp_max(float v) { for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o)); return v; }
__device__ float block_sum(float v, float* red) {
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_sum(v); __syncthreads(); if (lane == 0) red[w] = v; __syncthreads();
  v = (threadIdx.x < nw) ? red[threadIdx.x] : 0.f; if (w == 0) v = warp_sum(v);
  if (threadIdx.x == 0) red[32] = v; __syncthreads(); return red[32];
}
__device__ float block_max(float v, float* red) {
  int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  v = warp_max(v); __syncthreads(); if (lane == 0) red[w] = v; __syncthreads();
  v = (threadIdx.x < nw) ? red[threadIdx.x] : -INF; if (w == 0) v = warp_max(v);
  if (threadIdx.x == 0) red[32] = v; __syncthreads(); return red[32];
}
__device__ __forceinline__ float fp8_e4m3(unsigned b) {
  unsigned s = (b >> 7) & 1, e = (b >> 3) & 15, m = b & 7;
  float v = e ? __int_as_float(((e + 120) << 23) | (m << 20)) : (float)m * 0.001953125f;  // 2^-9
  return s ? -v : v;
}
__device__ __forceinline__ void load_luts(float* lut4, float* lut8) {
  const float e2m1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
  for (int i = threadIdx.x; i < 256; i += blockDim.x) { lut8[i] = fp8_e4m3(i); if (i < 16) lut4[i] = e2m1[i]; }
}

// ---------------------------------------------------------------- row GEMV (FMT 0 = NVFP4, 1 = FP8, 2 = F32)
#ifndef ASG
#define ASG 0
#endif
extern "C" __global__ void rgemv(const long long* __restrict__ wptr, const long long* __restrict__ sptr,
    const float* __restrict__ gsc, const float* __restrict__ x, int xdiv, int xstride,
    float* __restrict__ y, int ydiv, int ystride, const float* __restrict__ wts,
    const int* __restrict__ mcount, int N, int K)
{
  extern __shared__ float sh[];
  float* lut4 = sh; float* lut8 = sh + 16; float* xs = sh + 16 + 256;
  const int m = blockIdx.y;
  if (mcount && m >= *mcount) return;
  load_luts(lut4, lut8);
  const float* xm = x + (size_t)(m / xdiv) * xstride;
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = xm[i];
  __syncthreads();
  const int lane = threadIdx.x & 31, n = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (n >= N) return;
  float acc = 0.f;
#if FMT == 0
  const uint2* W = (const uint2*)((const unsigned char*)wptr[m] + (size_t)n * (K >> 1));
  const unsigned char* S = (const unsigned char*)sptr[m] + (size_t)n * (K >> 4);
  const int G = K >> 4;
  for (int g0 = lane; g0 < G; g0 += 128) {
    uint2 q[4]; unsigned char sc[4];
#pragma unroll
    for (int u = 0; u < 4; u++) { int g = g0 + 32 * u; if (g < G) { q[u] = W[g]; sc[u] = S[g]; } }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      int g = g0 + 32 * u;
      if (g < G) {
        const float* xx = xs + g * 16;
        float p = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          p = fmaf(lut4[(q[u].x >> (4 * i)) & 15], xx[i], p);
          p = fmaf(lut4[(q[u].y >> (4 * i)) & 15], xx[8 + i], p);
        }
        acc = fmaf(p, lut8[sc[u]], acc);
      }
    }
  }
#elif FMT == 1
  const uint4* W = (const uint4*)((const unsigned char*)wptr[m] + (size_t)n * K);
  const int G = K >> 4;
  for (int g0 = lane; g0 < G; g0 += 128) {
    uint4 q[4];
#pragma unroll
    for (int u = 0; u < 4; u++) { int g = g0 + 32 * u; if (g < G) q[u] = W[g]; }
#pragma unroll
    for (int u = 0; u < 4; u++) {
      int g = g0 + 32 * u;
      if (g < G) {
        const float* xx = xs + g * 16;
        const unsigned v[4] = {q[u].x, q[u].y, q[u].z, q[u].w};
#pragma unroll
        for (int c = 0; c < 4; c++)
#pragma unroll
          for (int i = 0; i < 4; i++) acc = fmaf(lut8[(v[c] >> (8 * i)) & 255], xx[4 * c + i], acc);
      }
    }
  }
#elif FMT == 3
  const uint4* W = (const uint4*)((const unsigned short*)wptr[m] + (size_t)n * K);   // bf16: 8 per uint4
  for (int g = lane; g < (K >> 3); g += 32) {
    uint4 q = W[g]; const float* xx = xs + g * 8;
    const unsigned v[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
    for (int c = 0; c < 4; c++) {
      acc = fmaf(__int_as_float(v[c] << 16), xx[2 * c], acc);
      acc = fmaf(__int_as_float(v[c] & 0xffff0000u), xx[2 * c + 1], acc);
    }
  }
#else
  const float4* W = (const float4*)((const float*)wptr[m] + (size_t)n * K);
  for (int g = lane; g < (K >> 2); g += 32) {
    float4 w = W[g]; const float* xx = xs + g * 4;
    acc = fmaf(w.x, xx[0], fmaf(w.y, xx[1], fmaf(w.z, xx[2], fmaf(w.w, xx[3], acc))));
  }
#endif
  acc = warp_sum(acc);
  if (lane == 0) {
    float s = gsc ? gsc[m] : 1.f;
    if (wts) s *= wts[m];
#if ASG
    y[(size_t)(m / ydiv) * ystride + n] = acc * s;     // single writer per element: store (identical to 0 + a)
#else
    atomicAdd(y + (size_t)(m / ydiv) * ystride + n, acc * s);
#endif
  }
}

// ---------------------------------------------------------------- row GEMV v2: x in registers, R rows per warp
// lane owns groups g = lane + 32*u (16 weights each); its 16*NG x-values live in registers for all R rows.
// FMT 0 = NVFP4, 1 = FP8 (arithmetic decode). KK = K (compile time) so NG = KK/16/32 is static.
__device__ __forceinline__ float fp8f(unsigned b) {
  unsigned a = b & 0x7f;
  float v = (a & 0x78) ? __int_as_float((a << 20) + (120u << 23)) : (float)(a & 7) * 0.001953125f;
  return (b & 0x80) ? -v : v;
}
#ifndef KK
#define KK 2048
#endif
#define NG ((KK / 16 + 31) / 32)
#ifndef RW
#define RW 4
#endif
#ifndef FP8LUT
#define FP8LUT 0
#endif
#ifndef PRE
#define PRE 0
#endif
extern "C" __global__ void rgemv2(const long long* __restrict__ wptr, const long long* __restrict__ sptr,
    const float* __restrict__ gsc, const float* __restrict__ x, int xdiv, int xstride,
    float* __restrict__ y, int ydiv, int ystride, const float* __restrict__ wts,
    const int* __restrict__ mcount, int N, const long long* __restrict__ wdst, const long long* __restrict__ sdst)
{
  __shared__ float lut4[16];
  __shared__ float lut8s[256];
  const int m = blockIdx.y;
  if (mcount && m >= *mcount) return;
#if FP8LUT
  for (int i = threadIdx.x; i < 256; i += blockDim.x) lut8s[i] = fp8_e4m3(i);
#endif
  if (threadIdx.x < 16) {
    const float e2m1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    lut4[threadIdx.x] = e2m1[threadIdx.x];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31;
  const int row0 = (blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5)) * RW;
  if (row0 >= N) return;
  const int G = KK / 16;
  const float* xm = x + (size_t)(m / xdiv) * xstride;
  float xr[NG][16];
#pragma unroll
  for (int u = 0; u < NG; u++) {
    int g = lane + 32 * u;
    if (g < G) {
      const float4* xv = (const float4*)(xm + g * 16);
#pragma unroll
      for (int c = 0; c < 4; c++) { float4 t = xv[c]; xr[u][4 * c] = t.x; xr[u][4 * c + 1] = t.y; xr[u][4 * c + 2] = t.z; xr[u][4 * c + 3] = t.w; }
    }
  }
  float acc[RW];
#pragma unroll
  for (int r = 0; r < RW; r++) acc[r] = 0.f;
#if FMT == 0 && PRE
  // PRE: issue the loads of all RW rows before any math (more bytes in flight); same arithmetic order as below
  const unsigned char* Wb = (const unsigned char*)wptr[m];
  const unsigned char* Sb = (const unsigned char*)sptr[m];
  uint2 qa[RW][NG]; unsigned char sa[RW][NG];
#pragma unroll
  for (int r = 0; r < RW; r++) {
    const int n = row0 + r;
    const uint2* W = (const uint2*)(Wb + (size_t)n * (KK / 2));
    const unsigned char* S = Sb + (size_t)n * (KK / 16);
#pragma unroll
    for (int u = 0; u < NG; u++) { int g = lane + 32 * u; if (g < G && n < N) { qa[r][u] = W[g]; sa[r][u] = S[g]; } }
  }
  if (wdst) {
    long long wd = wdst[m], sd = sdst[m];
    if (wd) {
#pragma unroll
      for (int r = 0; r < RW; r++) {
        const int n = row0 + r;
        uint2* Wd = (uint2*)((unsigned char*)wd + (size_t)n * (KK / 2));
        unsigned char* Sd = (unsigned char*)sd + (size_t)n * (KK / 16);
#pragma unroll
        for (int u = 0; u < NG; u++) { int g = lane + 32 * u; if (g < G && n < N) { Wd[g] = qa[r][u]; Sd[g] = sa[r][u]; } }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RW; r++) {
    if (row0 + r < N) {
#pragma unroll
      for (int u = 0; u < NG; u++) {
        int g = lane + 32 * u;
        if (g < G) {
          float p = 0.f;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            p = fmaf(lut4[(qa[r][u].x >> (4 * i)) & 15], xr[u][i], p);
            p = fmaf(lut4[(qa[r][u].y >> (4 * i)) & 15], xr[u][8 + i], p);
          }
          acc[r] = fmaf(p, fp8f(sa[r][u]), acc[r]);
        }
      }
    }
  }
#elif FMT == 0
  const unsigned char* Wb = (const unsigned char*)wptr[m];
  const unsigned char* Sb = (const unsigned char*)sptr[m];
#pragma unroll
  for (int r = 0; r < RW; r++) {
    const int n = row0 + r;
    if (n < N) {
      const uint2* W = (const uint2*)(Wb + (size_t)n * (KK / 2));
      const unsigned char* S = Sb + (size_t)n * (KK / 16);
      uint2 q[NG]; unsigned char sc[NG];
#pragma unroll
      for (int u = 0; u < NG; u++) { int g = lane + 32 * u; if (g < G) { q[u] = W[g]; sc[u] = S[g]; } }
      if (wdst) {                                // copy-on-read: write the expert bytes just read into a VRAM cache slot
        long long wd = wdst[m], sd = sdst[m];
        if (wd) {
          uint2* Wd = (uint2*)((unsigned char*)wd + (size_t)n * (KK / 2));
          unsigned char* Sd = (unsigned char*)sd + (size_t)n * (KK / 16);
#pragma unroll
          for (int u = 0; u < NG; u++) { int g = lane + 32 * u; if (g < G) { Wd[g] = q[u]; Sd[g] = sc[u]; } }
        }
      }
#pragma unroll
      for (int u = 0; u < NG; u++) {
        int g = lane + 32 * u;
        if (g < G) {
          float p = 0.f;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            p = fmaf(lut4[(q[u].x >> (4 * i)) & 15], xr[u][i], p);
            p = fmaf(lut4[(q[u].y >> (4 * i)) & 15], xr[u][8 + i], p);
          }
          acc[r] = fmaf(p, fp8f(sc[u]), acc[r]);
        }
      }
    }
  }
#elif PRE
  // FP8, PRE: loads of all RW rows first, then the same arithmetic as below
  const unsigned char* Wb = (const unsigned char*)wptr[m];
  uint4 qa[RW][NG];
#pragma unroll
  for (int r = 0; r < RW; r++) {
    const int n = row0 + r;
    const uint4* W = (const uint4*)(Wb + (size_t)n * KK);
#pragma unroll
    for (int u = 0; u < NG; u++) { int g = lane + 32 * u; if (g < G && n < N) qa[r][u] = W[g]; }
  }
#pragma unroll
  for (int r = 0; r < RW; r++) {
    if (row0 + r < N) {
#pragma unroll
      for (int u = 0; u < NG; u++) {
        int g = lane + 32 * u;
        if (g < G) {
          const unsigned v[4] = {qa[r][u].x, qa[r][u].y, qa[r][u].z, qa[r][u].w};
#pragma unroll
          for (int c = 0; c < 4; c++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
#if FP8LUT
              acc[r] = fmaf(lut8s[(v[c] >> (8 * i)) & 255], xr[u][4 * c + i], acc[r]);
#else
              acc[r] = fmaf(fp8f((v[c] >> (8 * i)) & 255), xr[u][4 * c + i], acc[r]);
#endif
            }
        }
      }
    }
  }
#else
  const unsigned char* Wb = (const unsigned char*)wptr[m];
#pragma unroll
  for (int r = 0; r < RW; r++) {
    const int n = row0 + r;
    if (n < N) {
      const uint4* W = (const uint4*)(Wb + (size_t)n * KK);
      uint4 q[NG];
#pragma unroll
      for (int u = 0; u < NG; u++) { int g = lane + 32 * u; if (g < G) q[u] = W[g]; }
#pragma unroll
      for (int u = 0; u < NG; u++) {
        int g = lane + 32 * u;
        if (g < G) {
          const unsigned v[4] = {q[u].x, q[u].y, q[u].z, q[u].w};
#pragma unroll
          for (int c = 0; c < 4; c++)
#pragma unroll
            for (int i = 0; i < 4; i++) {
#if FP8LUT
              acc[r] = fmaf(lut8s[(v[c] >> (8 * i)) & 255], xr[u][4 * c + i], acc[r]);
#else
              acc[r] = fmaf(fp8f((v[c] >> (8 * i)) & 255), xr[u][4 * c + i], acc[r]);
#endif
            }
        }
      }
    }
  }
#endif
  const float sgl = (gsc ? gsc[m] : 1.f) * (wts ? wts[m] : 1.f);
#pragma unroll
  for (int r = 0; r < RW; r++) {
    float a = warp_sum(acc[r]);
#if ASG
    if (lane == 0 && row0 + r < N) y[(size_t)(m / ydiv) * ystride + row0 + r] = a * sgl;
#else
    if (lane == 0 && row0 + r < N) atomicAdd(y + (size_t)(m / ydiv) * ystride + row0 + r, a * sgl);
#endif
  }
}

// ---------------------------------------------------------------- NVFP4 row GEMV, split over SPL warps (K = KK = 2048)
// Bit-identical to rgemv2 (FMT 0, PRE 0): warp kp computes the group sums p_u of the groups u = kp*NGP .. (it uses the same lane ->
// group mapping g = lane + 32*u), parks p_u and the group scale in shared memory, and one warp per row then runs the same fma chain
// over u = 0..NG-1 and the same warp_sum. Each thread holds only NG/SPL groups of x -> ~40 instead of 118 registers -> 3x the warps
// per SM (the fast GEMV is limited by loads in flight, not by arithmetic; internal benchmark).
#ifndef SPL
#define SPL 4
#endif
#if (KK / 16 + 31) / 32 >= SPL                               // only where every warp gets at least one group
extern "C" __global__ void rgemv2s(const long long* __restrict__ wptr, const long long* __restrict__ sptr,
    const float* __restrict__ gsc, const float* __restrict__ x, int xdiv, int xstride,
    float* __restrict__ y, int ydiv, int ystride, const float* __restrict__ wts,
    const int* __restrict__ mcount, int N, const long long* __restrict__ wdst, const long long* __restrict__ sdst)
{
  const int NGP = NG / SPL, RG = 128 / 32 / SPL;          // groups per warp, row groups per block (block = 128 threads)
  __shared__ float lut4[16];
  __shared__ float ps[RG][RW][NG][32];
  const int m = blockIdx.y;
  if (mcount && m >= *mcount) return;                       // uniform per block
  if (threadIdx.x < 16) {
    const float e2m1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    lut4[threadIdx.x] = e2m1[threadIdx.x];
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, kp = w % SPL, rg = w / SPL;
  const int row0 = (blockIdx.x * RG + rg) * RW;
  const int G = KK / 16;
  const float* xm = x + (size_t)(m / xdiv) * xstride;
  float xr[NGP][16];
#pragma unroll
  for (int v = 0; v < NGP; v++) {
    int g = lane + 32 * (kp * NGP + v);
    const float4* xv = (const float4*)(xm + g * 16);
#pragma unroll
    for (int c = 0; c < 4; c++) { float4 t = xv[c]; xr[v][4 * c] = t.x; xr[v][4 * c + 1] = t.y; xr[v][4 * c + 2] = t.z; xr[v][4 * c + 3] = t.w; }
  }
  const unsigned char* Wb = (const unsigned char*)wptr[m];
  const unsigned char* Sb = (const unsigned char*)sptr[m];
  const long long wd = wdst ? wdst[m] : 0, sd = wdst ? sdst[m] : 0;
  uint2 q[RW][NGP]; unsigned char sc[RW][NGP];               // all loads first (few registers per row here)
#pragma unroll
  for (int r = 0; r < RW; r++) {
    const int n = row0 + r;
#pragma unroll
    for (int v = 0; v < NGP; v++) {
      int g = lane + 32 * (kp * NGP + v);
      if (n < N) { q[r][v] = ((const uint2*)(Wb + (size_t)n * (KK / 2)))[g]; sc[r][v] = (Sb + (size_t)n * (KK / 16))[g]; }
    }
  }
  if (wd) {                                                  // copy-on-read: every warp writes the bytes it read
#pragma unroll
    for (int r = 0; r < RW; r++) {
      const int n = row0 + r;
      if (n < N) {
#pragma unroll
        for (int v = 0; v < NGP; v++) {
          int g = lane + 32 * (kp * NGP + v);
          ((uint2*)((unsigned char*)wd + (size_t)n * (KK / 2)))[g] = q[r][v]; ((unsigned char*)sd + (size_t)n * (KK / 16))[g] = sc[r][v];
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RW; r++) {
    if (row0 + r < N) {
#pragma unroll
      for (int v = 0; v < NGP; v++) {
        float p = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          p = fmaf(lut4[(q[r][v].x >> (4 * i)) & 15], xr[v][i], p);
          p = fmaf(lut4[(q[r][v].y >> (4 * i)) & 15], xr[v][8 + i], p);
        }
        ps[rg][r][kp * NGP + v][lane] = p;
      }
    }
  }
  __syncthreads();
  const float sgl = (gsc ? gsc[m] : 1.f) * (wts ? wts[m] : 1.f);
  for (int r = kp; r < RW; r += SPL) {           // rows spread over the SPL warps of the row group
    const int n = row0 + r;
    if (n >= N) continue;
    const unsigned char* S = Sb + (size_t)n * (KK / 16);      // group scales again (just read by this block: L2 hit)
    unsigned char sc[NG];
#pragma unroll
    for (int u = 0; u < NG; u++) sc[u] = S[lane + 32 * u];
    float acc = 0.f;
#pragma unroll
    for (int u = 0; u < NG; u++) acc = fmaf(ps[rg][r][u][lane], fp8f(sc[u]), acc);
    float a = warp_sum(acc);
#if ASG
    if (lane == 0) y[(size_t)(m / ydiv) * ystride + n] = a * sgl;
#else
    if (lane == 0) atomicAdd(y + (size_t)(m / ydiv) * ystride + n, a * sgl);
#endif
  }
}
#endif

// ---------------------------------------------------------------- FP8 row GEMV, loads split over SPL warps (K = KK, KK/16/32 groups)
// Bit-identical to rgemv2 (FMT 1, PRE 0): the fp8 sum of a lane is ONE fma chain over its groups u = 0..NG-1, so the warps load
// their part of every row at the same time (loads in flight are the limit), then run the chain one after another: warp kp continues
// the running sums left in shared memory by warp kp-1. The last warp does the same warp_sum as rgemv2.
#if (KK / 16 + 31) / 32 >= SPL
extern "C" __global__ void rgemv2s8(const long long* __restrict__ wptr, const long long* __restrict__ sptr,
    const float* __restrict__ gsc, const float* __restrict__ x, int xdiv, int xstride,
    float* __restrict__ y, int ydiv, int ystride, const float* __restrict__ wts,
    const int* __restrict__ mcount, int N, const long long* __restrict__ wdst, const long long* __restrict__ sdst)
{
  const int NGP = NG / SPL;
  __shared__ float lut8s[256];
  __shared__ float ch[RW][32];
  const int m = blockIdx.y;
  if (mcount && m >= *mcount) return;
  for (int i = threadIdx.x; i < 256; i += blockDim.x) lut8s[i] = fp8_e4m3(i);
  const int lane = threadIdx.x & 31, kp = threadIdx.x >> 5;   // block = SPL warps = one row group
  const int row0 = blockIdx.x * RW;
  const float* xm = x + (size_t)(m / xdiv) * xstride;
  float xr[NGP][16];
#pragma unroll
  for (int v = 0; v < NGP; v++) {
    int g = lane + 32 * (kp * NGP + v);
    const float4* xv = (const float4*)(xm + g * 16);
#pragma unroll
    for (int c = 0; c < 4; c++) { float4 t = xv[c]; xr[v][4 * c] = t.x; xr[v][4 * c + 1] = t.y; xr[v][4 * c + 2] = t.z; xr[v][4 * c + 3] = t.w; }
  }
  const unsigned char* Wb = (const unsigned char*)wptr[m];
  uint4 q[RW][NGP];
#pragma unroll
  for (int r = 0; r < RW; r++)
#pragma unroll
    for (int v = 0; v < NGP; v++)
      if (row0 + r < N) q[r][v] = ((const uint4*)(Wb + (size_t)(row0 + r) * KK))[lane + 32 * (kp * NGP + v)];
  __syncthreads();                                            // lut8s ready
  const float sgl = (gsc ? gsc[m] : 1.f) * (wts ? wts[m] : 1.f);
  for (int k = 0; k < SPL; k++) {
    if (kp == k) {
#pragma unroll
      for (int r = 0; r < RW; r++) {
        if (row0 + r >= N) continue;
        float acc = k ? ch[r][lane] : 0.f;
#pragma unroll
        for (int v = 0; v < NGP; v++) {
          const unsigned w4[4] = {q[r][v].x, q[r][v].y, q[r][v].z, q[r][v].w};
#pragma unroll
          for (int c = 0; c < 4; c++)
#pragma unroll
            for (int i = 0; i < 4; i++) acc = fmaf(lut8s[(w4[c] >> (8 * i)) & 255], xr[v][4 * c + i], acc);
        }
        if (k < SPL - 1) ch[r][lane] = acc;
        else {
          float a = warp_sum(acc);
#if ASG
          if (lane == 0) y[(size_t)(m / ydiv) * ystride + row0 + r] = a * sgl;
#else
          if (lane == 0) atomicAdd(y + (size_t)(m / ydiv) * ystride + row0 + r, a * sgl);
#endif
        }
      }
    }
    __syncthreads();
  }
}
#endif

// ---------------------------------------------------------------- RMSNorm (zero-centred: out = norm(x) * (1 + w))
extern "C" __global__ void rmsnorm1p(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int n, float eps) {
  __shared__ float red[33];
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
  float r = rsqrtf(block_sum(s, red) / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = x[i] * r * (1.f + w[i]);
}

// ---------------------------------------------------------------- Gated DeltaNet: conv + l2norm + gates
// qkv[8192] = [q 16x128 | k 16x128 | v 32x128]; conv_state [8192][3] (oldest first); conv_w [8192][4]
// grid 64 blocks x 128 threads: 0..15 q heads, 16..31 k heads, 32..63 v heads. ab[64] = [a(32) | b(32)]
extern "C" __global__ void dn_prep(const float* __restrict__ qkv, float* __restrict__ conv_state,
    const float* __restrict__ conv_w, const float* __restrict__ ab, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ qkv_out, float* __restrict__ gb)
{
  __shared__ float red[33];
  const int b = blockIdx.x, t = threadIdx.x, c = b * 128 + t;
  float* st = conv_state + (size_t)c * 3;
  const float* w = conv_w + (size_t)c * 4;
  float cur = qkv[c];
  float o = w[0] * st[0] + w[1] * st[1] + w[2] * st[2] + w[3] * cur;
  st[0] = st[1]; st[1] = st[2]; st[2] = cur;
  o = o / (1.f + __expf(-o));                       // silu
  if (b < 32) {                                     // q or k head: l2 normalise over 128 dims
    float inv = rsqrtf(block_sum(o * o, red) + 1e-6f);
    o *= inv;
    if (b < 16) o *= 0.08838834764831845f;         // q / sqrt(128)
  }
  qkv_out[c] = o;
  if (b == 0 && t < 32) {
    float a = ab[t] + dt_bias[t];
    float sp = a > 20.f ? a : log1pf(__expf(a));    // softplus
    gb[t] = -__expf(A_log[t]) * sp;                  // g (log decay)
    gb[32 + t] = 1.f / (1.f + __expf(-ab[32 + t]));  // beta
  }
}

// recurrent gated delta rule + gated RMSNorm. grid 32 (value heads) x 128 threads (value dim j)
// state S[h][k][j] fp32. value head h uses key head h/2.
extern "C" __global__ void dn_recur(const float* __restrict__ qkv, const float* __restrict__ gb,
    float* __restrict__ S, const float* __restrict__ z, const float* __restrict__ normw,
    float* __restrict__ out, float eps)
{
  __shared__ float qs[128], ks[128];
  __shared__ float red[33];
  const int h = blockIdx.x, j = threadIdx.x, kh = h >> 1;
  qs[j] = qkv[kh * 128 + j];
  ks[j] = qkv[2048 + kh * 128 + j];
  __syncthreads();
  const float decay = __expf(gb[h]), beta = gb[32 + h];
  float* Sh = S + (size_t)h * 128 * 128;
  float col[128];
  float kv = 0.f;
#pragma unroll
  for (int k = 0; k < 128; k++) { float s = Sh[k * 128 + j] * decay; col[k] = s; kv = fmaf(s, ks[k], kv); }
  const float delta = (qkv[4096 + h * 128 + j] - kv) * beta;
  float o = 0.f;
#pragma unroll
  for (int k = 0; k < 128; k++) { float s = fmaf(ks[k], delta, col[k]); Sh[k * 128 + j] = s; o = fmaf(s, qs[k], o); }
  float r = rsqrtf(block_sum(o * o, red) / 128.f + eps);
  float zz = z[h * 128 + j];
  out[h * 128 + j] = o * r * normw[j] * (zz / (1.f + __expf(-zz)));
}

// ---------------------------------------------------------------- full attention prologue
// qf[8192] = per head [q 256 | gate 256]; kv[1024] = [k 2x256 | v 2x256]
// grid 18 x 256: blocks 0..15 q heads, 16..17 kv heads. partial RoPE on dims 0..63 (rotate_half within 64)
// KV cache: KV8=0 -> fp16 [kvh][T][256];  KV8=1 -> int8 [kvh][T][256] + fp32 scale [kvh][T]
extern "C" __global__ void attn_prep(const float* __restrict__ qf, const float* __restrict__ kv,
    const float* __restrict__ qn, const float* __restrict__ kn, const float* __restrict__ cosT, const float* __restrict__ sinT,
    const long long* __restrict__ pos_p, float* __restrict__ qr, float* __restrict__ gate,
    void* __restrict__ Kc, void* __restrict__ Vc, __half* __restrict__ Ks, __half* __restrict__ Vs, int maxT, float eps,
    __half* __restrict__ Kr, __half* __restrict__ Vr, const int* __restrict__ rp)
{
  __shared__ float xs[256];
  __shared__ float red[33];
  const int b = blockIdx.x, t = threadIdx.x, pos = (int)pos_p[0];
  const int rpos = rp ? rp[pos] : pos;          // RoPE row (differs from the cache slot after images: 3D positions)
  const bool isq = b < 16;
  const int h = isq ? b : b - 16;
  float x = isq ? qf[h * 512 + t] : kv[h * 256 + t];
  if (isq) gate[h * 256 + t] = qf[h * 512 + 256 + t];
  float r = rsqrtf(block_sum(x * x, red) / 256.f + eps);
  x = x * r * (1.f + (isq ? qn[t] : kn[t]));
  xs[t] = x;
  __syncthreads();
  if (t < 64) {
    const int i = t & 31;
    float c = cosT[rpos * 32 + i], s = sinT[rpos * 32 + i];
    x = t < 32 ? xs[t] * c - xs[t + 32] * s : xs[t] * c + xs[t - 32] * s;
  }
  if (isq) { qr[h * 256 + t] = x; return; }
  float v = kv[512 + h * 256 + t];
  size_t base = KROW(h, pos) * 256 + t;
#if KV8
  // one scale per 32 dims (= one warp) so outlier channels do not crush the others
  float ka = warp_max(fabsf(x)), va = warp_max(fabsf(v));
  // scales stored as fp16; quantise with the rounded scale so the stored scale is exact
  __half kh = __float2half(ka > 0.f ? ka / 127.f : 1.f), vh = __float2half(va > 0.f ? va / 127.f : 1.f);
  float ksc = __half2float(kh), vsc = __half2float(vh);
  ((signed char*)Kc)[base] = (signed char)max(-127, min(127, __float2int_rn(x / ksc)));
  ((signed char*)Vc)[base] = (signed char)max(-127, min(127, __float2int_rn(v / vsc)));
  if ((t & 31) == 0) { size_t sb = SROW(h, pos) * 8 + (t >> 5); Ks[sb] = kh; Vs[sb] = vh; }
  size_t rb = ((size_t)h * RING + (pos % RING)) * 256 + t;          // exact fp16 copy of the recent window
  Kr[rb] = __float2half(x); Vr[rb] = __float2half(v);
#else
  ((__half*)Kc)[base] = __float2half(x);
  ((__half*)Vc)[base] = __float2half(v);
#endif
}

// flash decoding, head_dim 256, GQA group 8: grid (2 kv heads, nsplit) x 256 threads (8 warps = 8 q heads)
#define CH 64
__device__ __forceinline__ void load8(const void* base, size_t idx, int lane, float* o) {
#if KV8
  uint2 q = *((const uint2*)((const signed char*)base + idx) + lane);
  const signed char* c = (const signed char*)&q;
#pragma unroll
  for (int i = 0; i < 8; i++) o[i] = (float)c[i];
#else
  uint4 q = *((const uint4*)((const __half*)base + idx) + lane);
  const __half2* hh = (const __half2*)&q;
#pragma unroll
  for (int i = 0; i < 4; i++) { float2 f = __half22float2(hh[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
#endif
}
__device__ __forceinline__ void load8h(const __half* base, int lane, float* o) {
  uint4 q = *((const uint4*)base + lane);
  const __half2* hh = (const __half2*)&q;
#pragma unroll
  for (int i = 0; i < 4; i++) { float2 f = __half22float2(hh[i]); o[2 * i] = f.x; o[2 * i + 1] = f.y; }
}
extern "C" __global__ void attn_partial(const float* __restrict__ qr, const void* __restrict__ Kc, const void* __restrict__ Vc,
    const __half* __restrict__ Ks, const __half* __restrict__ Vs, const long long* __restrict__ pos_p,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc, int maxT, int nsplit,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr)
{
  __shared__ float sc[8][CH];
  const int g = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int T = (int)pos_p[0] + 1, t0 = sp * CH;
  if (t0 >= T) return;
  const int t1 = min(t0 + CH, T), n = t1 - t0, h = g * 8 + w;
  float q[8];
#pragma unroll
  for (int i = 0; i < 8; i++) q[i] = qr[h * 256 + lane * 8 + i] * 0.0625f;   // 1/sqrt(256)
  const size_t hb = (size_t)g * maxT;
  for (int t = t0; t < t1; t++) {
    float d = 0.f;
#if KV8
    if (t >= T - WIN) {
      float kk[8]; load8h(Kr + ((size_t)g * RING + (t % RING)) * 256, lane, kk);
#pragma unroll
      for (int i = 0; i < 8; i++) d = fmaf(q[i], kk[i], d);
    } else {
      float kk[8]; load8(Kc, KROW(g, t) * 256, lane, kk);
#pragma unroll
      for (int i = 0; i < 8; i++) d = fmaf(q[i], kk[i], d);
      d *= __half2float(Ks[SROW(g, t) * 8 + (lane >> 2)]);
    }
#else
    { float kk[8]; load8(Kc, KROW(g, t) * 256, lane, kk);
#pragma unroll
      for (int i = 0; i < 8; i++) d = fmaf(q[i], kk[i], d); }
#endif
    d = warp_sum(d);
    if (lane == 0) sc[w][t - t0] = d;
  }
  __syncwarp();
  float mx = -INF;
  for (int i = lane; i < n; i += 32) mx = fmaxf(mx, sc[w][i]);
  mx = warp_max(mx);
  float sum = 0.f;
  for (int i = lane; i < n; i += 32) { float e = __expf(sc[w][i] - mx); sc[w][i] = e; sum += e; }
  sum = warp_sum(sum);
  __syncwarp();
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int t = t0; t < t1; t++) {
    float vv[8];
    float p = sc[w][t - t0];
#if KV8
    if (t >= T - WIN) load8h(Vr + ((size_t)g * RING + (t % RING)) * 256, lane, vv);
    else { load8(Vc, KROW(g, t) * 256, lane, vv); p *= __half2float(Vs[SROW(g, t) * 8 + (lane >> 2)]); }
#else
    load8(Vc, KROW(g, t) * 256, lane, vv);
#endif
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = fmaf(p, vv[i], acc[i]);
  }
  const size_t o = (size_t)h * nsplit + sp;
#pragma unroll
  for (int i = 0; i < 8; i++) pacc[o * 256 + lane * 8 + i] = acc[i];
  if (lane == 0) { pm[o] = mx; pl[o] = sum; }
}
// decode attention v2: each lane scores ONE key (no per-key shuffle chain). Block = 64 keys as 2 sub-tiles of 32,
// 8 warps = the 8 q heads of one kv group. K tile in shared memory (fp32, row stride 257 -> conflict-free).
extern "C" __global__ void attn_partial2(const float* __restrict__ qr, const void* __restrict__ Kc, const void* __restrict__ Vc,
    const __half* __restrict__ Ks, const __half* __restrict__ Vs, const long long* __restrict__ pos_p,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc, int maxT, int nsplit,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr)
{
  __shared__ float Kt[32 * 257];
  __shared__ __align__(16) float qs[8 * 256];
  const int g = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int T = (int)pos_p[0] + 1, t0 = sp * CH;
  if (t0 >= T) return;
  const int h = g * 8 + w;
  for (int i = threadIdx.x; i < 8 * 256; i += 256) qs[i] = qr[g * 8 * 256 + i] * 0.0625f;
  const size_t hb = (size_t)g * maxT;
  float mx = -INF, sum = 0.f, acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int s0 = t0; s0 < min(t0 + CH, T); s0 += 32) {
    const int n = min(32, T - s0);
    __syncthreads();
    for (int i = threadIdx.x; i < 32 * 256; i += 256) {
      int j = i >> 8, d = i & 255, t = s0 + j;
      float kv = 0.f;
      if (j < n) {
#if KV8
        if (t >= T - WIN) kv = __half2float(Kr[((size_t)g * RING + (t % RING)) * 256 + d]);
        else kv = (float)((const signed char*)Kc)[KROW(g, t) * 256 + d] * __half2float(Ks[SROW(g, t) * 8 + (d >> 5)]);
#else
        kv = __half2float(((const __half*)Kc)[KROW(g, t) * 256 + d]);
#endif
      }
      Kt[j * 257 + d] = kv;
    }
    __syncthreads();
    // score of key `lane` for head w
    const float* kr = Kt + lane * 257;
    const float* q = qs + w * 256;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll 8
    for (int d = 0; d < 256; d += 4) {
      a0 = fmaf(q[d], kr[d], a0); a1 = fmaf(q[d + 1], kr[d + 1], a1);
      a2 = fmaf(q[d + 2], kr[d + 2], a2); a3 = fmaf(q[d + 3], kr[d + 3], a3);
    }
    float sc = lane < n ? (a0 + a1) + (a2 + a3) : -INF;
    float tm = warp_max(sc);
    float mn = fmaxf(mx, tm);
    float corr = __expf(mx - mn);
    float p = lane < n ? __expf(sc - mn) : 0.f;
    sum = sum * corr + warp_sum(p);
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] *= corr;
    mx = mn;
    // P.V: lane owns dims [8*lane, 8*lane+8)
    for (int j = 0; j < n; j++) {
      float pj = __shfl_sync(0xffffffffu, p, j);
      const int t = s0 + j;
      float vv[8];
#if KV8
      if (t >= T - WIN) load8h(Vr + ((size_t)g * RING + (t % RING)) * 256, lane, vv);
      else { load8(Vc, KROW(g, t) * 256, lane, vv); pj *= __half2float(Vs[SROW(g, t) * 8 + (lane >> 2)]); }
#else
      load8(Vc, KROW(g, t) * 256, lane, vv);
#endif
#pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = fmaf(pj, vv[i], acc[i]);
    }
  }
  const size_t o = (size_t)h * nsplit + sp;
#pragma unroll
  for (int i = 0; i < 8; i++) pacc[o * 256 + lane * 8 + i] = acc[i];
  if (lane == 0) { pm[o] = mx; pl[o] = sum; }
}

// decode attention v3 = v1 layout (8 warps = 8 heads, lane owns 8 dims) but 4 keys per step:
// four independent load + dot + shuffle-reduce chains -> latency hidden.
extern "C" __global__ void attn_partial3(const float* __restrict__ qr, const void* __restrict__ Kc, const void* __restrict__ Vc,
    const __half* __restrict__ Ks, const __half* __restrict__ Vs, const long long* __restrict__ pos_p,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc, int maxT, int nsplit,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr)
{
  __shared__ float sc[8][CH];
  const int g = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int T = (int)pos_p[0] + 1, t0 = sp * CH;
  if (t0 >= T) return;
  const int t1 = min(t0 + CH, T), n = t1 - t0, h = g * 8 + w;
  float q[8];
#pragma unroll
  for (int i = 0; i < 8; i++) q[i] = qr[h * 256 + lane * 8 + i] * 0.0625f;
  const size_t hb = (size_t)g * maxT;
  for (int tb = t0; tb < t1; tb += 4) {
    float d[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int t = min(tb + u, t1 - 1);
      float kk[8], s = 1.f;
#if KV8
      if (t >= T - WIN) load8h(Kr + ((size_t)g * RING + (t % RING)) * 256, lane, kk);
      else { load8(Kc, KROW(g, t) * 256, lane, kk); s = __half2float(Ks[SROW(g, t) * 8 + (lane >> 2)]); }
#else
      load8(Kc, KROW(g, t) * 256, lane, kk);
#endif
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) a = fmaf(q[i], kk[i], a);
      d[u] = a * s;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
#pragma unroll
      for (int u = 0; u < 4; u++) d[u] += __shfl_xor_sync(0xffffffffu, d[u], o);
    }
    if (lane < 4 && tb + lane < t1) sc[w][tb + lane - t0] = d[0] * (lane == 0) + d[1] * (lane == 1) + d[2] * (lane == 2) + d[3] * (lane == 3);
  }
  __syncwarp();
  float mx = -INF;
  for (int i = lane; i < n; i += 32) mx = fmaxf(mx, sc[w][i]);
  mx = warp_max(mx);
  float sum = 0.f;
  for (int i = lane; i < n; i += 32) { float e = __expf(sc[w][i] - mx); sc[w][i] = e; sum += e; }
  sum = warp_sum(sum);
  __syncwarp();
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0}, acc2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int t = t0; t < t1; t += 2) {
#pragma unroll
    for (int u = 0; u < 2; u++) {
      const int tt = t + u;
      if (tt >= t1) break;
      float vv[8];
      float p = sc[w][tt - t0];
#if KV8
      if (tt >= T - WIN) load8h(Vr + ((size_t)g * RING + (tt % RING)) * 256, lane, vv);
      else { load8(Vc, KROW(g, tt) * 256, lane, vv); p *= __half2float(Vs[SROW(g, tt) * 8 + (lane >> 2)]); }
#else
      load8(Vc, KROW(g, tt) * 256, lane, vv);
#endif
      float* a = u ? acc2 : acc;
#pragma unroll
      for (int i = 0; i < 8; i++) a[i] = fmaf(p, vv[i], a[i]);
    }
  }
  const size_t o = (size_t)h * nsplit + sp;
#pragma unroll
  for (int i = 0; i < 8; i++) pacc[o * 256 + lane * 8 + i] = acc[i] + acc2[i];
  if (lane == 0) { pm[o] = mx; pl[o] = sum; }
}

// decode attention v4: tiles that lie completely before the exact window are staged ONCE per block in shared
// memory (int8 K/V + scales, coalesced 16-byte loads) and read by all 8 head-warps from there.
extern "C" __global__ void attn_partial4(const float* __restrict__ qr, const void* __restrict__ Kc, const void* __restrict__ Vc,
    const __half* __restrict__ Ks, const __half* __restrict__ Vs, const long long* __restrict__ pos_p,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc, int maxT, int nsplit,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr)
{
  __shared__ float sc[8][CH];
  __shared__ __align__(16) signed char Kt8[CH * 256], Vt8[CH * 256];
  __shared__ float Kst[CH * 8], Vst[CH * 8];
  const int g = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int T = (int)pos_p[0] + 1, t0 = sp * CH;
  if (t0 >= T) return;
  const int t1 = min(t0 + CH, T), n = t1 - t0, h = g * 8 + w;
  const size_t hb = (size_t)g * maxT;
#if KV8
  const bool staged = t1 - 1 < T - WIN;          // whole tile is int8 (older than the exact window)
#else
  const bool staged = false;
#endif
  if (staged) {
    const uint4* ks = (const uint4*)((const signed char*)Kc + KROW(g, t0) * 256);
    const uint4* vs = (const uint4*)((const signed char*)Vc + KROW(g, t0) * 256);
    for (int i = threadIdx.x; i < n * 16; i += 256) { ((uint4*)Kt8)[i] = ks[i]; ((uint4*)Vt8)[i] = vs[i]; }
    for (int i = threadIdx.x; i < n * 8; i += 256) { Kst[i] = __half2float(Ks[SROW(g, t0) * 8 + i]); Vst[i] = __half2float(Vs[SROW(g, t0) * 8 + i]); }
    __syncthreads();
  }
  float q[8];
#pragma unroll
  for (int i = 0; i < 8; i++) q[i] = qr[h * 256 + lane * 8 + i] * 0.0625f;
  for (int tb = t0; tb < t1; tb += 4) {
    float d[4];
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int t = min(tb + u, t1 - 1);
      float kk[8], s = 1.f;
      if (staged) {
        uint2 qq = *((const uint2*)(Kt8 + (t - t0) * 256) + lane);
        const signed char* c = (const signed char*)&qq;
#pragma unroll
        for (int i = 0; i < 8; i++) kk[i] = (float)c[i];
        s = Kst[(t - t0) * 8 + (lane >> 2)];
      } else {
#if KV8
        if (t >= T - WIN) load8h(Kr + ((size_t)g * RING + (t % RING)) * 256, lane, kk);
        else { load8(Kc, KROW(g, t) * 256, lane, kk); s = __half2float(Ks[SROW(g, t) * 8 + (lane >> 2)]); }
#else
        load8(Kc, KROW(g, t) * 256, lane, kk);
#endif
      }
      float a = 0.f;
#pragma unroll
      for (int i = 0; i < 8; i++) a = fmaf(q[i], kk[i], a);
      d[u] = a * s;
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) {
#pragma unroll
      for (int u = 0; u < 4; u++) d[u] += __shfl_xor_sync(0xffffffffu, d[u], o);
    }
    if (lane < 4 && tb + lane < t1) sc[w][tb + lane - t0] = lane == 0 ? d[0] : lane == 1 ? d[1] : lane == 2 ? d[2] : d[3];
  }
  __syncwarp();
  float mx = -INF;
  for (int i = lane; i < n; i += 32) mx = fmaxf(mx, sc[w][i]);
  mx = warp_max(mx);
  float sum = 0.f;
  for (int i = lane; i < n; i += 32) { float e = __expf(sc[w][i] - mx); sc[w][i] = e; sum += e; }
  sum = warp_sum(sum);
  __syncwarp();
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int t = t0; t < t1; t++) {
    float vv[8];
    float p = sc[w][t - t0];
    if (staged) {
      uint2 qq = *((const uint2*)(Vt8 + (t - t0) * 256) + lane);
      const signed char* c = (const signed char*)&qq;
#pragma unroll
      for (int i = 0; i < 8; i++) vv[i] = (float)c[i];
      p *= Vst[(t - t0) * 8 + (lane >> 2)];
    } else {
#if KV8
      if (t >= T - WIN) load8h(Vr + ((size_t)g * RING + (t % RING)) * 256, lane, vv);
      else { load8(Vc, KROW(g, t) * 256, lane, vv); p *= __half2float(Vs[SROW(g, t) * 8 + (lane >> 2)]); }
#else
      load8(Vc, KROW(g, t) * 256, lane, vv);
#endif
    }
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = fmaf(p, vv[i], acc[i]);
  }
  const size_t o = (size_t)h * nsplit + sp;
#pragma unroll
  for (int i = 0; i < 8; i++) pacc[o * 256 + lane * 8 + i] = acc[i];
  if (lane == 0) { pm[o] = mx; pl[o] = sum; }
}

// combine splits, multiply by sigmoid(gate). grid 16 x 256
// attn_partial5: like attn_partial4, but every block walks SUB consecutive tiles of CH tokens with an online softmax
// (running max / sum / acc per warp), so there are SUB times fewer partial results for attn_combine5 to merge.
#ifndef SUB
#define SUB 8
#endif
extern "C" __global__ void attn_partial5(const float* __restrict__ qr, const void* __restrict__ Kc, const void* __restrict__ Vc,
    const __half* __restrict__ Ks, const __half* __restrict__ Vs, const long long* __restrict__ pos_p,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc, int maxT, int nsplit,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr)
{
  __shared__ float sc[8][CH];
  __shared__ __align__(16) signed char Kt8[CH * 256], Vt8[CH * 256];
  __shared__ float Kst[CH * 8], Vst[CH * 8];
  const int g = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int T = (int)pos_p[0] + 1, T0 = sp * CH * SUB;
  if (T0 >= T) return;
  const int h = g * 8 + w;
  float q[8];
#pragma unroll
  for (int i = 0; i < 8; i++) q[i] = qr[h * 256 + lane * 8 + i] * 0.0625f;
  float M = -INF, Lsum = 0.f;
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int sub = 0; sub < SUB; sub++) {
    const int t0 = T0 + sub * CH;
    if (t0 >= T) break;
    const int t1 = min(t0 + CH, T), n = t1 - t0;
#if KV8
    const bool staged = t1 - 1 < T - WIN;
#else
    const bool staged = false;
#endif
    __syncthreads();                             // previous tile fully consumed
    if (staged) {
      const uint4* ks = (const uint4*)((const signed char*)Kc + KROW(g, t0) * 256);
      const uint4* vs = (const uint4*)((const signed char*)Vc + KROW(g, t0) * 256);
      for (int i = threadIdx.x; i < n * 16; i += 256) { ((uint4*)Kt8)[i] = ks[i]; ((uint4*)Vt8)[i] = vs[i]; }
      for (int i = threadIdx.x; i < n * 8; i += 256) { Kst[i] = __half2float(Ks[SROW(g, t0) * 8 + i]); Vst[i] = __half2float(Vs[SROW(g, t0) * 8 + i]); }
      __syncthreads();
    }
    for (int tb = t0; tb < t1; tb += 4) {
      float d[4];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int t = min(tb + u, t1 - 1);
        float kk[8], s = 1.f;
        if (staged) {
          uint2 qq = *((const uint2*)(Kt8 + (t - t0) * 256) + lane);
          const signed char* c = (const signed char*)&qq;
#pragma unroll
          for (int i = 0; i < 8; i++) kk[i] = (float)c[i];
          s = Kst[(t - t0) * 8 + (lane >> 2)];
        } else {
#if KV8
          if (t >= T - WIN) load8h(Kr + ((size_t)g * RING + (t % RING)) * 256, lane, kk);
          else { load8(Kc, KROW(g, t) * 256, lane, kk); s = __half2float(Ks[SROW(g, t) * 8 + (lane >> 2)]); }
#else
          load8(Kc, KROW(g, t) * 256, lane, kk);
#endif
        }
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) a = fmaf(q[i], kk[i], a);
        d[u] = a * s;
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
#pragma unroll
        for (int u = 0; u < 4; u++) d[u] += __shfl_xor_sync(0xffffffffu, d[u], o);
      }
      if (lane < 4 && tb + lane < t1) sc[w][tb + lane - t0] = lane == 0 ? d[0] : lane == 1 ? d[1] : lane == 2 ? d[2] : d[3];
    }
    __syncwarp();
    float mt = -INF;
    for (int i = lane; i < n; i += 32) mt = fmaxf(mt, sc[w][i]);
    mt = warp_max(mt);
    const float Mn = fmaxf(M, mt), alpha = __expf(M - Mn);
    float sum = 0.f;
    for (int i = lane; i < n; i += 32) { float e = __expf(sc[w][i] - Mn); sc[w][i] = e; sum += e; }
    sum = warp_sum(sum);
    Lsum = Lsum * alpha + sum;
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] *= alpha;
    M = Mn;
    __syncwarp();
    for (int t = t0; t < t1; t++) {
      float vv[8];
      float p = sc[w][t - t0];
      if (staged) {
        uint2 qq = *((const uint2*)(Vt8 + (t - t0) * 256) + lane);
        const signed char* c = (const signed char*)&qq;
#pragma unroll
        for (int i = 0; i < 8; i++) vv[i] = (float)c[i];
        p *= Vst[(t - t0) * 8 + (lane >> 2)];
      } else {
#if KV8
        if (t >= T - WIN) load8h(Vr + ((size_t)g * RING + (t % RING)) * 256, lane, vv);
        else { load8(Vc, KROW(g, t) * 256, lane, vv); p *= __half2float(Vs[SROW(g, t) * 8 + (lane >> 2)]); }
#else
        load8(Vc, KROW(g, t) * 256, lane, vv);
#endif
      }
#pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = fmaf(p, vv[i], acc[i]);
    }
  }
  const size_t o = (size_t)h * nsplit + sp;
#pragma unroll
  for (int i = 0; i < 8; i++) pacc[o * 256 + lane * 8 + i] = acc[i];
  if (lane == 0) { pm[o] = M; pl[o] = Lsum; }
}

extern "C" __global__ void attn_partial6(const float* __restrict__ qr, const void* __restrict__ Kc, const void* __restrict__ Vc,
    const __half* __restrict__ Ks, const __half* __restrict__ Vs, const long long* __restrict__ pos_p,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc, int maxT, int nsplit,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr)
{
  __shared__ float sc[8][CH];
  __shared__ __align__(16) signed char Kt8[CH * 256], Vt8[CH * 256];
  __shared__ float Kst[CH * 8], Vst[CH * 8];
  const int g = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int T = (int)pos_p[0] + 1, T0 = sp * CH * SUB;
  if (T0 >= T) return;
  const int h = g * 8 + w;
  float q[8];
#pragma unroll
  for (int i = 0; i < 8; i++) q[i] = qr[h * 256 + lane * 8 + i] * 0.0625f;
  float M = -INF, Lsum = 0.f;
  uint4 pk[4], pv[4]; float pks[2], pvs[2]; bool have = false;   // CH=64: 64*256 B = 4 uint4 per thread
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int sub = 0; sub < SUB; sub++) {
    const int t0 = T0 + sub * CH;
    if (t0 >= T) break;
    const int t1 = min(t0 + CH, T), n = t1 - t0;
#if KV8
    const bool staged = t1 - 1 < T - WIN;
#else
    const bool staged = false;
#endif
    __syncthreads();                             // previous tile fully consumed
    if (staged) {
      if (sub == 0 || !have) {                   // first tile (or nothing prefetched): load directly
        const uint4* ks = (const uint4*)((const signed char*)Kc + KROW(g, t0) * 256);
        const uint4* vs = (const uint4*)((const signed char*)Vc + KROW(g, t0) * 256);
        for (int i = threadIdx.x; i < n * 16; i += 256) { ((uint4*)Kt8)[i] = ks[i]; ((uint4*)Vt8)[i] = vs[i]; }
        for (int i = threadIdx.x; i < n * 8; i += 256) { Kst[i] = __half2float(Ks[SROW(g, t0) * 8 + i]); Vst[i] = __half2float(Vs[SROW(g, t0) * 8 + i]); }
      } else {                                   // tile was prefetched into registers during the previous tile's math
#pragma unroll
        for (int r = 0; r < 4; r++) { ((uint4*)Kt8)[threadIdx.x + 256 * r] = pk[r]; ((uint4*)Vt8)[threadIdx.x + 256 * r] = pv[r]; }
#pragma unroll
        for (int r = 0; r < 2; r++) { Kst[threadIdx.x + 256 * r] = pks[r]; Vst[threadIdx.x + 256 * r] = pvs[r]; }
      }
      __syncthreads();
    }
    have = false;                                // prefetch the next tile if it is a full staged one
    {
      const int u0 = t0 + CH, u1 = min(u0 + CH, T);
      if (sub + 1 < SUB && u0 < T && u1 - u0 == CH && u1 - 1 < T - WIN) {
        const uint4* ks = (const uint4*)((const signed char*)Kc + KROW(g, u0) * 256);
        const uint4* vs = (const uint4*)((const signed char*)Vc + KROW(g, u0) * 256);
#pragma unroll
        for (int r = 0; r < 4; r++) { pk[r] = ks[threadIdx.x + 256 * r]; pv[r] = vs[threadIdx.x + 256 * r]; }
#pragma unroll
        for (int r = 0; r < 2; r++) { pks[r] = __half2float(Ks[SROW(g, u0) * 8 + threadIdx.x + 256 * r]); pvs[r] = __half2float(Vs[SROW(g, u0) * 8 + threadIdx.x + 256 * r]); }
        have = true;
      }
    }
    for (int tb = t0; tb < t1; tb += 4) {
      float d[4];
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int t = min(tb + u, t1 - 1);
        float kk[8], s = 1.f;
        if (staged) {
          uint2 qq = *((const uint2*)(Kt8 + (t - t0) * 256) + lane);
          const signed char* c = (const signed char*)&qq;
#pragma unroll
          for (int i = 0; i < 8; i++) kk[i] = (float)c[i];
          s = Kst[(t - t0) * 8 + (lane >> 2)];
        } else {
#if KV8
          if (t >= T - WIN) load8h(Kr + ((size_t)g * RING + (t % RING)) * 256, lane, kk);
          else { load8(Kc, KROW(g, t) * 256, lane, kk); s = __half2float(Ks[SROW(g, t) * 8 + (lane >> 2)]); }
#else
          load8(Kc, KROW(g, t) * 256, lane, kk);
#endif
        }
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) a = fmaf(q[i], kk[i], a);
        d[u] = a * s;
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
#pragma unroll
        for (int u = 0; u < 4; u++) d[u] += __shfl_xor_sync(0xffffffffu, d[u], o);
      }
      if (lane < 4 && tb + lane < t1) sc[w][tb + lane - t0] = lane == 0 ? d[0] : lane == 1 ? d[1] : lane == 2 ? d[2] : d[3];
    }
    __syncwarp();
    float mt = -INF;
    for (int i = lane; i < n; i += 32) mt = fmaxf(mt, sc[w][i]);
    mt = warp_max(mt);
    const float Mn = fmaxf(M, mt), alpha = __expf(M - Mn);
    float sum = 0.f;
    for (int i = lane; i < n; i += 32) { float e = __expf(sc[w][i] - Mn); sc[w][i] = e; sum += e; }
    sum = warp_sum(sum);
    Lsum = Lsum * alpha + sum;
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] *= alpha;
    M = Mn;
    __syncwarp();
    for (int t = t0; t < t1; t++) {
      float vv[8];
      float p = sc[w][t - t0];
      if (staged) {
        uint2 qq = *((const uint2*)(Vt8 + (t - t0) * 256) + lane);
        const signed char* c = (const signed char*)&qq;
#pragma unroll
        for (int i = 0; i < 8; i++) vv[i] = (float)c[i];
        p *= Vst[(t - t0) * 8 + (lane >> 2)];
      } else {
#if KV8
        if (t >= T - WIN) load8h(Vr + ((size_t)g * RING + (t % RING)) * 256, lane, vv);
        else { load8(Vc, KROW(g, t) * 256, lane, vv); p *= __half2float(Vs[SROW(g, t) * 8 + (lane >> 2)]); }
#else
        load8(Vc, KROW(g, t) * 256, lane, vv);
#endif
      }
#pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = fmaf(p, vv[i], acc[i]);
    }
  }
  const size_t o = (size_t)h * nsplit + sp;
#pragma unroll
  for (int i = 0; i < 8; i++) pacc[o * 256 + lane * 8 + i] = acc[i];
  if (lane == 0) { pm[o] = M; pl[o] = Lsum; }
}

#ifndef STAGE_ONLY
#define STAGE_ONLY 0
#endif
#ifndef QK_ONLY
#define QK_ONLY 0
#endif
#ifndef SKIP_WIN
#define SKIP_WIN 0
#endif
#ifndef QK8
#define QK8 0
#endif
// exact int8 -> float without I2F (quarter rate on Pascal): byte ^ 0x80 into the mantissa of 2^23, subtract 2^23 + 128
__device__ __forceinline__ void cvt8(uint2 w, float* f) {
  const unsigned a = w.x ^ 0x80808080u, b = w.y ^ 0x80808080u;
  f[0] = __int_as_float(__byte_perm(a, 0x4B000000u, 0x7440)) - 8388736.f; f[1] = __int_as_float(__byte_perm(a, 0x4B000000u, 0x7441)) - 8388736.f;
  f[2] = __int_as_float(__byte_perm(a, 0x4B000000u, 0x7442)) - 8388736.f; f[3] = __int_as_float(__byte_perm(a, 0x4B000000u, 0x7443)) - 8388736.f;
  f[4] = __int_as_float(__byte_perm(b, 0x4B000000u, 0x7440)) - 8388736.f; f[5] = __int_as_float(__byte_perm(b, 0x4B000000u, 0x7441)) - 8388736.f;
  f[6] = __int_as_float(__byte_perm(b, 0x4B000000u, 0x7442)) - 8388736.f; f[7] = __int_as_float(__byte_perm(b, 0x4B000000u, 0x7443)) - 8388736.f;
}

// decode attention v7 = v6 (tiles of CH staged once per block, SUB tiles per split, register prefetch) with 16-key rounds:
// reduce-scatter (1 shuffle per key instead of 5) and exact byte-permute int8 conversion. Same partial layout -> attn_combine5.
extern "C" __global__ void attn_partial7(const float* __restrict__ qr, const void* __restrict__ Kc, const void* __restrict__ Vc,
    const __half* __restrict__ Ks, const __half* __restrict__ Vs, const long long* __restrict__ pos_p,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc, int maxT, int nsplit,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr)
{
  __shared__ float sc[8][CH];
  __shared__ __align__(16) signed char Kt8[CH * 256], Vt8[CH * 256];
  __shared__ float Kst[CH * 8], Vst[CH * 8];
  const int g = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int T = (int)pos_p[0] + 1, T0 = sp * CH * SUB;
  if (T0 >= T) return;
  const int h = g * 8 + w;
  float q[8];
#pragma unroll
  for (int i = 0; i < 8; i++) q[i] = qr[h * 256 + lane * 8 + i] * 0.0625f;
#if QK8
  // QK8 (slightly lossy): q as int8 with one scale per 32 dims (= 4 lanes); Q.K of int8 keys with dp4a, no conversions
  float qmx = 0.f;
#pragma unroll
  for (int i = 0; i < 8; i++) qmx = fmaxf(qmx, fabsf(q[i]));
  qmx = fmaxf(qmx, __shfl_xor_sync(0xffffffffu, qmx, 1)); qmx = fmaxf(qmx, __shfl_xor_sync(0xffffffffu, qmx, 2));
  const float qsc = qmx > 0.f ? qmx / 127.f : 1.f;
  int q8[2];
#pragma unroll
  for (int w2 = 0; w2 < 2; w2++) {
    int pk = 0;
#pragma unroll
    for (int b4 = 0; b4 < 4; b4++) pk |= (__float2int_rn(q[w2 * 4 + b4] / qsc) & 255) << (8 * b4);
    q8[w2] = pk;
  }
#endif
  float M = -INF, Lsum = 0.f;
  uint4 pk[4], pv[4]; float pks[2], pvs[2]; bool have = false;   // CH=64: 64*256 B = 4 uint4 per thread
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int sub = 0; sub < SUB; sub++) {
    const int t0 = T0 + sub * CH;
    if (t0 >= T) break;
    const int t1 = min(t0 + CH, T), n = t1 - t0;
#if KV8
    const bool staged = t1 - 1 < T - WIN;
#else
    const bool staged = false;
#endif
    __syncthreads();                             // previous tile fully consumed
    if (staged) {
      if (sub == 0 || !have) {                   // first tile (or nothing prefetched): load directly
        const uint4* ks = (const uint4*)((const signed char*)Kc + KROW(g, t0) * 256);
        const uint4* vs = (const uint4*)((const signed char*)Vc + KROW(g, t0) * 256);
        for (int i = threadIdx.x; i < n * 16; i += 256) { ((uint4*)Kt8)[i] = ks[i]; ((uint4*)Vt8)[i] = vs[i]; }
        for (int i = threadIdx.x; i < n * 8; i += 256) { Kst[i] = __half2float(Ks[SROW(g, t0) * 8 + i]); Vst[i] = __half2float(Vs[SROW(g, t0) * 8 + i]); }
      } else {                                   // tile was prefetched into registers during the previous tile's math
#pragma unroll
        for (int r = 0; r < 4; r++) { ((uint4*)Kt8)[threadIdx.x + 256 * r] = pk[r]; ((uint4*)Vt8)[threadIdx.x + 256 * r] = pv[r]; }
#pragma unroll
        for (int r = 0; r < 2; r++) { Kst[threadIdx.x + 256 * r] = pks[r]; Vst[threadIdx.x + 256 * r] = pvs[r]; }
      }
      __syncthreads();
    }
    have = false;                                // prefetch the next tile if it is a full staged one
    {
      const int u0 = t0 + CH, u1 = min(u0 + CH, T);
      if (sub + 1 < SUB && u0 < T && u1 - u0 == CH && u1 - 1 < T - WIN) {
        const uint4* ks = (const uint4*)((const signed char*)Kc + KROW(g, u0) * 256);
        const uint4* vs = (const uint4*)((const signed char*)Vc + KROW(g, u0) * 256);
#pragma unroll
        for (int r = 0; r < 4; r++) { pk[r] = ks[threadIdx.x + 256 * r]; pv[r] = vs[threadIdx.x + 256 * r]; }
#pragma unroll
        for (int r = 0; r < 2; r++) { pks[r] = __half2float(Ks[SROW(g, u0) * 8 + threadIdx.x + 256 * r]); pvs[r] = __half2float(Vs[SROW(g, u0) * 8 + threadIdx.x + 256 * r]); }
        have = true;
      }
    }
#if STAGE_ONLY
    if (staged) { acc[0] += (float)Kt8[lane] + (float)Vt8[lane]; continue; }
#endif
#if SKIP_WIN
    if (!staged) continue;
#endif
    for (int tb = t0; tb < t1; tb += 16) {       // 16 keys per round: one reduce-scatter instead of 5 shuffles per key
      float* Kf = (float*)Kt8; float* Vf = (float*)Vt8;   // non-int8 tiles: 16 keys staged as float (exact fp16 ring / int8*scale)
      if (!staged) {
        __syncthreads();
        for (int idx = threadIdx.x; idx < 16 * 64; idx += 256) {      // 16 keys x 64 groups of 4 dims
          const int u = idx >> 6, d4 = (idx & 63) * 4, t = tb + u;
          float4 kq = make_float4(0.f, 0.f, 0.f, 0.f), vq = kq;
          if (t < t1) {
#if KV8
            if (t >= T - WIN) {
              const __half* kr = Kr + ((size_t)g * RING + (t % RING)) * 256 + d4; const __half* vr = Vr + ((size_t)g * RING + (t % RING)) * 256 + d4;
              kq = make_float4(__half2float(kr[0]), __half2float(kr[1]), __half2float(kr[2]), __half2float(kr[3]));
              vq = make_float4(__half2float(vr[0]), __half2float(vr[1]), __half2float(vr[2]), __half2float(vr[3]));
            } else {
              const signed char* kc = (const signed char*)Kc + KROW(g, t) * 256 + d4; const signed char* vc = (const signed char*)Vc + KROW(g, t) * 256 + d4;
              const float ksc = __half2float(Ks[SROW(g, t) * 8 + (d4 >> 5)]), vsc = __half2float(Vs[SROW(g, t) * 8 + (d4 >> 5)]);
              kq = make_float4((float)kc[0] * ksc, (float)kc[1] * ksc, (float)kc[2] * ksc, (float)kc[3] * ksc);
              vq = make_float4((float)vc[0] * vsc, (float)vc[1] * vsc, (float)vc[2] * vsc, (float)vc[3] * vsc);
            }
#else
            const __half* kr = (const __half*)Kc + KROW(g, t) * 256 + d4; const __half* vr = (const __half*)Vc + KROW(g, t) * 256 + d4;
            kq = make_float4(__half2float(kr[0]), __half2float(kr[1]), __half2float(kr[2]), __half2float(kr[3]));
            vq = make_float4(__half2float(vr[0]), __half2float(vr[1]), __half2float(vr[2]), __half2float(vr[3]));
#endif
          }
          *(float4*)(Kf + u * 256 + d4) = kq; *(float4*)(Vf + u * 256 + d4) = vq;
        }
        __syncthreads();
      }
      float d[16];
#pragma unroll
      for (int u = 0; u < 16; u++) {
        const int t = tb + u;
        float a = 0.f;
        if (t < t1) {
          float kk[8], s = 1.f;
          if (staged) {
#if QK8
            const uint2 w8 = *((const uint2*)(Kt8 + (t - t0) * 256) + lane);
            d[u] = (float)__dp4a(q8[1], (int)w8.y, __dp4a(q8[0], (int)w8.x, 0)) * (qsc * Kst[(t - t0) * 8 + (lane >> 2)]);
            continue;
#else
            cvt8(*((const uint2*)(Kt8 + (t - t0) * 256) + lane), kk);
            s = Kst[(t - t0) * 8 + (lane >> 2)];
#endif
          } else {
            const float4 k0 = *(const float4*)(Kf + (t - tb) * 256 + lane * 8), k1 = *(const float4*)(Kf + (t - tb) * 256 + lane * 8 + 4);
            kk[0] = k0.x; kk[1] = k0.y; kk[2] = k0.z; kk[3] = k0.w; kk[4] = k1.x; kk[5] = k1.y; kk[6] = k1.z; kk[7] = k1.w;
          }
#pragma unroll
          for (int i = 0; i < 8; i++) a = fmaf(q[i], kk[i], a);
          a *= s;
        }
        d[u] = a;
      }
      // reduce-scatter over the 32 lanes: afterwards lanes 2j and 2j+1 hold the score of key kl(lane)
#define RS_STAGE(N, O) { const bool up = (lane & (O)) != 0;                                           \
        _Pragma("unroll") for (int j = 0; j < (N); j++) {                                             \
          const float snd = up ? d[j] : d[j + (N)], kp = up ? d[j + (N)] : d[j];                     \
          d[j] = kp + __shfl_xor_sync(0xffffffffu, snd, (O)); } }
      RS_STAGE(8, 16) RS_STAGE(4, 8) RS_STAGE(2, 4) RS_STAGE(1, 2)
#undef RS_STAGE
      d[0] += __shfl_xor_sync(0xffffffffu, d[0], 1);
      const int kl = ((lane >> 1) & 1) + ((lane >> 2) & 1) * 2 + ((lane >> 3) & 1) * 4 + ((lane >> 4) & 1) * 8;
      const float sct = tb + kl < t1 ? d[0] : -INF;
      const float Mn = fmaxf(M, warp_max(sct));
      const float alpha = __expf(M - Mn);
      const float ex = sct == -INF ? 0.f : __expf(sct - Mn);
      Lsum = Lsum * alpha + warp_sum((lane & 1) ? 0.f : ex);
#pragma unroll
      for (int i = 0; i < 8; i++) acc[i] *= alpha;
      M = Mn;
      if (!(lane & 1)) sc[w][kl] = ex;
      __syncwarp();
#if QK_ONLY
      const int nu = 0;
#else
      const int nu = min(16, t1 - tb);
#endif
      for (int u = 0; u < nu; u++) {
        const int t = tb + u;
        float vv[8], p = sc[w][u];
        if (staged) {
          cvt8(*((const uint2*)(Vt8 + (t - t0) * 256) + lane), vv);
          p *= Vst[(t - t0) * 8 + (lane >> 2)];
        } else {
          const float4 v0 = *(const float4*)(Vf + u * 256 + lane * 8), v1 = *(const float4*)(Vf + u * 256 + lane * 8 + 4);
          vv[0] = v0.x; vv[1] = v0.y; vv[2] = v0.z; vv[3] = v0.w; vv[4] = v1.x; vv[5] = v1.y; vv[6] = v1.z; vv[7] = v1.w;
        }
#pragma unroll
        for (int i = 0; i < 8; i++) acc[i] = fmaf(p, vv[i], acc[i]);
      }
      __syncwarp();
    }
  }
  const size_t o = (size_t)h * nsplit + sp;
#pragma unroll
  for (int i = 0; i < 8; i++) pacc[o * 256 + lane * 8 + i] = acc[i];
  if (lane == 0) { pm[o] = M; pl[o] = Lsum; }
}


// decode attention v8: a warp computes 4 heads (hg) for a quarter (kq) of every 64-key tile, so every int8 K/V value it
// loads and converts serves 4 heads instead of 1 (v6: each of the 8 head-warps converts the same tile). Online softmax per
// warp; at the end the 4 key quarters of each head are merged in shared memory. Same partial layout -> attn_combine5.
extern "C" __global__ void __launch_bounds__(256, 2) attn_partial8(const float* __restrict__ qr, const void* __restrict__ Kc, const void* __restrict__ Vc,
    const __half* __restrict__ Ks, const __half* __restrict__ Vs, const long long* __restrict__ pos_p,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc, int maxT, int nsplit,
    const __half* __restrict__ Kr, const __half* __restrict__ Vr)
{
  __shared__ __align__(16) signed char KVb[2 * CH * 256];   // K tile | V tile; reused as one 32 KB block for the merge
  signed char* const Kt8 = KVb; signed char* const Vt8 = KVb + CH * 256;
  __shared__ float Kst[CH * 8], Vst[CH * 8];
  __shared__ __align__(16) float sc[8][16];
  const int g = blockIdx.x, sp = blockIdx.y, lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  const int hg = w & 1, kq = w >> 1;
  const int T = (int)pos_p[0] + 1, T0 = sp * CH * SUB;
  if (T0 >= T) return;
  float q[4][8];
#pragma unroll
  for (int j = 0; j < 4; j++)
#pragma unroll
    for (int i = 0; i < 8; i++) q[j][i] = qr[(g * 8 + hg * 4 + j) * 256 + lane * 8 + i] * 0.0625f;
  float M[4], L[4], acc[4][8];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    M[j] = -INF; L[j] = 0.f;
#pragma unroll
    for (int i = 0; i < 8; i++) acc[j][i] = 0.f;
  }
  uint4 pk[4], pv[4]; float pks[2], pvs[2]; bool have = false;
  for (int sub = 0; sub < SUB; sub++) {
    const int t0 = T0 + sub * CH;
    if (t0 >= T) break;
    const int t1 = min(t0 + CH, T), n = t1 - t0;
#if KV8
    const bool staged = t1 - 1 < T - WIN;
#else
    const bool staged = false;
#endif
    __syncthreads();                             // previous tile fully consumed
    if (staged) {
      if (sub == 0 || !have) {                   // first tile (or nothing prefetched): load directly
        const uint4* ks = (const uint4*)((const signed char*)Kc + KROW(g, t0) * 256);
        const uint4* vs = (const uint4*)((const signed char*)Vc + KROW(g, t0) * 256);
        for (int i = threadIdx.x; i < n * 16; i += 256) { ((uint4*)Kt8)[i] = ks[i]; ((uint4*)Vt8)[i] = vs[i]; }
        for (int i = threadIdx.x; i < n * 8; i += 256) { Kst[i] = __half2float(Ks[SROW(g, t0) * 8 + i]); Vst[i] = __half2float(Vs[SROW(g, t0) * 8 + i]); }
      } else {                                   // tile was prefetched into registers during the previous tile's math
#pragma unroll
        for (int r = 0; r < 4; r++) { ((uint4*)Kt8)[threadIdx.x + 256 * r] = pk[r]; ((uint4*)Vt8)[threadIdx.x + 256 * r] = pv[r]; }
#pragma unroll
        for (int r = 0; r < 2; r++) { Kst[threadIdx.x + 256 * r] = pks[r]; Vst[threadIdx.x + 256 * r] = pvs[r]; }
      }
      __syncthreads();
    }
    have = false;                                // prefetch the next tile if it is a full staged one
    {
      const int u0 = t0 + CH, u1 = min(u0 + CH, T);
      if (sub + 1 < SUB && u0 < T && u1 - u0 == CH && u1 - 1 < T - WIN) {
        const uint4* ks = (const uint4*)((const signed char*)Kc + KROW(g, u0) * 256);
        const uint4* vs = (const uint4*)((const signed char*)Vc + KROW(g, u0) * 256);
#pragma unroll
        for (int r = 0; r < 4; r++) { pk[r] = ks[threadIdx.x + 256 * r]; pv[r] = vs[threadIdx.x + 256 * r]; }
#pragma unroll
        for (int r = 0; r < 2; r++) { pks[r] = __half2float(Ks[SROW(g, u0) * 8 + threadIdx.x + 256 * r]); pvs[r] = __half2float(Vs[SROW(g, u0) * 8 + threadIdx.x + 256 * r]); }
        have = true;
      }
    }
    const int jg = lane >> 3;                    // after the reduce-scatter: lane holds head jg, key ((lane >> 1) & 3)
    for (int tb = t0 + kq * 16; tb < min(t0 + kq * 16 + 16, t1); tb += 4) {     // my 16 keys of the tile, 4 per round
      float d[16];                               // [head j][key u] -> j * 4 + u
#pragma unroll
      for (int u = 0; u < 4; u++) {
        const int t = tb + u;
        float kk[8], s = 1.f;
        if (t < t1) {
          if (staged) { cvt8(*((const uint2*)(Kt8 + (t - t0) * 256) + lane), kk); s = Kst[(t - t0) * 8 + (lane >> 2)]; }
          else {
#if KV8
            if (t >= T - WIN) load8h(Kr + ((size_t)g * RING + (t % RING)) * 256, lane, kk);
            else { load8(Kc, KROW(g, t) * 256, lane, kk); s = __half2float(Ks[SROW(g, t) * 8 + (lane >> 2)]); }
#else
            load8(Kc, KROW(g, t) * 256, lane, kk);
#endif
          }
        } else {
#pragma unroll
          for (int i = 0; i < 8; i++) kk[i] = 0.f;
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
          float a = 0.f;
#pragma unroll
          for (int i = 0; i < 8; i++) a = fmaf(q[j][i], kk[i], a);
          d[j * 4 + u] = a * s;
        }
      }
#define RS_STAGE(N, O) { const bool up = (lane & (O)) != 0;                                           \
        _Pragma("unroll") for (int x = 0; x < (N); x++) {                                             \
          const float snd = up ? d[x] : d[x + (N)], kp = up ? d[x + (N)] : d[x];                     \
          d[x] = kp + __shfl_xor_sync(0xffffffffu, snd, (O)); } }
      RS_STAGE(8, 16) RS_STAGE(4, 8) RS_STAGE(2, 4) RS_STAGE(1, 2)
#undef RS_STAGE
      d[0] += __shfl_xor_sync(0xffffffffu, d[0], 1);
      const int ku = (lane >> 1) & 3;
      const float sct = tb + ku < t1 ? d[0] : -INF;
      float mt = sct;                            // max / sum within the 8 lanes of head jg
      mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2)); mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 4));
      float Mo = M[0];
#pragma unroll
      for (int j = 1; j < 4; j++) if (jg == j) Mo = M[j];
      const float Mn = fmaxf(Mo, mt);
      const float ex = sct == -INF ? 0.f : __expf(sct - Mn);
      float gs = (lane & 1) ? 0.f : ex;
      gs += __shfl_xor_sync(0xffffffffu, gs, 1); gs += __shfl_xor_sync(0xffffffffu, gs, 2); gs += __shfl_xor_sync(0xffffffffu, gs, 4);
      if (!(lane & 1)) sc[w][ku * 4 + jg] = ex;    // key-major: the 4 heads of a key are one float4
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const float mn = __shfl_sync(0xffffffffu, Mn, j * 8), sj = __shfl_sync(0xffffffffu, gs, j * 8);
        const float al = __expf(M[j] - mn);
        L[j] = L[j] * al + sj; M[j] = mn;
#pragma unroll
        for (int i = 0; i < 8; i++) acc[j][i] *= al;
      }
      __syncwarp();
      const int nu = min(4, t1 - tb);
      for (int u = 0; u < nu; u++) {
        const int t = tb + u;
        float vv[8], sv = 1.f;
        if (staged) { cvt8(*((const uint2*)(Vt8 + (t - t0) * 256) + lane), vv); sv = Vst[(t - t0) * 8 + (lane >> 2)]; }
        else {
#if KV8
          if (t >= T - WIN) load8h(Vr + ((size_t)g * RING + (t % RING)) * 256, lane, vv);
          else { load8(Vc, KROW(g, t) * 256, lane, vv); sv = __half2float(Vs[SROW(g, t) * 8 + (lane >> 2)]); }
#else
          load8(Vc, KROW(g, t) * 256, lane, vv);
#endif
        }
        const float4 p4 = *(const float4*)&sc[w][u * 4];
        const float pj[4] = {p4.x * sv, p4.y * sv, p4.z * sv, p4.w * sv};
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int i = 0; i < 8; i++) acc[j][i] = fmaf(pj[j], vv[i], acc[j][i]);
      }
      __syncwarp();
    }
  }
  // merge the 4 key quarters of every head (online-softmax merge) through shared memory, then write the split partial
  __syncthreads();
  float* sacc = (float*)Kt8;                     // [8 warps][4 heads][256] = 32 KB over Kt8 + Vt8 (contiguous)
  float* sML = Kst;                              // [8 warps][4 heads][2]
#pragma unroll
  for (int j = 0; j < 4; j++) {
#pragma unroll
    for (int i = 0; i < 8; i++) sacc[(w * 4 + j) * 256 + lane * 8 + i] = acc[j][i];
    if (lane == 0) { sML[(w * 4 + j) * 2] = M[j]; sML[(w * 4 + j) * 2 + 1] = L[j]; }
  }
  __syncthreads();
  {                                              // 256 threads: head (8) x 32 lanes, 8 dims each
    const int hh = threadIdx.x >> 5, hgg = hh >> 2, jj = hh & 3;
    float Mx = -INF;
#pragma unroll
    for (int qq = 0; qq < 4; qq++) Mx = fmaxf(Mx, sML[((qq * 2 + hgg) * 4 + jj) * 2]);
    float Ls = 0.f, o[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int qq = 0; qq < 4; qq++) {
      const int src = (qq * 2 + hgg) * 4 + jj;
      const float m = sML[src * 2];
      const float c = m == -INF ? 0.f : __expf(m - Mx);
      Ls += c * sML[src * 2 + 1];
#pragma unroll
      for (int i = 0; i < 8; i++) o[i] += c * sacc[src * 256 + lane * 8 + i];
    }
    const int h = g * 8 + hh;
    const size_t oo = (size_t)h * nsplit + sp;
#pragma unroll
    for (int i = 0; i < 8; i++) pacc[oo * 256 + lane * 8 + i] = o[i];
    if (lane == 0) { pm[oo] = Mx; pl[oo] = Ls; }
  }
}

// attn_combine5: merge the partials of attn_partial5 (CH*SUB tokens each); 16 heads x 256 dims, splits in parallel chunks
extern "C" __global__ void attn_combine5(const long long* __restrict__ pos_p, const float* __restrict__ pm,
    const float* __restrict__ pl, const float* __restrict__ pacc, const float* __restrict__ gate,
    float* __restrict__ out, int nsplit)
{
  const int h = blockIdx.x, d = threadIdx.x;
  const int ns = ((int)pos_p[0] + 1 + CH * SUB - 1) / (CH * SUB);
  float M = -INF;
  for (int s = 0; s < ns; s++) M = fmaxf(M, pm[h * nsplit + s]);
  float num = 0.f, den = 0.f;
  for (int s = 0; s < ns; s++) {
    size_t o = (size_t)h * nsplit + s;
    float c = __expf(pm[o] - M);
    num += c * pacc[o * 256 + d];
    den += c * pl[o];
  }
  float gg = gate[h * 256 + d];
  out[h * 256 + d] = num / den * (1.f / (1.f + __expf(-gg)));
}

extern "C" __global__ void attn_combine(const long long* __restrict__ pos_p, const float* __restrict__ pm,
    const float* __restrict__ pl, const float* __restrict__ pacc, const float* __restrict__ gate,
    float* __restrict__ out, int nsplit)
{
  const int h = blockIdx.x, d = threadIdx.x;
  const int ns = ((int)pos_p[0] + 1 + CH - 1) / CH;
  float M = -INF;
  for (int s = 0; s < ns; s++) M = fmaxf(M, pm[h * nsplit + s]);
  float num = 0.f, den = 0.f;
  for (int s = 0; s < ns; s++) {
    size_t o = (size_t)h * nsplit + s;
    float c = __expf(pm[o] - M);
    num += c * pacc[o * 256 + d];
    den += c * pl[o];
  }
  float gg = gate[h * 256 + d];
  out[h * 256 + d] = num / den * (1.f / (1.f + __expf(-gg)));
}

// ---------------------------------------------------------------- MoE router (+ shared expert gate, adaptive k)
// one block of 256 threads (= experts). Expert blob layout: see engine (offsets passed in).
// Outputs: weights w[0..k), w[k] = sigmoid(shared gate); routes[8] (-1 padded); counts[0] = 2*(k+1), counts[1] = k+1
// pointer arrays: gu_w[2*(k+1)], gu_s[..], gu_g[..] (global scales), dn_w[k+1], dn_s, dn_g
extern "C" __global__ void router(const float* __restrict__ Wr, const float* __restrict__ wsg, const float* __restrict__ h,
    const long long* __restrict__ loc, long long shared_blob, int l, int E, int H, int topk, float thr,
    float* __restrict__ wout, long long* __restrict__ routes, int* __restrict__ counts,
    long long* __restrict__ gu_w, long long* __restrict__ gu_s, float* __restrict__ gu_g,
    long long* __restrict__ dn_w, long long* __restrict__ dn_s, float* __restrict__ dn_g,
    long long o_gw, long long o_uw, long long o_dw, long long o_gs, long long o_us, long long o_ds, long long o_g2, int write_ptrs)
{
  __shared__ float lg[256];
  __shared__ float red[33];
  __shared__ int sel[16];
  __shared__ float selw[16];
  __shared__ int keff;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  // Wr is now the precomputed logit vector [E + 1] (router logits + shared-expert gate logit)
  if (threadIdx.x < E) lg[threadIdx.x] = Wr[threadIdx.x];
  float sg = Wr[E];
  __syncthreads();
  if (threadIdx.x == 0) red[32] = 0.f;
  float v = threadIdx.x < E ? lg[threadIdx.x] : -INF;
  float mx = block_max(v, red);
  float ex = threadIdx.x < E ? __expf(v - mx) : 0.f;
  float sum = block_sum(ex, red);
  float p = threadIdx.x < E ? ex / sum : -1.f;
  __syncthreads();
  if (threadIdx.x == 0) red[0] = sg;           // keep shared gate logit
  __syncthreads();
  sg = red[0];
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
    if (thr > 0.f) {                            // adaptive: smallest k whose (renormalised) mass >= thr
      float c = 0.f;
      for (k = 0; k < topk; ) { c += selw[k] / s; k++; if (c >= thr) break; }
      s = 0.f;
      for (int j = 0; j < k; j++) s += selw[j];
    }
    keff = k;
    for (int j = 0; j < k; j++) wout[j] = selw[j] / s;
    wout[k] = 1.f / (1.f + __expf(-sg));
    counts[0] = 2 * (k + 1); counts[1] = k + 1;
  }
  __syncthreads();
  const int k = keff;
  if (threadIdx.x < topk) routes[threadIdx.x] = threadIdx.x < k ? sel[threadIdx.x] : -1;
  if (write_ptrs && threadIdx.x <= k) {
    const int j = threadIdx.x;
    const long long b = j < k ? loc[(long long)l * E + sel[j]] : shared_blob;
    gu_w[2 * j] = b + o_gw; gu_w[2 * j + 1] = b + o_uw;
    gu_s[2 * j] = b + o_gs; gu_s[2 * j + 1] = b + o_us;
    const float* g2 = (const float*)(b + o_g2);
    gu_g[2 * j] = g2[0]; gu_g[2 * j + 1] = g2[1];
    dn_w[j] = b + o_dw; dn_s[j] = b + o_ds; dn_g[j] = g2[2];
  }
}

extern "C" __global__ void router2(const float* __restrict__ Wr, const float* __restrict__ wsg, const float* __restrict__ h,
    const long long* __restrict__ loc, long long shared_blob, int l, int E, int H, int topk, float thr,
    float* __restrict__ wout, long long* __restrict__ routes, int* __restrict__ counts,
    long long* __restrict__ gu_w, long long* __restrict__ gu_s, float* __restrict__ gu_g,
    long long* __restrict__ dn_w, long long* __restrict__ dn_s, float* __restrict__ dn_g,
    long long o_gw, long long o_uw, long long o_dw, long long o_gs, long long o_us, long long o_ds, long long o_g2, int write_ptrs,
    const float* __restrict__ g2tab, const float* __restrict__ g2sh, long long* __restrict__ routesB, float* __restrict__ wB,
    const volatile long long* __restrict__ tokseq, volatile long long* __restrict__ flag, float skipB)
{
  __shared__ float lg[256];
  __shared__ float red[33];
  __shared__ int sel[16];
  __shared__ float selw[16];
  __shared__ int keff;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  // Wr is now the precomputed logit vector [E + 1] (router logits + shared-expert gate logit)
  if (threadIdx.x < E) lg[threadIdx.x] = Wr[threadIdx.x];
  float sg = Wr[E];
  __syncthreads();
  if (threadIdx.x == 0) red[32] = 0.f;
  float v = threadIdx.x < E ? lg[threadIdx.x] : -INF;
  float mx = block_max(v, red);
  float ex = threadIdx.x < E ? __expf(v - mx) : 0.f;
  float sum = block_sum(ex, red);
  float p = threadIdx.x < E ? ex / sum : -1.f;
  __syncthreads();
  if (threadIdx.x == 0) red[0] = sg;           // keep shared gate logit
  __syncthreads();
  sg = red[0];
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
    if (thr > 0.f) {                            // adaptive: smallest k whose (renormalised) mass >= thr
      float c = 0.f;
      for (k = 0; k < topk; ) { c += selw[k] / s; k++; if (c >= thr) break; }
      s = 0.f;
      for (int j = 0; j < k; j++) s += selw[j];
    }
    keff = k;
    red[1] = s;
  }
  __syncthreads();
  const int k = keff;
  if (threadIdx.x < topk) routes[threadIdx.x] = threadIdx.x < k ? sel[threadIdx.x] : -1;
  __threadfence_system();
  __syncthreads();
  // router2: experts resident in VRAM (loc[l*E+e] != 0, a pinned host table) go straight into the A pointer arrays
  // (computed by the GPU right away, together with the shared expert at the end); the others go to routesB/wB (host)
  // in selection order for the CPU to resolve.
  if (threadIdx.x == 0) {
    int a = 0, nb = 0;
    // skipB > 0 (lossy, cache-aware): an expert that is not in VRAM and has less than skipB of the routed mass is left out
    // (no PCIe read); the remaining weights are renormalised. Never drops the top expert.
    // skipB < 0: same with threshold -skipB, but the kept weights are NOT renormalised (the dropped term is simply missing)
    unsigned keep = 0xffffu;
    if (skipB != 0.f) {
      const float th = fabsf(skipB);
      float s2 = 0.f;
      for (int j = 0; j < k; j++) {
        if ((j > 0 || th >= 1.f) && !loc[(long long)l * E + sel[j]] && selw[j] / red[1] < th) keep &= ~(1u << j);   // th >= 1: VRAM only
        else s2 += selw[j];
      }
      if (skipB > 0.f) red[1] = s2;
    }
    for (int j = 0; j < k; j++) {
      if (!(keep >> j & 1)) continue;
      const int e = sel[j];
      const long long b = loc[(long long)l * E + e];
      const float wj = selw[j] / red[1];
      if (b) {
        gu_w[2 * a] = b + o_gw; gu_w[2 * a + 1] = b + o_uw; gu_s[2 * a] = b + o_gs; gu_s[2 * a + 1] = b + o_us;
        const float* g2 = g2tab + ((long long)l * E + e) * 3;
        gu_g[2 * a] = g2[0]; gu_g[2 * a + 1] = g2[1]; dn_w[a] = b + o_dw; dn_s[a] = b + o_ds; dn_g[a] = g2[2]; wout[a] = wj;
        a++;
      } else { routesB[nb] = e; wB[nb] = wj; nb++; }
    }
    const long long b = shared_blob;
    gu_w[2 * a] = b + o_gw; gu_w[2 * a + 1] = b + o_uw; gu_s[2 * a] = b + o_gs; gu_s[2 * a + 1] = b + o_us;
    gu_g[2 * a] = g2sh[0]; gu_g[2 * a + 1] = g2sh[1]; dn_w[a] = b + o_dw; dn_s[a] = b + o_ds; dn_g[a] = g2sh[2];
    wout[a] = 1.f / (1.f + __expf(-sg));
    counts[0] = 2 * (a + 1); counts[1] = a + 1;
    for (int q = nb; q < 8; q++) routesB[q] = -1;
    __threadfence_system();
    if (flag) { flag[l] = *tokseq; __threadfence_system(); }   // CPU polls this instead of an event
  }
}

extern "C" __global__ void router3(const float* __restrict__ Wr, const float* __restrict__ wsg, const float* __restrict__ h,
    const long long* __restrict__ loc, long long shared_blob, int l, int E, int H, int topk, float thr,
    float* __restrict__ wout, long long* __restrict__ routes, int* __restrict__ counts,
    long long* __restrict__ gu_w, long long* __restrict__ gu_s, float* __restrict__ gu_g,
    long long* __restrict__ dn_w, long long* __restrict__ dn_s, float* __restrict__ dn_g,
    long long o_gw, long long o_uw, long long o_dw, long long o_gs, long long o_us, long long o_ds, long long o_g2, int write_ptrs,
    const float* __restrict__ g2tab, const float* __restrict__ g2sh,
    const volatile long long* __restrict__ tokseq, volatile long long* __restrict__ flag,
    long long* __restrict__ gu_wd, long long* __restrict__ gu_sd, long long* __restrict__ dn_wd, long long* __restrict__ dn_sd,
    const long long* __restrict__ vict, volatile long long* __restrict__ ctl, long long* __restrict__ inslog,
    volatile long long* __restrict__ req, const volatile long long* __restrict__ resp)
{
  __shared__ float lg[256];
  __shared__ float red[33];
  __shared__ int sel[16];
  __shared__ float selw[16];
  __shared__ int keff;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  // Wr is now the precomputed logit vector [E + 1] (router logits + shared-expert gate logit)
  if (threadIdx.x < E) lg[threadIdx.x] = Wr[threadIdx.x];
  float sg = Wr[E];
  __syncthreads();
  if (threadIdx.x == 0) red[32] = 0.f;
  float v = threadIdx.x < E ? lg[threadIdx.x] : -INF;
  float mx = block_max(v, red);
  float ex = threadIdx.x < E ? __expf(v - mx) : 0.f;
  float sum = block_sum(ex, red);
  float p = threadIdx.x < E ? ex / sum : -1.f;
  __syncthreads();
  if (threadIdx.x == 0) red[0] = sg;           // keep shared gate logit
  __syncthreads();
  sg = red[0];
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
    if (thr > 0.f) {                            // adaptive: smallest k whose (renormalised) mass >= thr
      float c = 0.f;
      for (k = 0; k < topk; ) { c += selw[k] / s; k++; if (c >= thr) break; }
      s = 0.f;
      for (int j = 0; j < k; j++) s += selw[j];
    }
    keff = k;
    red[1] = s;
  }
  __syncthreads();
  const int k = keff;
  if (threadIdx.x < topk) routes[threadIdx.x] = threadIdx.x < k ? sel[threadIdx.x] : -1;
  __threadfence_system();
  __syncthreads();
  // router3: every expert is resolved on the GPU. loc[l*E+e] = VRAM blob address, or pinned-RAM address | 1, or 0 (miss).
  // RAM experts are read zero-copy and, if a victim VRAM slot is available, copied into it by the GEMV (copy-on-read).
  // A miss is handed to the CPU: req = {code, seq}; the block spins until resp = {addr | 1, seq}.
  // ctl (pinned): [0] victims taken (index into vict), [1] number of victims, [2] inserts logged, [3] miss seq, [4] nospin
  if (threadIdx.x == 0) {
    int a = 0;
    long long vi = ctl[0];
    const long long nv = ctl[1];
    for (int j = 0; j < k; j++) {
      const int e = sel[j];
      const long long code = (long long)l * E + e;
      long long b = ((const volatile long long*)loc)[code];   // the CPU updates loc while the graph runs
      if (!b) {                                   // miss: ask the CPU
        if (ctl[4]) { b = shared_blob | 1; }      // warm-up run before graph capture: dummy pointer, result discarded
        else {
          const long long sq = ctl[3] + 1; ctl[3] = sq;
          req[0] = code; __threadfence_system(); req[1] = sq; __threadfence_system();
          while (resp[1] != sq) { }
          __threadfence_system();
          b = resp[0];
        }
      }
      long long d = 0;
      if (b & 1) {                                // pinned RAM: zero-copy, try copy-on-read into a victim slot
        b &= ~1LL;
        while (vi < nv) {
          const long long vslot = vict[2 * vi], vold = vict[2 * vi + 1];
          vi++;
          bool clash = false;                     // victim belongs to an expert this layer is using right now
          if (vold >= 0 && vold / E == l) for (int q = 0; q < k; q++) if (sel[q] == (int)(vold % E)) clash = true;
          if (clash) continue;
          if (vold >= 0) ((volatile long long*)loc)[vold] = 0;   // the victim's old expert is not in VRAM any more (CPU fixes it up later)
          d = vslot;
          const long long n = ctl[2]; inslog[2 * n] = code; inslog[2 * n + 1] = vslot; ctl[2] = n + 1;
          break;
        }
      }
      gu_w[2 * a] = b + o_gw; gu_w[2 * a + 1] = b + o_uw; gu_s[2 * a] = b + o_gs; gu_s[2 * a + 1] = b + o_us;
      const float* g2 = g2tab + code * 3;
      gu_g[2 * a] = g2[0]; gu_g[2 * a + 1] = g2[1]; dn_w[a] = b + o_dw; dn_s[a] = b + o_ds; dn_g[a] = g2[2];
      wout[a] = selw[j] / red[1];
      gu_wd[2 * a] = d ? d + o_gw : 0; gu_wd[2 * a + 1] = d ? d + o_uw : 0; gu_sd[2 * a] = d ? d + o_gs : 0; gu_sd[2 * a + 1] = d ? d + o_us : 0;
      dn_wd[a] = d ? d + o_dw : 0; dn_sd[a] = d ? d + o_ds : 0;
      a++;
    }
    ctl[0] = vi;
    const long long b = shared_blob;
    gu_w[2 * a] = b + o_gw; gu_w[2 * a + 1] = b + o_uw; gu_s[2 * a] = b + o_gs; gu_s[2 * a + 1] = b + o_us;
    gu_g[2 * a] = g2sh[0]; gu_g[2 * a + 1] = g2sh[1]; dn_w[a] = b + o_dw; dn_s[a] = b + o_ds; dn_g[a] = g2sh[2];
    gu_wd[2 * a] = gu_wd[2 * a + 1] = gu_sd[2 * a] = gu_sd[2 * a + 1] = 0; dn_wd[a] = dn_sd[a] = 0;
    wout[a] = 1.f / (1.f + __expf(-sg));
    counts[0] = 2 * (a + 1); counts[1] = a + 1;
    __threadfence_system();
    flag[l] = *tokseq; __threadfence_system();
  }
}

extern "C" __global__ void router4(const float* __restrict__ Wr, const float* __restrict__ wsg, const float* __restrict__ h,
    const long long* __restrict__ loc, long long shared_blob, int l, int E, int H, int topk, float thr,
    float* __restrict__ wout, long long* __restrict__ routes, int* __restrict__ counts,
    long long* __restrict__ gu_w, long long* __restrict__ gu_s, float* __restrict__ gu_g,
    long long* __restrict__ dn_w, long long* __restrict__ dn_s, float* __restrict__ dn_g,
    long long o_gw, long long o_uw, long long o_dw, long long o_gs, long long o_us, long long o_ds, long long o_g2, int write_ptrs,
    const float* __restrict__ g2tab, const float* __restrict__ g2sh,
    const volatile long long* __restrict__ tokseq, volatile long long* __restrict__ flag,
    long long* __restrict__ gu_wd, long long* __restrict__ gu_sd, long long* __restrict__ dn_wd, long long* __restrict__ dn_sd,
    const long long* __restrict__ vict, volatile long long* __restrict__ ctl, long long* __restrict__ inslog,
    volatile long long* __restrict__ mreq, float* __restrict__ mw, volatile long long* __restrict__ nmiss, long long* __restrict__ lsel)
{
  __shared__ float lg[256];
  __shared__ float red[33];
  __shared__ int sel[16];
  __shared__ float selw[16];
  __shared__ int keff;
  const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
  // Wr is now the precomputed logit vector [E + 1] (router logits + shared-expert gate logit)
  if (threadIdx.x < E) lg[threadIdx.x] = Wr[threadIdx.x];
  float sg = Wr[E];
  __syncthreads();
  if (threadIdx.x == 0) red[32] = 0.f;
  float v = threadIdx.x < E ? lg[threadIdx.x] : -INF;
  float mx = block_max(v, red);
  float ex = threadIdx.x < E ? __expf(v - mx) : 0.f;
  float sum = block_sum(ex, red);
  float p = threadIdx.x < E ? ex / sum : -1.f;
  __syncthreads();
  if (threadIdx.x == 0) red[0] = sg;           // keep shared gate logit
  __syncthreads();
  sg = red[0];
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
    if (thr > 0.f) {                            // adaptive: smallest k whose (renormalised) mass >= thr
      float c = 0.f;
      for (k = 0; k < topk; ) { c += selw[k] / s; k++; if (c >= thr) break; }
      s = 0.f;
      for (int j = 0; j < k; j++) s += selw[j];
    }
    keff = k;
    red[1] = s;
  }
  __syncthreads();
  const int k = keff;
  if (threadIdx.x < topk) routes[threadIdx.x] = threadIdx.x < k ? sel[threadIdx.x] : -1;
  __threadfence_system();
  __syncthreads();
  // router3: every expert is resolved on the GPU. loc[l*E+e] = VRAM blob address, or pinned-RAM address | 1, or 0 (miss).
  // RAM experts are read zero-copy and, if a victim VRAM slot is available, copied into it by the GEMV (copy-on-read).
  // A miss is handed to the CPU: req = {code, seq}; the block spins until resp = {addr | 1, seq}.
  // ctl (pinned): [0] victims taken (index into vict), [1] number of victims, [2] inserts logged, [3] miss seq, [4] nospin
  if (threadIdx.x == 0) {
    int a = 0, nm = 0;
    long long vi = ctl[0];
    const long long nv = ctl[1];
    for (int j = 0; j < k; j++) {
      const int e = sel[j];
      const long long code = (long long)l * E + e;
      long long b = ((const volatile long long*)loc)[code];   // the CPU updates loc while the graph runs
      lsel[j] = code;
      if (!b) {                                   // miss: computed later by moe_late (after the CPU loaded it), not waited for here
        mreq[l * 8 + nm] = code; mw[l * 8 + nm] = selw[j] / red[1]; nm++;
        continue;
      }
      long long d = 0;
      if (b & 1) {                                // pinned RAM: zero-copy, try copy-on-read into a victim slot
        b &= ~1LL;
        while (vi < nv) {
          const long long vslot = vict[2 * vi], vold = vict[2 * vi + 1];
          vi++;
          bool clash = false;                     // victim belongs to an expert this layer is using right now
          if (vold >= 0 && vold / E == l) for (int q = 0; q < k; q++) if (sel[q] == (int)(vold % E)) clash = true;
          if (clash) continue;
          if (vold >= 0) ((volatile long long*)loc)[vold] = 0;   // the victim's old expert is not in VRAM any more (CPU fixes it up later)
          d = vslot;
          const long long n = ctl[2]; inslog[2 * n] = code; inslog[2 * n + 1] = vslot; ctl[2] = n + 1;
          break;
        }
      }
      gu_w[2 * a] = b + o_gw; gu_w[2 * a + 1] = b + o_uw; gu_s[2 * a] = b + o_gs; gu_s[2 * a + 1] = b + o_us;
      const float* g2 = g2tab + code * 3;
      gu_g[2 * a] = g2[0]; gu_g[2 * a + 1] = g2[1]; dn_w[a] = b + o_dw; dn_s[a] = b + o_ds; dn_g[a] = g2[2];
      wout[a] = selw[j] / red[1];
      gu_wd[2 * a] = d ? d + o_gw : 0; gu_wd[2 * a + 1] = d ? d + o_uw : 0; gu_sd[2 * a] = d ? d + o_gs : 0; gu_sd[2 * a + 1] = d ? d + o_us : 0;
      dn_wd[a] = d ? d + o_dw : 0; dn_sd[a] = d ? d + o_ds : 0;
      a++;
    }
    ctl[0] = vi;
    const long long b = shared_blob;
    gu_w[2 * a] = b + o_gw; gu_w[2 * a + 1] = b + o_uw; gu_s[2 * a] = b + o_gs; gu_s[2 * a + 1] = b + o_us;
    gu_g[2 * a] = g2sh[0]; gu_g[2 * a + 1] = g2sh[1]; dn_w[a] = b + o_dw; dn_s[a] = b + o_ds; dn_g[a] = g2sh[2];
    gu_wd[2 * a] = gu_wd[2 * a + 1] = gu_sd[2 * a] = gu_sd[2 * a + 1] = 0; dn_wd[a] = dn_sd[a] = 0;
    wout[a] = 1.f / (1.f + __expf(-sg));
    counts[0] = 2 * (a + 1); counts[1] = a + 1;
    for (int q = k; q < 8; q++) lsel[q] = -1;
    nmiss[l] = nm;
    __threadfence_system();
    flag[l] = *tokseq; __threadfence_system();
  }
}

// moe_late_ptrs: pointer arrays for the experts router4 could not resolve (layer l). Spins until the CPU has published
// each one (rseq[l*8+q] == tokseq, raddr = pinned RAM address | 1), takes copy-on-read victims like router4.
extern "C" __global__ void moe_late_ptrs(int l, int E, const volatile long long* __restrict__ tokseq,
    const volatile long long* __restrict__ nmiss, const volatile long long* __restrict__ mreq, const float* __restrict__ mw,
    const volatile long long* __restrict__ raddr, const volatile long long* __restrict__ rseq, const long long* __restrict__ lsel,
    const float* __restrict__ g2tab, const long long* __restrict__ vict, volatile long long* __restrict__ ctl, long long* __restrict__ inslog,
    long long* __restrict__ loc, int* __restrict__ counts, float* __restrict__ wout,
    long long* __restrict__ gu_w, long long* __restrict__ gu_s, float* __restrict__ gu_g,
    long long* __restrict__ dn_w, long long* __restrict__ dn_s, float* __restrict__ dn_g,
    long long* __restrict__ gu_wd, long long* __restrict__ gu_sd, long long* __restrict__ dn_wd, long long* __restrict__ dn_sd,
    long long o_gw, long long o_uw, long long o_dw, long long o_gs, long long o_us, long long o_ds)
{
  if (threadIdx.x != 0) return;
  const int nm = (int)nmiss[l];
  const long long ts = *tokseq;
  long long vi = ctl[0];
  const long long nv = ctl[1];
  for (int q = 0; q < nm; q++) {
    const long long code = mreq[l * 8 + q];
    if (ctl[4]) { counts[0] = counts[1] = 0; return; }        // warm-up before graph capture: skip
    while (rseq[l * 8 + q] != ts) { }
    __threadfence_system();
    const long long b = raddr[l * 8 + q] & ~1LL;
    long long d = 0;
    while (vi < nv) {
      const long long vslot = vict[2 * vi], vold = vict[2 * vi + 1];
      vi++;
      bool clash = false;
      for (int j = 0; j < 8; j++) if (lsel[j] == vold && vold >= 0) clash = true;
      if (clash) continue;
      if (vold >= 0) ((volatile long long*)loc)[vold] = 0;
      d = vslot;
      const long long n = ctl[2]; inslog[2 * n] = code; inslog[2 * n + 1] = vslot; ctl[2] = n + 1;
      break;
    }
    gu_w[2 * q] = b + o_gw; gu_w[2 * q + 1] = b + o_uw; gu_s[2 * q] = b + o_gs; gu_s[2 * q + 1] = b + o_us;
    const float* g2 = g2tab + code * 3;
    gu_g[2 * q] = g2[0]; gu_g[2 * q + 1] = g2[1]; dn_w[q] = b + o_dw; dn_s[q] = b + o_ds; dn_g[q] = g2[2];
    wout[q] = mw[l * 8 + q];
    gu_wd[2 * q] = d ? d + o_gw : 0; gu_wd[2 * q + 1] = d ? d + o_uw : 0; gu_sd[2 * q] = d ? d + o_gs : 0; gu_sd[2 * q + 1] = d ? d + o_us : 0;
    dn_wd[q] = d ? d + o_dw : 0; dn_sd[q] = d ? d + o_ds : 0;
  }
  ctl[0] = vi;
  counts[0] = 2 * nm; counts[1] = nm;
}

extern "C" __global__ void cpy(const long long* __restrict__ a, long long* __restrict__ b) { if (threadIdx.x < 16) b[threadIdx.x] = a[threadIdx.x]; }
// n 8-byte words a -> b (a may be pinned host memory: replaces a memcpy node, which costs ~60 us in a graph on WDDM)
extern "C" __global__ void cpyn(const long long* __restrict__ a, long long* __restrict__ b, int n) { for (int i = threadIdx.x; i < n; i += blockDim.x) b[i] = a[i]; }

extern "C" __global__ void silu_mul(const float* __restrict__ gu, float* __restrict__ h, const int* __restrict__ cnt, int I) {
  int i = blockIdx.x * blockDim.x + threadIdx.x, e = i / I, j = i % I;
  if (e >= cnt[1]) return;
  float g = gu[(2 * e) * I + j], u = gu[(2 * e + 1) * I + j];
  h[i] = g / (1.f + __expf(-g)) * u;
}
'''

_mods = {}


def mod(**defs):
    key = tuple(sorted(defs.items()))
    if key not in _mods:
        opts = ('-std=c++14',) + tuple(f'-D{k}={v}' for k, v in key)
        _mods[key] = cp.RawModule(code=_SRC, options=opts)
    return _mods[key]


def fn(name, **defs):
    return mod(**defs).get_function(name)


def stream():
    return cp.cuda.ExternalStream(torch.cuda.current_stream().cuda_stream)


i32, f32, i64 = cp.int32, cp.float32, cp.int64
FMT = {'nvfp4': 0, 'fp8': 1, 'f32': 2, 'bf16': 3}
PRE_DOWN = int(__import__('os').environ.get('PRE_DOWN', '1'))


GEMV_SPLIT = os.environ.get('GEMV_SPLIT', '1') == '1'
GEMV_SPLIT_RW = int(os.environ.get('GEMV_SPLIT_RW', 8))
GEMV_SPLIT8 = os.environ.get('GEMV_SPLIT8', '0') == '1'
GEMV_SPLIT8_RW = int(os.environ.get('GEMV_SPLIT8_RW', 4))
GEMV_BT = int(os.environ.get('GEMV_BT', 128))   # threads per block of the fast GEMV (not K=512): 128 measured +2-8 % over 256 (internal benchmark, bit-identical)


def rgemv(fmt, wptr, sptr, gsc, x, y, N, K, M, xdiv=1, xstride=0, ydiv=1, ystride=0, wts=None, mcount=None, wdst=None, sdst=None, assign=False):
    """y[m/ydiv*ystride + n] += gsc[m] * wts[m] * sum_k W_m[n,k] x[m/xdiv*xstride + k]"""
    p = lambda t: 0 if t is None else t.data_ptr()
    if fmt in ('nvfp4', 'fp8') and K % 512 == 0:       # fast path: x in registers (tuned on GTX 1070)
        rw, lut = (4, 0) if fmt == 'nvfp4' else (1, 1)
        pre = int(fmt == 'nvfp4' and K == 512 and PRE_DOWN)   # MoE down: loads of all rows first (bit-identical, +22 %)
        if fmt == 'nvfp4' and K == 2048 and GEMV_SPLIT:   # split over 4 warps (bit-identical, fewer registers -> more warps per SM)
            rws = GEMV_SPLIT_RW
            fn('rgemv2s', FMT=0, KK=K, KV8=0, RW=rws, SPL=4, ASG=int(assign))(((N + rws - 1) // rws, M), (128,),
                (wptr.data_ptr(), p(sptr), p(gsc), x.data_ptr(), i32(xdiv), i32(xstride), y.data_ptr(), i32(ydiv), i32(ystride),
                 p(wts), p(mcount), i32(N), p(wdst), p(sdst)), stream=stream())
            return
        if fmt == 'fp8' and GEMV_SPLIT8 and K % 2048 == 0:   # loads split over 4 warps, chain order kept (bit-identical)
            rws = GEMV_SPLIT8_RW
            fn('rgemv2s8', FMT=1, KK=K, KV8=0, RW=rws, SPL=4, ASG=int(assign))(((N + rws - 1) // rws, M), (128,),
                (wptr.data_ptr(), p(sptr), p(gsc), x.data_ptr(), i32(xdiv), i32(xstride), y.data_ptr(), i32(ydiv), i32(ystride),
                 p(wts), p(mcount), i32(N), p(wdst), p(sdst)), stream=stream())
            return
        bt = 256 if K == 512 else GEMV_BT                  # MoE down (K=512): 256 stays faster (-3 % with 128)
        rows_per_block = bt // 32 * rw
        fn('rgemv2', FMT=FMT[fmt], KK=K, KV8=0, RW=rw, FP8LUT=lut, PRE=pre, ASG=int(assign))(((N + rows_per_block - 1) // rows_per_block, M), (bt,),
            (wptr.data_ptr(), p(sptr), p(gsc), x.data_ptr(), i32(xdiv), i32(xstride), y.data_ptr(), i32(ydiv), i32(ystride),
             p(wts), p(mcount), i32(N), p(wdst), p(sdst)), stream=stream())
        return
    fn('rgemv', FMT=FMT[fmt], KV8=0, ASG=int(assign))(((N + 7) // 8, M), (256,),
        (wptr.data_ptr(), p(sptr), p(gsc), x.data_ptr(), i32(xdiv), i32(xstride), y.data_ptr(), i32(ydiv), i32(ystride),
         p(wts), p(mcount), i32(N), i32(K)), shared_mem=(16 + 256 + K) * 4, stream=stream())
