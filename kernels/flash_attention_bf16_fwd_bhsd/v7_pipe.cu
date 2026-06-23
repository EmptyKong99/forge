#include "ops/flash_attention_bf16_fwd_bhsd/interface.h"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

// v7 pipe — v5 + cp.async double-buffered K/V (hide load latency without growing the
// tile; v6 showed bigger tiles regress on occupancy). K/V now stored CONTIGUOUSLY
// (Ks[key][d], Vs[key][d]) so cp.async can copy them; PV's B-operand therefore uses
// ldmatrix.TRANS on Vs (ldm.trans from [key][d] == non-trans from the transposed
// [d][key] of v5). Double buffer: compute on tile cur while prefetching cur+1.
// 4-warp block shares K/V. D=128, S%16==0.

#define DM 128
#define DP 136
#define BN 16
#define WARPS 4
#define QPB (WARPS * 16)

__device__ __forceinline__ uint32_t sm(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldm4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(s));
}
__device__ __forceinline__ void ldm2t(uint32_t& a, uint32_t& b, uint32_t s) {   // transposing
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1},[%2];\n" : "=r"(a), "=r"(b) : "r"(s));
}
__device__ __forceinline__ void ldm2(uint32_t& a, uint32_t& b, uint32_t s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];\n" : "=r"(a), "=r"(b) : "r"(s));
}
__device__ __forceinline__ void mma(float& d0, float& d1, float& d2, float& d3,
                                     uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1,
                                     float c0, float c1, float c2, float c3) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}
__device__ __forceinline__ float grp_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1)); v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2)); return v;
}
__device__ __forceinline__ float grp_sum(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 1); v += __shfl_xor_sync(0xffffffff, v, 2); return v;
}

__global__ void __launch_bounds__(WARPS * 32)
fa_v7(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
      const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
      int B, int H, int S_q, int S_kv, int D, int causal, float scale,
      long qsb, long qsh, long qss, long qsd, long ksb, long ksh, long kss, long ksd,
      long vsb, long vsh, long vss, long vsd, long osb, long osh, long oss, long osd) {
  const int b = blockIdx.z, h = blockIdx.y;
  const int qblock = blockIdx.x * QPB;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int group = lane >> 2, tidg = lane & 3;
  const int qbase = qblock + warp * 16;

  __shared__ __nv_bfloat16 Qs[QPB][DP];
  __shared__ __nv_bfloat16 Ks[2][BN][DP];
  __shared__ __nv_bfloat16 Vs[2][BN][DP];      // contiguous V[key][d]
  __shared__ __nv_bfloat16 Ps[QPB][BN + 8];

  const long qb = b * qsb + h * qsh, kb = b * ksb + h * ksh, vb = b * vsb + h * vsh;
  for (int i = tid; i < QPB * DM; i += WARPS * 32) {
    int r = i / DM, c = i % DM;
    Qs[r][c] = (qblock + r < S_q) ? Q[qb + (long)(qblock + r) * qss + c * qsd] : __float2bfloat16(0.f);
  }

  // cp.async one K/V tile (16x128) into buffer `buf`, key origin `kt`
  auto load = [&](int buf, int kt) {
    for (int i = tid; i < BN * DM; i += WARPS * 32) {
      int r = i / DM, c = i % DM; int kp = kt + r;          // 8 bf16 = 16B per chunk
      if ((c & 7) == 0) {
        if (kp < S_kv) {
          __pipeline_memcpy_async(&Ks[buf][r][c], &K[kb + (long)kp * kss + c * ksd], 16);
          __pipeline_memcpy_async(&Vs[buf][r][c], &V[vb + (long)kp * vss + c * vsd], 16);
        } else {
          *reinterpret_cast<int4*>(&Ks[buf][r][c]) = make_int4(0, 0, 0, 0);
          *reinterpret_cast<int4*>(&Vs[buf][r][c]) = make_int4(0, 0, 0, 0);
        }
      }
    }
    __pipeline_commit();
  };

  float o_acc[16][4];
  #pragma unroll
  for (int n = 0; n < 16; ++n) o_acc[n][0] = o_acc[n][1] = o_acc[n][2] = o_acc[n][3] = 0.f;
  float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.f, l1 = 0.f;

  const int kmax = causal ? min(qblock + QPB - 1, S_kv - 1) : (S_kv - 1);
  const int ntile = kmax / BN + 1;

  load(0, 0);
  for (int t = 0; t < ntile; ++t) {
    int cur = t & 1, kt = t * BN;
    if (t + 1 < ntile) load((t + 1) & 1, (t + 1) * BN);
    __pipeline_wait_prior(t + 1 < ntile ? 1 : 0);
    __syncthreads();

    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, u0 = 0, u1 = 0, u2 = 0, u3 = 0;
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      int dcol = kk * 16 + ((lane >> 4) & 1) * 8;
      uint32_t a0, a1, a2, a3; ldm4(a0, a1, a2, a3, sm(&Qs[warp * 16 + (lane & 15)][dcol]));
      int kcol = kk * 16 + ((lane >> 3) & 1) * 8;
      uint32_t b0, b1;
      ldm2(b0, b1, sm(&Ks[cur][(lane & 7)][kcol]));        mma(s0, s1, s2, s3, a0, a1, a2, a3, b0, b1, s0, s1, s2, s3);
      ldm2(b0, b1, sm(&Ks[cur][8 + (lane & 7)][kcol]));    mma(u0, u1, u2, u3, a0, a1, a2, a3, b0, b1, u0, u1, u2, u3);
    }
    float sc[8] = {s0, s1, u0, u1, s2, s3, u2, u3};
    int kcol0 = 2 * tidg;
    int keyidx[4] = {kcol0, kcol0 + 1, 8 + kcol0, 8 + kcol0 + 1};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int kpos = kt + keyidx[j];
      sc[j]     = (causal && kpos > qbase + group)     ? -FLT_MAX : sc[j] * scale;
      sc[j + 4] = (causal && kpos > qbase + group + 8) ? -FLT_MAX : sc[j + 4] * scale;
    }
    float mt0 = grp_max(fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3])));
    float mt1 = grp_max(fmaxf(fmaxf(sc[4], sc[5]), fmaxf(sc[6], sc[7])));
    float nm0 = fmaxf(m0, mt0), nm1 = fmaxf(m1, mt1);
    float c0 = __expf(m0 - nm0), c1 = __expf(m1 - nm1);
    float p[8];
    #pragma unroll
    for (int j = 0; j < 4; ++j) { p[j] = __expf(sc[j] - nm0); p[j + 4] = __expf(sc[j + 4] - nm1); }
    l0 = l0 * c0 + grp_sum(p[0] + p[1] + p[2] + p[3]);
    l1 = l1 * c1 + grp_sum(p[4] + p[5] + p[6] + p[7]);
    m0 = nm0; m1 = nm1;
    #pragma unroll
    for (int n = 0; n < 16; ++n) { o_acc[n][0] *= c0; o_acc[n][1] *= c0; o_acc[n][2] *= c1; o_acc[n][3] *= c1; }

    int pr0 = warp * 16 + group, pr1 = warp * 16 + group + 8;
    Ps[pr0][keyidx[0]] = __float2bfloat16(p[0]); Ps[pr0][keyidx[1]] = __float2bfloat16(p[1]);
    Ps[pr0][keyidx[2]] = __float2bfloat16(p[2]); Ps[pr0][keyidx[3]] = __float2bfloat16(p[3]);
    Ps[pr1][keyidx[0]] = __float2bfloat16(p[4]); Ps[pr1][keyidx[1]] = __float2bfloat16(p[5]);
    Ps[pr1][keyidx[2]] = __float2bfloat16(p[6]); Ps[pr1][keyidx[3]] = __float2bfloat16(p[7]);
    __syncwarp();

    uint32_t pa0, pa1, pa2, pa3; ldm4(pa0, pa1, pa2, pa3, sm(&Ps[warp * 16 + (lane & 15)][((lane >> 4) & 1) * 8]));
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
      uint32_t vb0, vb1; ldm2t(vb0, vb1, sm(&Vs[cur][lane & 15][n * 8]));   // trans: [key][d]->B[key,d]
      mma(o_acc[n][0], o_acc[n][1], o_acc[n][2], o_acc[n][3], pa0, pa1, pa2, pa3, vb0, vb1,
          o_acc[n][0], o_acc[n][1], o_acc[n][2], o_acc[n][3]);
    }
  }

  float inv0 = (l0 > 0.f) ? 1.f / l0 : 0.f, inv1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
  const long ob = b * osb + h * osh;
  int r0 = qbase + group, r1 = qbase + group + 8;
  #pragma unroll
  for (int n = 0; n < 16; ++n) {
    int dc = n * 8 + 2 * tidg;
    if (r0 < S_q) { O[ob + (long)r0 * oss + (long)dc * osd] = __float2bfloat16(o_acc[n][0] * inv0);
                    O[ob + (long)r0 * oss + (long)(dc + 1) * osd] = __float2bfloat16(o_acc[n][1] * inv0); }
    if (r1 < S_q) { O[ob + (long)r1 * oss + (long)dc * osd] = __float2bfloat16(o_acc[n][2] * inv1);
                    O[ob + (long)r1 * oss + (long)(dc + 1) * osd] = __float2bfloat16(o_acc[n][3] * inv1); }
  }
}

extern "C" cudaError_t openkernels_launch_flash_attention_bf16_fwd_bhsd(
    const OpenKernelsFlashAttentionBF16FwdBHSDArgs* a, cudaStream_t stream) {
  if (a == nullptr) return cudaErrorInvalidValue;
  if (a->D != DM || a->S_q % 16 != 0 || a->S_kv % BN != 0) return cudaErrorNotSupported;
  dim3 grid((a->S_q + QPB - 1) / QPB, a->H, a->B);
  fa_v7<<<grid, WARPS * 32, 0, stream>>>(
      a->q, a->k, a->v, a->o, a->B, a->H, a->S_q, a->S_kv, a->D, a->causal, a->scale,
      a->q_stride_b, a->q_stride_h, a->q_stride_s, a->q_stride_d,
      a->k_stride_b, a->k_stride_h, a->k_stride_s, a->k_stride_d,
      a->v_stride_b, a->v_stride_h, a->v_stride_s, a->v_stride_d,
      a->o_stride_b, a->o_stride_h, a->o_stride_s, a->o_stride_d);
  return cudaGetLastError();
}
