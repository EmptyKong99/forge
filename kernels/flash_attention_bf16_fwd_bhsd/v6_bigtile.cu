#include "ops/flash_attention_bf16_fwd_bhsd/interface.h"
#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

// v6 bigtile — v5 with a larger key tile (BN). v5 used BN=16: for S=8192 that's 512
// tiles, each paying 2 __syncthreads (synchronous K/V load), a softmax reduction, and
// a P shared round-trip. Bigger BN amortizes all that fixed per-tile overhead over
// more keys → fewer synchronous load stalls. Parametric in BN (= 8·NSUB, k-steps
// NK=BN/16). Same tensor-core math + 4-warp shared-K/V as v5. D=128, S%16==0.

#define DM 128
#define DP 136
#define BN 32                 // key tile (sweep 16/32/64)
#define NSUB (BN / 8)         // n-subtiles for QK^T
#define NK (BN / 16)          // k-steps for PV
#define BNP (BN + 8)
#define WARPS 4
#define QPB (WARPS * 16)

__device__ __forceinline__ uint32_t sm(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void ldm4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(s));
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
fa_v6(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
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
  __shared__ __nv_bfloat16 Ks[BN][DP];
  __shared__ __nv_bfloat16 Vt[DM][BNP];
  __shared__ __nv_bfloat16 Ps[QPB][BNP];

  const long qb = b * qsb + h * qsh, kb = b * ksb + h * ksh, vb = b * vsb + h * vsh;
  for (int i = tid; i < QPB * DM; i += WARPS * 32) {
    int r = i / DM, c = i % DM;
    Qs[r][c] = (qblock + r < S_q) ? Q[qb + (long)(qblock + r) * qss + c * qsd] : __float2bfloat16(0.f);
  }

  float o_acc[16][4];
  #pragma unroll
  for (int n = 0; n < 16; ++n) o_acc[n][0] = o_acc[n][1] = o_acc[n][2] = o_acc[n][3] = 0.f;
  float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.f, l1 = 0.f;

  const int kmax = causal ? min(qblock + QPB - 1, S_kv - 1) : (S_kv - 1);

  for (int kt = 0; kt <= kmax; kt += BN) {
    __syncthreads();
    for (int i = tid; i < BN * DM; i += WARPS * 32) {
      int r = i / DM, c = i % DM; int kp = kt + r;
      Ks[r][c] = (kp < S_kv) ? K[kb + (long)kp * kss + c * ksd] : __float2bfloat16(0.f);
    }
    for (int i = tid; i < DM * BN; i += WARPS * 32) {
      int d = i / BN, key = i % BN; int kp = kt + key;
      Vt[d][key] = (kp < S_kv) ? V[vb + (long)kp * vss + d * vsd] : __float2bfloat16(0.f);
    }
    __syncthreads();

    // QK^T: NSUB n-subtiles, each accumulated over 8 k-steps
    float s_acc[NSUB][4];
    #pragma unroll
    for (int ns = 0; ns < NSUB; ++ns) { s_acc[ns][0] = s_acc[ns][1] = s_acc[ns][2] = s_acc[ns][3] = 0.f; }
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      int dcol = kk * 16 + ((lane >> 4) & 1) * 8;
      uint32_t a0, a1, a2, a3; ldm4(a0, a1, a2, a3, sm(&Qs[warp * 16 + (lane & 15)][dcol]));
      int kcol = kk * 16 + ((lane >> 3) & 1) * 8;
      #pragma unroll
      for (int ns = 0; ns < NSUB; ++ns) {
        uint32_t b0, b1; ldm2(b0, b1, sm(&Ks[ns * 8 + (lane & 7)][kcol]));
        mma(s_acc[ns][0], s_acc[ns][1], s_acc[ns][2], s_acc[ns][3],
            a0, a1, a2, a3, b0, b1, s_acc[ns][0], s_acc[ns][1], s_acc[ns][2], s_acc[ns][3]);
      }
    }
    // scale + causal mask; row R0 = s_acc[ns][0,1], row R1 = s_acc[ns][2,3]
    #pragma unroll
    for (int ns = 0; ns < NSUB; ++ns) {
      int k0 = ns * 8 + 2 * tidg;
      int kk0 = kt + k0, kk1 = kt + k0 + 1;
      s_acc[ns][0] = (causal && kk0 > qbase + group)     ? -FLT_MAX : s_acc[ns][0] * scale;
      s_acc[ns][1] = (causal && kk1 > qbase + group)     ? -FLT_MAX : s_acc[ns][1] * scale;
      s_acc[ns][2] = (causal && kk0 > qbase + group + 8) ? -FLT_MAX : s_acc[ns][2] * scale;
      s_acc[ns][3] = (causal && kk1 > qbase + group + 8) ? -FLT_MAX : s_acc[ns][3] * scale;
    }
    float lm0 = -FLT_MAX, lm1 = -FLT_MAX;
    #pragma unroll
    for (int ns = 0; ns < NSUB; ++ns) {
      lm0 = fmaxf(lm0, fmaxf(s_acc[ns][0], s_acc[ns][1]));
      lm1 = fmaxf(lm1, fmaxf(s_acc[ns][2], s_acc[ns][3]));
    }
    float mt0 = grp_max(lm0), mt1 = grp_max(lm1);
    float nm0 = fmaxf(m0, mt0), nm1 = fmaxf(m1, mt1);
    float c0 = __expf(m0 - nm0), c1 = __expf(m1 - nm1);
    float ps0 = 0.f, ps1 = 0.f;
    #pragma unroll
    for (int ns = 0; ns < NSUB; ++ns) {
      float a = __expf(s_acc[ns][0] - nm0), bb = __expf(s_acc[ns][1] - nm0);
      float cc = __expf(s_acc[ns][2] - nm1), dd = __expf(s_acc[ns][3] - nm1);
      s_acc[ns][0] = a; s_acc[ns][1] = bb; s_acc[ns][2] = cc; s_acc[ns][3] = dd;
      ps0 += a + bb; ps1 += cc + dd;
    }
    l0 = l0 * c0 + grp_sum(ps0); l1 = l1 * c1 + grp_sum(ps1); m0 = nm0; m1 = nm1;
    #pragma unroll
    for (int n = 0; n < 16; ++n) { o_acc[n][0] *= c0; o_acc[n][1] *= c0; o_acc[n][2] *= c1; o_acc[n][3] *= c1; }

    int pr0 = warp * 16 + group, pr1 = warp * 16 + group + 8;
    #pragma unroll
    for (int ns = 0; ns < NSUB; ++ns) {
      int k0 = ns * 8 + 2 * tidg;
      Ps[pr0][k0] = __float2bfloat16(s_acc[ns][0]); Ps[pr0][k0 + 1] = __float2bfloat16(s_acc[ns][1]);
      Ps[pr1][k0] = __float2bfloat16(s_acc[ns][2]); Ps[pr1][k0 + 1] = __float2bfloat16(s_acc[ns][3]);
    }
    __syncwarp();

    // PV: accumulate over NK k-steps of 16 keys; 16 d-subtiles
    #pragma unroll
    for (int kk = 0; kk < NK; ++kk) {
      uint32_t pa0, pa1, pa2, pa3;
      ldm4(pa0, pa1, pa2, pa3, sm(&Ps[warp * 16 + (lane & 15)][kk * 16 + ((lane >> 4) & 1) * 8]));
      #pragma unroll
      for (int n = 0; n < 16; ++n) {
        uint32_t vb0, vb1; ldm2(vb0, vb1, sm(&Vt[n * 8 + (lane & 7)][kk * 16 + ((lane >> 3) & 1) * 8]));
        mma(o_acc[n][0], o_acc[n][1], o_acc[n][2], o_acc[n][3], pa0, pa1, pa2, pa3, vb0, vb1,
            o_acc[n][0], o_acc[n][1], o_acc[n][2], o_acc[n][3]);
      }
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
  fa_v6<<<grid, WARPS * 32, 0, stream>>>(
      a->q, a->k, a->v, a->o, a->B, a->H, a->S_q, a->S_kv, a->D, a->causal, a->scale,
      a->q_stride_b, a->q_stride_h, a->q_stride_s, a->q_stride_d,
      a->k_stride_b, a->k_stride_h, a->k_stride_s, a->k_stride_d,
      a->v_stride_b, a->v_stride_h, a->v_stride_s, a->v_stride_d,
      a->o_stride_b, a->o_stride_h, a->o_stride_s, a->o_stride_d);
  return cudaGetLastError();
}
