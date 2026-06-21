#include "ops/flash_attention_bf16_fwd_bhsd/interface.h"
#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

// v3 warp-per-query — fix v2's register spill + low occupancy.
// v2 was one THREAD per query with acc[128] -> spills, and 64 threads/block -> low
// occupancy. v3: one WARP per query, the 32 lanes split D (lane L owns dims
// {L, L+32, L+64, L+96} = 4 each). So acc is 4 regs/lane (no spill), 256 threads/
// block (8 queries), and the Q·K dot is a warp-shuffle reduction. K/V tiles still
// cached in shared, reused across the block's 8 queries. D fixed at 128 (= 32 lanes
// × 4). Lesson: split the feature dim across the warp to kill per-thread acc[D].

#define DMAX 128
#define WPB 8           // warps (= queries) per block
#define BN 64           // keys per shared tile
#define THREADS (WPB * 32)

__device__ __forceinline__ float warp_sum(float v) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  return v;   // all lanes get the full sum
}

__global__ void __launch_bounds__(THREADS)
fa_v3(const __nv_bfloat16* __restrict__ Q,
      const __nv_bfloat16* __restrict__ K,
      const __nv_bfloat16* __restrict__ V,
      __nv_bfloat16* __restrict__ O,
      int B, int H, int S_q, int S_kv, int D, int causal, float scale,
      long qsb, long qsh, long qss, long qsd,
      long ksb, long ksh, long kss, long ksd,
      long vsb, long vsh, long vss, long vsd,
      long osb, long osh, long oss, long osd) {
  const int b = blockIdx.z, h = blockIdx.y;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int qpos = blockIdx.x * WPB + warp;
  const bool active = qpos < S_q;

  __shared__ __nv_bfloat16 Ks[BN][DMAX];
  __shared__ __nv_bfloat16 Vs[BN][DMAX];

  // this lane owns dims d0,d1,d2,d3 = lane + 32*{0,1,2,3}
  const long qbase = b * qsb + h * qsh + (long)qpos * qss;
  float q[4], acc[4] = {0.f, 0.f, 0.f, 0.f};
  if (active) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = __bfloat162float(Q[qbase + (long)(lane + 32 * i) * qsd]);
  }
  float m = -FLT_MAX, l = 0.f;

  const int kmax = causal ? min(blockIdx.x * WPB + WPB - 1, S_kv - 1) : (S_kv - 1);
  const long kb0 = b * ksb + h * ksh;
  const long vb0 = b * vsb + h * vsh;

  for (int kt = 0; kt <= kmax; kt += BN) {
    #pragma unroll
    for (int i = threadIdx.x; i < BN * DMAX; i += THREADS) {
      int row = i / DMAX, col = i % DMAX, kpos = kt + row;
      __nv_bfloat16 kk = __float2bfloat16(0.f), vv = kk;
      if (kpos < S_kv) {
        kk = K[kb0 + (long)kpos * kss + col * ksd];
        vv = V[vb0 + (long)kpos * vss + col * vsd];
      }
      Ks[row][col] = kk; Vs[row][col] = vv;
    }
    __syncthreads();

    if (active) {
      int kcount = min(BN, S_kv - kt);
      for (int j = 0; j < kcount; ++j) {
        int kpos = kt + j;
        if (causal && kpos > qpos) break;
        float part = 0.f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) part += q[i] * __bfloat162float(Ks[j][lane + 32 * i]);
        float s = warp_sum(part) * scale;           // full Q·K, every lane has it
        float m_new = fmaxf(m, s);
        float corr = __expf(m - m_new);
        float p = __expf(s - m_new);
        l = l * corr + p;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
          acc[i] = acc[i] * corr + p * __bfloat162float(Vs[j][lane + 32 * i]);
        m = m_new;
      }
    }
    __syncthreads();
  }

  if (active) {
    const long obase = b * osb + h * osh + (long)qpos * oss;
    float inv = (l > 0.f) ? (1.f / l) : 0.f;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      O[obase + (long)(lane + 32 * i) * osd] = __float2bfloat16(acc[i] * inv);
  }
}

extern "C" cudaError_t openkernels_launch_flash_attention_bf16_fwd_bhsd(
    const OpenKernelsFlashAttentionBF16FwdBHSDArgs* a, cudaStream_t stream) {
  if (a == nullptr) return cudaErrorInvalidValue;
  if (a->D != DMAX) return cudaErrorNotSupported;
  dim3 grid((a->S_q + WPB - 1) / WPB, a->H, a->B);
  fa_v3<<<grid, THREADS, 0, stream>>>(
      a->q, a->k, a->v, a->o, a->B, a->H, a->S_q, a->S_kv, a->D, a->causal, a->scale,
      a->q_stride_b, a->q_stride_h, a->q_stride_s, a->q_stride_d,
      a->k_stride_b, a->k_stride_h, a->k_stride_s, a->k_stride_d,
      a->v_stride_b, a->v_stride_h, a->v_stride_s, a->v_stride_d,
      a->o_stride_b, a->o_stride_h, a->o_stride_s, a->o_stride_d);
  return cudaGetLastError();
}
