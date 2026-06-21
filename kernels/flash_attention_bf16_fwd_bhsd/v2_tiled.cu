#include "ops/flash_attention_bf16_fwd_bhsd/interface.h"
#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

// v2 tiled — flash-attention with K/V cached in shared memory.
// v1's waste: each query re-streamed ALL of K/V from global (S_q times over). v2
// processes a BLOCK of BM queries together: load each K/V tile [BN keys] into shared
// ONCE, reuse it across all BM queries, online-softmax across tiles. causal early-
// exits per query (keys are ordered, so break once kpos>qpos). Still SIMT (one
// thread per query, scalar dot) — tensor cores are the next rung. D fixed at 128.
//   reuse: K/V global reads drop ~BM×. acc[128]/thread spills (fixed in v3).

#define DMAX 128
#define BM 64      // queries per block (= threads)
#define BN 32      // keys per shared tile

__global__ void fa_v2(const __nv_bfloat16* __restrict__ Q,
                      const __nv_bfloat16* __restrict__ K,
                      const __nv_bfloat16* __restrict__ V,
                      __nv_bfloat16* __restrict__ O,
                      int B, int H, int S_q, int S_kv, int D, int causal, float scale,
                      long qsb, long qsh, long qss, long qsd,
                      long ksb, long ksh, long kss, long ksd,
                      long vsb, long vsh, long vss, long vsd,
                      long osb, long osh, long oss, long osd) {
  const int b = blockIdx.z, h = blockIdx.y;
  const int q0 = blockIdx.x * BM;
  const int tid = threadIdx.x;
  const int qpos = q0 + tid;
  const bool active = qpos < S_q;

  __shared__ __nv_bfloat16 Ks[BN][DMAX];
  __shared__ __nv_bfloat16 Vs[BN][DMAX];

  const long qbase = b * qsb + h * qsh + (long)qpos * qss;
  float q[DMAX], acc[DMAX];
  if (active) {
    #pragma unroll
    for (int d = 0; d < DMAX; ++d) { q[d] = __bfloat162float(Q[qbase + d * qsd]); acc[d] = 0.f; }
  }
  float m = -FLT_MAX, l = 0.f;

  // causal: no query in this block attends past its own position; cap the key sweep
  // at the block's last query (min with S_kv-1).
  const int kmax = causal ? min(q0 + BM - 1, S_kv - 1) : (S_kv - 1);
  const long kb0 = b * ksb + h * ksh;
  const long vb0 = b * vsb + h * vsh;

  for (int kt = 0; kt <= kmax; kt += BN) {
    // cooperative load of K/V tile [kt, kt+BN) into shared (all threads, active or not)
    #pragma unroll
    for (int i = tid; i < BN * DMAX; i += BM) {
      int row = i / DMAX, col = i % DMAX;
      int kpos = kt + row;
      __nv_bfloat16 kv0 = __float2bfloat16(0.f), vv0 = kv0;
      if (kpos < S_kv) {
        kv0 = K[kb0 + (long)kpos * kss + col * ksd];
        vv0 = V[vb0 + (long)kpos * vss + col * vsd];
      }
      Ks[row][col] = kv0; Vs[row][col] = vv0;
    }
    __syncthreads();

    if (active) {
      int kcount = min(BN, S_kv - kt);
      for (int j = 0; j < kcount; ++j) {
        int kpos = kt + j;
        if (causal && kpos > qpos) break;          // ordered keys -> rest masked
        float s = 0.f;
        #pragma unroll
        for (int d = 0; d < DMAX; ++d) s += q[d] * __bfloat162float(Ks[j][d]);
        s *= scale;
        float m_new = fmaxf(m, s);
        float corr = __expf(m - m_new);
        float p = __expf(s - m_new);
        l = l * corr + p;
        #pragma unroll
        for (int d = 0; d < DMAX; ++d) acc[d] = acc[d] * corr + p * __bfloat162float(Vs[j][d]);
        m = m_new;
      }
    }
    __syncthreads();
  }

  if (active) {
    const long obase = b * osb + h * osh + (long)qpos * oss;
    float inv = (l > 0.f) ? (1.f / l) : 0.f;
    #pragma unroll
    for (int d = 0; d < DMAX; ++d) O[obase + d * osd] = __float2bfloat16(acc[d] * inv);
  }
}

extern "C" cudaError_t openkernels_launch_flash_attention_bf16_fwd_bhsd(
    const OpenKernelsFlashAttentionBF16FwdBHSDArgs* a, cudaStream_t stream) {
  if (a == nullptr) return cudaErrorInvalidValue;
  if (a->D != DMAX) return cudaErrorNotSupported;
  dim3 grid((a->S_q + BM - 1) / BM, a->H, a->B);
  fa_v2<<<grid, BM, 0, stream>>>(
      a->q, a->k, a->v, a->o, a->B, a->H, a->S_q, a->S_kv, a->D, a->causal, a->scale,
      a->q_stride_b, a->q_stride_h, a->q_stride_s, a->q_stride_d,
      a->k_stride_b, a->k_stride_h, a->k_stride_s, a->k_stride_d,
      a->v_stride_b, a->v_stride_h, a->v_stride_s, a->v_stride_d,
      a->o_stride_b, a->o_stride_h, a->o_stride_s, a->o_stride_d);
  return cudaGetLastError();
}
