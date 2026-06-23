#include "ops/flash_attention_bf16_fwd_bhsd/interface.h"
#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>

// v1 naive — correctness floor for flash_attention_bf16_fwd_bhsd.
// One THREAD per query position (b,h,s): loop over all kv with online softmax,
// accumulate O. No tensor cores, no tiling — obviously correct, slow. This is the
// baseline the anvil base-arm should match, and the floor the tensor-core versions
// climb from. D fixed at 128 (required_5); else NotSupported.
//   O[b,h,s,:] = softmax_k( scale * Q[b,h,s,:]·K[b,h,k,:] ) · V[b,h,k,:]
//   causal: query s attends to k in [0, s].

#define DMAX 128

__global__ void fa_v1(const __nv_bfloat16* __restrict__ Q,
                      const __nv_bfloat16* __restrict__ K,
                      const __nv_bfloat16* __restrict__ V,
                      __nv_bfloat16* __restrict__ O,
                      int B, int H, int S_q, int S_kv, int D, int causal, float scale,
                      long qsb, long qsh, long qss, long qsd,
                      long ksb, long ksh, long kss, long ksd,
                      long vsb, long vsh, long vss, long vsd,
                      long osb, long osh, long oss, long osd) {
  long qid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = (long)B * H * S_q;
  if (qid >= total) return;

  int s = qid % S_q;
  long t = qid / S_q;
  int h = t % H;
  int b = t / H;

  const long qbase = b * qsb + h * qsh + s * qss;
  const long kbase0 = b * ksb + h * ksh;
  const long vbase0 = b * vsb + h * vsh;

  float q[DMAX], acc[DMAX];
  #pragma unroll
  for (int d = 0; d < DMAX; ++d) { q[d] = __bfloat162float(Q[qbase + d * qsd]); acc[d] = 0.f; }

  float m = -FLT_MAX, l = 0.f;
  int kv_end = causal ? (s + 1) : S_kv;
  for (int k = 0; k < kv_end; ++k) {
    const long kb = kbase0 + (long)k * kss;
    const long vb = vbase0 + (long)k * vss;
    float score = 0.f;
    #pragma unroll
    for (int d = 0; d < DMAX; ++d) score += q[d] * __bfloat162float(K[kb + d * ksd]);
    score *= scale;

    float m_new = fmaxf(m, score);
    float corr = expf(m - m_new);
    float p = expf(score - m_new);
    l = l * corr + p;
    #pragma unroll
    for (int d = 0; d < DMAX; ++d) acc[d] = acc[d] * corr + p * __bfloat162float(V[vb + d * vsd]);
    m = m_new;
  }

  const long obase = b * osb + h * osh + s * oss;
  float inv = (l > 0.f) ? (1.f / l) : 0.f;
  #pragma unroll
  for (int d = 0; d < DMAX; ++d) O[obase + d * osd] = __float2bfloat16(acc[d] * inv);
}

extern "C" cudaError_t openkernels_launch_flash_attention_bf16_fwd_bhsd(
    const OpenKernelsFlashAttentionBF16FwdBHSDArgs* a, cudaStream_t stream) {
  if (a == nullptr) return cudaErrorInvalidValue;
  if (a->D != DMAX) return cudaErrorNotSupported;
  long total = (long)a->B * a->H * a->S_q;
  int threads = 128;
  long blocks = (total + threads - 1) / threads;
  fa_v1<<<(unsigned)blocks, threads, 0, stream>>>(
      a->q, a->k, a->v, a->o, a->B, a->H, a->S_q, a->S_kv, a->D, a->causal, a->scale,
      a->q_stride_b, a->q_stride_h, a->q_stride_s, a->q_stride_d,
      a->k_stride_b, a->k_stride_h, a->k_stride_s, a->k_stride_d,
      a->v_stride_b, a->v_stride_h, a->v_stride_s, a->v_stride_d,
      a->o_stride_b, a->o_stride_h, a->o_stride_s, a->o_stride_d);
  return cudaGetLastError();
}
