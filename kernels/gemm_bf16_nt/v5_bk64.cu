#include "ops/gemm_bf16_nt/interface.h"
#include <cuda_bf16.h>
#include <mma.h>
#include <cuda_pipeline.h>

using namespace nvcuda;

// v5: push the wmma ceiling. v4 + BK=64 (more mma per sync, fewer syncs, better
// cp.async batching). Shared (double-buffered, padded) is 72KB, so it uses
// dynamic shared memory with an opt-in (cudaFuncAttributeMaxDynamicSharedMemorySize).
// 128x128 tile, 8 warps (2x4), each warp 4x2 of 16x16 wmma frags.
// Requires M,N % 128 == 0, K % 64 == 0, contiguous-k layout; else NotSupported.

#define BM 128
#define BN 128
#define BK 64
#define BKP 72            // BK + 8 padding
#define THREADS 256
#define WARPS 8

// dynamic shared layout: As[2][BM][BKP] then Bs[2][BN][BKP]
#define AS(b,r,k) smem[(b)*(BM*BKP) + (r)*BKP + (k)]
#define BS(b,r,k) smem[2*(BM*BKP) + (b)*(BN*BKP) + (r)*BKP + (k)]

__global__ void __launch_bounds__(THREADS)
gemm_v5(const __nv_bfloat16* __restrict__ A,
        const __nv_bfloat16* __restrict__ B,
        __nv_bfloat16* __restrict__ C,
        int M, int N, int K,
        long asm_, long bsn, long csm, long csn,
        float alpha, float beta) {
  extern __shared__ __nv_bfloat16 smem[];
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int warp_row = warp >> 2;
  const int warp_col = warp & 3;
  const int lane = tid & 31;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][2];
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    #pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(c_frag[i][j], 0.f);

  auto load_tile = [&](int b, int kbase) {
    #pragma unroll
    for (int t = 0; t < (BM * BK / 8) / THREADS; ++t) {   // 1024/256 = 4 int4/thread
      int v = tid + t * THREADS;
      int row = v >> 3, kk8 = (v & 7) << 3;               // 8 int4 per 64-wide row
      __pipeline_memcpy_async(&AS(b, row, kk8),
                              &A[(long)(block_row + row) * asm_ + (kbase + kk8)],
                              sizeof(int4));
    }
    #pragma unroll
    for (int t = 0; t < (BN * BK / 8) / THREADS; ++t) {
      int v = tid + t * THREADS;
      int row = v >> 3, kk8 = (v & 7) << 3;
      __pipeline_memcpy_async(&BS(b, row, kk8),
                              &B[(long)(block_col + row) * bsn + (kbase + kk8)],
                              sizeof(int4));
    }
  };

  const int nk = K / BK;
  load_tile(0, 0);
  __pipeline_commit();

  for (int t = 0; t < nk; ++t) {
    int cur = t & 1;
    if (t + 1 < nk) { load_tile((t + 1) & 1, (t + 1) * BK); __pipeline_commit(); }
    __pipeline_wait_prior(t + 1 < nk ? 1 : 0);
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag[2];
      #pragma unroll
      for (int wm = 0; wm < 4; ++wm)
        wmma::load_matrix_sync(a_frag[wm], &AS(cur, warp_row * 64 + wm * 16, kk), BKP);
      #pragma unroll
      for (int wn = 0; wn < 2; ++wn)
        wmma::load_matrix_sync(b_frag[wn], &BS(cur, warp_col * 32 + wn * 16, kk), BKP);
      #pragma unroll
      for (int wm = 0; wm < 4; ++wm)
        #pragma unroll
        for (int wn = 0; wn < 2; ++wn)
          wmma::mma_sync(c_frag[wm][wn], a_frag[wm], b_frag[wn], c_frag[wm][wn]);
    }
    __syncthreads();
  }

  // epilogue: reuse the start of smem as the fp32 staging buffer
  float (*Cstage)[16][16] = reinterpret_cast<float (*)[16][16]>(&smem[0]);
  #pragma unroll
  for (int wm = 0; wm < 4; ++wm) {
    #pragma unroll
    for (int wn = 0; wn < 2; ++wn) {
      wmma::store_matrix_sync(&Cstage[warp][0][0], c_frag[wm][wn], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        int idx = lane + e * 32;
        int rr = idx >> 4, cc = idx & 15;
        int row = block_row + warp_row * 64 + wm * 16 + rr;
        int col = block_col + warp_col * 32 + wn * 16 + cc;
        long cidx = (long)row * csm + (long)col * csn;
        float prev = (beta != 0.f) ? __bfloat162float(C[cidx]) : 0.f;
        C[cidx] = __float2bfloat16(alpha * Cstage[warp][rr][cc] + beta * prev);
      }
      __syncwarp();
    }
  }
}

extern "C" cudaError_t openkernels_launch_gemm_bf16_nt(
    const OpenKernelsGemmBF16NTArgs* args, cudaStream_t stream) {
  if (args == nullptr) return cudaErrorInvalidValue;
  const int M = args->m, N = args->n, K = args->k;
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) return cudaErrorNotSupported;
  if (args->a_stride_k != 1 || args->b_stride_k != 1) return cudaErrorNotSupported;

  const int smem_bytes = (2 * BM * BKP + 2 * BN * BKP) * (int)sizeof(__nv_bfloat16);
  static bool configured = false;
  if (!configured) {
    cudaError_t e = cudaFuncSetAttribute(
        gemm_v5, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (e != cudaSuccess) return e;
    configured = true;
  }
  dim3 block(THREADS);
  dim3 grid(N / BN, M / BM);
  gemm_v5<<<grid, block, smem_bytes, stream>>>(
      args->a, args->b, args->c, M, N, K,
      args->a_stride_m, args->b_stride_n, args->c_stride_m, args->c_stride_n,
      args->alpha, args->beta);
  return cudaGetLastError();
}
