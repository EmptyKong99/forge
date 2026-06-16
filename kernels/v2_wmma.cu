#include "ops/gemm_bf16_nt/interface.h"
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// v2: shared-memory tiled BF16 GEMM NT using tensor cores (wmma 16x16x16).
// C[M,N] = alpha * A[M,K] @ B[N,K]^T + beta * C, fp32 accumulate.
// Block tile 64x64, K-step 32. 4 warps (128 threads), 2x2 warp grid; each warp
// owns a 32x32 output = 2x2 of 16x16 fragments.
//
// B is stored [N,K] row-major == the K x N matrix in column-major, which is
// exactly what wmma's matrix_b(col_major) wants, so NT maps cleanly.
// Assumes M,N % 64 == 0 and K % 32 == 0 (the suite shapes); else NotSupported.

#define BM 64
#define BN 64
#define BK 32
#define WARPS 4          // 2x2
#define THREADS 128

__global__ void __launch_bounds__(THREADS)
gemm_wmma(const __nv_bfloat16* __restrict__ A,
          const __nv_bfloat16* __restrict__ B,
          __nv_bfloat16* __restrict__ C,
          int M, int N, int K,
          long asm_, long ask, long bsn, long bsk, long csm, long csn,
          float alpha, float beta) {
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;            // 0..3
  const int warp_row = warp >> 1;       // 0..1
  const int warp_col = warp & 1;        // 0..1

  __shared__ __nv_bfloat16 As[BM][BK];  // As[m][k]
  __shared__ __nv_bfloat16 Bs[BN][BK];  // Bs[n][k]
  __shared__ float Cs[BM][BN];          // staging for fp32 -> bf16 store

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[2][2];
  #pragma unroll
  for (int i = 0; i < 2; ++i)
    #pragma unroll
    for (int j = 0; j < 2; ++j)
      wmma::fill_fragment(c_frag[i][j], 0.0f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    // load A tile (64x32 = 2048) and B tile (2048): 16 elems/thread each
    #pragma unroll
    for (int t = 0; t < (BM * BK) / THREADS; ++t) {
      int idx = tid + t * THREADS;
      int r = idx / BK, k = idx % BK;
      As[r][k] = A[(long)(block_row + r) * asm_ + (long)(k0 + k) * ask];
    }
    #pragma unroll
    for (int t = 0; t < (BN * BK) / THREADS; ++t) {
      int idx = tid + t * THREADS;
      int r = idx / BK, k = idx % BK;
      Bs[r][k] = B[(long)(block_col + r) * bsn + (long)(k0 + k) * bsk];
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag[2];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag[2];
      #pragma unroll
      for (int wm = 0; wm < 2; ++wm)
        wmma::load_matrix_sync(a_frag[wm], &As[warp_row * 32 + wm * 16][kk], BK);
      #pragma unroll
      for (int wn = 0; wn < 2; ++wn)
        wmma::load_matrix_sync(b_frag[wn], &Bs[warp_col * 32 + wn * 16][kk], BK);
      #pragma unroll
      for (int wm = 0; wm < 2; ++wm)
        #pragma unroll
        for (int wn = 0; wn < 2; ++wn)
          wmma::mma_sync(c_frag[wm][wn], a_frag[wm], b_frag[wn], c_frag[wm][wn]);
    }
    __syncthreads();
  }

  // stage accumulators to shared, then convert to bf16 with alpha/beta
  #pragma unroll
  for (int wm = 0; wm < 2; ++wm)
    #pragma unroll
    for (int wn = 0; wn < 2; ++wn)
      wmma::store_matrix_sync(&Cs[warp_row * 32 + wm * 16][warp_col * 32 + wn * 16],
                              c_frag[wm][wn], BN, wmma::mem_row_major);
  __syncthreads();

  #pragma unroll
  for (int idx = tid; idx < BM * BN; idx += THREADS) {
    int r = idx / BN, c = idx % BN;
    int row = block_row + r, col = block_col + c;
    long cidx = (long)row * csm + (long)col * csn;
    float prev = (beta != 0.f) ? __bfloat162float(C[cidx]) : 0.f;
    C[cidx] = __float2bfloat16(alpha * Cs[r][c] + beta * prev);
  }
}

extern "C" cudaError_t openkernels_launch_gemm_bf16_nt(
    const OpenKernelsGemmBF16NTArgs* args, cudaStream_t stream) {
  if (args == nullptr) return cudaErrorInvalidValue;
  const int M = args->m, N = args->n, K = args->k;
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) return cudaErrorNotSupported;
  dim3 block(THREADS);
  dim3 grid(N / BN, M / BM);
  gemm_wmma<<<grid, block, 0, stream>>>(
      args->a, args->b, args->c, M, N, K,
      args->a_stride_m, args->a_stride_k, args->b_stride_n, args->b_stride_k,
      args->c_stride_m, args->c_stride_n, args->alpha, args->beta);
  return cudaGetLastError();
}
