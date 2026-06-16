#include "ops/gemm_bf16_nt/interface.h"
#include <cuda_bf16.h>

// v1: shared-memory tiled + register-blocked BF16 GEMM NT, fp32 accumulate.
// C[M,N] = alpha * A[M,K] @ B[N,K]^T + beta * C
// Block tile BM x BN, K-step BK; each thread computes TM x TN outputs.
// Assumes the suite shapes (M,N multiples of 128, K multiple of 16); otherwise
// reports cudaErrorNotSupported (allowed by the ABI) to keep the hot path branchless.

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8
// threads/block = (BM/TM)*(BN/TN) = 16*16 = 256

__global__ void __launch_bounds__(256)
gemm_regblock(const __nv_bfloat16* __restrict__ A,
              const __nv_bfloat16* __restrict__ B,
              __nv_bfloat16* __restrict__ C,
              int M, int N, int K,
              long asm_, long ask, long bsn, long bsk, long csm, long csn,
              float alpha, float beta) {
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int tid = threadIdx.x;
  const int tx = tid % (BN / TN);   // 0..15, column group
  const int ty = tid / (BN / TN);   // 0..15, row group

  __shared__ float As[BK][BM];      // As[k][m]  (transposed for the inner product)
  __shared__ float Bs[BK][BN];      // Bs[k][n]

  float acc[TM][TN];
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // each thread loads (BM*BK)/256 = 8 elems of A and of B
    #pragma unroll
    for (int t = 0; t < (BM * BK) / 256; ++t) {
      int idx = tid + t * 256;          // 0..2047
      int r  = idx / BK;                // row in tile 0..127
      int kk = idx % BK;                // 0..15
      As[kk][r] = __bfloat162float(A[(long)(block_row + r) * asm_ + (long)(k0 + kk) * ask]);
    }
    #pragma unroll
    for (int t = 0; t < (BN * BK) / 256; ++t) {
      int idx = tid + t * 256;
      int n  = idx / BK;
      int kk = idx % BK;
      Bs[kk][n] = __bfloat162float(B[(long)(block_col + n) * bsn + (long)(k0 + kk) * bsk]);
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; ++kk) {
      float a_reg[TM], b_reg[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) a_reg[i] = As[kk][ty * TM + i];
      #pragma unroll
      for (int j = 0; j < TN; ++j) b_reg[j] = Bs[kk][tx * TN + j];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j)
          acc[i][j] += a_reg[i] * b_reg[j];
    }
    __syncthreads();
  }

  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    int row = block_row + ty * TM + i;
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      int col = block_col + tx * TN + j;
      long cidx = (long)row * csm + (long)col * csn;
      float prev = (beta != 0.f) ? __bfloat162float(C[cidx]) : 0.f;
      C[cidx] = __float2bfloat16(alpha * acc[i][j] + beta * prev);
    }
  }
}

extern "C" cudaError_t openkernels_launch_gemm_bf16_nt(
    const OpenKernelsGemmBF16NTArgs* args, cudaStream_t stream) {
  if (args == nullptr) return cudaErrorInvalidValue;
  const int M = args->m, N = args->n, K = args->k;
  if (M % BM != 0 || N % BN != 0 || K % BK != 0) return cudaErrorNotSupported;
  dim3 block(256);
  dim3 grid(N / BN, M / BM);
  gemm_regblock<<<grid, block, 0, stream>>>(
      args->a, args->b, args->c, M, N, K,
      args->a_stride_m, args->a_stride_k, args->b_stride_n, args->b_stride_k,
      args->c_stride_m, args->c_stride_n, args->alpha, args->beta);
  return cudaGetLastError();
}
