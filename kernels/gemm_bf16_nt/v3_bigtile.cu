#include "ops/gemm_bf16_nt/interface.h"
#include <cuda_bf16.h>
#include <mma.h>

using namespace nvcuda;

// v3: 128x128 block tile, wmma tensor cores, vectorized (128-bit) global loads,
// shared-memory K padding to cut bank conflicts. Single-buffered (no cp.async yet).
// 8 warps (256 threads) in a 2x4 grid; each warp owns 64x32 = 4x2 of 16x16 frags.
// Requires M,N % 128 == 0, K % 32 == 0, and contiguous-k layout (a_stride_k ==
// b_stride_k == 1); else cudaErrorNotSupported.

#define BM 128
#define BN 128
#define BK 32
#define BKP 40            // padded shared leading dim (BK + 8)
#define THREADS 256
#define WARPS 8

__global__ void __launch_bounds__(THREADS)
gemm_v3(const __nv_bfloat16* __restrict__ A,
        const __nv_bfloat16* __restrict__ B,
        __nv_bfloat16* __restrict__ C,
        int M, int N, int K,
        long asm_, long bsn, long csm, long csn,
        float alpha, float beta) {
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int warp_row = warp >> 2;       // 0..1  -> 64-row band
  const int warp_col = warp & 3;        // 0..3  -> 32-col band
  const int lane = tid & 31;

  __shared__ __nv_bfloat16 As[BM][BKP];
  __shared__ __nv_bfloat16 Bs[BN][BKP];

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[4][2];
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    #pragma unroll
    for (int j = 0; j < 2; ++j) wmma::fill_fragment(c_frag[i][j], 0.f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    // vectorized load: BM*BK/8 = 512 int4 vectors, 256 threads -> 2 each
    #pragma unroll
    for (int t = 0; t < (BM * BK / 8) / THREADS; ++t) {
      int v = tid + t * THREADS;        // 0..511
      int row = v >> 2;                 // 0..127  (4 vectors per 32-wide row)
      int kk8 = (v & 3) << 3;           // {0,8,16,24}
      const __nv_bfloat16* gp = &A[(long)(block_row + row) * asm_ + (k0 + kk8)];
      *reinterpret_cast<int4*>(&As[row][kk8]) = *reinterpret_cast<const int4*>(gp);
    }
    #pragma unroll
    for (int t = 0; t < (BN * BK / 8) / THREADS; ++t) {
      int v = tid + t * THREADS;
      int row = v >> 2;
      int kk8 = (v & 3) << 3;
      const __nv_bfloat16* gp = &B[(long)(block_col + row) * bsn + (k0 + kk8)];
      *reinterpret_cast<int4*>(&Bs[row][kk8]) = *reinterpret_cast<const int4*>(gp);
    }
    __syncthreads();

    #pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag[4];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag[2];
      #pragma unroll
      for (int wm = 0; wm < 4; ++wm)
        wmma::load_matrix_sync(a_frag[wm], &As[warp_row * 64 + wm * 16][kk], BKP);
      #pragma unroll
      for (int wn = 0; wn < 2; ++wn)
        wmma::load_matrix_sync(b_frag[wn], &Bs[warp_col * 32 + wn * 16][kk], BKP);
      #pragma unroll
      for (int wm = 0; wm < 4; ++wm)
        #pragma unroll
        for (int wn = 0; wn < 2; ++wn)
          wmma::mma_sync(c_frag[wm][wn], a_frag[wm], b_frag[wn], c_frag[wm][wn]);
    }
    __syncthreads();
  }

  // store: stage each 16x16 fragment through a small per-warp shared buffer,
  // then convert to bf16 with alpha/beta and write to C.
  __shared__ float Cstage[WARPS][16][16];
  #pragma unroll
  for (int wm = 0; wm < 4; ++wm) {
    #pragma unroll
    for (int wn = 0; wn < 2; ++wn) {
      wmma::store_matrix_sync(&Cstage[warp][0][0], c_frag[wm][wn], 16, wmma::mem_row_major);
      __syncwarp();
      #pragma unroll
      for (int e = 0; e < 8; ++e) {
        int idx = lane + e * 32;        // 0..255
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
  dim3 block(THREADS);
  dim3 grid(N / BN, M / BM);
  gemm_v3<<<grid, block, 0, stream>>>(
      args->a, args->b, args->c, M, N, K,
      args->a_stride_m, args->b_stride_n, args->c_stride_m, args->c_stride_n,
      args->alpha, args->beta);
  return cudaGetLastError();
}
