#include "ops/gemm_bf16_nt/interface.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Row-major BF16 GEMM: C = alpha * A[M,K] * B[N,K]^T + beta * C
//
// Tile size: BM = BN = 128, BK = 8
// Thread block: 16 x 16 = 256 threads, each computes an 8x8 tile of C.
// Shared memory: A_tile[128][8] + B_tile[128][8] (loaded B tile as NxK,
//   accessed as B^T during accumulation).
// ---------------------------------------------------------------------------

constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 8;

__global__ void gemm_bf16_nt_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    int64_t a_stride_m, int64_t a_stride_k,
    int64_t b_stride_n, int64_t b_stride_k,
    int64_t c_stride_m, int64_t c_stride_n,
    float alpha, float beta) {

  // Shared memory split into A tile and B tile (transposed view of B)
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* smem_A = smem;
  __nv_bfloat16* smem_B = smem + BM * BK;

  // Identity
  const int bx = blockIdx.x;   // tile along N dimension
  const int by = blockIdx.y;   // tile along M dimension
  const int tx = threadIdx.x;  // local column (0..15)
  const int ty = threadIdx.y;  // local row
  const int tid = ty * blockDim.x + tx;  // linear thread id within block

  // Each thread computes this 8x8 sub-tile
  const int sub_row = ty * 8;  // local row within C tile (0..120)
  const int sub_col = tx * 8;  // local column

  // Accumulator array (fp32)
  float acc[8][8] = {};

  // Loop over the K dimension in tiles of BK
  for (int kb = 0; kb < K; kb += BK) {
    // ---------- cooperative load of A tile (BM x BK) ----------
    // Zero the assigned rows in smem_A
    if (tid < BM) {
      #pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        smem_A[tid * BK + kk] = 0;
      }
    }
    // Load valid rows
    if (tid < BM) {
      int global_row = by * BM + tid;               // row in A
      int global_k_start = kb;                      // column start in A
      if (global_row < M) {
        int remaining = K - global_k_start;
        if (remaining >= BK) {
          // Full vectorised load (8 elements = 16 bytes)
          const int4* src = reinterpret_cast<const int4*>(
              A + global_row * a_stride_m + global_k_start * a_stride_k);
          int4 vec = *src;
          __nv_bfloat16 vals[BK];
          __builtin_memcpy(vals, &vec, sizeof(vec));
          #pragma unroll
          for (int kk = 0; kk < BK; ++kk) {
            smem_A[tid * BK + kk] = vals[kk];
          }
        } else {
          // Partial load – load only the valid elements, others stay 0
          #pragma unroll
          for (int kk = 0; kk < BK; ++kk) {
            if (kk < remaining) {
              smem_A[tid * BK + kk] =
                  A[global_row * a_stride_m + (global_k_start + kk) * a_stride_k];
            }
          }
        }
      }
    }

    // ---------- cooperative load of B tile (BN x BK) ----------
    // Zero the assigned rows in smem_B
    if (tid < BN) {
      #pragma unroll
      for (int kk = 0; kk < BK; ++kk) {
        smem_B[tid * BK + kk] = 0;
      }
    }
    // Load valid rows (N dimension)
    if (tid < BN) {
      int global_n = bx * BN + tid;                 // row in B (N index)
      int global_k_start = kb;
      if (global_n < N) {
        int remaining = K - global_k_start;
        if (remaining >= BK) {
          const int4* src = reinterpret_cast<const int4*>(
              B + global_n * b_stride_n + global_k_start * b_stride_k);
          int4 vec = *src;
          __nv_bfloat16 vals[BK];
          __builtin_memcpy(vals, &vec, sizeof(vec));
          #pragma unroll
          for (int kk = 0; kk < BK; ++kk) {
            smem_B[tid * BK + kk] = vals[kk];
          }
        } else {
          #pragma unroll
          for (int kk = 0; kk < BK; ++kk) {
            if (kk < remaining) {
              smem_B[tid * BK + kk] =
                  B[global_n * b_stride_n + (global_k_start + kk) * b_stride_k];
            }
          }
        }
      }
    }

    __syncthreads();   // all shared memory is ready

    // ---------- compute the 8x8 contribution for this thread ----------
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      int a_row = sub_row + i;                     // local row in smem_A
      #pragma unroll
      for (int j = 0; j < 8; ++j) {
        float sum = 0.0f;
        int b_row = sub_col + j;                   // local row in smem_B (corresponds to column n)
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
          sum += __bfloat162float(smem_A[a_row * BK + kk]) *
                 __bfloat162float(smem_B[b_row * BK + kk]);
        }
        acc[i][j] += sum;
      }
    }

    __syncthreads();   // ensure all threads finish reading before next tile overwrites smem
  } // end for kb

  // ---------- write back results ----------
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    int row = by * BM + sub_row + i;
    if (row >= M) continue;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      int col = bx * BN + sub_col + j;
      if (col >= N) continue;

      float c_old = 0.0f;
      if (beta != 0.0f) {
        c_old = __bfloat162float(C[row * c_stride_m + col * c_stride_n]);
      }
      float result = alpha * acc[i][j] + beta * c_old;
      C[row * c_stride_m + col * c_stride_n] = __float2bfloat16_rn(result);
    }
  }
}

// ---------------------------------------------------------------------------
// Host launch function
// ---------------------------------------------------------------------------
extern "C" cudaError_t openkernels_launch_gemm_bf16_nt(
    const OpenKernelsGemmBF16NTArgs* args,
    cudaStream_t stream) {

  dim3 block(16, 16);
  dim3 grid((args->n + BN - 1) / BN, (args->m + BM - 1) / BM);
  size_t smem = (BM * BK + BN * BK) * sizeof(__nv_bfloat16);

  gemm_bf16_nt_kernel<<<grid, block, smem, stream>>>(
      args->a, args->b, args->c,
      args->m, args->n, args->k,
      args->a_stride_m, args->a_stride_k,
      args->b_stride_n, args->b_stride_k,
      args->c_stride_m, args->c_stride_n,
      args->alpha, args->beta);

  return cudaGetLastError();
}