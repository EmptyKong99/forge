#include "ops/gemm_bf16_nt/interface.h"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdint>

// v8: v7 + software-pipelined fragment loads (load both k-substeps, then mma).
// v7: raw PTX. ldmatrix + mma.sync.m16n8k16 with register-resident accumulators,
// replacing the wmma wrappers (which plateaued at 0.88x). cp.async double-buffer
// + 128x128 tile kept from v4. mma/ldmatrix/cp.async are sm_80+ -> valid on sm_120.
//
// Warp owns 64 rows x 32 cols of C = (4 M-tiles of 16) x (4 N-tiles of 8) mma ops.
// BK=32 -> 2 k-substeps of 16 per shared tile.
// Requires M,N % 128 == 0, K % 32 == 0, contiguous-k layout; else NotSupported.

#define BM 128
#define BN 128
#define BK 32
#define BKP 40
#define THREADS 256

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                       uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t& r0, uint32_t& r1, uint32_t a) {
  // non-trans: Bs[n][k] is already col-major K x N, exactly what mma .col wants
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void mma16816(float& d0, float& d1, float& d2, float& d3,
                                         uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1,
                                         float c0, float c1, float c2, float c3) {
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}

__global__ void __launch_bounds__(THREADS)
gemm_v8(const __nv_bfloat16* __restrict__ A,
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

  __shared__ __nv_bfloat16 As[2][BM][BKP];
  __shared__ __nv_bfloat16 Bs[2][BN][BKP];

  float acc[4][4][4];   // [M-tile][N-tile][c0..c3]
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      #pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  auto load_tile = [&](int b, int kbase) {
    #pragma unroll
    for (int t = 0; t < (BM * BK / 8) / THREADS; ++t) {
      int v = tid + t * THREADS; int row = v >> 2, kk8 = (v & 3) << 3;
      __pipeline_memcpy_async(&As[b][row][kk8],
                              &A[(long)(block_row + row) * asm_ + (kbase + kk8)], sizeof(int4));
    }
    #pragma unroll
    for (int t = 0; t < (BN * BK / 8) / THREADS; ++t) {
      int v = tid + t * THREADS; int row = v >> 2, kk8 = (v & 3) << 3;
      __pipeline_memcpy_async(&Bs[b][row][kk8],
                              &B[(long)(block_col + row) * bsn + (kbase + kk8)], sizeof(int4));
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

    // load fragments for BOTH k-substeps first, then issue all mma — gives the
    // scheduler room to overlap ldmatrix latency with tensor-core math.
    uint32_t a[2][4][4], b[2][4][2];
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
      int ks = s * 16;
      #pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        int rowbase = warp_row * 64 + mt * 16 + (lane & 15);
        int kcol = ks + ((lane >> 4) & 1) * 8;
        ldm_x4(a[s][mt][0], a[s][mt][1], a[s][mt][2], a[s][mt][3],
               smem_addr(&As[cur][rowbase][kcol]));
      }
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        int nrow = warp_col * 32 + nt * 8 + (lane & 7);
        int kcol = ks + ((lane >> 3) & 1) * 8;
        ldm_x2(b[s][nt][0], b[s][nt][1], smem_addr(&Bs[cur][nrow][kcol]));
      }
    }
    #pragma unroll
    for (int s = 0; s < 2; ++s)
      #pragma unroll
      for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt)
          mma16816(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3],
                   a[s][mt][0], a[s][mt][1], a[s][mt][2], a[s][mt][3], b[s][nt][0], b[s][nt][1],
                   acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3]);
    __syncthreads();
  }

  // epilogue: mma m16n8 C layout -> group=lane/4, tidg=lane%4
  const int group = lane >> 2, tidg = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      int row = block_row + warp_row * 64 + mt * 16 + group;
      int col = block_col + warp_col * 32 + nt * 8 + tidg * 2;
      float* a4 = acc[mt][nt];
      // (row, col), (row, col+1), (row+8, col), (row+8, col+1)
      int rr[4] = {row, row, row + 8, row + 8};
      int cc[4] = {col, col + 1, col, col + 1};
      #pragma unroll
      for (int e = 0; e < 4; ++e) {
        long cidx = (long)rr[e] * csm + (long)cc[e] * csn;
        float prev = (beta != 0.f) ? __bfloat162float(C[cidx]) : 0.f;
        C[cidx] = __float2bfloat16(alpha * a4[e] + beta * prev);
      }
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
  gemm_v8<<<grid, block, 0, stream>>>(
      args->a, args->b, args->c, M, N, K,
      args->a_stride_m, args->b_stride_n, args->c_stride_m, args->c_stride_n,
      args->alpha, args->beta);
  return cudaGetLastError();
}
