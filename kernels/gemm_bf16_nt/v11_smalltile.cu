#include "ops/gemm_bf16_nt/interface.h"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdint>

// v11 (PTX route): smaller N-tile + 3-stage pipeline (independent reviewer's #1).
// v9's 3-stage regressed because the 128x128 tile made each stage 40KB -> 3 stages
// = 60KB -> 1 block/SM (occupancy cliff). Fix: shrink the block to 128x64 so each
// warp owns 64x16 (acc[4][2] = 32 regs, half of v8's 64) and each stage is ~15KB ->
// 3 stages fit in 45KB -> still 2 blocks/SM. Deeper prefetch at v8's occupancy =
// hides the DRAM latency that hurts big shapes (8192^2). Tests the "smaller tile +
// higher occupancy beats bigger tile" hypothesis v3/v5 anchored us against.
// Requires M % 128 == 0, N % 64 == 0, K % 32 == 0, contiguous-k; else NotSupported.

#define BM 128
#define BN 64
#define BK 32
#define BKP 40
#define THREADS 256
#define STAGES 3

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                       uint32_t& r3, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ void ldm_x2(uint32_t& r0, uint32_t& r1, uint32_t a) {
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
gemm_v11(const __nv_bfloat16* __restrict__ A,
         const __nv_bfloat16* __restrict__ B,
         __nv_bfloat16* __restrict__ C,
         int M, int N, int K,
         long asm_, long bsn, long csm, long csn,
         float alpha, float beta) {
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* As = smem;                        // [STAGES][BM][BKP]
  __nv_bfloat16* Bs = smem + STAGES * BM * BKP;    // [STAGES][BN][BKP]
  auto Aat = [&](int s, int r, int k) -> __nv_bfloat16* { return &As[(s * BM + r) * BKP + k]; };
  auto Bat = [&](int s, int r, int k) -> __nv_bfloat16* { return &Bs[(s * BN + r) * BKP + k]; };

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int warp_row = warp >> 2;       // 0..1 -> 64-row band
  const int warp_col = warp & 3;        // 0..3 -> 16-col band
  const int lane = tid & 31;

  float acc[4][2][4];                   // [M-tile][N-tile][c0..c3]
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    #pragma unroll
    for (int j = 0; j < 2; ++j)
      #pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  auto load_tile = [&](int s, int kbase) {
    #pragma unroll
    for (int t = 0; t < (BM * BK / 8) / THREADS; ++t) {
      int v = tid + t * THREADS; int row = v >> 2, kk8 = (v & 3) << 3;
      __pipeline_memcpy_async(Aat(s, row, kk8),
                              &A[(long)(block_row + row) * asm_ + (kbase + kk8)], sizeof(int4));
    }
    // BN*BK/8 = 64*32/8 = 256 = THREADS -> one int4 each (rows 0..63)
    {
      int v = tid; int row = v >> 2, kk8 = (v & 3) << 3;
      __pipeline_memcpy_async(Bat(s, row, kk8),
                              &B[(long)(block_col + row) * bsn + (kbase + kk8)], sizeof(int4));
    }
  };

  const int nk = K / BK;
  #pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < nk) load_tile(s, s * BK);
    __pipeline_commit();
  }

  for (int t = 0; t < nk; ++t) {
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    int cur = t % STAGES;

    uint32_t a[2][4][4], b[2][2][2];
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
      int ks = s * 16;
      #pragma unroll
      for (int mt = 0; mt < 4; ++mt) {
        int rowbase = warp_row * 64 + mt * 16 + (lane & 15);
        int kcol = ks + ((lane >> 4) & 1) * 8;
        ldm_x4(a[s][mt][0], a[s][mt][1], a[s][mt][2], a[s][mt][3],
               smem_addr(Aat(cur, rowbase, kcol)));
      }
      #pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        int nrow = warp_col * 16 + nt * 8 + (lane & 7);
        int kcol = ks + ((lane >> 3) & 1) * 8;
        ldm_x2(b[s][nt][0], b[s][nt][1], smem_addr(Bat(cur, nrow, kcol)));
      }
    }
    #pragma unroll
    for (int s = 0; s < 2; ++s)
      #pragma unroll
      for (int mt = 0; mt < 4; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt)
          mma16816(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3],
                   a[s][mt][0], a[s][mt][1], a[s][mt][2], a[s][mt][3], b[s][nt][0], b[s][nt][1],
                   acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3]);
    __syncthreads();

    int nxt = t + STAGES - 1;
    if (nxt < nk) load_tile(nxt % STAGES, nxt * BK);
    __pipeline_commit();
  }

  const int group = lane >> 2, tidg = lane & 3;
  #pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      int row = block_row + warp_row * 64 + mt * 16 + group;
      int col = block_col + warp_col * 16 + nt * 8 + tidg * 2;
      float* a4 = acc[mt][nt];
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
  size_t smem = (size_t)STAGES * (BM + BN) * BKP * sizeof(__nv_bfloat16);  // ~45KB
  static bool set = false;
  if (!set) {
    cudaFuncSetAttribute(gemm_v11, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    set = true;
  }
  dim3 block(THREADS);
  dim3 grid(N / BN, M / BM);
  gemm_v11<<<grid, block, smem, stream>>>(
      args->a, args->b, args->c, M, N, K,
      args->a_stride_m, args->b_stride_n, args->c_stride_m, args->c_stride_n,
      args->alpha, args->beta);
  return cudaGetLastError();
}
