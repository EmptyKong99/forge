#include "ops/gemm_bf16_nt/interface.h"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdint>

// v15 (PTX route): v13 (XOR-swizzled, no pad) + 3-stage cp.async pipeline.
// v9's 3-stage died because v8's padded 40KB tiles made 3 stages = 60KB -> dynamic
// shared -> 1 block/SM (occupancy cliff). v13's swizzle removed the pad: each 2-stage
// tile is 32KB. So 3 stages = 48KB (As/Bs, dynamic) + 2KB Cs (static) = 50KB/block.
// On sm_120 (~100KB shared/SM) that's STILL 2 blocks/SM -> the deeper prefetch is
// finally affordable. Targets the big shapes (8192^2/16384 still ~0.95-0.97x =
// DRAM-latency-bound). swizzle + stmatrix epilogue are byte-for-byte v13.
// Requires M,N % 128 == 0, K % 32 == 0, contiguous-k; else NotSupported.

#define BM 128
#define BN 128
#define BK 32
#define BKP 32
#define THREADS 256
#define STAGES 3

__device__ __forceinline__ int swz(int row, int k) {
  return ((k >> 3) ^ (row & 3)) * 8 + (k & 7);
}
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
__device__ __forceinline__ void stm_x2(uint32_t a, uint32_t r0, uint32_t r1) {
  asm volatile("stmatrix.sync.aligned.m8n8.x2.shared.b16 [%0], {%1,%2};\n"
               :: "r"(a), "r"(r0), "r"(r1));
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
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  __nv_bfloat162 v = __halves2bfloat162(__float2bfloat16(lo), __float2bfloat16(hi));
  return *reinterpret_cast<uint32_t*>(&v);
}

__global__ void __launch_bounds__(THREADS)
gemm_v15(const __nv_bfloat16* __restrict__ A,
         const __nv_bfloat16* __restrict__ B,
         __nv_bfloat16* __restrict__ C,
         int M, int N, int K,
         long asm_, long bsn, long csm, long csn,
         float alpha, float beta) {
  extern __shared__ __nv_bfloat16 smem[];
  __nv_bfloat16* As = smem;                          // [STAGES][BM][BKP]
  __nv_bfloat16* Bs = smem + STAGES * BM * BKP;      // [STAGES][BN][BKP]
  auto Aat = [&](int s, int r, int k) -> __nv_bfloat16* { return &As[(s * BM + r) * BKP + swz(r, k)]; };
  auto Bat = [&](int s, int r, int k) -> __nv_bfloat16* { return &Bs[(s * BN + r) * BKP + swz(r, k)]; };
  __shared__ __nv_bfloat16 Cs[8][2][8][8];           // static epilogue scratch (2KB)

  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int warp_row = warp >> 2;
  const int warp_col = warp & 3;
  const int lane = tid & 31;

  float acc[4][4][4];
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      #pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.f;

  auto load_tile = [&](int s, int kbase) {
    #pragma unroll
    for (int t = 0; t < (BM * BK / 8) / THREADS; ++t) {
      int v = tid + t * THREADS; int row = v >> 2, kk8 = (v & 3) << 3;
      __pipeline_memcpy_async(Aat(s, row, kk8),
                              &A[(long)(block_row + row) * asm_ + (kbase + kk8)], sizeof(int4));
    }
    #pragma unroll
    for (int t = 0; t < (BN * BK / 8) / THREADS; ++t) {
      int v = tid + t * THREADS; int row = v >> 2, kk8 = (v & 3) << 3;
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

    uint32_t a[2][4][4], b[2][4][2];
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
      for (int nt = 0; nt < 4; ++nt) {
        int nrow = warp_col * 32 + nt * 8 + (lane & 7);
        int kcol = ks + ((lane >> 3) & 1) * 8;
        ldm_x2(b[s][nt][0], b[s][nt][1], smem_addr(Bat(cur, nrow, kcol)));
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

    int nxt = t + STAGES - 1;
    if (nxt < nk) load_tile(nxt % STAGES, nxt * BK);
    __pipeline_commit();
  }

  const int group = lane >> 2, tidg = lane & 3;
  if (beta == 0.f) {
    uint32_t st_addr = smem_addr(&Cs[warp][(lane >> 3) & 1][lane & 7][0]);
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        float* a4 = acc[mt][nt];
        uint32_t r_lo = pack2(alpha * a4[0], alpha * a4[1]);
        uint32_t r_hi = pack2(alpha * a4[2], alpha * a4[3]);
        stm_x2(st_addr, r_lo, r_hi);
        __syncwarp();
        int base_row = block_row + warp_row * 64 + mt * 16;
        int base_col = block_col + warp_col * 32 + nt * 8;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int e = lane + i * 32;
          int m = e >> 6, rc = e & 63;
          int r = rc >> 3, c = rc & 7;
          int grow = base_row + m * 8 + r, gcol = base_col + c;
          C[(long)grow * csm + (long)gcol * csn] = Cs[warp][m][r][c];
        }
        __syncwarp();
      }
    }
  } else {
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
      #pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        int row = block_row + warp_row * 64 + mt * 16 + group;
        int col = block_col + warp_col * 32 + nt * 8 + tidg * 2;
        float* a4 = acc[mt][nt];
        int rr[4] = {row, row, row + 8, row + 8};
        int cc[4] = {col, col + 1, col, col + 1};
        #pragma unroll
        for (int e = 0; e < 4; ++e) {
          long cidx = (long)rr[e] * csm + (long)cc[e] * csn;
          float prev = __bfloat162float(C[cidx]);
          C[cidx] = __float2bfloat16(alpha * a4[e] + beta * prev);
        }
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
  size_t smem = (size_t)STAGES * (BM + BN) * BKP * sizeof(__nv_bfloat16);  // 48KB
  static bool set = false;
  if (!set) {
    cudaFuncSetAttribute(gemm_v15, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    set = true;
  }
  dim3 block(THREADS);
  dim3 grid(N / BN, M / BM);
  gemm_v15<<<grid, block, smem, stream>>>(
      args->a, args->b, args->c, M, N, K,
      args->a_stride_m, args->b_stride_n, args->c_stride_m, args->c_stride_n,
      args->alpha, args->beta);
  return cudaGetLastError();
}
