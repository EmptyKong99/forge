#include "ops/gemm_bf16_nt/interface.h"

#define BLOCK_M 128
#define BLOCK_N 128
#define BK 32
#define TM 8
#define TN 8
#define THREADS_PER_BLOCK ((BLOCK_M / TM) * (BLOCK_N / TN)) // 256

__global__ void gemm_bf16_nt_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ c,
    const int M, const int N, const int K,
    const int64_t a_stride_m, const int64_t a_stride_k,
    const int64_t b_stride_n, const int64_t b_stride_k,
    const int64_t c_stride_m, const int64_t c_stride_n,
    const float alpha, const float beta)
{
    const int block_m = blockIdx.y;
    const int block_n = blockIdx.x;

    const int tid_x = threadIdx.x;  // 0..15
    const int tid_y = threadIdx.y;  // 0..15
    const int tid   = tid_y * 16 + tid_x; // 0..255

    const int row_start = block_m * BLOCK_M + tid_y * TM;
    const int col_start = block_n * BLOCK_N + tid_x * TN;

    // Accumulators (row-major TM x TN), initially zero
    float acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    __shared__ __nv_bfloat16 As[BLOCK_M * BK];
    __shared__ __nv_bfloat16 Bs[BLOCK_N * BK];  // stored N-major: Bs[n][k]

    // Iterate over K in tiles of BK
    for (int k_block = 0; k_block < K; k_block += BK) {
        // Cooperatively load A tile into As, B tile into Bs.
        // Each thread loads 16 elements (2 vector loads of 8 BF16 each).
        #pragma unroll
        for (int step = 0; step < 16; step += 8) {
            const int lid = tid * 16 + step;   // linear index inside the tile
            const int row = lid / BK;
            const int col = lid % BK;

            // ---- load A ----
            {
                const int global_row = block_m * BLOCK_M + row;
                const int global_col = k_block + col;
                uint4 data = make_uint4(0, 0, 0, 0);
                if (global_row < M && global_col + 7 < K) {
                    const __nv_bfloat16* src = a + global_row * a_stride_m + global_col * a_stride_k;
                    data = *reinterpret_cast<const uint4*>(src);
                }
                // Always store (sets zeros for out-of-bounds)
                *reinterpret_cast<uint4*>(&As[lid]) = data;
            }

            // ---- load B ----
            {
                const int col_n = lid / BK;   // N index inside B tile
                const int col_k = lid % BK;  // K index inside B tile
                const int global_n = block_n * BLOCK_N + col_n;
                const int global_k = k_block + col_k;
                uint4 data = make_uint4(0, 0, 0, 0);
                if (global_n < N && global_k + 7 < K) {
                    const __nv_bfloat16* src = b + global_n * b_stride_n + global_k * b_stride_k;
                    data = *reinterpret_cast<const uint4*>(src);
                }
                *reinterpret_cast<uint4*>(&Bs[lid]) = data;
            }
        }

        __syncthreads();

        // Compute partial products for this K tile
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Preload A register block (8 values)
            float a_reg[TM];
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                const int row = tid_y * TM + tm;
                a_reg[tm] = __bfloat162float(As[row * BK + k]);
            }

            // Preload B register block (8 values)
            float b_reg[TN];
            #pragma unroll
            for (int tn = 0; tn < TN; ++tn) {
                const int col = tid_x * TN + tn;
                b_reg[tn] = __bfloat162float(Bs[col * BK + k]);
            }

            // Outer product update
            #pragma unroll
            for (int tm = 0; tm < TM; ++tm) {
                const float av = a_reg[tm];
                #pragma unroll
                for (int tn = 0; tn < TN; ++tn) {
                    acc[tm][tn] += av * b_reg[tn];
                }
            }
        }

        __syncthreads();
    }

    // Write back to C, applying alpha and beta
    #pragma unroll
    for (int tm = 0; tm < TM; ++tm) {
        const int global_row = row_start + tm;
        if (global_row >= M) continue;
        #pragma unroll
        for (int tn = 0; tn < TN; ++tn) {
            const int global_col = col_start + tn;
            if (global_col >= N) continue;

            // Read original C only when beta != 0
            float c_val = 0.0f;
            if (beta != 0.0f) {
                c_val = __bfloat162float(c[global_row * c_stride_m + global_col * c_stride_n]);
            }

            const float result = alpha * acc[tm][tn] + beta * c_val;
            __nv_bfloat16 res = __float2bfloat16(result);
            c[global_row * c_stride_m + global_col * c_stride_n] = res;
        }
    }
}

extern "C" cudaError_t openkernels_launch_gemm_bf16_nt(
    const OpenKernelsGemmBF16NTArgs* args,
    cudaStream_t stream)
{
    if (!args) return cudaErrorInvalidValue;

    const int M = args->m, N = args->n, K = args->k;

    // Configure grid
    dim3 block(16, 16);  // 256 threads
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N,
              (M + BLOCK_M - 1) / BLOCK_M);

    gemm_bf16_nt_kernel<<<grid, block, 0, stream>>>(
        args->a, args->b, args->c,
        M, N, K,
        args->a_stride_m, args->a_stride_k,
        args->b_stride_n, args->b_stride_k,
        args->c_stride_m, args->c_stride_n,
        args->alpha, args->beta);

    return cudaGetLastError();
}