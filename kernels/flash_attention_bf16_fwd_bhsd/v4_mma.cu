#include "ops/flash_attention_bf16_fwd_bhsd/interface.h"
#include <cuda_bf16.h>
#include <cfloat>
#include <cmath>
#include <cstdint>

// v4 mma — tensor-core flash attention. One warp owns 16 queries; loop key tiles of
// BN=16. QK^T and PV both run on mma.m16n8k16 (the gemm NT facts transfer):
//   S = Q·K^T : NT gemm, M=16 q, N=16 keys, K=D=128. Q=A, K=B (both non-trans, like
//               gemm — K[keys][d] row-major IS col-major [d,keys]).
//   O = P·V   : NN, but store V TRANSPOSED in shared (Vs_t[d][key]) so it's B
//               non-trans too; P (after softmax) is A.
// Hard middle: row-softmax lives in the C-accumulator fragment (row=query group),
// reduce over keys across the 4-lane group via shfl; then write P(bf16) to shared and
// ldmatrix it back as the PV A-operand. D=128, S_q/S_kv % 16 == 0 (required_5).
// One warp / block here (correctness first; occupancy is a later rung).

#define DM 128
#define DP 136          // D + 8 pad
#define BN 16
#define BNP 24          // BN + 8 pad

__device__ __forceinline__ uint32_t sm(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldm4(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
               : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(s));
}
__device__ __forceinline__ void ldm2(uint32_t& a, uint32_t& b, uint32_t s) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];\n"
               : "=r"(a), "=r"(b) : "r"(s));
}
__device__ __forceinline__ void mma(float& d0, float& d1, float& d2, float& d3,
                                     uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                     uint32_t b0, uint32_t b1,
                                     float c0, float c1, float c2, float c3) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                 "f"(c0), "f"(c1), "f"(c2), "f"(c3));
}
// reduce a value across the 4-lane group {4g..4g+3} (same query row): all 4 get the result
__device__ __forceinline__ float grp_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 1));
  v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, 2));
  return v;
}
__device__ __forceinline__ float grp_sum(float v) {
  v += __shfl_xor_sync(0xffffffff, v, 1);
  v += __shfl_xor_sync(0xffffffff, v, 2);
  return v;
}

__global__ void __launch_bounds__(32)
fa_v4(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
      const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O,
      int B, int H, int S_q, int S_kv, int D, int causal, float scale,
      long qsb, long qsh, long qss, long qsd, long ksb, long ksh, long kss, long ksd,
      long vsb, long vsh, long vss, long vsd, long osb, long osh, long oss, long osd) {
  const int b = blockIdx.z, h = blockIdx.y;
  const int qbase = blockIdx.x * 16;         // first query of this warp's tile
  const int lane = threadIdx.x & 31;
  const int group = lane >> 2, tidg = lane & 3;   // owns query rows R0=group, R1=group+8

  __shared__ __nv_bfloat16 Qs[16][DP];
  __shared__ __nv_bfloat16 Ks[16][DP];
  __shared__ __nv_bfloat16 Vt[DM][BNP];      // V transposed: Vt[d][key]
  __shared__ __nv_bfloat16 Ps[16][BNP];

  const long qb = b * qsb + h * qsh, kb = b * ksb + h * ksh, vb = b * vsb + h * vsh;
  // load Q tile (16 x 128) once
  for (int i = lane; i < 16 * DM; i += 32) {
    int r = i / DM, c = i % DM;
    Qs[r][c] = Q[qb + (long)(qbase + r) * qss + c * qsd];
  }

  float o_acc[16][4];
  #pragma unroll
  for (int n = 0; n < 16; ++n) { o_acc[n][0] = o_acc[n][1] = o_acc[n][2] = o_acc[n][3] = 0.f; }
  float m0 = -FLT_MAX, m1 = -FLT_MAX, l0 = 0.f, l1 = 0.f;

  const int kmax = causal ? min(qbase + 15, S_kv - 1) : (S_kv - 1);
  __syncwarp();

  for (int kt = 0; kt <= kmax; kt += BN) {
    // load K tile (16 x 128) and V transposed (128 x 16)
    for (int i = lane; i < 16 * DM; i += 32) {
      int r = i / DM, c = i % DM;
      Ks[r][c] = K[kb + (long)(kt + r) * kss + c * ksd];
    }
    for (int i = lane; i < DM * BN; i += 32) {
      int d = i / BN, key = i % BN;
      Vt[d][key] = V[vb + (long)(kt + key) * vss + d * vsd];
    }
    __syncwarp();

    // ---- QK^T : S[16 x 16] in two n8 fragments, accumulate over D (8 k-steps) ----
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0;   // n-subtile 0 (keys 0..7)
    float t0 = 0, t1 = 0, t2 = 0, t3 = 0;   // n-subtile 1 (keys 8..15)
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
      int dcol = kk * 16 + ((lane >> 4) & 1) * 8;
      uint32_t a0, a1, a2, a3;
      ldm4(a0, a1, a2, a3, sm(&Qs[lane & 15][dcol]));
      int kcol = kk * 16 + ((lane >> 3) & 1) * 8;
      uint32_t b0, b1;
      ldm2(b0, b1, sm(&Ks[0 * 8 + (lane & 7)][kcol]));
      mma(s0, s1, s2, s3, a0, a1, a2, a3, b0, b1, s0, s1, s2, s3);
      ldm2(b0, b1, sm(&Ks[1 * 8 + (lane & 7)][kcol]));
      mma(t0, t1, t2, t3, a0, a1, a2, a3, b0, b1, t0, t1, t2, t3);
    }
    // s0,s1->(R0,key 2tidg),(R0,2tidg+1); s2,s3->(R1,..);  t* -> keys 8+...
    // scale + causal mask
    float sc[8] = {s0, s1, t0, t1, s2, s3, t2, t3};   // [R0: k=2t,2t+1,8+2t,8+2t+1][R1: same]
    int kcol0 = 2 * tidg;
    int keyidx[4] = {kcol0, kcol0 + 1, 8 + kcol0, 8 + kcol0 + 1};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      int kpos = kt + keyidx[j];
      sc[j]     = (causal && kpos > qbase + group)     ? -FLT_MAX : sc[j] * scale;       // R0
      sc[j + 4] = (causal && kpos > qbase + group + 8) ? -FLT_MAX : sc[j + 4] * scale;   // R1
    }
    // row softmax (R0 = sc[0..3], R1 = sc[4..7]) reduced across the 4-lane group
    float mt0 = grp_max(fmaxf(fmaxf(sc[0], sc[1]), fmaxf(sc[2], sc[3])));
    float mt1 = grp_max(fmaxf(fmaxf(sc[4], sc[5]), fmaxf(sc[6], sc[7])));
    float nm0 = fmaxf(m0, mt0), nm1 = fmaxf(m1, mt1);
    float c0 = __expf(m0 - nm0), c1 = __expf(m1 - nm1);
    float p[8];
    #pragma unroll
    for (int j = 0; j < 4; ++j) { p[j] = __expf(sc[j] - nm0); p[j + 4] = __expf(sc[j + 4] - nm1); }
    l0 = l0 * c0 + grp_sum(p[0] + p[1] + p[2] + p[3]);
    l1 = l1 * c1 + grp_sum(p[4] + p[5] + p[6] + p[7]);
    m0 = nm0; m1 = nm1;
    // rescale running O
    #pragma unroll
    for (int n = 0; n < 16; ++n) { o_acc[n][0] *= c0; o_acc[n][1] *= c0; o_acc[n][2] *= c1; o_acc[n][3] *= c1; }

    // write P (bf16) to shared at [row][key]
    Ps[group][keyidx[0]]     = __float2bfloat16(p[0]);
    Ps[group][keyidx[1]]     = __float2bfloat16(p[1]);
    Ps[group][keyidx[2]]     = __float2bfloat16(p[2]);
    Ps[group][keyidx[3]]     = __float2bfloat16(p[3]);
    Ps[group + 8][keyidx[0]] = __float2bfloat16(p[4]);
    Ps[group + 8][keyidx[1]] = __float2bfloat16(p[5]);
    Ps[group + 8][keyidx[2]] = __float2bfloat16(p[6]);
    Ps[group + 8][keyidx[3]] = __float2bfloat16(p[7]);
    __syncwarp();

    // ---- PV : O[16 x 128] += P[16 x 16] · V[16 x 128], 16 n-subtiles, 1 k-step ----
    uint32_t pa0, pa1, pa2, pa3;
    ldm4(pa0, pa1, pa2, pa3, sm(&Ps[lane & 15][((lane >> 4) & 1) * 8]));
    #pragma unroll
    for (int n = 0; n < 16; ++n) {
      uint32_t vb0, vb1;
      ldm2(vb0, vb1, sm(&Vt[n * 8 + (lane & 7)][((lane >> 3) & 1) * 8]));
      mma(o_acc[n][0], o_acc[n][1], o_acc[n][2], o_acc[n][3],
          pa0, pa1, pa2, pa3, vb0, vb1,
          o_acc[n][0], o_acc[n][1], o_acc[n][2], o_acc[n][3]);
    }
    __syncwarp();
  }

  // normalize + write O. o_acc[n]: (R0, n*8+2tidg),(R0,+1),(R1,n*8+2tidg),(R1,+1)
  float inv0 = (l0 > 0.f) ? 1.f / l0 : 0.f, inv1 = (l1 > 0.f) ? 1.f / l1 : 0.f;
  const long ob = b * osb + h * osh;
  int r0 = qbase + group, r1 = qbase + group + 8;
  #pragma unroll
  for (int n = 0; n < 16; ++n) {
    int dc = n * 8 + 2 * tidg;
    if (r0 < S_q) {
      O[ob + (long)r0 * oss + (long)dc * osd]       = __float2bfloat16(o_acc[n][0] * inv0);
      O[ob + (long)r0 * oss + (long)(dc + 1) * osd] = __float2bfloat16(o_acc[n][1] * inv0);
    }
    if (r1 < S_q) {
      O[ob + (long)r1 * oss + (long)dc * osd]       = __float2bfloat16(o_acc[n][2] * inv1);
      O[ob + (long)r1 * oss + (long)(dc + 1) * osd] = __float2bfloat16(o_acc[n][3] * inv1);
    }
  }
}

extern "C" cudaError_t openkernels_launch_flash_attention_bf16_fwd_bhsd(
    const OpenKernelsFlashAttentionBF16FwdBHSDArgs* a, cudaStream_t stream) {
  if (a == nullptr) return cudaErrorInvalidValue;
  if (a->D != DM || a->S_q % 16 != 0 || a->S_kv % BN != 0) return cudaErrorNotSupported;
  dim3 grid(a->S_q / 16, a->H, a->B);
  fa_v4<<<grid, 32, 0, stream>>>(
      a->q, a->k, a->v, a->o, a->B, a->H, a->S_q, a->S_kv, a->D, a->causal, a->scale,
      a->q_stride_b, a->q_stride_h, a->q_stride_s, a->q_stride_d,
      a->k_stride_b, a->k_stride_h, a->k_stride_s, a->k_stride_d,
      a->v_stride_b, a->v_stride_h, a->v_stride_s, a->v_stride_d,
      a->o_stride_b, a->o_stride_h, a->o_stride_s, a->o_stride_d);
  return cudaGetLastError();
}
