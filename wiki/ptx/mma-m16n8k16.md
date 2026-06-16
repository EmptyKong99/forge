# Wiki / PTX: `mma.sync.m16n8k16` + `ldmatrix` (bf16 in, f32 acc)

Knowledge card. Verified empirically on RTX 5090 (sm_120) by
`kernels/gemm_bf16_nt/v7_mma.cu` and `v8_pipe.cu` (0.92× cuBLAS, okbench-checked).

## Instructions & per-thread registers
```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {d0..d3},{a0..a3},{b0,b1},{c0..c3};
ldmatrix.sync.aligned.m8n8.x4.shared.b16            {r0..r3},[addr];  // A (16x16)
ldmatrix.sync.aligned.m8n8.x2.shared.b16            {r0,r1},[addr];   // B (16x8)
```
- A = 4×b32 (8 bf16), B = 2×b32 (4 bf16), C/D = 4×f32 per thread.
- `addr` is a **shared** address from `__cvta_generic_to_shared(ptr)` (cast u32).

## Fragment layouts
- **C/D accumulator** (needed for the epilogue store): `group=lane/4`, `tidg=lane%4`
  - d0 → (row `group`,   col `2*tidg`)
  - d1 → (row `group`,   col `2*tidg+1`)
  - d2 → (row `group+8`, col `2*tidg`)
  - d3 → (row `group+8`, col `2*tidg+1`)
- **ldmatrix addressing** — each lane gives the address of one 8-wide row of one
  8×8 tile; x4 uses lanes 0–31 (4 tiles), x2 uses lanes 0–15 (2 tiles):
  - A (row-major M×K in shared): `addr = &As[rowbase + lane%16][k + (lane/16)*8]`
  - B (col-major K×N in shared):  `addr = &Bs[nbase + lane%8][k + ((lane/8)&1)*8]`

## ⚠️ Gotcha (cost one iteration, caught by okbench)
For NT GEMM (`C = A·Bᵀ`), the input B `[N,K]` row-major, stored in shared as
`Bs[n][k]`, **is already column-major K×N** — exactly what the `.col` B operand
wants → load with **plain `ldmatrix` (NOT `.trans`)**. `.trans` compiles and runs
fast but is numerically wrong. Rule:
- B already col-major K×N in shared → `ldmatrix` (no trans)
- B stored row-major N×K, want col-major → `ldmatrix.trans`

## Target architectures
`mma.sync` / `ldmatrix` / `cp.async` are **sm_80+** → valid on sm_120 (consumer
Blackwell uses this Ampere/Ada-style path, *not* sm_90 `wgmma` or sm_100
`tcgen05`). Confirm any instruction against the PTX ISA §9.7 SM-support table.
