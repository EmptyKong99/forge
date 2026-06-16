# Skill: BF16 tensor-core GEMM with raw PTX (`ldmatrix` + `mma.sync`)

Distilled from `kernels/gemm_bf16_nt/v7_mma.cu`, which broke past the wmma ceiling
(0.88× → 0.915× cuBLAS) on RTX 5090 (sm_120). All three instructions below are
**sm_80+** and work on sm_120 (consumer Blackwell uses the Ampere/Ada-style
`mma.sync` path — *not* sm_90 `wgmma` or sm_100 `tcgen05`).

## When to use
After a wmma kernel plateaus. wmma's `load_matrix_sync` is a generic strided load
and schedules poorly; `ldmatrix` is a single warp instruction that lands data in
exactly the registers `mma.sync` wants, and keeping the accumulator in registers
removes wmma's shared round-trip. Net: more throughput, tighter scheduling.

## The instructions (m16n8k16, bf16 in / f32 acc)
```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {d0..d3},{a0..a3},{b0,b1},{c0..c3};
ldmatrix.sync.aligned.m8n8.x4.shared.b16        {r0..r3},[addr];   // A (16x16)
ldmatrix.sync.aligned.m8n8.x2.shared.b16        {r0,r1},[addr];    // B (16x8)
```
Per thread: A = 4×b32 (8 bf16), B = 2×b32 (4 bf16), C/D = 4×f32. `addr` is a
**shared-memory** address from `__cvta_generic_to_shared(ptr)` (cast to u32).

## Fragment layouts that matter
- **C/D accumulator** (for the epilogue store): `group = lane/4`, `tidg = lane%4`.
  - d0 → (row `group`,   col `2*tidg`)
  - d1 → (row `group`,   col `2*tidg+1`)
  - d2 → (row `group+8`, col `2*tidg`)
  - d3 → (row `group+8`, col `2*tidg+1`)
- **ldmatrix addressing**: each lane supplies the address of one 8-wide row of one
  8×8 tile. x4 uses lanes 0–31 (four tiles), x2 uses lanes 0–15 (two tiles).
  - A (row-major MxK in shared): `addr = &As[rowbase + (lane%16)][k + (lane/16)*8]`
  - B (see gotcha): `addr = &Bs[nbase + (lane%8)][k + ((lane/8)&1)*8]`

## ⚠️ The key gotcha (cost one iteration)
For NT GEMM (`C = A·Bᵀ`), B is the input `[N,K]` row-major. Stored in shared as
`Bs[n][k]`, that array **is already column-major K×N**, which is exactly what the
`.col` B operand wants → load it with **plain `ldmatrix` (NOT `.trans`)**.
Using `.trans` compiles and runs fast but gives wrong results. Rule of thumb:
- B already col-major K×N in shared → `ldmatrix` (no trans)
- B stored row-major N×K and you want col-major → `ldmatrix.trans`

## Correctness oracle
Don't trust the layout from memory — okbench checks every shape against cuBLAS.
v7 attempt 1 ran at 0.92× but `correct=False`; flipping B to non-trans fixed it
with no speed loss. Write it, bench it, fix the layout from the pass/fail signal.

## Next levers (not yet done)
- `ldmatrix`-based or vectorized epilogue (still scalar register store).
- software-pipeline the `ldmatrix` of the next k-substep with the current `mma`.
- avoid shared bank conflicts feeding `ldmatrix` (swizzle instead of +8 pad).
