# Fact / PTX: `ldmatrix` family (sm_80+)

**Type:** fact (verified). Backed by gemm_bf16_nt v7/v8 on RTX 5090 (sm_120),
okbench-checked. The `.trans` rule below cost one bench iteration to learn.

## What it does
Loads 8×8 b16 matrix tiles from **shared** straight into the registers that
`mma.sync` expects — one cooperative warp instruction, no per-thread index math,
conflict-aware. Replaces a hand-rolled shared→register gather.

```
ldmatrix.sync.aligned.m8n8.x1.shared.b16 {r0},          [addr];  // 1 tile
ldmatrix.sync.aligned.m8n8.x2.shared.b16 {r0,r1},       [addr];  // 2 tiles
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {r0,r1,r2,r3}, [addr];  // 4 tiles
```
- `addr` = a **shared** address via `__cvta_generic_to_shared(ptr)` cast to u32.
- Each participating lane supplies the address of **one 8-wide row** of one 8×8
  tile: x1 uses lanes 0–7, x2 lanes 0–15, x4 lanes 0–31.
- `x4` for the A operand (16×16 = four 8×8), `x2` for the B operand (16×8) in the
  m16n8k16 GEMM.

## Addressing (verified, m16n8k16 bf16)
- A (row-major M×K in shared): `addr = &As[rowbase + (lane%16)][k + (lane/16)*8]`
- B (col-major K×N in shared):  `addr = &Bs[nbase + (lane%8)][k + ((lane/8)&1)*8]`

## ⚠️ The `.trans` rule (cost one iteration, caught by okbench)
`.trans` transposes the 8×8 on load. **Use it only when the shared data is in the
*wrong* major order** for the mma operand:
- data already col-major K×N in shared, operand wants `.col` → **plain ldmatrix**
  (NT GEMM's B: `Bs[n][k]` is already col-major → NO `.trans`).
- data row-major N×K, want col-major → **`.trans`**.

`.trans` compiles and runs at full speed even when wrong — only the *numbers* are
wrong. okbench's per-shape correctness check is what catches it; speed won't.

## `stmatrix` (the store counterpart) — VERIFIED → see `stmatrix.md`
`stmatrix.sync.aligned.m8n8.{x1,x2,x4}.shared.b16` writes registers → shared.
Verified on sm_120 by gemm_bf16_nt v12 (coalesced epilogue, 0.9598×). Its register
data layout matches the mma C-accumulator, so accumulators stmatrix directly with no
shuffle. Full layout + gotchas: `stmatrix.md`.

## SM support
sm_80+ → valid on sm_120.

## Cross-refs
- fact: `mma-m16n8k16.md` (the consumer of these fragments)
- menu: `../menu/warp-matrix-mma.md`
