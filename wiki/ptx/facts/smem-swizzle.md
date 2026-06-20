# Fact / shared memory: XOR swizzle for conflict-free `ldmatrix` (no padding)

**Type:** fact (verified). Backed by gemm_bf16_nt **v13_swizzle** on RTX 5090
(sm_120), okbench-checked: 5/5 correct, geomean **0.9900×** (square_4096 **1.0257×**
— over cuBLAS), 3 runs 0.9882–0.9901. Replaces padding (`BKP=BK+8`) entirely.

## The working swizzle (verified, bf16, BK=32)
Each shared row = BK=32 bf16 = 4 chunks of 8 (one 16B cp.async / one ldmatrix row
each). Permute the chunk index by the low 2 bits of the row:
```cpp
__device__ __forceinline__ int swz(int row, int k) {  // k multiple of 8 at call sites
  return ((k >> 3) ^ (row & 3)) * 8 + (k & 7);          // -> col in [0,32)
}
// store: &As[b][row][swz(row, kk8)]   (cp.async 16B dst)
// load : &As[cur][rowbase][swz(rowbase, kcol)]  (ldmatrix addr, kcol multiple of 8)
```
`As`/`Bs` are `[2][BM][BK]` with **BK=32, no `+8` pad** → 32KB shared (was 40KB),
buying occupancy for free.

## Why it removes conflicts
Within an 8-row `ldmatrix` group the rows differ in their low bits, so
`chunk ^ (row&3)` sends the 4 chunks of consecutive rows to **different** banks —
the same conflict-avoidance padding gave, but with zero extra shared (occupancy-
neutral). It's the cure v9 (chased occupancy) and v10 (chased no-pad) each got only
half of.

## ⚠️ The one gotcha that makes this dangerous
The **store and the load must apply the exact same `swz`.** If they disagree the
kernel **compiles and runs at full speed but is numerically wrong** — there is no
crash, no slowdown, only wrong numbers. okbench's per-shape correctness check is the
only thing that catches it. (v13 got it right on the first bench because the swizzle
is a single shared helper called from both sites — keep it that way; never inline
two copies that could drift.)

## Generalizing
- The `& 3` period matches 4 chunks (BK=32, 8-elem chunks). For other BK/element
  widths the mask/shift change — re-derive so each `ldmatrix` group's chunks spread
  across banks, and **re-verify with okbench** (this is a fact card for *this* shape).
- Hopper `wgmma`/TMA use hardware *canonical* swizzle modes in the matrix descriptor
  instead — different mechanism, sm_90+ only → `../menu/smem-layout-swizzle.md`.

## Cross-refs
- heuristic: `../heuristics/padding-vs-swizzle.md` (when padding vs swizzle)
- fact: `ldmatrix-family.md`, `cp-async.md` (the store/load this sits between)
- menu: `../menu/smem-layout-swizzle.md`
