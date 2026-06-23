# Fact / PTX: `stmatrix` (register → shared, sm_120 verified)

**Type:** fact (verified). Backed by gemm_bf16_nt **v12_stmatrix** on RTX 5090
(sm_120), okbench-checked: 5/5 shapes correct, geomean **0.9598×** (3 runs
0.9594–0.9612). Promoted from a `menu/` pointer by the survey→use→distill loop.

## Resolved uncertainty (why this was worth verifying)
The menu card listed stmatrix as "sm_80+" (from a doc sweep) but historically it
shipped with sm_90 — so whether it even **exists on sm_120** (consumer Blackwell,
which *drops* some Hopper features like `wgmma`) was unknown. **Verified: it
compiles and runs correct on sm_120.** Do not assume the inverse for `wgmma`.

## What it does
Inverse of `ldmatrix`: each lane provides data in **mma-fragment register layout**
and an address in **row layout**; the instruction permutes registers → shared.
Lets you write an mma accumulator straight to shared with **no warp shuffle**.

```
stmatrix.sync.aligned.m8n8.x2.shared.b16 [addr], {r0,r1};   // 2 matrices
// (x1 = {r0}; x4 = {r0,r1,r2,r3})
```

## Verified layout (m8n8.x2, b16)
- **Data register** — lane T's `rM` holds the b16 pair at `(row = T/4, col =
  2*(T%4))` of matrix M: **low16 = col `2*(T%4)`, high16 = col `2*(T%4)+1`**.
  This is **identical to the mma m16n8 C-accumulator 8×8 layout** (verified by v8),
  so you can pack accumulators directly:
  - `r0 = pack(bf16(d0), bf16(d1))` → upper 8×8 (acc rows 0–7)
  - `r1 = pack(bf16(d2), bf16(d3))` → lower 8×8 (acc rows 8–15)
  - pack with `__halves2bfloat162(lo, hi)` (`.x`→low16, `.y`→high16).
- **Address** — lane T gives the base of **row `T%8` of matrix `T/8`**. For x2 only
  lanes 0–15 matter, but **lanes 16–31 must still pass an in-bounds address**
  (`&Cs[(lane>>3)&1][lane&7][0]` keeps them in [0,1]).
- `addr` is a **shared** address via `__cvta_generic_to_shared` (u32), like ldmatrix.

## ⚠️ Gotchas
- Convert f32 acc → bf16 **before** packing; stmatrix only moves `.b16`.
- Follow with `__syncwarp()` before the warp reads the scratch back (stmatrix is a
  warp-collective store into shared, not visible to the lane until synced).
- `alpha` folds into the value before packing; `beta != 0` needs a global read of
  C → v12 keeps a scalar fallback for that path (okbench's score uses beta=0).

## Why it was a +3.5pp win (not the expected "neutral")
v8's scalar epilogue had each thread store its 4 accumulators to **scattered**
global positions `(r,c),(r,c+1),(r+8,c),(r+8,c+1)` → poorly coalesced. v12 routes
acc → shared (stmatrix) → **contiguous** shared→global copy = coalesced C writes.
On large outputs (square_8192 C = 128MB) write bandwidth is a real slice of
runtime → 0.9245× → **0.9598×**. When this matters is a judgment →
`../heuristics/epilogue-coalescing.md`.

## SM support
Verified sm_120. (Listed sm_80+ in docs; trust the verification, not the sweep.)

## Cross-refs
- fact: `ldmatrix-family.md` (the inverse), `mma-m16n8k16.md` (the fragment source)
- menu: `../menu/warp-matrix-mma.md`
- heuristic: `../heuristics/epilogue-coalescing.md`
