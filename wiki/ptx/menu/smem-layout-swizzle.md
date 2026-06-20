# Menu / PTX: shared-memory layout, matrix descriptors, swizzle

**Type:** menu (breadth, **UNVERIFIED**). Source: PTX ISA §9.7. Purpose: catalog
the smem-layout tools so the agent knows the *options* for conflict-free / async
loads. The chosen one must be verified into `facts/` before trust.

## Bank-conflict avoidance (sm_80+, applies to ldmatrix/mma path)
- **Padding** — add columns so consecutive smem rows land in different banks.
  Simple, costs shared. **Verified in use** (v8 `BKP=BK+8`).
- **XOR swizzle** — permute each row's column index by XOR-ing in some row bits,
  so the same access pattern spreads across banks with *zero* padding. Occupancy-
  neutral. **VERIFIED + champion** (gemm v13, 0.99×, beats cuBLAS on square_4096) →
  exact working function in `../facts/smem-swizzle.md`.
- Choice between them is a judgment → `[[padding-vs-swizzle]]`.

## Matrix descriptors (sm_90+ wgmma/TMA) — UNVERIFIED, off-target
`wgmma` and `cp.async.bulk` don't take raw pointers; they take a 64-bit **matrix
descriptor** encoding the shared base address, leading/stride byte offsets, and a
**canonical swizzle mode** (e.g. 32B/64B/128B swizzle). These swizzle modes are a
*hardware* contract on Hopper, distinct from the hand-rolled XOR swizzle above.
**sm_90+ only — not on the sm_120 target.** Listed so the agent doesn't confuse
"swizzle" the hand technique with "swizzle mode" the descriptor field.

## Reading guide
- On **sm_120 (our hardware):** you only have padding and hand-XOR-swizzle. The
  descriptor/canonical-swizzle machinery is Hopper and irrelevant here.
- Don't import a Hopper smem layout into an sm_80-path kernel — the descriptor
  swizzle modes assume `wgmma`, which sm_120 doesn't have.

## Cross-refs
- heuristic: `../heuristics/padding-vs-swizzle.md`
- menu: `warp-matrix-mma.md` (wgmma needs these descriptors), `async-copy-model.md`
