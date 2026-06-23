# Heuristic / GEMM: coalesce the epilogue (don't just vectorize it)

**Type:** heuristic (condition → technique). Provenance: gemm_bf16_nt v8 (scalar
scattered stores) vs v12 (stmatrix → shared → coalesced copy) on RTX 5090 (sm_120),
okbench `required_5`. v8 = 0.9245×, **v12 = 0.9598×** (+3.5pp).

## The mechanism
An mma accumulator fragment is laid out **scattered** across the output tile —
lane T owns `(row, col),(row, col+1),(row+8, col),(row+8, col+1)`. Writing those
straight to global gives **uncoalesced** stores. The output C is `M·N·2` bytes;
for large outputs that write traffic is a non-trivial slice of runtime, so the
coalescing pays off even though the epilogue is "only one pass."

| Regime | Prefer | Why |
|---|---|---|
| **Large output** (M·N large, e.g. ≥4096²; C ≫ tens of MB) | **coalesce** via stmatrix→shared→contiguous copy | C-write bandwidth is real here (square_8192 C=128MB); scattered stores waste it |
| **Small output, huge K** (tiny M·N, K dominates) | **don't bother** | epilogue is a negligible fraction; the shared round-trip + 2KB scratch just adds occupancy pressure for ~0 gain |
| **β≠0 (read-modify-write C)** | scalar/direct | you must read prev C anyway; the coalescing win shrinks and the code is simpler scalar (v12 keeps a scalar fallback here) |

## The sharper lesson: coalesce ≠ vectorize
Widening stores (`int4`, 8-wide) is **not** the same as coalescing them. If the
8 elements a thread writes are still at scattered rows, a wide store doesn't help —
that is why an earlier "vectorized epilogue" attempt (v6, wmma route) was neutral.
v12 wins because it **reorders** the data through shared so the global writes are
*contiguous across lanes*, not merely wider per lane.

## ⚠️ Not "always add an stmatrix epilogue"
The win is conditional on output size and β=0. For skinny/small-output or
β≠0 GEMMs it's neutral-to-negative (extra shared, extra sync). And it cost 2KB of
shared scratch — on a kernel already at the occupancy edge that 2KB could itself
regress (it didn't here; v12 stayed correct and faster — okbench confirmed).

## Cross-refs
- fact: `../facts/stmatrix.md` (the verified instruction + layout)
- `[[tile-size-vs-shape]]` (the other large-output lever)
