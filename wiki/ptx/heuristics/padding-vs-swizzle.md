# Heuristic / shared memory: padding vs. XOR-swizzle for bank conflicts

**Type:** heuristic (condition → technique). Provenance: gemm_bf16_nt v8 (padded)
vs v10 (no pad) on RTX 5090 (sm_120), okbench `required_5`.

## The problem

`ldmatrix` / vectorized smem loads conflict when many lanes hit the same 32 banks.
Two cures:

| Technique | What it costs | When it's right |
|---|---|---|
| **Padding** (`BKP = BK + 8`) | extra shared per tile → can lower occupancy | simplest; good default when shared budget is loose. Used by v8 (`BKP=40`) |
| **XOR swizzle** (permute the smem column by `row` bits) | zero extra shared, some index arithmetic | **the winner here (v13, 0.99×)** — occupancy-neutral conflict cure; use when shared is tight and you need occupancy *and* no conflicts |
| **Neither** (raw `BK` stride) | bank conflicts | almost never on these access patterns — see v10 |

## The evidence

- **v8** padded (`BKP=40`, 40KB shared) = **0.9245×**.
- **v10** dropped the pad (`BKP=32`, 32KB shared), hoping less shared → 3 blocks/SM
  = **0.8997×** ↓. The padding-free layout reintroduced `ldmatrix` bank conflicts
  that cost ~3pp *more* than the occupancy it bought. **Lesson:** on the raw-PTX
  path the `+8` pad is doing real work — you cannot just shrink shared for
  occupancy and expect to win.

## The swizzle lever — RESOLVED (v13, it won)

The **XOR swizzle** was the one move that keeps the tile size *and* buys occupancy
*and* removes conflicts — the successor to both v9 (occupancy) and v10 (conflicts).
It was predicted "3–8%, unbenched." **v13 benched it: +3pp (v12 0.96 → v13 0.99),
correct on the first try, square_4096 over cuBLAS.** So for *this* GEMM the ranking
is: **swizzle > padding > nothing**. The exact working swizzle + the (silent-wrong)
gotcha are now a fact: `../facts/smem-swizzle.md`. Note the prediction's *direction*
was right but it's still just luck until benched — v11's prediction was equally
confident and wrong.

## Meta-note (the recurring lesson across v9/v10/v11)
Every confident prediction in this op — author's *and* an independent reviewer's —
was wrong by data at least once. okbench (compile + per-shape correctness vs
cuBLAS + timing) is the only arbiter. Heuristic cards record *tendencies and their
flip-conditions*, never verdicts; the bench decides each instance.

## Cross-refs
- `[[pipeline-depth-vs-occupancy]]` · `[[tile-size-vs-shape]]`
- menu: `../menu/smem-layout-swizzle.md`
