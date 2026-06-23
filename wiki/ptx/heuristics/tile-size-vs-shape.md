# Heuristic / GEMM: block-tile size vs. problem shape

**Type:** heuristic (condition → technique). Not a fact — a *judgment* whose
answer flips with the workload. Provenance: gemm_bf16_nt v8 vs v11 on RTX 5090
(sm_120), okbench `required_5`.

## The map

| Regime | Prefer | Why |
|---|---|---|
| **Large, squarish, big-K** (M,N ≥ 4096, K ≥ 4096) | **big tile** (128×128) | A/B are re-read by few blocks → arithmetic intensity / reuse dominates; you want each loaded tile amortized over as much math as possible |
| **Skinny / small / batched** (small M *or* N, or many tiny GEMMs) | **small tile** (e.g. 128×64) | a big tile wastes SMs (few blocks, poor load balance) and the matmul is latency- not compute-bound; more, smaller blocks = more occupancy to hide latency |
| **Memory-bound** (tiny K, or low compute-per-byte) | **small tile + more blocks** | reuse can't save you; throughput comes from occupancy |

## The evidence (don't trust the rule, trust the bench)

- **v8** 128×128 = **0.9245×** cuBLAS. Per-shape: ~0.96 on 4096², ~0.91 on the
  big 8192²/16384 shapes.
- **v11** 128×64 + 3-stage = **0.6954×** — the *worst* of the whole ladder.
  Shrinking N to 64 roughly doubled how many blocks re-load each strip of B →
  arithmetic intensity collapsed. This was an **independent reviewer's #1 pick**
  (predicted 0.93–0.95×); the bench falsified it.

## ⚠️ Do not read this as "always use 128×128"

That conclusion is **overfit to okbench's shape set**, which is *entirely* large
big-K matrices — the exact regime that rewards big tiles. v11's 128×64 is not a
"bad kernel"; it is the *right* kernel for a regime okbench doesn't sample (skinny
/ small / batched). If the shape distribution changes, re-bench before assuming
the ranking holds. **The deliverable is the map above, not a winner.**

## Cross-refs
- `[[pipeline-depth-vs-occupancy]]` — the other axis v11 traded on
- fact: `../facts/mma-m16n8k16.md` — the instruction these tiles are built from
