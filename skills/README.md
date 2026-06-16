# skills/ — agent-callable tools

A **skill** here = an AI-agent-callable tool/script with a clear input/output and
structured result (the team's definition of a Kernel Skill). Distinct from
`wiki/`, which is knowledge the agent *reads*; skills are things the agent *runs*.

## Index
| skill | role | usage | output |
|---|---|---|---|
| `bench.sh` | Verify + Benchmark (the okbench harness) | `bench.sh <op> <variant> [device]` on the server | per-shape correctness + speedup vs cuBLAS, geomean, TFLOPS; JSON in `runs/` |

## Planned (roadmap)
- `profile.sh` — NCU wrapper (SpeedOfLight / Occupancy / SASS hotspots) → JSON, for
  evidence-driven optimization instead of guessing the bottleneck (survey P0).
- `compile.sh` — nvcc/NVRTC wrapper with structured error parsing.
- `review` — Codex code-review pass (the KDA verifier: Claude Code writes, Codex reviews).

## Design principles (from the team's skill survey)
independent CLI/script · structured (JSON) output · graded execution · idempotent ·
artifacts under a run dir · no secrets in code · vendor-abstracted where possible.
