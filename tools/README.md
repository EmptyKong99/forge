# tools/ — executables

Mechanical, agent-callable executables (bash/python) with clear input/output. A
`skills/` playbook orchestrates these; `wiki/` is the knowledge they're used with.

| tool | role | usage | output |
|---|---|---|---|
| `bench.sh` | deploy a kernel + run okbench (compile + correctness vs cuBLAS + timing) | `bench.sh <op> <variant> [device]` (on the server) | per-shape correctness + speedup, geomean, TFLOPS; JSON in `runs/` |

## Planned
- `profile.sh` — NCU wrapper (SpeedOfLight / Occupancy / SASS hotspots) → JSON (survey P0).
- `compile.sh` — nvcc/NVRTC wrapper with structured error parsing.
- `ptxdoc` — queryable PTX ISA lookup (syntax + SM table per instruction).
