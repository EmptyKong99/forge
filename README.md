# forge — an agent that writes high-performance GPU kernels

Claude (the coding agent) writes GPU kernels, deploys them to an RTX 5090 server,
and benchmarks them against the vendor library (cuBLAS) through the OpenKernels
`okbench` harness — then reads the result and iterates. **The agent is the kernel
generator**; the server only compiles and measures.

(Sister project `anvil` automates the same loop with an LLM *API* as the
generator. Here the generator is the agent driving the session directly.)

## Status
op `gemm_bf16_nt` (C = A @ Bᵀ, BF16, RTX 5090): **0.78× cuBLAS** (166.6 TFLOPS),
all 5 suite shapes correct. Progression: 0.03 → 0.20 → 0.53 → 0.78 (see `NOTES.md`).

**Goal: meet / beat cuBLAS** by dropping to PTX-level instructions
(`mma.sync`, `ldmatrix`, `cp.async`); PTX know-how gets distilled into `skills/`.

## How it works
- `kernels/<op>/<variant>.cu` — agent-authored kernels (pure CUDA / inline PTX)
- `bench.sh <op> <variant> [device]` — deploy into the OpenKernels submission tree,
  run okbench (compile + correctness + timing on real hardware), print the score
- `runs/<op>__<variant>.json` — okbench result JSONs (kept on the server)
- `NOTES.md` — per-version results log + bottleneck analysis
- `skills/` — distilled, reusable how-tos learned along the way (e.g. PTX mma usage)

## Generality (axes — current vs intended)
The design is deliberately kept extensible along these axes; only the first column
is built so far.

| axis | now | intended |
|---|---|---|
| op | `gemm_bf16_nt` | + fp8 gemm, flash-attention, linear-attention, … |
| shapes | okbench `required_5` | per-op suites |
| language | pure CUDA / PTX | + Triton, TileLang |
| backend | `sm_120` (RTX 5090) | + `sm_100`, … |

## One iteration
1. agent writes/edits `kernels/<op>/<variant>.cu`
2. `scp` to the server, then `ssh server 'bash forge/bench.sh <op> <variant> <device>'`
3. read the geomean score + per-shape numbers, decide the next change, commit & push

## Setup it depends on (on the GPU server, in the user's own work dir)
- the OpenKernels repo (provides `okbench` + the op specs/reference)
- a venv with torch (cu128, sm_120)
- nvcc 13, g++-12 host compiler
