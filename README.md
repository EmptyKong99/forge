# forge — an agent that writes high-performance GPU kernels

Claude (the coding agent) writes GPU kernels, deploys them to an RTX 5090 server,
and benchmarks them against the vendor library (cuBLAS) through the OpenKernels
`okbench` harness — then reads the result and iterates. **The agent is the kernel
generator**; the server only compiles and measures.

(Sister project `anvil` automates the same loop with an LLM *API* as the
generator. Here the generator is the agent driving the session directly.)

## Results — `gemm_bf16_nt` (C = A @ Bᵀ, BF16, RTX 5090)

Score = **geometric mean** of the per-shape speedup vs cuBLAS (`torch.matmul`,
≈ 210 TFLOPS = 1.0×) over the 5 fixed `required_5` shapes (all K=4096):

| shape  | M | N |
|---|---|---|
| square | 4096 | 4096 |
| tall   | 8192 | 4096 |
| wide   | 4096 | 8192 |
| square | 8192 | 8192 |
| tall   | 16384 | 4096 |

| variant | approach | geomean vs cuBLAS | TFLOPS |
|---|---|---|---|
| (baseline) | naive 16×16 tiled (anvil smoke) | 0.0296× | ~6 |
| v1_regblock | 128×128 block, 8×8 register tile, SIMT fp32 acc | 0.2035× | 42.8 |
| v2_wmma | 64×64 block, wmma 16×16×16 tensor cores | 0.5266× | 111.9 |
| v3_bigtile | 128×128 + 128-bit vectorized loads + smem K-padding | 0.7803× | 166.6 |
| **v4_pipeline** | + cp.async double-buffering | **0.8847×** | **188.7** |
| v5_bk64 | BK=64, dynamic 72KB smem (experiment) | 0.8322× ↓ | 177.1 |
| v6_epilogue | vectorized bf16 epilogue (no smem growth) | 0.8843× ≈ | 188.7 |
| **v7_mma** | **raw PTX: `ldmatrix` + `mma.sync.m16n8k16`, register-resident C** | **0.9150×** | **195.1** |

All variants pass correctness on all 5 shapes. **v7 (raw PTX) breaks past the wmma
ceiling** — see `skills/ptx-mma.md` for the recipe and the key gotcha.

**Goal: meet / beat cuBLAS.** Strategy (this week's task): push the wmma path to
its ceiling first, then drop to raw PTX so the PTX jump clearly demonstrates its
value. **Finding: the wmma path firmly plateaus at ~0.88×.** Three levers tried
beyond v4, none helped:
- v5 BK=64 → 72KB smem → occupancy 1 block/SM → 0.83× (regress)
- `__launch_bounds__(256,2)` occupancy hint → register spills → 0.85× (regress)
- vectorized bf16 epilogue (v6) → 0.884× (neutral; epilogue is a tiny fraction)

So the next real gain must come from **raw PTX**: `ldmatrix` +
`mma.sync.m16n8k16` with register-staged accumulators (fewer shared round-trips
and tighter scheduling than the wmma wrappers). PTX know-how → `skills/`.

## How it works
- `kernels/<op>/<variant>.cu` — agent-authored kernels (pure CUDA / inline PTX)
- `bench.sh <op> <variant> [device]` — deploy into the OpenKernels submission tree,
  run okbench (compile + correctness + timing on real hardware), print the score
- `runs/<op>__<variant>.json` — okbench result JSONs (kept on the server)
- `skills/` — distilled, reusable how-tos learned along the way (e.g. PTX mma usage)

### One iteration
1. agent writes/edits `kernels/<op>/<variant>.cu`
2. `scp` to the server, then `ssh <server> 'bash <forge-dir>/bench.sh <op> <variant> <device>'`
3. read the geomean + per-shape numbers, decide the next change, commit & push

Branches: `main` = stable milestones (tagged), `dev` = daily iteration.

## Generality (axes — current vs intended)
Kept extensible along these axes; only the first column is built so far.

| axis | now | intended |
|---|---|---|
| op | `gemm_bf16_nt` | + fp8 gemm, flash-attention, linear-attention, … |
| shapes | okbench `required_5` | per-op suites |
| language | pure CUDA / PTX | + Triton, TileLang |
| backend | `sm_120` (RTX 5090) | + `sm_100`, … |

## Depends on (on the GPU server, in the user's own work dir)
- the OpenKernels repo (provides `okbench` + the op specs/reference)
- a venv with torch (cu128, sm_120); nvcc 13, g++-12 host compiler
