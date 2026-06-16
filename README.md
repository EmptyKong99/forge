# forge — an agent that writes high-performance GPU kernels

**Claude Code (the agent) writes the kernel**; an RTX 5090 server only compiles and
benchmarks it against the vendor library (cuBLAS) via the OpenKernels `okbench`
harness; the agent reads the score and iterates. The agent is the *coder*; okbench
is the correctness/perf *verifier*. (In the team's KDA framing — "Claude Code
writes, Codex reviews" — a Codex code-review pass is the planned second verifier.)

Sister project `anvil` is the automated version of this loop (an LLM API as the
coder instead of the agent driving the session).

## Results — `gemm_bf16_nt` (C = A·Bᵀ, BF16, RTX 5090)

Score = geometric mean of per-shape speedup vs cuBLAS (`torch.matmul`, 1.0× ≈ 210
TFLOPS) over the 5 `required_5` shapes (all K=4096): square 4096², tall 8192×4096,
wide 4096×8192, square 8192², tall 16384×4096.

| variant | approach | geomean vs cuBLAS | TFLOPS |
|---|---|---|---|
| baseline | naive 16×16 tiled | 0.0296× | ~6 |
| v1_regblock | 128×128 block, 8×8 register tile, SIMT | 0.2035× | 42.8 |
| v2_wmma | wmma 16×16×16 tensor cores | 0.5266× | 111.9 |
| v3_bigtile | 128×128 + vectorized loads + smem padding | 0.7803× | 166.6 |
| v4_pipeline | + cp.async double-buffering | 0.8847× | 188.7 |
| v5 / v6 | BK=64 / epilogue tweaks (probe the ceiling) | 0.83 / 0.88 | — |
| v7_mma | raw PTX `ldmatrix` + `mma.sync.m16n8k16` | 0.9150× | 195.1 |
| **v8_pipe** | **+ software-pipelined fragment loads** | **0.9245×** | **196.9** |

All variants pass correctness on all 5 shapes. The story: good SIMT → tensor cores
→ the wmma path **plateaus at ~0.88×** (v5/v6 confirm) → **raw PTX breaks past it**
to ~0.92×. Full per-version detail: `kernels/gemm_bf16_nt/README.md`.

This is "learn CUDA / hand-write performant kernels" (meeting 260513) demonstrated
by the agent — and the substrate for the automated loop (`anvil`).

## Layout
```
skills/    agent-callable TOOLS (input/output, structured result)
           bench.sh = deploy + okbench (compile + correctness vs cuBLAS + timing)
wiki/      agent-read KNOWLEDGE (KernelWiki)
  ptx/       per-instruction fact cards (verified, not doc dumps)
  methods/   playbooks the agent follows (e.g. how to use a new PTX instruction)
kernels/<op>/<variant>.cu   agent-authored kernels + a per-op README history
runs/      okbench result JSONs (on the server; gitignored)
```
**skill = run it · wiki = read it.** Running a method (wiki) produces fact cards
(wiki) and uses tools (skills); cards make the next iteration faster.

## One iteration
1. write/edit `kernels/<op>/<variant>.cu`
2. `scp` to the server, then `ssh <server> 'bash <forge>/skills/bench.sh <op> <variant> <device>'`
3. read the geomean + per-shape numbers, decide the next change, commit & push

Branches: `main` = stable tagged milestones, `dev` = daily iteration.

## Generality (axes — current vs intended)
| axis | now | intended |
|---|---|---|
| op | `gemm_bf16_nt` | + fp8 gemm, flash-attention, … |
| shapes | okbench `required_5` | per-op suites |
| language | pure CUDA / PTX | + Triton, TileLang |
| backend | `sm_120` (RTX 5090) | + `sm_100`, … |

## Depends on (GPU server, in the user's own work dir)
OpenKernels repo (okbench + op specs/reference) · torch (cu128, sm_120) · nvcc 13 · g++-12.
