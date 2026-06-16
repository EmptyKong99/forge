# forge — kernels written by the agent (RTX 5090)

Kernels here are authored by Claude (the coding agent) — the agent *is* the
generator. The 5090 server only compiles/benchmarks them via okbench, and the
agent reads the score and iterates. (`anvil` is the sister track where an LLM API
is the generator instead.)

## Layout
- `kernels/<op>/<variant>.cu` — kernel sources (one file per attempt)
- `bench.sh <op> <variant> [device]` — deploy + run okbench + print summary (on the server)
- `runs/<op>__<variant>.json` — okbench result JSONs (kept on the server)

## Workflow (one iteration)
1. write/edit `kernels/<op>/<variant>.cu`
2. `scp` it to the server's `forge/kernels/<op>/`
3. `ssh <server> 'bash <forge-dir>/bench.sh <op> <variant> <device>'`
4. read the geomean / per-shape speedup, decide the next change

## Target
op `gemm_bf16_nt`: C = A[M,K] @ B[N,K]^T, BF16 in, fp32 accumulate, BF16 out.
Reference = torch.matmul (cuBLAS) = 1.0x ≈ 210 TFLOPS on the suite.
Score = **geometric mean** of the per-shape speedup (cuBLAS_ms / ours_ms) over the
5 fixed shapes in the `required_5` suite (all K=4096):

| shape | M | N |
|---|---|---|
| square | 4096 | 4096 |
| tall | 8192 | 4096 |
| wide | 4096 | 8192 |
| square | 8192 | 8192 |
| tall | 16384 | 4096 |

## Results log
| variant | approach | geomean vs cuBLAS | TFLOPS | notes |
|---|---|---|---|---|
| (baseline) | naive 16×16 tiled (anvil smoke) | 0.0296x | ~6 | correctness/plumbing only |
| v1_regblock | 128×128 block, 8×8 reg tile, fp32 acc, SIMT | 0.2035x | 42.8 | all 5 shapes correct |
| v2_wmma | 64×64 block, wmma 16×16×16 tensor cores, BK=32 | 0.5266x | 111.9 | all correct; small tile, no double-buffer |
| v3_bigtile | 128×128 block, wmma, 128-bit vectorized loads, smem K-padding | 0.7803x | 166.6 | all correct; single-buffered |
| v4_pipeline | v3 + cp.async double-buffering (cuda_pipeline intrinsics) | 0.8847x | 188.7 | all correct; ~88% of cuBLAS |
| v5 (next) | push the wmma ceiling: BK=64 / 3-stage pipeline / vectorized bf16 epilogue | — | — | find where wmma plateaus, *then* drop to PTX |

## Plan: find the wmma ceiling, then PTX
Strategy (per the week's task): squeeze the wmma path to its limit first, so the
later jump to raw PTX (`mma.sync`/`ldmatrix`/`cp.async`) clearly demonstrates its
value. Levers left on the wmma path:
- larger K-step (BK=64) and/or 128×256 tile for more reuse / fewer syncs
- 3-stage (triple-buffer) cp.async pipeline
- vectorized bf16 epilogue (store 8 at a time) instead of scalar per-element
Then: raw `mma.sync` + `ldmatrix` + register-staged C to break past the wmma ceiling.
