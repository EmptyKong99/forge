# forge — kernels written by the agent (RTX 5090)

Kernels here are authored by Claude (the coding agent) — the agent *is* the
generator. The 5090 server only compiles/benchmarks them via okbench, and the
agent reads the score and iterates. (`anvil` is the sister track where an LLM API
is the generator instead.)

## Layout
- `kernels/<variant>.cu` — kernel sources (one file per attempt)
- `bench.sh` — deploy + run okbench + print summary (runs on the server)
- `runs/<variant>.json` — okbench result JSONs (on the server, under gucheng)

## Workflow (one iteration)
1. write/edit `kernels/<variant>.cu` on the Mac
2. `scp` it to the server's `forge/kernels/`
3. `ssh server 'bash /nvme/share/gucheng/forge/bench.sh <variant> <device>'`
4. read the geomean / per-shape speedup, decide the next change

## Target
op `gemm_bf16_nt`: C = A[M,K] @ B[N,K]^T, BF16 in, fp32 accumulate, BF16 out.
Reference = torch.matmul (cuBLAS) = 1.0x ≈ 210 TFLOPS on the suite.
Score = **geometric mean** of the per-shape speedup (cuBLAS_ms / ours_ms) over the
5 fixed shapes in the `required_5` suite (all K=4096):
1. square 4096×4096   2. tall 8192×4096   3. wide 4096×8192
4. square 8192×8192   5. tall 16384×4096

## Results log
| variant | approach | geomean vs cuBLAS | TFLOPS | notes |
|---|---|---|---|---|
| (baseline) | naive 16×16 tiled (anvil smoke) | 0.0296x | ~6 | correctness/plumbing only |
| v1_regblock | 128×128 block, 8×8 reg tile, fp32 acc, SIMT | 0.2035x | 42.8 | all 5 shapes correct |
| v2_wmma | 64×64 block, wmma 16×16×16 tensor cores, BK=32 | 0.5266x | 111.9 | all correct; small tile, no double-buffer |
| v3_bigtile | 128×128 block, wmma, 128-bit vectorized loads, smem K-padding | 0.7803x | 166.6 | all correct; single-buffered |
| v4 (next) | cp.async double-buffering (overlap global load with mma) | — | — | hide load latency |

## Bottlenecks to attack next (v4+)
- No pipelining → double-buffer shared tiles with `cp.async` to overlap global
  loads with tensor-core math (likely the biggest remaining win).
- Try larger K-step (BK=64) and/or 128×256 tile for more reuse.
- Consider raw `mma.sync` + register-staged C to cut the shared store/convert cost.
- Epilogue: vectorized bf16 stores instead of scalar per-element.
