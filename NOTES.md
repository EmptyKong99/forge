# forge — hand-written CUDA kernels for OpenKernels (RTX 5090)

Kernels are written by hand; the 5090 server only compiles/benchmarks them via
okbench. Separate from `anvil` (which automates kernel generation with an LLM) —
here the kernels are authored directly.

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
| v3 (next) | bigger tile + cp.async double-buffering + smem padding + vectorized loads | — | — | close the gap to cuBLAS |

## Bottlenecks to attack next (v3+)
- Small 64×64 tile → low arithmetic intensity / reuse. Go 128×128 (or 128×256).
- Scalar global loads → use 128-bit vectorized loads (int4 = 8 bf16) + `cp.async`.
- No pipelining → double-buffer shared tiles to overlap load with mma.
- wmma shared loads likely bank-conflicted → pad shared leading dim (BK+8).
