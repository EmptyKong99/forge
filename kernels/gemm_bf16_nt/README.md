# gemm_bf16_nt — detailed version history

**Op.** `C = α · A·Bᵀ + β · C`, BF16 inputs, **fp32 accumulate**, BF16 output.
"NT" is the BLAS transpose convention (N = not transposed, T = transposed): A is
**N**, B is **T**, so `C = A·Bᵀ`. A is `[M,K]`, B is `[N,K]` (both row-major), C is
`[M,N]`. Because both A and B are read along K (contiguous), the NT layout is
friendly: `C[m,n] = Σ_k A[m,k]·B[n,k]`.

**Reference / score.** Reference is `torch.matmul` (cuBLAS), = 1.0× ≈ 210 TFLOPS.
Score = geometric mean of per-shape speedup (cuBLAS_ms / ours_ms) over `required_5`
(all K=4096): square 4096², tall 8192×4096, wide 4096×8192, square 8192², tall
16384×4096. Correctness is checked per-shape against cuBLAS by okbench.

**ABI.** Implement `extern "C" cudaError_t openkernels_launch_gemm_bf16_nt(const
OpenKernelsGemmBF16NTArgs*, cudaStream_t)`; args carry pointers, m/n/k, the six
strides, alpha, beta. Unsupported shapes/layouts may return `cudaErrorNotSupported`
(we require M,N%tile==0, K%BK==0, contiguous-k for the fast paths).

## Versions

### baseline — naive 16×16 tiled — 0.0296×
The anvil smoke kernel: one thread per output, 16×16 shared tiles, fp32 accumulate.
Correct but ~34× slower than cuBLAS. Exists only to prove the harness.

### v1_regblock — register-blocked SIMT — 0.2035× (42.8 TFLOPS)
128×128 block tile, each thread computes an 8×8 register tile; A staged transposed
in shared for a clean inner product. No tensor cores → caps where SIMT FMA caps.
**Lesson:** register blocking is the first big lever (7× over naive) but SIMT can't
touch tensor-core throughput.

### v2_wmma — tensor cores via wmma — 0.5266× (111.9 TFLOPS)
Switched the inner product to `nvcuda::wmma` 16×16×16 fragments. 64×64 block, 4
warps. The NT layout maps cleanly: B stored `[N,K]` is column-major K×N, exactly
what `matrix_b, col_major` wants. **Lesson:** tensor cores ≈ 2.6× over good SIMT.

### v3_bigtile — bigger tile + vectorized loads + smem padding — 0.7803× (166.6)
128×128 block, 8 warps. 128-bit (`int4`, 8 bf16) vectorized global→shared loads;
shared K-dim padded (BK+8) to cut bank conflicts on the wmma loads. **Lesson:**
arithmetic intensity (bigger tile) + load width + bank-conflict-free shared are the
classic GEMM levers; +0.25 over v2.

### v4_pipeline — cp.async double-buffering — 0.8847× (188.7) ← wmma champion
Prefetch the next K-tile into a second shared buffer with `cp.async`
(`<cuda_pipeline.h>`) while the tensor cores work on the current one. +0.10 by
hiding global-load latency.

### v5_bk64 — BK=64, 72KB dynamic smem — 0.8322× ↓ (regression)
Larger K-step needs 72KB shared (opt-in), which drops occupancy to 1 block/SM and
*regresses*. **Lesson:** past a point, bigger shared hurts more (occupancy) than the
extra reuse helps.

### v6_epilogue — vectorized bf16 epilogue — 0.8843× (≈ neutral)
int4 (8-wide) output stores on the β=0 fast path. Neutral — the epilogue is a tiny
fraction of a K=4096 GEMM. Also tried `__launch_bounds__(256,2)`: regressed (register
spills). **Conclusion: the wmma path firmly plateaus at ~0.88×.**

### v7_mma — raw PTX `ldmatrix` + `mma.sync.m16n8k16` — 0.9150× (195.1) ← champion
Dropped below wmma to raw PTX: `ldmatrix` lands shared data directly in the
registers `mma.sync` expects, accumulators stay in registers (no shared round-trip),
and the warp does 4×4 m16n8k16 mma tiles for its 64×32 output. Breaks past the wmma
ceiling. **Key gotcha (cost one bench):** B is already column-major K×N in shared, so
load it with **non-trans** `ldmatrix` — `.trans` runs fast but is *wrong*. okbench's
correctness check caught it. See `../../wiki/ptx/mma-m16n8k16.md`.

### v8_pipe — software-pipelined fragment loads — 0.9245× (196.9) ← champion
Same instructions/layouts as v7, but load the `ldmatrix` fragments for *both*
k-substeps up front, then issue all `mma` — giving the scheduler room to overlap
load latency with tensor-core math. +0.01, correct on the first try (no layout
change). Applied straight from `skills/ptx-mma.md`'s "next levers" list.

### Remaining gap to cuBLAS (~0.92× → 1.0×)
Bigger shapes (8192²) sit lower (~0.91×) than small ones (~0.96×) → still some
tensor-core feeding stalls. Likely next levers: deeper (3-stage) cp.async pipeline
within the register budget, swizzled shared layout to kill `ldmatrix` bank
conflicts, and a wider warp tile. Diminishing returns; matching cuBLAS on every
shape is hard.
