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

## Two routes (split for finding the limit + distilling skills)

The ladder forks at v6 into two optimization *paradigms*, each pushed to its own
ceiling (a single linear chain commits you to one path):

- **wmma route** — v1→v6, `nvcuda::wmma` wrappers. **Firmly plateaus ~0.88×** (v5/v6
  proved it). Squeezing it further (swizzle/occupancy) is diminishing; kept as the
  clean per-technique distillation spine.
- **PTX route** — v7→…, raw `ldmatrix`+`mma.sync`. Broke past wmma to 0.92×; this is
  where the remaining headroom is. New work lives here (`v9+`), branches off v8.

Caveat on "no-skill vs skill": these two routes are **NOT** a clean ablation — the
author's context already knows the PTX recipe, can't un-know it. The clean no-skill
baseline is anvil/DeepSeek (EXP-001/002/003). Here the routes mean *wmma-limit vs
PTX-limit*, both distilled into skills.

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

### v9_stage3 — 3-stage cp.async pipeline — 0.8297× ↓↓ (regression, dead branch)
PTX route. v8 + a 3rd in-flight K-tile (deeper prefetch) to hide global latency on
big shapes. Needs 60KB shared (>48KB static cap) → moved As/Bs to dynamic shared →
**1 block/SM, occupancy collapses**. All shapes drop uniformly to ~0.81–0.87 (vs v8's
0.96 small / 0.91 big) — the occupancy loss swamps the prefetch gain. **Same wall as
v5's BK=64.** Correct, but a clear regression. **Lesson (→ skill card):** on sm_120,
deeper pipeline / bigger shared trades occupancy for reuse and *loses*; the next
lever must be **occupancy-neutral** (swizzle, register reduction), not more shared.

### v10_nopad — drop the BK padding (BKP 40→32) — 0.8997× ↓ (regression)
PTX route, occupancy probe. Less shared (32KB → maybe 3 blocks/SM), but no padding
reintroduces `ldmatrix` bank conflicts that cost MORE than the occupancy gained
(all shapes drop ~3pp). **Lesson:** the `+8` pad is doing real work on the raw-PTX
path too — you can't just shrink shared for occupancy.

### v11_smalltile — 128×64 tile + 3-stage pipeline — 0.6954× ↓↓↓ (worst)
PTX route. An **independent fresh-context reviewer's top pick**: shrink the block so
`acc[4][2]`=32 regs and a 3-stage pipeline fits at 2 blocks/SM. Predicted 0.93–0.95×.
Benched **0.70×** — the smaller N-tile crushed **arithmetic intensity** (B re-loaded
by far more blocks), which dominates the occupancy gain. v3/v5's "bigger tile = more
reuse" holds here. **Meta-lesson:** even a clean independent reviewer's confident
prediction was wrong by data — predictions (author's *or* reviewer's) are cheap;
okbench is the only truth.

### Verdict: v8 (0.9245×) is a robust local optimum
Three post-v8 levers all regressed — v9 (+shared → occupancy cliff), v10 (−pad →
bank conflicts), v11 (−tile → arithmetic-intensity collapse). On sm_120 the dominant
force for this GEMM is **arithmetic intensity / reuse**; v8's 128×128 padded
double-buffer balances it, and every "obvious" next lever trades away the thing that
matters. The only untried lever that keeps the tile size is a **proper XOR swizzle**
(occupancy-neutral, shaves `ldmatrix` bank conflicts; maybe 3–8%) — hard, uncertain.
cuBLAS's last ~8% is near-optimal hand-tuned SASS; matching it is brutal and likely
not worth more flailing. **The deliverable here is the ladder + the three
failed-branch lessons + the meta-lesson, not one more percent.**
