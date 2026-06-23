# Fact / PTX: `cp.async` global→shared async copy (sm_80+)

**Type:** fact (verified). Backed by gemm_bf16_nt v4/v8 on RTX 5090 (sm_120),
okbench-checked (0.88×/0.92×). The CUDA C wrappers, not raw asm, were used.

## What it does
Copies global→shared **without** going through registers, and **without blocking**
the issuing thread — the copy proceeds in the background while the SM does math.
This is what makes a prefetch pipeline possible (load tile t+1 while computing t).

## CUDA C surface (`<cuda_pipeline.h>`) — what forge actually calls
```cpp
__pipeline_memcpy_async(dst_smem, src_global, sizeof(int4));  // 16B = 8 bf16
__pipeline_commit();                 // seal a group of the above
__pipeline_wait_prior(N);            // block until all but newest N groups land
```
- **16B copies** (`int4`) are the sweet spot: one instruction moves 8 bf16, and
  16B is the widest `cp.async` granularity.
- `dst` must be a real shared address; `src` global. Each thread issues its own.

## The double-buffer pattern (v8)
```cpp
load_tile(0);  __pipeline_commit();           // prime
for (t = 0; t < nk; ++t) {
  if (t+1 < nk) { load_tile(t+1); __pipeline_commit(); }  // prefetch next
  __pipeline_wait_prior(t+1 < nk ? 1 : 0);     // keep ≤1 group in flight
  __syncthreads();
  ... ldmatrix + mma on tile t ...
  __syncthreads();
}
```
`wait_prior(1)` = "let the newest 1 group keep loading, block on the rest" → exactly
one tile in flight = double-buffer. `wait_prior(2)` + a 3rd buffer = 3-stage (v9).

## Underlying PTX (for reference)
- `cp.async.cg.shared.global [dst], [src], 16;` — cache-global hint, 16B.
- `cp.async.ca...` — cache-at-all-levels (use for smaller/ reused copies).
- `cp.async.commit_group;` · `cp.async.wait_group N;` · `cp.async.wait_all;`

## ⚠️ Gotchas
- A `cp.async` copy is **not visible** until you `wait` *and* `__syncthreads()` —
  the wait orders the async engine; the barrier orders the threads.
- You must size the wait to your buffer count; `wait_prior(1)` with only 1 buffer
  is a correctness/perf bug. Depth is an occupancy tradeoff →
  `[[pipeline-depth-vs-occupancy]]`.

## SM support
sm_80+ → valid on sm_120. The bulk/TMA successor (`cp.async.bulk`) is sm_90 only
→ `../menu/async-copy-model.md`.

## Cross-refs
- heuristic: `../heuristics/pipeline-depth-vs-occupancy.md`
- menu: `../menu/async-copy-model.md`
