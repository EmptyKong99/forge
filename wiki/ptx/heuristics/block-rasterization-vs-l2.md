# Heuristic / GEMM: L2 threadblock rasterization vs. a large L2

**Type:** heuristic (condition → technique). Provenance: gemm_bf16_nt v13 (default
2D grid) vs v14 (1D grid + GROUP-strip rasterization) on RTX 5090 (sm_120), okbench
`required_5`. v13 = **0.9900×**; v14 = **0.9683–0.9716×** across GROUP ∈ {4,8,16,32}
— a uniform regression, **no GROUP wins**.

## What rasterization does
Remap the linear block id → `(tile_m, tile_n)` so that *concurrent* CTAs cover a
compact 2D region of C (here: GROUP-tall column strips → consecutive CTAs share a
`tile_n` = the same B columns). The classic CUTLASS trick to raise L2 hit rate on
the streamed operand.

| Regime | Rasterize? | Why |
|---|---|---|
| **Small L2 relative to working set** (older GPUs; or problems ≫ L2) | **yes** | default scheduling streams the shared operand from DRAM; forcing locality wins back bandwidth |
| **Large L2 (sm_120 / consumer Blackwell), working set ~fits** | **no** | the L2 already captures the reuse; the manual remap only adds per-block div/mod overhead and disrupts the natural access order |

## The evidence (a clean negative result)
v14 regressed **uniformly, including compute-bound square_4096** (1.0257× →
1.0062×). A compute-bound shape can't benefit from L2 locality, so its 2pp drop is
**pure overhead** — proof the remap cost is real and the L2-reuse benefit is absent
on this hardware. The 5090's large L2 is already doing the job rasterization was
invented to do.

## ⚠️ Not "rasterization is useless"
It's a staple that *wins* on small-L2 architectures and on problems far larger than
L2. It loses **here** because consumer Blackwell pairs a big L2 with these
moderate (≤16384²-ish) shapes. **Lesson:** don't port CUTLASS-era scheduling tricks
to a new arch on faith — the memory hierarchy moved under them. Re-bench; the L2
size is the hidden variable.

## Cross-refs
- `[[tile-size-vs-shape]]`, `[[epilogue-coalescing]]` (the other large-output levers)
- fact: `../facts/smem-swizzle.md` (the v13 champion this was measured against)
