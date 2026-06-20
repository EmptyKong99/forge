# Heuristic / GEMM: cp.async pipeline depth vs. occupancy

**Type:** heuristic (condition → technique). Provenance: gemm_bf16_nt v8 (2-stage)
vs v9 (3-stage) on RTX 5090 (sm_120), okbench `required_5`.

## The tradeoff

A deeper software pipeline (more in-flight K-tiles) hides more global-load
latency — but each extra stage costs one more shared-memory tile, and past the
static 48KB cap you fall to **dynamic shared**, which can drop you to **1 block/SM**.
Occupancy and pipeline depth pull against each other.

| Regime | Prefer | Why |
|---|---|---|
| **Compute-bound, big tile already filling shared** (this GEMM on sm_120) | **shallow** (2-stage double-buffer) | the SM is already busy; a 3rd stage buys latency-hiding you don't need and pays an occupancy cliff you can't afford |
| **Latency-bound** (small K per tile, memory-bound, or small tiles leaving shared headroom) | **deep** (3–5 stage) | math can't hide the DRAM latency; more in-flight tiles keep the tensor cores fed. This is why CUTLASS/Hopper default to deep pipelines |
| **sm_90+ with TMA + large shared** | **deep + cp.async.bulk/mbarrier** | the async-copy hardware and bigger shared budget remove the occupancy penalty that bites on sm_80/sm_120 |

## The evidence

- **v8** 2-stage = **0.9245×**. Shared = 40KB (static) → 2 blocks/SM.
- **v9** 3-stage = **0.8297×** ↓↓. Needed 60KB → dynamic shared → **1 block/SM**;
  *all* shapes dropped uniformly to ~0.81–0.87. The occupancy loss swamped the
  prefetch gain. Same wall as the earlier wmma `v5_bk64` (72KB → 1 block/SM).
- **v15** 3-stage on the *swizzled* base (no pad, 48KB) = **0.8371×** ↓↓. Built to
  test whether v9 only failed from padding bloat — it did **not**: same ~16pp
  collapse even with the leaner tiles. **This is the strong evidence:** deep prefetch
  is wrong for *this* (compute-bound, large-shape, sm_120) GEMM intrinsically, not
  because of one version's shared budget. Don't keep re-trying depth here.

## ⚠️ Not "deep pipelines are bad"

Deep pipelines are **correct and standard** when shared/occupancy budget allows
(small tiles, big-shared archs, latency-bound shapes). v9 lost *here* because v8's
128×128 tile already spent the shared budget on sm_120 — depth and the existing
tile competed for the same 48KB. On an arch with more shared, or with a smaller
tile, the 3rd stage is a win. **The next occupancy-neutral lever on this exact
config is swizzle, not more stages** — see `[[padding-vs-swizzle]]`.

## Cross-refs
- `[[tile-size-vs-shape]]` · `[[padding-vs-swizzle]]`
- menu: `../menu/async-copy-model.md` (cp.async vs cp.async.bulk vs mbarrier)
