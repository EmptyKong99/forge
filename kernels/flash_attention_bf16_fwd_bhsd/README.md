# flash_attention_bf16_fwd_bhsd — version history

**Op.** `O = softmax(scale · Q·Kᵀ [+ causal mask]) · V`, BF16 in/out, fp32 accumulate.
Layout **BHSD**: Q `[B,H,S_q,D]`, K/V `[B,H,S_kv,D]`, O `[B,H,S_q,D]`. `required_5` =
D=128, B=1, H=32, S ∈ {512(causal), 512(full), 1024, 4096, 8192(causal)} — i.e.
**4 causal + 1 full**. Reference = PyTorch SDPA(cuDNN). Score = geomean speedup.

## ⚠️ Correctness gate caveat (read this first)
okbench's default `correct` for this op = `allclose_vs_cudnn` at **atol=0.002**, which
is **smaller than 1 bf16 ULP** (0.0156 at magnitude≈2). Two *correct* bf16 kernels
that sum the PV reduction in different block orders disagree by ~1 ULP on a few
large-magnitude elements (causal early queries) → a mathematically-correct kernel
**fails the cuDNN gate**. okbench also computes `sampled_vs_fp32_math_allclose` (vs
the fp32 ground truth, atol 0.008) — **that is the sound gate, and all our versions
pass it on all 5 shapes**. We score against fp32-math; speed is still vs cuDNN. See
`../../wiki/ptx/facts/flash-attention-forward.md`. (Flagged to the team — draft op.)

## Versions

### v1_naive — one thread per query — 1.6× (0.016× cuDNN), floor
Thread per (b,h,query): loop all kv, online softmax, accumulate O. No tiling, no
reuse — re-streams ALL of K/V from global for *every* query. Obviously correct
(passes fp32-math on all shapes), ~60× slower than cuDNN. The correctness floor.

### v2_tiled — K/V cached in shared — 2.5× (0.025× cuDNN)
Process a block of BM=64 queries together: load each BN=32-key K/V tile into shared
**once**, reuse across all 64 queries, online-softmax across tiles; causal early-exits
per query (ordered keys → break once kpos>qpos). +0.6× over v1 from cutting K/V
global traffic ~64×. **Bottleneck now:** still SIMT (scalar Q·K dot) and `acc[128]`
per thread → heavy register spills. **Next lever:** tensor cores (transfer the gemm
`mma`/`ldmatrix` facts to QKᵀ and PV) + warp-level tiling to kill the spills.

### v3_warp — warp-per-query, D split across lanes — 6.3× (0.063× cuDNN)
v2's killers: `acc[128]` per thread (register spills) + 64 threads/block (low
occupancy). v3: one **warp** per query, the 32 lanes split D (lane L owns dims
`{L, L+32, L+64, L+96}` = 4 each) → `acc` is **4 regs/lane, no spill**; 256
threads/block; the Q·K dot is a `__shfl_xor` warp reduction. +2.5× over v2.
**Small shapes now competitive** (fa0 causal 13.1%, fa1 full 7.2%) — but the **big
shapes drag** (s8192 = 3.4%): 8192 sequential keys × a per-key warp-reduce is the
wall. **Lesson:** split the feature dim across the warp to kill per-thread `acc[D]`.
Now beats the naive ladder *and* matches the anvil/DeepSeek agent's median (~7.7%,
EXP-005) on the small shapes — but scalar per-key attention can't touch cuDNN on long
sequences. **Next lever:** tensor cores (16 keys per `mma`, not one per warp-reduce).

### v4_mma — tensor-core QKᵀ and PV — 9.9× (0.099× cuDNN) ← champion
The big jump: both matmuls on `mma.m16n8k16`. One warp owns 16 queries, loops key
tiles of BN=16. **QKᵀ = NT gemm** (Q=A, K=B, both non-trans — the gemm v8 layout
verbatim). **PV**: store V *transposed* in shared (`Vt[d][key]`) so it's B non-trans
too; P (after softmax) is A. The two hard middles, both landed **first try**:
(1) **row softmax inside the C-accumulator** — each query row's keys are split across
a 4-lane group, reduced with `shfl_xor` (offsets 1,2); (2) **P repack** — write P
(bf16) to shared, `ldmatrix` it back as the PV A-operand. +1.6× over v3, and the
**big shapes finally move** (s8192 3.4%→7.3%, s4096→7.0%) because mma does 16 keys at
once instead of one scalar warp-reduce. Correct under fp32 gate (sampled_max_abs
≤0.003). **Beats the anvil/DeepSeek agent** (EXP-005 median 7.7%) — forge is now a
real ceiling. **This is the hand-proof that the gemm PTX facts generalize to FA.**
Still **1 warp/block (32 threads) = terrible occupancy** → v5 is multi-warp +
cp.async + swizzle, lots of headroom.

### v5_block — 4-warp block, shared K/V — 32.6× (0.326× cuDNN, 26 TFLOPS) ← champion
v4 was 1 warp/block = shared-limited, awful occupancy. v5: a **block of 4 warps = 64
queries**; K and Vᵀ load into shared **once per block**, reused by all 4 query-warps.
Per-warp compute is byte-for-byte v4 — only Q/P index by warp, K/V are block-shared.
**+3.3× over v4** (the biggest jump in the ladder): halves shared-per-query (2×
occupancy) *and* gives 4 warps to hide load/ldmatrix/mma latency. fp32-correct on all
5 shapes. Now ~33% of cuDNN — **4× the anvil/DeepSeek agent** (EXP-005 median 7.7%).
**Lesson:** the occupancy/reuse structure (load once, share across warps) beats raw
math tweaks — same instructions as v4, 3.3× faster.

### v6_bigtile — larger key tile (BN 32/64) — 26.7% / 21.0% ↓ (regression)
Hypothesis: bigger BN amortizes per-tile overhead (sync, softmax reduce, P round-trip)
over more keys → fewer synchronous-load stalls. **Falsified:** BN=32 → 26.7%, BN=64 →
21.0%, both **below v5's 32.6%** (BN=16). The extra shared (Ks/Vt/Ps all scale with BN)
**drops occupancy**, and on sm_120 occupancy wins — the **exact same lever as the gemm
ladder** (v5_bk64/v9: bigger shared → occupancy cliff). v5 (BN=16) is the sweet spot.
**Lesson:** don't grow the tile to cut overhead here; the real next lever is hiding
load latency *without* more shared = **cp.async double-buffer** (v7).

### Gap to cuDNN
cuDNN fuses the two matmuls on tensor cores with warp-specialized pipelines. We're at
2.5%; the jump needs (1) tensor-core QKᵀ and PV, (2) the P=exp(S) fragment repacked
between the two mma ops, (3) online softmax in fragment layout. That's the v3 target —
and the hand-proof that the gemm PTX facts generalize here (cf. anvil EXP-005, where
those facts gave the agent a ~1.7× cross-op nudge).
