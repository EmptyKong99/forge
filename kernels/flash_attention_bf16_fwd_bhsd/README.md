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

### Gap to cuDNN
cuDNN fuses the two matmuls on tensor cores with warp-specialized pipelines. We're at
2.5%; the jump needs (1) tensor-core QKᵀ and PV, (2) the P=exp(S) fragment repacked
between the two mma ops, (3) online softmax in fragment layout. That's the v3 target —
and the hand-proof that the gemm PTX facts generalize here (cf. anvil EXP-005, where
those facts gave the agent a ~1.7× cross-op nudge).
