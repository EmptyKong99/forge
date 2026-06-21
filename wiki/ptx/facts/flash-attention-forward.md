# Fact / FlashAttention forward (BHSD, bf16) — structure + the cuDNN-tolerance trap

**Type:** fact (verified). Backed by gemm_bf16_nt's PTX facts (reused) + forge
`flash_attention_bf16_fwd_bhsd` v1/v2 on RTX 5090 (sm_120), fp32-math-checked. This
is the *first* op-specific FA card; the matmul primitives come from the gemm facts.

## The op
`O[b,h,s,:] = Σ_k softmax_k(scale · Q[b,h,s,:]·K[b,h,k,:]) · V[b,h,k,:]`. Causal:
query s attends keys `[0, s]`. Two matmuls (QKᵀ, PV) with a **softmax in between** —
that middle step is what makes it not just two GEMMs.

## Online softmax (the core recurrence — must be numerically stable)
Stream keys in tiles; keep per-query running `m` (max), `l` (denom), `acc[D]`:
```
for each key k:
  s     = scale · dot(Q, K_k)
  m_new = max(m, s)
  corr  = exp(m - m_new)        # rescale old state to the new max
  p     = exp(s - m_new)
  l     = l·corr + p
  acc   = acc·corr + p·V_k      # elementwise over D
  m     = m_new
O = acc / l                      # normalize once at the end
```
**Why:** never materialize the full S×S scores; subtracting the running max keeps
`exp` from overflowing. Reordering keys is mathematically identical (→ see the
tolerance trap below for why bf16 still disagrees by 1 ULP).

## Tiling for reuse (v1→v2 lesson)
Naive = one thread per query re-streams all K/V from global (S× waste). Tile it:
a block of BM queries loads each BN-key K/V tile into **shared once**, reuses across
all BM queries. Causal: cap the key sweep at the block's last query, and per-query
`break` once `kpos>qpos` (keys are ordered). Cut K/V global traffic ~BM×.

## ⚠️ The cuDNN-tolerance trap (cost real time to diagnose)
okbench's default `correct` = `allclose_vs_cudnn` at **atol=0.002 < 1 bf16 ULP**
(0.0156 at magnitude≈2). The PV sum, computed in fp32 then rounded to bf16, depends
on **block-summation order**; two correct kernels disagree by ~1 ULP on a few
large-magnitude elements (causal early queries with few keys). So a
**mathematically-correct kernel fails the cuDNN gate** (full shapes pass — their
outputs are small-magnitude averages, 1 ULP < 0.002). **Use the fp32-math gate**
(`sampled_vs_fp32_math_allclose`, atol 0.008, vs the true answer) — okbench computes
it; all our kernels pass it. Don't chase bit-matching cuDNN: it's closed, version-
dependent, and matching another impl's rounding ≠ being correct. anvil judges FA by
this field (`okbench_runner._CORRECT_FIELD_BY_OP`).

## Tensor-core path (the open rung)
QKᵀ and PV are both `mma.sync.m16n8k16.bf16` (the gemm facts apply directly). The FA-
specific hard part: the first mma yields S in the **C-accumulator fragment layout**;
you apply `exp` to it, then must **repack P (bf16) into the A-operand layout** for the
second mma, all while carrying the online-softmax rescale per row. Not yet built
(v2 is SIMT). This is where the gemm→FA primitive transfer is proven by hand.

## Cross-refs
- fact: `mma-m16n8k16.md`, `ldmatrix-family.md`, `cp-async.md`, `smem-swizzle.md`
  (the primitives QKᵀ/PV reuse)
- kernels: `../../kernels/flash_attention_bf16_fwd_bhsd/` (v1 floor, v2 tiled)
