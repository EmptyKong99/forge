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

## Tensor-core path — VERIFIED (v4_mma, 9.9× cuDNN, first try)
Both matmuls are `mma.sync.m16n8k16.bf16` and **map onto the gemm NT layout** — the
gemm facts transfer directly:
- **QKᵀ** is literally NT gemm: `S[q,k]=Σ_d Q[q,d]K[k,d]` → Q=A, K=B, **both
  non-trans** (K[keys][d] row-major IS col-major [d,keys], exactly like gemm's B).
- **PV** is NN, but **store V transposed in shared** (`Vt[d][key]`) → V is B non-trans
  too; P is A. (Transposing at the shared-store avoids `ldmatrix.trans`.)

Two FA-specific hard parts (both worked first try, see v4):
1. **Row softmax inside the C-accumulator.** After QKᵀ, S sits in the m16n8 C-frag
   (row=query `group=lane/4`, cols=keys spread across `lane%4`). Each query row's keys
   are held by the **4-lane group** `{4g..4g+3}` → reduce max/sum with
   `__shfl_xor_sync` offsets **1,2** (stays inside the group). The lane owns 2 rows
   (`group`, `group+8`).
2. **Repack P → A-operand.** P (=exp(S), bf16) is in the C-frag layout; the PV mma
   wants it as an A operand. **Write P to shared `Ps[16][BN]`, `ldmatrix` it back.**
   Carry the online-softmax rescale (`O *= corr` per row) before each PV accumulate.

This is the concrete proof that gemm-derived PTX facts generalize across ops (cf.
anvil EXP-005, where injecting them gave the agent a ~1.7× cross-op nudge — by hand
they take FA from a 6.3× SIMT kernel to 9.9×). Open: occupancy (v4 is 1 warp/block).

## Cross-refs
- fact: `mma-m16n8k16.md`, `ldmatrix-family.md`, `cp-async.md`, `smem-swizzle.md`
  (the primitives QKᵀ/PV reuse)
- kernels: `../../kernels/flash_attention_bf16_fwd_bhsd/` (v1 floor, v2 tiled)
