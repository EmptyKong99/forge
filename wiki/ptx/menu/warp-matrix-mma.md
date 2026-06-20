# Menu / PTX: warp-level matrix instructions (the full shape/type space)

**Type:** menu (breadth, **mostly UNVERIFIED**). Distilled from the PTX ISA §9.7
(Warp Level Matrix). Purpose: tell the agent *what exists* so it can pick a target,
then verify the chosen one into a `facts/` card. **Only `mma.m16n8k16.bf16` is
empirically verified** here (→ `../facts/mma-m16n8k16.md`); everything else is a
pointer, not a trusted layout.

## The three generations (pick by target SM)

| Family | Min SM | Granularity | Notes |
|---|---|---|---|
| `wmma.*` | sm_70 | warp, opaque fragments | easiest, slowest; wrappers hide layout. forge wmma route plateaued ~0.88× |
| `mma.sync` | sm_80 | warp, explicit registers | the workhorse on sm_80–sm_120 (incl. consumer Blackwell sm_120). forge PTX route uses this → 0.92× |
| `wgmma.mma_async` | sm_90 | **warpgroup** (4 warps), async | Hopper datacenter only; needs matrix descriptors + `wgmma.fence/commit/wait`. **Not** on sm_120 |
| `tcgen05.*` | sm_100 | tensor-core gen5 | Blackwell datacenter only |

## `mma.sync` shape space (sm_80+)

- **m8n8**: k4 (f16/f64), k16, k32, k128
- **m16n8**: k4, k8, **k16**, k32, k64, k128, k256
- Types: `.f16 .f32 .f64 .tf32 .s8 .u8 .e4m3 .e5m2` (fp8 e4m3/e5m2 are sm_89+).
  Accumulate is typically `.f32` (or `.s32` for int).
- **Rule of thumb:** bigger k = more work per instruction = fewer issues, but
  needs more registers (A/B fragments) and shared per step. bf16/fp16 → `k16`;
  int8/fp8 → `k32`; the v7/v8 GEMM uses **m16n8k16.bf16**.

## `ldmatrix` / `stmatrix` (sm_80+)

- `ldmatrix.sync.aligned.m8n8.{x1,x2,x4}[.trans].shared.b16` — load 1/2/4 8×8
  tiles from shared straight into mma-shaped registers. `.trans` transposes on
  load (see the trans gotcha in the facts card — fast but wrong if the data is
  already in the wanted major order).
- `stmatrix` — the store counterpart (registers → shared); useful for epilogue
  re-tiling / writing back through shared. x1/x2/x4, b16. **VERIFIED on sm_120**
  (gemm v12, coalesced epilogue 0.9598×) → `../facts/stmatrix.md`. Note: docs say
  sm_80+ but it historically shipped sm_90 — the verify confirmed sm_120 has it.

## How to use this card
1. Pick the shape/type your op needs from the space above.
2. If it's not the verified `m16n8k16.bf16`, **derive a candidate layout and mark
   it unverified**, then run `skills/use-ptx-instruction` to bench it into a
   `facts/` card. Do NOT trust a layout you only read here.

## Cross-refs
- fact (verified): `../facts/mma-m16n8k16.md`, `../facts/ldmatrix-family.md`
- skill: `skills/use-ptx-instruction` (verify one), `skills/survey-ptx-knowledge` (how this menu was built)
