---
name: use-ptx-instruction
description: Use when writing or debugging a CUDA kernel that needs raw PTX (e.g. mma.sync, ldmatrix, cp.async). Gives the procedure to turn an unfamiliar PTX instruction into a verified, working kernel — using the okbench bench tool as a correctness oracle and the wiki/ptx cards as the layout source.
---

# Use an unfamiliar PTX instruction in a kernel

A procedure the agent follows. It calls `tools/` (executables) and reads/writes
`wiki/` (knowledge). **skill = follow it · tool = run it · wiki = read it.**

## Procedure

1. **Scope** — pick the ONE instruction to verify now. (For the breadth question
   "what instructions even exist for my target SM?" run `survey-ptx-knowledge`
   first; it produces the `wiki/ptx/menu/` map you pick from. This skill verifies
   one menu entry into a fact.)
2. **Syntax + SM support** — get the per-instruction syntax line and SM-support
   table (these are regular/parseable); confirm the target arch (e.g. `sm_120`) is
   listed.
3. **Layout (the dangerous part)** — do NOT trust plain-text extraction of the
   fragment/register tables (nor the `menu/` cards, which are UNVERIFIED). Prefer a
   verified `wiki/ptx/facts/*` card; if none exists, derive from the instruction
   family and mark it **unverified**.
4. **Write a minimal kernel; benchmark it** with `tools/bench.sh <op> <variant>` —
   correctness is checked against cuBLAS by okbench.
5. **If fast but wrong → it's a layout bug.** Flip ONE variable at a time
   (trans/no-trans, operand order, addressing offset) and re-bench.
6. **Once correct → write/update a `wiki/ptx/facts/` card** (syntax + verified
   layout + the gotcha that bit you + SM support). If the sweep also taught you a
   *when-to-use* lesson across variants, that goes to `distill-heuristic`, not here.

Feedback loop: each run adds cards, so step 3 increasingly hits a verified card
instead of trial-and-error.

## Correctness oracle

Low-level layouts are error-prone to read; trust = **doc + empirical
confirmation**, not doc alone. `tools/bench.sh`'s per-shape check vs the reference
(cuBLAS for gemm) is the oracle — write it, bench it, fix from pass/fail. (This
caught a B `.trans` bug in one iteration, with no speed loss.)

### …but first check the gate is sound (the oracle can lie)
A "fail" is only trustworthy if the gate's tolerance is. **The default reference
gate can be wrong:** flash-attention's `allclose_vs_cudnn` uses atol=0.002, which is
**below 1 bf16 ULP** (0.0156) — two *correct* bf16 kernels that sum the reduction in
different orders disagree by ~1 ULP and the gate red-flags a correct kernel. Before
trusting a fail: (1) is the tolerance ≥ 1 ULP at the output magnitude? (2) does the
bench also expose an **fp32-ground-truth** gate (`sampled_vs_fp32_math_allclose`)?
Judge correctness against fp32 math, not against another bf16 reference. If the gate
is unsound, fix/flag it (anvil keys this per-op in `okbench_runner._CORRECT_FIELD_BY_OP`)
— don't burn iterations "fixing" a kernel that was already correct.

## The deliverable is the verified card — not a doc tool

The PTX ISA page is one giant HTML and the exact fragment-layout tables
(§9.7.15.5) don't survive a generic fetch — but the fix is **not** to build a
PTX-doc extraction tool. The methodology is **practice-first**: write the kernel,
let okbench verify it, then distill what you learned (syntax + the layout that
*actually worked* + the gotcha that bit you) into a `wiki/ptx/facts/` card carrying
**provenance** (which kernel/measurement backs it). The doc is raw material; the
verified card — e.g. `wiki/ptx/facts/mma-m16n8k16.md`, backed by v7/v8 on the 5090 —
is the deliverable.
