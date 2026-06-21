---
name: distill-heuristic
description: Use AFTER you have benchmarked two or more variants of a kernel (e.g. a champion plus some regressions) and want to bank the lesson. Turns cross-variant okbench results into a condition→technique "heuristic" card — NOT a verdict. Enforces the honesty rule that prevents overfitting to one benchmark's shapes: never write "technique X failed"; write "X loses in regime R because C, and flips when ¬C".
---

# Distill experiments into a heuristic card

**skill = follow it · tool = run it · wiki = read it.** Produces
`wiki/ptx/heuristics/*` cards — the *judgment* layer that fact/menu cards can't hold.

## The trap this skill exists to stop
After a benchmark sweep it is tempting to write "v9 (3-stage) FAILED, v11 (small
tile) FAILED → always use v8." That conclusion is **overfit to the benchmark's
shape distribution.** okbench `required_5` is *all* large big-K matrices — a biased
sample that rewards big tiles and shallow pipelines. A technique that loses there
(small tile, deep pipeline) is the *correct* technique in a regime okbench doesn't
sample (skinny / batched / latency-bound). Banking "X failed" teaches the next
agent to overfit. Bank the **flip-condition** instead.

## Procedure
1. **Collect ≥2 variants that differ on ONE axis** and their per-shape okbench
   numbers (not just the aggregate score — the per-shape texture is where the
   regime signal is). One should usually be the current champion.
2. **Name the axis** as a tradeoff, not a winner: "tile size", "pipeline depth vs
   occupancy", "padding vs swizzle". The card title is the *axis*, never a version.
3. **Find the mechanism, not the verdict.** Why did the loser lose *here*? (e.g.
   "60KB shared → 1 block/SM → occupancy cliff".) The mechanism is what predicts
   the flip.
4. **Write the regime → technique table.** Rows = regimes (large-square / skinny /
   memory-bound / big-shared-arch …). For each, which technique and *why the
   mechanism favors it there*. The champion occupies ONE row, not all of them.
5. **Add the mandatory `⚠️ Not "always X"` block** naming the exact regime where
   the okbench-loser would win. If you can't name one, you haven't found the
   mechanism yet — go back to step 3.
6. **Record provenance + the meta-note.** Which kernels/measurements back it; and
   the standing reminder that predictions (author's *and* reviewers') are cheap and
   have been falsified here — okbench decides each instance, the card only states
   tendencies.

## Honesty rules (hard constraints)
- ❌ "technique X failed / is a dead end" → ✅ "X loses in regime R because C".
- ❌ a card titled after a version (`v9`) → ✅ titled after the axis.
- ❌ a single recommendation → ✅ a regime→technique map with ≥2 populated rows.
- Every heuristic card MUST carry its flip-condition; a card with one row is a
  fact or a verdict, not a heuristic.

## A different op is a regime too (the transfer axis)
The most over-claimed regime row is **"another op."** Two classes of fact behave
oppositely across ops, so the card must state which it is:
- **Instruction facts** (`mma.sync`, `ldmatrix`, `cp.async`, the `.trans` layout) —
  transfer cross-op **cleanly, first-try**. Proof: the gemm v8 mma/ldmatrix/cp.async
  facts landed flash-attention v4 and v7 on the first attempt.
- **Tuning facts** (swizzle-vs-padding, tile size, pipeline depth) — are
  regime-specific, and **"different op" IS a regime**: they do not auto-transfer.
  Proof: `padding-vs-swizzle` — swizzle is the gemm v13 champion lever, but on FA it
  broke correctness *and* lost to padding. Same instruction set, opposite verdict.
So: a tuning card must carry the row "**another op?** → re-derive, don't assume"; an
instruction-fact card should note "transfers cross-op (verified on …)". Never let a
tuning win earned on one op masquerade as a general rule.

## The line between this and the other skills
- **survey** → what exists → `menu/`.
- **use-ptx-instruction** → make one work → `facts/`.
- **distill-heuristic** (this) → when to use which → `heuristics/`.

## Worked examples already in the wiki
- `../wiki/ptx/heuristics/tile-size-vs-shape.md` (v8 vs v11)
- `../wiki/ptx/heuristics/pipeline-depth-vs-occupancy.md` (v8 vs v9)
- `../wiki/ptx/heuristics/padding-vs-swizzle.md` (v8 vs v10)
