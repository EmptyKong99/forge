---
name: use-ptx-instruction
description: Use when writing or debugging a CUDA kernel that needs raw PTX (e.g. mma.sync, ldmatrix, cp.async). Gives the procedure to turn an unfamiliar PTX instruction into a verified, working kernel — using the okbench bench tool as a correctness oracle and the wiki/ptx cards as the layout source.
---

# Use an unfamiliar PTX instruction in a kernel

A procedure the agent follows. It calls `tools/` (executables) and reads/writes
`wiki/` (knowledge). **skill = follow it · tool = run it · wiki = read it.**

## Procedure

1. **Scope** — pick only the instructions this kernel needs; don't read the whole
   PTX chapter.
2. **Syntax + SM support** — get the per-instruction syntax line and SM-support
   table (these are regular/parseable); confirm the target arch (e.g. `sm_120`) is
   listed.
3. **Layout (the dangerous part)** — do NOT trust plain-text extraction of the
   fragment/register tables. Prefer a verified `wiki/ptx/*` card; if none exists,
   derive from the instruction family and mark it **unverified**.
4. **Write a minimal kernel; benchmark it** with `tools/bench.sh <op> <variant>` —
   correctness is checked against cuBLAS by okbench.
5. **If fast but wrong → it's a layout bug.** Flip ONE variable at a time
   (trans/no-trans, operand order, addressing offset) and re-bench.
6. **Once correct → write/update a `wiki/ptx/` card** (syntax + verified layout +
   the gotcha that bit you + SM support).

Feedback loop: each run adds cards, so step 3 increasingly hits a verified card
instead of trial-and-error.

## Correctness oracle

Low-level layouts are error-prone to read; trust = **doc + empirical
confirmation**, not doc alone. `tools/bench.sh`'s per-shape check vs cuBLAS is the
oracle — write it, bench it, fix from pass/fail. (This caught a B `.trans` bug in
one iteration, with no speed loss.)

## Honest gap → a deliverable

The PTX ISA page is one giant HTML; a generic fetch can't pull the exact
fragment-layout tables (§9.7.15.5). We need a queryable PTX-doc tool (`tools/`,
roadmap) that distills the doc into retrievable per-instruction records. Until then
the verified `wiki/ptx/` cards are the substitute.
