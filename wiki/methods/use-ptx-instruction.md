# Wiki / Method: using an unfamiliar PTX instruction in a kernel

A repeatable **procedure** (not instruction-specific). Running it produces facts
that get written back as wiki cards under `wiki/ptx/`. Procedure and knowledge are
separate on purpose: the agent *executes* this method and *looks up* the cards.

## Procedure
1. **Scope** — pick only the few instructions this kernel needs; don't read the
   whole chapter.
2. **Syntax + SM support** — from the per-instruction syntax line and the SM
   table (these are regular and parseable); confirm the target arch (e.g. sm_120)
   is listed.
3. **Layout** (the dangerous part) — do NOT trust plain-text extraction of the
   fragment/register tables. Prefer a verified `wiki/ptx/*` card; if none, derive
   from the instruction family and mark it **unverified**.
4. **Write a minimal kernel; bench with okbench** (`skills/bench.sh`) — correctness
   is checked against cuBLAS.
5. **If fast but wrong → it's a layout bug.** Flip ONE variable at a time
   (trans/no-trans, operand order, addressing offset) and re-bench.
6. **Once correct → write/update a `wiki/ptx/` card** (syntax + verified layout +
   the gotcha that bit you + SM support).

Positive feedback: each run adds cards, so step 3 increasingly hits a verified
card instead of trial-and-error.

## Correctness oracle
Low-level layouts are error-prone to read; trust = **doc + empirical
confirmation**, not doc alone. okbench's per-shape check vs cuBLAS is the oracle —
write it, bench it, fix from pass/fail. (This caught the B `.trans` bug in one
iteration, with no speed loss.)

## Honest gap → a deliverable
The PTX ISA page is one giant HTML; generic web-fetch can't pull the exact
fragment-layout tables (§9.7.15.5). So we need a **queryable PTX-doc tool** that
distills the doc into retrievable per-instruction records (syntax + SM table +
layouts) — that tool is itself a roadmap item ("ptx document tools"). Until then,
the verified `wiki/ptx/` cards are the substitute.
