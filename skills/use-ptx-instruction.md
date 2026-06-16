# Skill: use an unfamiliar PTX instruction in a kernel

A **skill** = a procedure the agent follows (a how-to prompt), which calls `tools/`
and reads `wiki/`. This one turns any unfamiliar PTX instruction into a verified,
usable kernel + a wiki fact card. (Skill = do it; wiki = read it; tool = run it.)

## Procedure
1. **Scope** — pick only the instructions this kernel needs; don't read the whole chapter.
2. **Syntax + SM support** — from the per-instruction syntax line and SM table
   (regular, parseable); confirm the target arch (e.g. sm_120) is listed.
3. **Layout** (the dangerous part) — do NOT trust plain-text extraction of the
   fragment/register tables. Prefer a verified `wiki/ptx/*` card; if none, derive
   from the instruction family and mark it **unverified**.
4. **Write a minimal kernel; benchmark it** with `tools/bench.sh` — correctness is
   checked against cuBLAS by okbench.
5. **If fast but wrong → it's a layout bug.** Flip ONE variable at a time
   (trans/no-trans, operand order, addressing offset) and re-bench.
6. **Once correct → write/update a `wiki/ptx/` card** (syntax + verified layout +
   the gotcha that bit you + SM support).

Feedback loop: each run adds cards, so step 3 increasingly hits a verified card
instead of trial-and-error.

## Correctness oracle
Low-level layouts are error-prone to read; trust = **doc + empirical confirmation**,
not doc alone. `tools/bench.sh`'s per-shape check vs cuBLAS is the oracle — write
it, bench it, fix from pass/fail. (This caught the B `.trans` bug in one iteration,
no speed loss.)

## Honest gap → a deliverable
The PTX ISA page is one giant HTML; generic fetch can't pull the exact
fragment-layout tables (§9.7.15.5). We need a **queryable PTX-doc tool** (`tools/`,
roadmap) that distills the doc into retrievable per-instruction records. Until then
the verified `wiki/ptx/` cards are the substitute.
