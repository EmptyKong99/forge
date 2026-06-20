---
name: survey-ptx-knowledge
description: Use BEFORE writing a kernel for an unfamiliar op, or when you only know the one instruction you happened to bump into. Breadth-first method to sweep the PTX ISA (or any HW doc) and produce a "menu" of what instructions/variants EXIST for a target SM — so the choice of technique is informed, not just whatever you tripped over. Complements use-ptx-instruction (which verifies ONE) and distill-heuristic (which records WHEN to use each).
---

# Survey a hardware doc into a knowledge menu

**skill = follow it · tool = run it · wiki = read it.** This skill produces
`wiki/ptx/menu/*` cards — the *breadth* layer.

## Why this exists (the failure it fixes)
"Practice-first" (write a kernel, distill what bit you) is great for *correctness*
but it only ever records the instruction you happened to hit. The wiki then becomes
a trace of one path, blind to better instructions that exist. This skill deliberately
goes **breadth-first**: map the whole option space *before* committing, so you don't
overfit to the first technique that worked.

## Procedure
1. **Fix the target SM first.** Everything downstream is gated by it (sm_120 has
   `mma.sync`/`cp.async` but NOT `wgmma`/`cp.async.bulk`/`tcgen05`). A menu without
   an SM filter is a trap — it lists instructions you can't run.
2. **Sweep the doc by *family*, not by your immediate need.** For PTX §9.7 the
   families are: warp-matrix (`wmma`/`mma.sync`/`wgmma`/`tcgen05`), async-copy
   (`cp.async`/`cp.async.bulk`/`mbarrier`), smem-layout (padding/swizzle/descriptors).
   Use WebFetch on the PTX ISA page; ask for *mnemonic + one-line purpose + variants
   + min SM* per family. (The big fragment-layout tables won't survive extraction —
   that's fine, breadth is the goal here, not layouts.)
3. **Write one menu card per family** under `wiki/ptx/menu/`. Each entry: what it
   is, variants/qualifiers, min SM, and an **on-target? yes/no** flag. Mark every
   layout/detail **UNVERIFIED** — a menu card is a map, not a trusted recipe.
4. **Tag the on-target subset.** The agent should be able to read one card and know
   "here are my 3 real options on this GPU," ignoring the off-target Hopper/Blackwell
   datacenter rows.
5. **Hand off, don't trust.** When a kernel needs a specific menu entry, pass it to
   `skills/use-ptx-instruction` to verify it into a `wiki/ptx/facts/` card. Trust is
   created by okbench, never by this survey.

## The line between this and the other skills
- **survey** (this) → *what exists* → `menu/` (unverified breadth).
- **use-ptx-instruction** → *make one work* → `facts/` (verified, exact layout).
- **distill-heuristic** → *when to use which* → `heuristics/` (regime → technique).

## Anti-overfit check (do this before declaring a survey "done")
Ask: "Is there an instruction in this family I'm ignoring only because my last
kernel didn't need it?" If yes, it still belongs in the menu — the next op may need
it. Breadth is the deliverable; the menu is allowed to list things forge has never
run, as long as they're flagged UNVERIFIED + on/off-target.
