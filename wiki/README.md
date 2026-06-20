# wiki/ — knowledge base (KernelWiki)

Knowledge the agent reads. **wiki = read it** (vs `skills/` = procedures you
follow, `tools/` = executables you run). Three *kinds* of knowledge, three subdirs
under `ptx/` — keeping them separate is what stops the wiki from collapsing into
"a trace of whatever the last kernel happened to hit."

## Three kinds of card (`ptx/`)

| Dir | Kind | Trust | Built by |
|---|---|---|---|
| `ptx/facts/` | **fact** — exact syntax, register/fragment layout, SM support, the gotcha that bit us | **verified** (okbench-backed, carries provenance) | `skills/use-ptx-instruction` |
| `ptx/menu/` | **menu** — breadth: what instructions/variants *exist* for a target SM | **UNVERIFIED** pointers (a map, not a recipe) | `skills/survey-ptx-knowledge` |
| `ptx/heuristics/` | **heuristic** — regime → technique judgment (when to use which) | **conditional** (tendency + flip-condition, decided per-instance by okbench) | `skills/distill-heuristic` |

The split matters: a *fact* you can verify true/false; a *heuristic* you cannot —
it flips with the workload, so it must carry its flip-condition, never a verdict.
Conflating them is how a kernel agent overfits one benchmark's shapes.

## Current contents
- `ptx/facts/` — `mma-m16n8k16.md`, `cp-async.md`, `ldmatrix-family.md`,
  `stmatrix.md` (backed by gemm_bf16_nt v4/v7/v8/v12 on RTX 5090 sm_120).
- `ptx/menu/` — `warp-matrix-mma.md`, `async-copy-model.md`,
  `smem-layout-swizzle.md` (swept from PTX ISA §9.7).
- `ptx/heuristics/` — `tile-size-vs-shape.md`, `pipeline-depth-vs-occupancy.md`,
  `padding-vs-swizzle.md`, `epilogue-coalescing.md` (distilled from gemm v8 vs
  v9/v10/v11/v12).

The `stmatrix.md` fact + `epilogue-coalescing.md` heuristic are the output of one
full **survey→use→distill** loop: stmatrix was an UNVERIFIED `menu/` entry → wrote
gemm v12 to use it → okbench verified it (correct, 0.9598×, new champion) → distilled
the layout (fact) and the when-to-coalesce judgment (heuristic). The loop closes.

(Procedures live in `skills/`, not here — *how to* survey/verify/distill is a skill;
the *cards* they produce land here.)
