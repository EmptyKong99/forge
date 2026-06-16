# wiki/ — knowledge base (KernelWiki)

Knowledge the agent **reads** (facts + methods), as opposed to `skills/`, which the
agent **runs**. Distilled from real, okbench-verified kernels — not raw doc dumps.

## Layout
- `ptx/` — per-instruction **fact cards**: syntax, register/fragment layouts, SM
  support, gotchas. Each card is verified empirically before it's trusted.
  - `mma-m16n8k16.md` — bf16 tensor-core mma + ldmatrix (used by gemm_bf16_nt v7/v8)
- `methods/` — **playbooks/procedures** the agent follows (instruction-agnostic).
  - `use-ptx-instruction.md` — how to turn any unfamiliar PTX instruction into a
    verified, usable card.

## Why split facts vs methods vs skills
- **wiki/ptx (facts)** — what to look up (a layout, a gotcha).
- **wiki/methods (procedure)** — how to work (the steps the agent runs each loop).
- **skills/ (tools)** — what the agent executes (bench, profile, compile).

Running a method produces facts (new cards); cards make the next run faster.
