# wiki/ — knowledge base (KernelWiki)

Declarative **facts** the agent reads. Distilled from real, okbench-verified
kernels — not raw doc dumps. **wiki = read it** (vs `skills/` = procedures you
follow, `tools/` = executables you run).

## Layout
- `ptx/` — per-instruction **fact cards**: syntax, register/fragment layouts, SM
  support, gotchas. Each card is verified empirically before it's trusted.
  - `mma-m16n8k16.md` — bf16 tensor-core mma + ldmatrix (used by gemm_bf16_nt v7/v8)

(Procedures live in `skills/`, not here — e.g. *how to* use a new PTX instruction
is `skills/use-ptx-instruction.md`; the *facts* it produces land here.)
