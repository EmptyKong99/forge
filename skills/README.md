# skills/ — procedures the agent follows (markdown how-tos)

A **skill** here = a markdown procedure/playbook (a how-to prompt) the agent
follows to accomplish a task; it calls `tools/` (executables) and reads `wiki/`
(knowledge). **skill = do it · tool = run it · wiki = read it.**

## Index
| skill | what it does |
|---|---|
| `use-ptx-instruction.md` | turn an unfamiliar PTX instruction into a verified, working kernel + a wiki card |

## Planned
- `optimize-kernel.md` — the AVO loop: bench → find bottleneck (profile tool) →
  pick the next lever → re-bench → keep best → stop on plateau.
