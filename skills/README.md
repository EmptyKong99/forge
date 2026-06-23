# skills/ — Agent Skills (SKILL.md procedures)

Each skill is a directory `<skill-name>/SKILL.md` in the standard Agent Skill
format: YAML frontmatter (`name`, `description`) + a markdown procedure the agent
follows. A skill calls `tools/` (executables) and reads `wiki/` (knowledge).
**skill = follow it · tool = run it · wiki = read it.**

> To make a skill actually invocable inside Claude Code it must live under
> `.claude/skills/<name>/SKILL.md`; the copies here are the repo's curated source
> of truth (symlink/copy into `.claude/skills/` to enable).

## Index — three skills, one per kind of knowledge
| skill | produces | description |
|---|---|---|
| `survey-ptx-knowledge/` | `wiki/ptx/menu/` | breadth-first sweep of a HW doc → a menu of what instructions/variants *exist* for a target SM (so technique choice is informed, not just whatever you tripped over) |
| `use-ptx-instruction/` | `wiki/ptx/facts/` | turn ONE unfamiliar PTX instruction into a verified, working kernel + a fact card (okbench as the correctness oracle) |
| `distill-heuristic/` | `wiki/ptx/heuristics/` | turn a cross-variant bench sweep into a regime→technique card — the flip-condition, never a "X failed" verdict |

These mirror the three wiki card kinds (menu / facts / heuristics) and form a loop:
**survey** (what exists) → **use** (make one work, verified) → **distill** (when to
use which). `survey` feeds candidates to `use`; repeated `use` results feed `distill`.

## Planned
- `optimize-kernel/` — the AVO loop: bench → find bottleneck (profile tool) →
  pick next lever → re-bench → keep best → stop on plateau.
