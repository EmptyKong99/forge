# skills/ — Agent Skills (SKILL.md procedures)

Each skill is a directory `<skill-name>/SKILL.md` in the standard Agent Skill
format: YAML frontmatter (`name`, `description`) + a markdown procedure the agent
follows. A skill calls `tools/` (executables) and reads `wiki/` (knowledge).
**skill = follow it · tool = run it · wiki = read it.**

> To make a skill actually invocable inside Claude Code it must live under
> `.claude/skills/<name>/SKILL.md`; the copies here are the repo's curated source
> of truth (symlink/copy into `.claude/skills/` to enable).

## Index
| skill | description |
|---|---|
| `use-ptx-instruction/` | turn an unfamiliar PTX instruction into a verified, working kernel + a wiki card |

## Planned
- `optimize-kernel/` — the AVO loop: bench → find bottleneck (profile tool) →
  pick next lever → re-bench → keep best → stop on plateau.
