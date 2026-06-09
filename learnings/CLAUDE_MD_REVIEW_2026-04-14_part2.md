<!-- continued from CLAUDE_MD_REVIEW_2026-04-14_part1.md -->

## D. Redundancies / Outdated — Proposed Removals

| Location | Content | Why Remove/Shorten |
|----------|---------|-------------------|
| Project CLAUDE.md, line 97 | `"We'll script it later" is not acceptable — later = never (confirmed by Dhiraj 2026-04-05).` | Date attribution adds no value. Rule stands alone. Remove parenthetical. |
| Project CLAUDE.md, line 127 | `Empirical basis (46 experiments, D=64 arch): 20ep scouts predict the final winner ~80% of the time (Spearman ρ=0.80). 30ep scouts hit 91%.` | D=64 arch context stale — now at D=16 with different dynamics. ρ=0.80 computed on different distribution. Remove D=64 qualifier or update to "~300 experiments". |
| Project CLAUDE.md, lines 141-147 | Full Python snippet for `TrainingDiagnostics` usage | Belongs in script template or `src/training/diagnostics.py` docstring, not CLAUDE.md. Replace with single line: `Integrate \`from src.training.diagnostics import TrainingDiagnostics\` — see module docstring.` |
| Project CLAUDE.md, lines 13-15 | "Two parallel research tracks" with "Accuracy track (N=4096, D=64)" | Accuracy track inactive. Replace with current-focus framing (see Gap 5 above). |
| Project CLAUDE.md, line 8 | Full final efficiency config with script path | Already in EXPERIMENT_QUEUE.md. Duplicate. Keep only key numbers (N, D, accuracy, FLOPs%). |
| Shared CLAUDE.md (IndraAstra), lines 26-36 | "When to add memories" / "When to search" bullet lists | Generic reminders Claude should internalize. Add length without decision criteria. Compress to one line each or remove. |
| STATE.md | Last updated: 2026-03-23 | Should reflect 2026-04-14 and current running experiments. Not CLAUDE.md content but indicates session-end checklist gap. |

---

## E. Style Drift

| Location | Current text | Karpathy equivalent style | Fix |
|----------|-------------|--------------------------|-----|
| Project CLAUDE.md, line 38 | Long tmux STRICT block with 4-item rationale list + incident history | "Use tmux for all launches. No bare nohup/&. tmux ls = ground truth." | Cut rationale list. Keep rule + consequence ("If tmux ls shows nothing, nothing is running"). Remove incident history (git has it). |
| Project CLAUDE.md, lines 131-149 | Diagnostics section: 20 lines including Python snippet, metric list, "why" explanation | "Every script must include TrainingDiagnostics (src/training/diagnostics.py). See module for usage." | Move Python snippet out. Keep metric names as 5-item list max. |
| Project CLAUDE.md, lines 102-113 | Evidence standards: 3 tag definitions + 5 numbered rules = 11 lines | Keep as-is — IS core intellectual contribution of this CLAUDE.md. Don't compress. | No change. |
| Project CLAUDE.md, Session Start (lines 19-24) | 4-item numbered list | Fine as-is. | No change, but add Session End (Gap 7). |
| Shared CLAUDE.md, lines 10-17 | Graphiti tool table with 4 rows | Fine — table format right. | Remove "When to add / When to search" prose below it (see Section D). |

Rough verdict: ~40-50 lines removable from project CLAUDE.md without losing decision-relevant content.

---

## F. Open Questions for the User

1. **Third machine documentation:** What hostname/alias for 5060ti machine? Working directory and tmux path on it? Should it appear in Training Machines table with explicit launch commands (like Studio/Mini), or intentionally informal? (Blocking Gap 3.)

2. **Accuracy track status:** D=64 accuracy work completely parked, or plan to return (e.g., paper baselines at D=64)? If parked, "Two parallel research tracks" section should become paper validation framing. (Blocking Gap 5.)

3. **Karpathy-style skill abstraction:** Their repo defines Claude Code plugin with 4 principles. Should we extract core methodological rules (evidence tagging, tier protocol, compounding rule) into reusable `skills/sgnnet-research/` structure for portability, or project-specific CLAUDE.md sufficient?

4. **STATE.md ownership:** STATE.md not updated since 2026-03-23 despite being in Session Start read list. Deprecate in favor of EXPERIMENT_QUEUE.md as single source of truth for running state? Or Session End checklist enforce updates?

5. **Script templates:** Given recurring pattern of scripts crashing on first run (step66, step306, step222, step232), should canonical `scripts/TEMPLATE_experiment.py` exist that new scripts copy from, pre-wired with correct imports, smoke-testable top-level block, diagnostics integration? Stronger fix than Gap 1's import-check rule.