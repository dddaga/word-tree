<!-- continued from CLAUDE_MD_REVIEW_2026-04-14_part1.md -->

## D. Redundancies / Outdated — Proposed Removals

| Location | Content | Why Remove/Shorten |
|----------|---------|-------------------|
| Project CLAUDE.md, line 97 | `"We'll script it later" is not acceptable — later = never (confirmed by Dhiraj 2026-04-05).` | The date attribution adds no value. The rule stands alone. Remove the parenthetical. |
| Project CLAUDE.md, line 127 | `Empirical basis (46 experiments, D=64 arch): 20ep scouts predict the final winner ~80% of the time (Spearman ρ=0.80). 30ep scouts hit 91%.` | The D=64 arch context is stale — we're now at D=16 with different dynamics. The ρ=0.80 figure was computed on a different distribution. Remove the D=64 qualifier or update to "~300 experiments". |
| Project CLAUDE.md, lines 141-147 | Full Python code snippet for `TrainingDiagnostics` usage | This belongs in the script template or `src/training/diagnostics.py` docstring, not in CLAUDE.md. Replace with a single line: `Integrate \`from src.training.diagnostics import TrainingDiagnostics\` — see module docstring.` |
| Project CLAUDE.md, lines 13-15 | "Two parallel research tracks" with "Accuracy track (N=4096, D=64)" | Accuracy track is inactive. Replace with current-focus framing (see Gap 5 above). |
| Project CLAUDE.md, line 8 | Full final efficiency config with script path | Already in EXPERIMENT_QUEUE.md. Duplicate. Keep only the key numbers (N, D, accuracy, FLOPs%). |
| Shared CLAUDE.md (IndraAstra), lines 26-36 | "When to add memories" / "When to search" bullet lists | These are generic reminders that Claude should internalize. They add length without adding decision criteria. Compress to one line each or remove. |
| STATE.md | Last updated: 2026-03-23 | Should reflect 2026-04-14 and current running experiments. Not CLAUDE.md content but indicates the session-end checklist gap. |

---

## E. Style Drift

| Location | Current text | Karpathy equivalent style | Fix |
|----------|-------------|--------------------------|-----|
| Project CLAUDE.md, line 38 | Long tmux STRICT block with 4-item rationale list + incident history | "Use tmux for all launches. No bare nohup/&. tmux ls = ground truth." | Cut rationale list. Keep rule + consequence ("If tmux ls shows nothing, nothing is running"). Remove incident history (git has it). |
| Project CLAUDE.md, lines 131-149 | Diagnostics section: 20 lines including Python snippet, metric list, "why" explanation | "Every script must include TrainingDiagnostics (src/training/diagnostics.py). See module for usage." | Move Python snippet out. Keep metric names as a 5-item list max. |
| Project CLAUDE.md, lines 102-113 | Evidence standards: 3 tag definitions + 5 numbered rules = 11 lines | Keep as-is — this IS the core intellectual contribution of this CLAUDE.md. Don't compress. | No change. |
| Project CLAUDE.md, Session Start (lines 19-24) | 4-item numbered list | Fine as-is. | No change, but add Session End (Gap 7). |
| Shared CLAUDE.md, lines 10-17 | Graphiti tool table with 4 rows | Fine — table format is right. | Remove the "When to add / When to search" prose below it (see Section D). |

Rough verdict: ~40-50 lines can be removed from the project CLAUDE.md without losing any decision-relevant content.

---

## F. Open Questions for the User

1. **Third machine documentation:** What is the hostname/alias for the 5060ti machine? What is the working directory and tmux path on it? Should it appear in the Training Machines table with explicit launch commands (like Studio/Mini), or is it intentionally informal? (Blocking Gap 3.)

2. **Accuracy track status:** Is D=64 accuracy work completely parked, or is there a plan to return to it (e.g., for paper baselines at D=64)? If parked, the "Two parallel research tracks" section should be replaced with the paper validation framing. (Blocking Gap 5.)

3. **Karpathy-style skill abstraction:** Their repo defines a Claude Code plugin with 4 principles. Should we extract our core methodological rules (evidence tagging, tier protocol, compounding rule) into a reusable `skills/sgnnet-research/` structure for portability to future projects, or is project-specific CLAUDE.md sufficient?

4. **STATE.md ownership:** STATE.md hasn't been updated since 2026-03-23 despite being in the Session Start read list. Should it be deprecated in favor of EXPERIMENT_QUEUE.md as the single source of truth for running state? Or should the Session End checklist enforce updates?

5. **Script templates:** Given the recurring pattern of scripts crashing on first run (step66, step306, step222, step232), should there be a canonical `scripts/TEMPLATE_experiment.py` that new scripts copy from, pre-wired with correct imports, smoke-testable top-level block, and diagnostics integration? This is a stronger fix than Gap 1's import-check rule.
