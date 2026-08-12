# Operating Contract — Selected Sections of `CLAUDE.md`

This is the operating contract Dhiraj and Claude Code maintain on the SGNNET project. It is a **living document** — edited as patterns emerge from observed friction across sessions, never as a one-time spec. Below are the load-bearing sections; the full file lives at `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/CLAUDE.md` plus the cold-path appendix `CLAUDE_reference.md`.

---

## Primary Goal

**Long-term goal (stubborn):** improve memory footprint + energy efficiency of deep learning.

**First milestone (stubborn):** ship paper. Vehicle = SGNNET as FC-replacement for VGG16 (≤1% params + ≤5% FLOPs + ≥95% accuracy — all met via step199 → step605).

**Vehicle (flexible):** pivot if better approach serves milestone. **Evaluate on full Pareto table (accuracy + params + FLOPs + wall-time + memory), never accuracy alone.** When SGNNET trails baseline, ask *"what closes gap so SGNNET wins on efficiency?"* — not *"narrow scope."*

---

## Core Tenet — Every Claim Needs Evidence

Writing paper. **Every hypothesis, speculation, design choice must be validated by experiment or cited research** — never intuition alone.

Tag every causal claim:

- **CONFIRMED** — clean ablation, one variable changed, control present
- **HYPOTHESIS** — post-hoc explanation, confounded experiment, or cited-only
- **STALE** — true on prior arch/scale that changed significantly

Rules:
1. Post-hoc failure explanations = `HYPOTHESIS`, not facts.
2. "KILLED" requires `CONFIRMED` evidence; else `KILLED (unvalidated)`.
3. Base changes → old conclusions become `STALE` — retest load-bearing ones first.
4. Before closing direction, verify evidence not confounded.

---

## Tier Protocol (STRICT)

| Tier | Budget | Data | Purpose | Advance rule |
|------|--------|------|---------|--------------|
| Tier 0 Scout | 20ep | 50% | Rejection filter | Configs NOT clearly failing → Tier 1 |
| Tier 1 Calibration | 75ep | 50% | Reliable comparison vs Ref | Winner ≥+0.5pp → defaults |
| Tier 2 Validation | 150ep | 100% | Paper-bound only | Only for publication |

Never skip Tier 0. Tier 0 = **rejection filter, not top-N selector** — advance every config with positive/neutral delta. Tier 0 predicts winner ~80% (ρ=0.80 over 46 early experiments); 30ep raises to 91%.

---

## Compounding Rule

Mechanisms on same signal path tend to cancel (gate-death, co-adaptation). Compounding safe only when mechanisms **orthogonal** (e.g., topology + signal modification; different W_* matrices). Before compounding, name signal path of each mechanism; if shared, isolation ablation first.

---

## Session Start (5 steps)

1. `mcp__graphiti__search_memory_facts("recent experiments results running", group_ids=["dhiraj"])`
2. Read `learnings/EXPERIMENT_QUEUE.md`
3. `scripts/slot_status.sh` — verify queue matches real tmux state; investigate ghost `RUNNING` entries before launching replacements.
4. **Meditation check** — read `learnings/meditations/TRACKER.md`. If DONE count since last meditation ≥ 25, surface one-line notice. Don't auto-run; user decides.
5. Create training monitor cron (5 min, recurring, background Agent). 5060ti completes T1 in ~2-4 min; 20 min missed most completions.

## Session End (4 steps)

1. Mark `DONE` in `learnings/EXPERIMENT_QUEUE.md`.
2. `mcp__graphiti__add_memory` — one episode per completed experiment (step, config, result, verdict).
3. Update relevant `learnings/concepts/*.md` if findings touch them.
4. `jj describe` with meaningful message; push if milestone-worthy.

---

## Selected attitude rules (from `MEMORY.md` index)

- **Gap-close attitude:** when trailing baseline: "what closes gap so SGNNET wins?" — NOT scope-narrow to avoid.
- **Stubborn problem solver:** obstacles increase excitement; first principles; never abandon after one failure.
- **Try and test:** try and test before rejecting; never close tool on theory alone.
- **Websearch on stuck:** after 2 failed attempts, immediately web-search using error logs/traces.
- **Action-driven meditation:** POC evidence + ≥3 committed scripts; never pure speculation.
- **Design-to-script rule:** every design discussion ends with runnable script + queue entry before session ends.
- **Never idle:** never let machines sit idle; always have next experiment ready.
- **No duplicate experiments:** single device for routine T0/T1; duplicate only for paper-claim verification.

---

## Maintaining this file

- Current truth — git tracks history. No logs in the file itself.
- Target ≤60 lines in root `CLAUDE.md`, cold-path content in `CLAUDE_reference.md`.
- Confirm structural changes with user before editing. Wording/query tweaks no confirmation needed.
- Observe workflow friction across sessions; synthesize before proposing.

---

The contract is text. The text is loaded into context every session. Patterns that produce friction get encoded into the contract. The contract has been edited approximately 40 times across the project's lifetime. This is what "building with AI" looks like in practice — not prompt engineering, but contract engineering.
