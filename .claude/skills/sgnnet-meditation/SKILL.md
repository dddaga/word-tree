---
name: sgnnet-meditation
description: Periodic action-driven reflection protocol for the neuro_graph research program. Re-anchors on the goal (improve memory footprint + energy efficiency of DL in general), reviews the last ~25 experiments, checks for drift and stuck-ness, generates reframes, **validates each with a ≤5-min POC in the conversation**, and closes with ≥3 committed experiment scripts + queue entries. Not speculation — tinkering with output. Invoke every 25 concluded experiments, or on demand when the user says "meditate" / "run a meditation" / "opus meditation".
---

# SGNNET Meditation — Strategic Reflection Protocol

A structured step back from tactical experiment-running. Produced by the prior 2026-04-15 three-model meditation which reopened the K=1 distillation direction (→ step604/605/606 winners) and named the dispatch-bound wall-time bottleneck. Make it a habit, not a one-off.

---

## 0. Core Value — The Anchor

> **Be stubborn with the goal. Be flexible with the approach.**

- **Goal (stubborn):** Improve memory footprint and energy efficiency of deep learning in general. SGNNET is the current vehicle, but the goal is architecture-agnostic. If SGNNET hits a wall the *answer is not* "tune SGNNET harder" — it is "what approach serves the goal right now?"
- **Approach (flexible):** specific mechanisms (ΔW, AH, polarizer), specific scales (N=2048, D=16), specific datasets (Imagenette), specific hardware (5060ti). All negotiable. All replaceable.

Every meditation checks both axes:
- **Drift check:** have we been flexible where we should have been stubborn? (Goal forgotten, side-quests taking over.)
- **Stuck check:** have we been stubborn where we should have been flexible? (Approach defended past the point of evidence.)

## 0.5 Second Law — Action Over Speculation

**A meditation that only produces speculation is a failed meditation.** The discipline is:

1. **Tinker while you meditate.** Each reframe and each candidate is pressure-tested with a ≤ 5-minute POC — a small script, a random-weight forward pass, a shape check, an analytic FLOP count, a fixture test. If a reframe sounds deep but can't survive a 30-line sanity check, it was speculation, not insight.
2. **No candidate graduates to P0 without evidence.** Evidence = one of: (a) a working POC snippet run in the meditation, (b) an existing codepath already at hand, (c) a cited paper with matched setup. "It should work" is not evidence.
3. **Output = a queue, not a wishlist.** Every meditation closes with ≥ 3 experiments appended to `EXPERIMENT_QUEUE.md` with step numbers, scripts named, slot/tier decided, and first-launch smoke-test done. If the queue did not move, the meditation did not happen.

The POC sandbox is explicit — see Stage 5b below. Treat the meditation like a research journal entry where the journal is a Jupyter notebook, not a notepad.

---

## 1. When to Meditate

| Trigger | Source | Action |
|---|---|---|
| **Auto — every 25 DONE experiments** | `learnings/meditations/TRACKER.md` vs `learnings/EXPERIMENT_QUEUE.md` DONE count | On session start, check delta; if ≥ 25, surface a notice before launching new work |
| **On demand** | User says "meditate", "run a meditation", "opus meditation", "step back", "time to reflect" | Run immediately |
| **Crisis trigger** | A paper-critical claim falsified (e.g. MLP_37 beat step199 on 2026-04-15), or ≥ 3 consecutive failed directions | Run before next experiment |

The auto-trigger is a *notice*, not a forced halt. User can defer with "not yet — queue it after stepXXX."

---

## 2. The Meditation Protocol (7 Stages)

Write one output file per meditation: `learnings/meditations/YYYY-MM-DD_NNN.md` (NNN = sequence number). Each stage is a section. Keep the whole doc ≤ 400 lines. Skip stages only with a one-line reason.

### Stage 1 — Goal Re-anchor (3–5 lines)
State the goal in the user's current words (ask if unsure). Explicitly contrast with whatever surrogate goal the last 25 experiments have been optimizing. *If the surrogate ≠ the goal, flag it in bold.*

### Stage 2 — Evidence Ledger (since last meditation)
Tight table. One row per notable CONFIRMED or HYPOTHESIS finding, one row per closed (or re-opened) direction. Columns: step, finding, tag, implication. No editorializing — only what the data says.

### Stage 3 — Drift Check (goal-backward)
For each major thread of work in the period, one line: *does this thread serve the goal, or has it drifted?* A thread may be valuable science but off-mission — say so. Recommend continue / pause / reframe.

### Stage 4 — Stuck Check (approach-forward)
For each mechanism / dataset / scale we've defended for ≥ 5 experiments, ask:
- What evidence would change our mind?
- Have we seen any of that evidence and explained it away?
- Is there an orthogonal approach we've avoided because it felt "not-SGNNET"?

Name anything defended by identity rather than evidence.

### Stage 5 — Reframes (5–7)
Each reframe is *one sentence*: "What if the real X is actually Y?" Good reframes invert a load-bearing assumption. Examples from 2026-04-15: *"What if K_iter loops are a DEQ fixed-point iteration, not a sequential mechanism?"* (→ reopened K=1 distillation, later 5.26× wall-time win.)

Bad reframes are tactical ("what if we try D=32?"). Those belong in the queue.

### Stage 5b — POC Sandbox (MANDATORY)
For each reframe from Stage 5, run a ≤ 5-minute POC *right now, in the conversation*. Acceptable POCs:

- A short Python/shell snippet that verifies the math or the shape (`python -c "..."` inline).
- A forward pass on random tensors to confirm the mechanism is expressible.
- An analytic FLOP/param count tabulated against the current best.
- A grep of the codebase to prove (or disprove) that we tried something adjacent already.
- A citation to a paper with the matched setup, verified by reading the abstract + method (WebFetch).

Record outcome per reframe: **survived / killed / inconclusive**. A reframe that survives the sandbox graduates to Stage 6. A reframe that's killed is logged with the killing snippet — future meditations read this to avoid re-raising the same idea.

If none of the reframes survive the sandbox, the meditation fails *honestly* — say so and continue with the existing roadmap rather than manufacturing fake signal.

### Stage 6 — Candidate Directions (15–25)
Short list. Each line: `ID | description | cost | expected signal | risk of zero | POC evidence`. The POC evidence column is non-empty for every row — even if it's just "shape-check passed" or "matches step237 existing code". Prioritize candidates that are *cheap to run and high-information-per-experiment*. A good candidate list is uncomfortable — it should include 2–3 ideas that initially feel wrong. The 2026-04-15 Opus meditation produced 26; several felt silly, one (consistency-DEQ) carried a paper-critical result.

### Stage 7 — Priorities for Next 25 Experiments (ACTIONABLE OUTPUT)
3–5 candidates promoted to P0. Each gets:
- A step number allocated *in this meditation*.
- A script filename committed (file created, even if empty scaffold).
- A slot pre-assigned and tier chosen.
- A smoke-test run: `d_env/bin/python3 scripts/train_stepNNN.py --help` must pass before the meditation closes.
- A one-line entry appended to `EXPERIMENT_QUEUE.md` with status `QUEUED`.

Any candidate not in the top 5 goes to a "Parking Lot" at the bottom of the doc — NOT into the queue. Parking lot is re-reviewed at the next meditation.

**Close criterion:** the meditation is complete when (a) the doc is saved, (b) ≥ 3 scripts exist on disk, (c) the queue has ≥ 3 new P0 entries, (d) TRACKER.md is updated. Not before.

---

## 3. Output Structure (Template)

```markdown
# Meditation NNN — YYYY-MM-DD

**Trigger:** auto-25 | on-demand | crisis
**Period covered:** stepXXX → stepYYY (N DONE experiments)
**Core value:** stubborn with goal, flexible with approach

---

## 1. Goal Re-anchor
(3–5 lines)

## 2. Evidence Ledger
| Step | Finding | Tag | Implication |
|---|---|---|---|

## 3. Drift Check
- Thread A: ...
- Thread B: ...

## 4. Stuck Check
- Mechanism X: defended N steps; change-of-mind evidence = Y
- Approach Z: ...

## 5. Reframes
1. What if ...
2. What if ...
...

## 5b. POC Sandbox — Evidence per Reframe
| Reframe | POC run | Outcome | Notes |
|---|---|---|---|
| 1 | `python -c "..."` output | survived / killed / inconclusive | ... |

## 6. Candidate Directions
| ID | Description | Cost | Expected Signal | Risk of Zero | POC Evidence |
|---|---|---|---|---|---|

## 7. Priorities for Next 25 (ACTIONABLE)
| Step # | Title | Script file | Slot | Tier | Smoke-test | Motivation |
|---|---|---|---|---|---|---|
| stepNNN | ... | `scripts/train_stepNNN_*.py` | 5060ti_cuda | T0 | ✅ / pending | ... |

Close checklist:
- [ ] ≥ 3 scripts written to `scripts/`
- [ ] All listed step scripts pass `--help` smoke test
- [ ] `EXPERIMENT_QUEUE.md` has ≥ 3 new `QUEUED` rows
- [ ] `TRACKER.md` history row appended with date + file path + P0 list
- [ ] Meditation file saved

## Parking Lot
- Candidate C: deferred because ...

---

*Next meditation due at step (last_step + 25). Update `TRACKER.md`.*
```

---

## 4. Tracker File — `learnings/meditations/TRACKER.md`

Lives at `learnings/meditations/TRACKER.md`. Minimal schema — human-readable, line-scannable.

```markdown
# Meditation Tracker

**Last meditation:** 2026-04-15 (Session 9 strategic synthesis, pre-skill)
**Last step at meditation:** step266
**Current DONE count since last:** (auto-computed)
**Next meditation due at:** step291

## History
| # | Date | Period | File |
|---|---|---|---|
| 0 | 2026-04-15 | pre-step266 | learnings/LEARNINGS_strategy_2026_04_15.md |
```

**Counting rule:** a DONE experiment = any row in `EXPERIMENT_QUEUE.md` whose status changed to `DONE` after the last meditation's timestamp. Broken experiments (`DONE (broken)`) count only if they produced learning. Close calls: count it.

---

## 5. Session Start Hook

At session start (after the existing graphiti search + queue read), check:

```bash
# Count DONE experiments since last meditation
grep -c "^| .* DONE " learnings/EXPERIMENT_QUEUE.md  # crude but fine
```

If (current_done_count − last_meditation_count) ≥ 25, surface **exactly one line** to the user:

> *"Meditation due — 25 experiments concluded since the last one. Run now, or defer?"*

Do not block. Do not auto-run. The user decides.

---

## 6. On-Demand Invocation

User triggers: "meditate", "run a meditation", "opus meditation", "time to step back", "let's reflect".

Response pattern:
1. Acknowledge in one line.
2. Open the meditation doc with a timestamped filename.
3. Walk through stages 1–7 *in the main conversation* (not a subagent — the user participates).
4. Save the final doc.
5. Update the tracker.
6. Offer to launch the P0 experiments immediately.

**Do NOT delegate a meditation to a subagent.** The whole point is to reflect *with the main context loaded*. A subagent without the full conversation history will produce generic output.

---

## 7. Anti-Patterns — What a Meditation Is NOT

- **Not a status report.** We have `EXPERIMENT_QUEUE.md` and `session_report` for that. Meditation is generative, not summative.
- **Not a tactical plan.** Tactical lives in the queue. Meditation questions strategy.
- **Not optimistic.** If the last 25 experiments produced no winners, say so in Stage 2 and let Stage 4 be harsh.
- **Not exhaustive.** ≤ 400 lines. If you need more, you've lost focus.
- **Not a one-model exercise.** When genuinely stuck, invoke three meditations in parallel (Haiku / Sonnet / Opus) and synthesize — the 2026-04-15 precedent. But cheap default is single-Opus.

---

## 8. Reference — Prior Meditations

- 2026-04-15: `learnings/LEARNINGS_strategy_2026_04_15.md` (three-model synthesis). Output: reopened K=1 distillation, named dispatch-bound bottleneck, 26 candidates. Direct ancestor of step604/605/606/bench_step608 wins.

---

## Quick Reference

```bash
# Check if meditation is due
tail -5 learnings/meditations/TRACKER.md

# Start a new meditation (manual)
# 1. Decide: DONE count since last → period covered → next step #
# 2. Create learnings/meditations/$(date +%Y-%m-%d)_NNN.md from template above
# 3. Work through stages 1–7 in main conversation
# 4. Update TRACKER.md last_step + date + row in history
# 5. Promote P0 candidates to EXPERIMENT_QUEUE.md with step numbers
```
