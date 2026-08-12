# Learnings Highlights

The `learnings/` directory is the long-term audit trail of the SGNNET project — dated design notes, experiment reports, meditations, and per-concept wikis. Four representative artifacts.

---

## 1. `learnings/meditations/2026-04-23_003.md` — severe drift detection

Action-driven meditation file. Re-anchors on the goal, scans 28 experiments since the prior meditation, produces an evidence ledger (CONFIRMED tags per step), runs a drift check across six workstreams, runs a stuck check on three architectural defenses, generates seven reframes, validates each with a ≤5-minute POC in the same conversation, and closes with three committed P0 scripts plus queue entries.

**Why it matters:** This is the meditation that fixed the "compute is running but the manuscript isn't moving" failure mode. Three consecutive meditations had flagged paper drift; this one made the call to stop mechanism search and switch to writing sprint. It also demonstrates the discipline of POC-validating reframes before committing them to priorities — no pure speculation.

---

## 2. `learnings/EXPERIMENT_REPORT.md` — sequential paper trail

The append-only experiment log. Every concluded experiment gets a row: step number, config, result, verdict tag (CONFIRMED / HYPOTHESIS / STALE), 1-2 sentence implication. Roughly 200+ rows by mid-2026-04.

**Why it matters:** This is the single source of truth that survives compaction events and session restarts. When a new session starts, the Graphiti search returns recent episodes, but `EXPERIMENT_REPORT.md` is what gets read in full to reconstruct context. The verdict tags make it trivial to filter: "show me all CONFIRMED kills involving multiplicative gating" returns the gate-death theorem evidence in one grep.

---

## 3. `learnings/LEARNINGS_design_2026_04_15.md` — design conversation captured

A single day's design conversation about ΔW-projection and K=1 student distillation, captured in real time. Includes the user's framing, the proposed mechanism, the ablation plan, the predicted outcome, the actual result, and the verdict.

**Why it matters:** This is what "design discussions always end with a script + queue entry in the same session" looks like in practice. No "we'll think about it later" — the design conversation produced step605, which produced the efficiency champion (95.95% @ 0.20M FLOPs, 5.26× speedup). The traceability from design intent to commit hash to result is direct.

---

## 4. `learnings/paper/` — paper-bound findings + baselines_needed

Subdirectory dedicated to manuscript work. `findings_log.md` aggregates the CONFIRMED claims that go into the paper. `baselines_needed.md` tracks gaps: claims that need a control experiment before they can be published. The paper audit from 2026-04-20 catalogued 7 HIGH-priority manuscript gaps; meditation 003 used this audit to set the writing sprint priorities.

**Why it matters:** Separating "findings worth publishing" from "experiments we ran" is itself a discipline. Many experiments produced informative negative results that didn't make the paper — the audit makes that explicit so the manuscript stays focused on the four empirical laws and the gate-death theorem, rather than padding with everything that ran.

---

## Reading order if you want one file

Read `learnings/meditations/2026-04-23_003.md` first. It is the densest single demonstration of the collaboration's working mode: goal re-anchor, evidence ledger, drift check, stuck check, reframes with POC validation, candidate directions table, and priorities table — all in ~160 lines.
