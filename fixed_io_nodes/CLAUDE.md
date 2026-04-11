# Project: word-tree / fixed_io_nodes — Sudarshan (group_id: "sudarshan")

## Project State
NativeNeurographLayer: sparse graph neural network replacing VGG16 FC layers, imagenette-10 classification.
Target: **85.81%** (run6 with FFN). All runs from run10+ drop the FFN — GNN must classify directly.
Active: gradient starvation diagnosed — softmax routing concentration (0.4% of nodes carry 50% gradient). run10 DONE (40.33% ep23). run11 RUNNING (fewer nodes). run12 PENDING (routing_temperature=2.0, single-variable fix).
Core docs: `vgg_training/learnings/EXPERIMENT_QUEUE.md`, `vgg_training/learnings/concepts/architecture.md`, `documentation/` (code-level reference)

**Primary goal:** Prove that a sparse graph neural network can match or exceed VGG16's dense MLP FC layers on imagenette-10 classification with fewer parameters. NativeNeurographLayer is the candidate architecture. imagenette-10 is the testbed.

---

## Session Start
Before responding to the user on every new session or after /clear:
1. `mcp__graphiti__search_memory_facts("recent experiments results running", group_ids=["sudarshan"])`
2. `mcp__graphiti__search_memory_facts("confirmed winners dead ends topology", group_ids=["sudarshan"])`
3. Read `vgg_training/learnings/EXPERIMENT_QUEUE.md`

---

## Training Machine

Mac Mini (local, MPS only). One device, one run at a time.

**Launch training:**
```bash
cd /Volumes/T9/IndraAstra/sudarshan/word-tree/fixed_io_nodes/vgg_training && tmux new-session -d -s runN '/Volumes/T9/IndraAstra/sudarshan/.venv/bin/python -u training.py training_runs/runN/config.yaml 2>&1 | tee training_runs/runN/train.log'
```

Python env: `/Volumes/T9/IndraAstra/sudarshan/.venv`
Working dir for training: `vgg_training/` (training.py resolves paths relative to its own location)

**Check if running:** `tmux ls`
**Attach to session:** `tmux attach -t runN`

---

## Knowledge Persistence

Three layers — always keep in sync:

| Layer | Purpose | How |
|---|---|---|
| Graphiti | Semantic search across sessions | `mcp__graphiti__add_memory` with `group_id="sudarshan"` |
| `EXPERIMENT_QUEUE.md` | Single-file overview of all runs (brief table) | Update run status/result row |
| `training_runs/runN/notes.md` | Per-run detailed notes (config delta, full results, analysis) | Create/update for that run |
| `vgg_training/learnings/LEARNINGS_p1.md` | Cross-cutting design notes only (no per-run content) | Append only for decisions that span multiple runs |
| `vgg_training/learnings/concepts/*.md` | Per-concept mechanism docs (no per-run tracking) | Update mechanism, root cause, fix sections |

| Event | Graphiti | EXPERIMENT_QUEUE | Run notes | Concepts |
|---|---|---|---|---|
| Experiment completed | run#, config delta, result, verdict | Update status + val_best | Create/update notes.md | Update results table if mechanism-relevant |
| Winner confirmed | gain, config, why | Mark status | Update notes.md | Note confirmed default |
| Approach killed | why it failed | Mark KILLED | Update notes.md | Update failure section |
| Design discussion | hypothesis, idea | — | — | Append to relevant concept page |

**File size rule:** No file in `learnings/` should exceed 250 lines. Split when approaching limit.

---

## Experiment Standards

**Before designing any experiment:** `mcp__graphiti__search_memory_facts("<mechanism>", group_ids=["sudarshan"])` — check if already tried.

**Every new run (run10+) requires:**
0. *(If diagnosing a problem)* Run and save a diagnostic analysis script in `vgg_training/` first (see `gradient_starvation_analysis.py` as the template). Reference it in the config comment and EXPERIMENT_QUEUE entry. Skip this step if the run is a straightforward config sweep with no preceding diagnosis.
1. A `config.yaml` in `training_runs/runN/` with the full config (not just deltas)
2. A comment in that config explaining what changed from the previous run and why
3. **Its own training script** — copy the previous run's `training_runN.py` to `training_runs/runN/training_runN.py` and modify it for that run. Never modify a completed run's script.
4. An entry added to `EXPERIMENT_QUEUE.md` before training starts
5. *(If a new parameter was added to shared code)* Update `documentation/architecture.md` config reference table and write to all knowledge layers (Graphiti, EXPERIMENT_QUEUE, notes.md, concepts).

Old runs (run1–run9) used the shared `training.py` and are not to be changed.

**Evidence tagging for all findings:**
- **CONFIRMED**: clean ablation — exactly one variable changed, control present
- **HYPOTHESIS**: post-hoc explanation, confounded experiment, or inferred from indirect evidence
- **STALE**: tested on a prior architecture/config that has since changed significantly

Rules:
1. Post-hoc explanations are HYPOTHESES, not facts. Never promote without a controlled experiment.
2. "KILLED" status requires CONFIRMED evidence. If evidence is HYPOTHESIS or STALE, mark "KILLED (unvalidated)".
3. When base config changes significantly, prior conclusions become STALE. Retest load-bearing ones first.

**Design discussions → scripts (STRICT):**
Every design discussion ends with a `config.yaml` for the next run + an EXPERIMENT_QUEUE entry. "We'll configure it later" = never.

---

## Maintaining This File

This file is current truth — git tracks history. Do not append logs.
- **Update** existing lines when context changes (new best accuracy, active investigation)
- **Add** only when a recurring pattern isn't covered
- **Remove** when an instruction is obsolete
