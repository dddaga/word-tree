---
name: sgnnet-research
description: Experimental research workflow for SGNNET — training-slot coordination across multiple machines and multiple teammates, evidence-tagged findings, tier-based experiment protocol. Invoke when starting an experiment, analyzing results, or onboarding a new teammate to this repo.
---

# SGNNET Research Workflow

Shared infrastructure skill for the SGNNET project. Multiple teammates work across 5 training slots on 3 machines. This skill encodes the safety rules, evidence standards, and coordination protocol so experiments don't collide.

---

## 0. Current State (as of 2026-04-16) — read first

- **Goal (long-term):** improve memory footprint + energy efficiency of DL in general.
- **Milestone 1:** ship a paper. Current vehicle: SGNNET as FC-replacement for VGG16.
- **Efficiency champion:** step605 K=1 soft-KD student — 95.95% @ 0.20M FLOPs, 12.7µs B=32 (**5.26× faster than VGG_FC**, 3418× fewer params). See `bench_step608` for the full Pareto table.
- **Standard model:** `src/sgnnet/model_smallworld.py::SGNNET_SmallWorld` with spatial precomputation. Legacy `model.py` / `model_wave.py` / `model_proximity_wave.py` emit DeprecationWarning — do not use in new experiments.
- **Evaluation rule:** always the full Pareto table (accuracy + params + FLOPs + wall-time + memory). Never accuracy alone.
- **Gap-close attitude:** when trailing a baseline on any dimension, first ask *"what closes the gap?"* — not *"narrow the scope."*
- **Meditation:** periodic action-driven reflection, `.claude/skills/sgnnet-meditation/SKILL.md`. Every 25 DONE experiments or on-demand. Always closes with ≥ 3 committed scripts.

---

## 1. Core Principle: Every Claim Needs Evidence

We are writing a paper. Every hypothesis, speculation, or design choice must be validated by **an experiment or a cited research paper** — not intuition. When you log a finding, tag it:

| Tag | Meaning | When to use |
|---|---|---|
| `CONFIRMED` | Clean ablation, exactly one variable changed, control present | Only after a valid comparison |
| `HYPOTHESIS` | Post-hoc explanation, confounded experiment, or cited but untested | Default for any intuitive leap |
| `STALE` | True on a prior arch/scale that has since changed significantly | When base assumptions shift |

Rule: Before closing a research direction (marking "KILLED"), verify the evidence is `CONFIRMED`. Post-hoc explanations of failures are `HYPOTHESIS`, not facts. One confounded experiment is not sufficient to kill a direction.

Create many hypotheses — but every hypothesis owes a planned experiment or a reference.

---

## 2. Training Slots (5 total across 3 machines)

| Slot ID | Machine | Device | Working dir | Python |
|---|---|---|---|---|
| `mini_mps` | Mac Mini (local) | MPS | `/Volumes/T9/IndraAstra/dhiraj/neuro_graph` | `d_env/bin/python3` |
| `mini_cpu` | Mac Mini (local) | CPU | same | same |
| `studio_mps` | Mac Studio (ssh `mac-studio`) | MPS | `/Users/admin/ml/dhiraj/qwen2_omni/testing` | `d_env/bin/python3` |
| `studio_cpu` | Mac Studio | CPU | same | same |
| `5060ti_cuda` | RTX 5060 Ti (ssh `5060ti`) | CUDA | `/home/indra/sgnnet_bench` | `venv/bin/python3` |

Shared RAM check before launch: Mac Studio ≥50 GB free+inactive; Mac Mini ≥20 GB; 5060ti ≥16 GB.

---

## 3. Multi-User Safety — Slot Coordination

**Problem:** Multiple teammates may launch experiments on the same 5 slots. Without coordination, two people can start training on the same device and silently corrupt each other's runs (MPS contention, OOM).

**Solution:** Every launch goes through `scripts/launch_slot.sh`. It maintains a per-slot lock file (`.claude/slots/<slot>.lock`) containing `<user> <session> <pid> <host>`. Tmux session names are prefixed with the user: `<user>-<step_name>`.

### Launch protocol (MANDATORY)

```bash
# Set your user prefix (once per shell session):
export SGNNET_USER="alice"   # or rely on $(whoami)

# Launch:
scripts/launch_slot.sh <slot_id> <script_path> [extra_args]

# Examples:
scripts/launch_slot.sh mini_mps     scripts/train_step400_cifar10_generalization.py
scripts/launch_slot.sh studio_cpu   scripts/train_step321_ah_alpha_sweep.py
scripts/launch_slot.sh 5060ti_cuda  scripts/bench_cuda_optimizations.py
```

The launcher does:
1. Read the lock file for that slot.
2. If pid is alive on the target host → **exit 2, slot occupied**. Pick a different slot.
3. If pid is dead (stale lock) → clean up the orphan tmux session, then launch yours.
4. Write a new lock file `<your_user> <session_name> <pid> <host>` on success.

**Exit codes:** 0 = launched; 1 = usage error; 2 = slot occupied by teammate; 3 = host unreachable / launch failed.

### Check slot status before launching
```bash
scripts/slot_status.sh
```
Shows all 5 slots as `[FREE]`, `[<user>]`, or `[STALE]` plus any undocumented tmux sessions.

### Cleaning up after yourself
The tmux session auto-terminates when the training script exits. The lock file is cleaned up by the `trap EXIT` handler in the session. If your ssh disconnects abruptly, the session survives but the lock may go stale — the next teammate's launcher will detect the stale pid and recover automatically.

### Undocumented sessions
If `scripts/slot_status.sh` shows tmux sessions not matching the `<user>-<step>` pattern, they were launched outside this skill (legacy scripts, manual tmux commands). Policy:
- If the session's process is alive → leave it (teammate using a legacy launch).
- If dead → `tmux kill-session -t <name>` to free it up.

---

## 4. Tier-Based Experiment Protocol

Every new mechanism follows this pyramid. Never skip Tier 0 to jump to Tier 1. Tier 2 (150ep, full data) is reserved for endgame validation.

| Tier | Budget | Data | Purpose | Advance rule |
|---|---|---|---|---|
| **Tier 0 Scout** | 20 epochs | 50% | Rejection filter | All configs not clearly failing → Tier 1 |
| **Tier 1 Calibration** | 75 epochs | 50% | Reliable comparison | Winner ≥ +0.5pp vs Ref → add to defaults |
| **Tier 2 Validation** | 150 epochs | 100% | Final ranking | Only for publication-bound winners |

Tier 0 is a **rejection filter**, not a top-2 selector. Advance all configs with positive or neutral delta.

Empirical basis: 20ep scouts predict the final winner ~80% of the time (Spearman ρ=0.80 measured over 46 early experiments). 30ep hits 91%.

---

## 5. Before Writing Any Script

Before writing a new experiment script, confirm:
- **Reference config:** what is the Ref baseline? (usually step199: N=2048 D=16 K_hh=2 K_iter=5 AH=1.0)
- **Scale:** N, D, K_hh, K_iter — or verify from the efficiency config if not specified
- **Tier:** Tier-0 / Tier-1 / Tier-2 (epochs + data fraction)
- **Ablation axis:** exactly ONE variable changes vs Ref (mechanism isolation)
- **Device target:** will this run on MPS, CPU, or CUDA?

If any of these are ambiguous: **ask.** Do not infer silently. `"We'll figure it out later"` means "later = never".

Start from `scripts/TEMPLATE_experiment.py`. Copy, rename, fill in the config dict.

---

## 6. Script Smoke Test (MANDATORY Before Launch)

Before `scripts/launch_slot.sh` on any new script, verify it parses:

```bash
d_env/bin/python3 scripts/SCRIPT.py --help
```

If that fails, fix the script before launching. Never launch a script you haven't sanity-checked. This catches: import errors, kwargs typos, broken class hierarchies, missing deps.

Root cause of step306, step66, step232, step222 all crashing on first run was skipping this step.

---

## 7. Results Logging (Triple Write)

Every result lands in THREE places:

1. **`results/<script_name>.json`** — raw numbers, written by the script
2. **`learnings/EXPERIMENT_QUEUE.md`** — live status table (update on DONE)
3. **Graphiti memory** — `mcp__graphiti__add_memory` with `group_id="dhiraj"`

For notable findings also update:
- **`learnings/concepts/<concept>.md`** — the consolidated per-concept wiki page
- **`learnings/paper/findings_log.md`** — if the result is paper-worthy

Triple write is non-negotiable. Missing any layer means the finding gets lost in the next session.

---

## 8. Compounding Rule

Mechanisms operating on the same signal path tend to cancel each other (gate-death, co-adaptation). Compounding is only safe when mechanisms are **orthogonal**:

- Topology (K_hh, RigL) + Signal modification (AH, ΔW): usually safe
- Two W_pos-dependent mechanisms: unsafe (double-sparsity)
- Two activation-modifying mechanisms: test with clean ablation first

Before writing a compound config, name the signal path of each mechanism. If they share one, do an isolation ablation first.

---

## 9. Session Start Checklist

At the start of every session or after `/clear`:

1. `mcp__graphiti__search_memory_facts("recent experiments", group_ids=["dhiraj"])`
2. `mcp__graphiti__search_memory_facts("confirmed winners dead ends", group_ids=["dhiraj"])`
3. Read `learnings/EXPERIMENT_QUEUE.md`
4. **Verify queue vs reality**: `scripts/slot_status.sh` — anything listed RUNNING in the queue without a matching active slot is a ghost entry. Investigate before launching replacements.

---

## 10. Session End Checklist

Before closing a session with completed experiments:

1. Mark completed experiments `DONE` in `learnings/EXPERIMENT_QUEUE.md`.
2. `mcp__graphiti__add_memory` — one episode per completed experiment (step, config, result, verdict).
3. Update any concept pages that the result touches.
4. If a script shipped, confirm it smoke-tested clean.
5. `jj describe` a sensible message; push if milestone-worthy.

---

## 11. Centralized Queue Controller

Instead of manually picking a slot and calling `launch_slot.sh`, submit experiments to the shared queue. The controller runs on Mac Mini, round-robins across users, honors device/slot preferences, and dispatches via `launch_slot.sh` when slots free up.

### 11.1 One-time setup (Mac Mini only)

```bash
# Background daemon. Default: :7433, tick every 30s.
nohup d_env/bin/python3 scripts/experiment_controller.py \
    > logs/experiment_controller.log 2>&1 &

# Or use the launchd template for always-on:
# cp scripts/experiment_controller.service.template ~/Library/LaunchAgents/...
```

The controller persists queue state to `.controller/queue.db` (SQLite, WAL mode). Safe to restart — queued entries survive; running entries recover via the `launched_slot` column.

### 11.2 Submitting an experiment

```bash
# Any-device, round-robin picks the next free slot
scripts/queue_submit.sh scripts/train_stepXXX.py --args "--epochs 20"

# CUDA-only
scripts/queue_submit.sh scripts/train_stepXXX.py --device-pref cuda

# Hard slot reservation — preempts any-pref entries on that slot
scripts/queue_submit.sh scripts/bench_stepXXX.py --slot-pref 5060ti_cuda
```

`--priority <int>` (higher runs first within a user's queue) and `--step-name` are also supported.

### 11.3 Identity — `.sgnnet_user` file

Experiments are tagged by user so the scheduler can round-robin fairly. Identity resolution (in order):

1. `--user <name>` CLI flag (one-off override)
2. `SGNNET_USER` env var (session-level override)
3. `.sgnnet_user` file in the current working directory ← **canonical**
4. **First-run interactive prompt** — if none of the above are set, `queue_submit.sh` asks once, suggests the parent directory name as default, and writes `.sgnnet_user` so subsequent runs are silent. Delete the file to re-configure.

**Why this matters:** multiple teammates may log into Mac Mini as the same OS user `indra`. Application-level identity (the `.sgnnet_user` file, one per working tree) is how the scheduler tells their queues apart.

### 11.4 Inspecting the queue

```bash
scripts/queue_list.sh                 # whole queue
scripts/queue_list.sh --user dhiraj   # just one user
curl http://mac-mini.local:7433/status | jq .   # scheduler internals
```

### 11.5 Scheduling rules (one screen)

- **Round-robin across users** — pointer stored in DB; survives controller restart.
- **Slot reservation** — if any queued entry has `slot_pref=X`, slot X will NOT launch any-pref entries once it frees; it waits for the reserved entry. Prevents a specific experiment from getting starved.
- **Device compatibility** — `--device-pref cuda` entries only run on `5060ti_cuda`. `mps` runs on either `*_mps`. `any` runs anywhere.
- **Priority** — within a user's own queue: higher priority first, then FIFO. Round-robin is orthogonal — it picks *which user's turn it is*, not which of their entries.

Full design doc: `docs/experiment_controller.md` (skill dir). Source: `scripts/experiment_controller.py`. Tests: `tests/test_experiment_controller.py` (23 unit tests, all scheduler edge cases).

### 11.6 Installing the skill in a new repo

The skill directory is self-contained. To adopt the workflow in a fresh project:

```bash
# Copy the skill
cp -r .claude/skills/sgnnet-research /path/to/new/repo/.claude/skills/

# Install the runtime scripts (symlinks OK)
cd /path/to/new/repo
ln -sf .claude/skills/sgnnet-research/scripts/queue_submit.sh           scripts/
ln -sf .claude/skills/sgnnet-research/scripts/queue_list.sh             scripts/
ln -sf .claude/skills/sgnnet-research/scripts/experiment_controller.py  scripts/
ln -sf .claude/skills/sgnnet-research/scripts/launch_slot.sh            scripts/
ln -sf .claude/skills/sgnnet-research/scripts/slot_status.sh            scripts/
```

`experiment_controller.py` uses `REPO_ROOT = Path(__file__).resolve().parent.parent`, so as long as it's in `scripts/` of the target repo, it finds the right `launch_slot.sh` and `.controller/queue.db`.

---

## 12. Broken Experiment Protocol

When a step is labeled `DONE (broken)`:

1. Log root cause in `learnings/LEARNINGS_ops.md` under "Known Failure Modes".
2. Create a follow-up step (e.g., `step222b`) in the queue with the fix.
3. Do NOT reuse the broken script without fixing the root cause first.
4. If a paper-critical experiment is broken, it blocks the paper — treat as P0.

---

## Quick Reference — Commands

```bash
# Submit to the shared queue (recommended)
scripts/queue_submit.sh scripts/train_stepXXX.py [--device-pref cuda] [--slot-pref 5060ti_cuda]
scripts/queue_list.sh

# Check slot status (low-level)
scripts/slot_status.sh

# Direct launch (bypasses queue — only when you know the slot is yours)
scripts/launch_slot.sh <slot> scripts/train_stepXXX.py

# Attach to watch live training
tmux attach -t <session_name>                          # local
ssh mac-studio -t /opt/homebrew/bin/tmux attach -t X   # Mac Studio
ssh 5060ti -t /usr/bin/tmux attach -t X                # 5060ti

# Manual cleanup of a stale slot (last resort)
rm .claude/slots/<slot>.lock
tmux kill-session -t <session_name>    # or via ssh

# Smoke test a new script
d_env/bin/python3 scripts/SCRIPT.py --help

# Take a checkpoint before risky changes
jj describe -m "checkpoint: before <thing>"
jj new -m "<thing>"
```
