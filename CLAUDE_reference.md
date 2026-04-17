# CLAUDE_reference.md — cold-path protocols

Referenced from CLAUDE.md. Load on demand (launching, scripting, recording results, session-start cron setup).

## Training Infrastructure (5 slots, 3 machines)
Full protocol (multi-user safety, lock files, orphan cleanup) in `.claude/skills/sgnnet-research/SKILL.md`.

| Slot | Machine | Device | Working dir | Python |
|---|---|---|---|---|
| `mini_mps`    | Mac Mini (local)    | MPS  | `/Volumes/T9/IndraAstra/dhiraj/neuro_graph`    | `d_env/bin/python3` |
| `mini_cpu`    | Mac Mini (local)    | CPU  | same                                           | same |
| `studio_mps`  | Mac Studio (ssh `mac-studio`) | MPS | `/Users/admin/ml/dhiraj/qwen2_omni/testing` | same |
| `studio_cpu`  | Mac Studio          | CPU  | same                                           | same |
| `5060ti_cuda` | RTX 5060 Ti (ssh `5060ti`) | CUDA | `/home/indra/sgnnet_bench`              | `venv/bin/python3` |

RAM threshold before launch: Mac Studio ≥50 GB, Mac Mini ≥20 GB. `vm_stat | grep -E 'free|inactive'` → (Pages free + Pages inactive) × 16384 / 1073741824.
**5060ti:** use `nvidia-smi` — if no process listed, slot is free. No RAM watermark needed.

**STRICT:** All launches through `scripts/launch_slot.sh <slot> <script> [args]`. The wrapper enforces lock files, cleans stale sessions, prevents collisions between teammates. No bare `tmux new-session`, no `nohup`, no `&`.

**Session prefix:** `${SGNNET_USER:-$(whoami)}-<step_name>`. Undocumented (non-prefixed) sessions may belong to a colleague using a legacy launch — leave them if alive, clean them up if dead.

## Script Hygiene
- Start every new experiment from `scripts/TEMPLATE_experiment.py`. Copy, rename, fill config dict.
- **Smoke-test before launch:** `d_env/bin/python3 scripts/SCRIPT.py --help`. If this fails, fix before launching. Mandatory — catches import bugs, kwarg typos, bad class hierarchies. (Root cause of step66/222/232/306 first-run crashes.)
- **Before writing:** confirm Ref baseline, scale (N/D/K_hh/K_iter), tier (0/1/2), ablation axis (one variable changes vs Ref), device target. If ambiguous, ask — don't infer silently.
- **Design discussions always end with a script + queue entry** in the same session. "We'll script it later" = later never happens.
- **Broken scripts:** log root cause in `learnings/LEARNINGS_ops.md` "Known Failure Modes". Create a follow-up step with the fix. Do not reuse the broken script.

## Knowledge Persistence — Triple Write
Every result/decision lands in **three** places:

| Layer | Purpose |
|---|---|
| Graphiti | Semantic memory across sessions (`mcp__graphiti__add_memory`, `group_id="dhiraj"`) |
| `learnings/LEARNINGS_*.md` | Append-only audit trail (dated entries) |
| `learnings/concepts/*.md` | Per-concept wiki, always current (use `[[concept_name]]` cross-links) |

Also update `learnings/EXPERIMENT_QUEUE.md` (live status) and `learnings/INDEX.md` (catalog).

**Before designing any experiment:** search Graphiti for prior attempts. Never retest without a new variable.

**Propagation rule:** when an experiment completes with a notable result, update pending/future scripts in-place — don't wait for all results.

**File size rule:** no file in `learnings/` exceeds 250 lines. Split with an index when approaching the limit.

**Paper tracking:** notable findings → `learnings/paper/findings_log.md`. Gaps → `learnings/paper/baselines_needed.md`.

## Training Diagnostics (MANDATORY)
Every training script integrates `src.training.diagnostics.TrainingDiagnostics(model, device, log_every=5)`. Metrics: effective rank of Z, neuron utilization %, W_pos cosine similarity, separability ratio, per-group gradient norms.

Loss/accuracy are lagging indicators. Diagnostics reveal why training works or fails — enabling data-driven next-experiment design.

## Training Monitor
Cron every 20 min, recurring. Created at session start. The cron prompt MUST launch a **background Agent** (`run_in_background=true`) for all tmux/ssh checks — never run these inline in the main context (pollutes conversation). Agent returns `SILENT` when idle; main context sees only notable events.
