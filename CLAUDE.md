# Project: SGNNET / neuro_graph — Dhiraj (group_id: "dhiraj")

## Project State
SGNNET: sparse O(N·K) graph neural network, Fourier encoding on S^{D-1}, Imagenette testbed.

- **Efficiency record:** 95.52% @ 0.98M FLOPs (0.79% of VGG16 FC), 67K params (0.05%) — step199.
- **D=16 ceiling:** 97.17% @ 1.97M FLOPs — step205/N=4096 and step209/N=8192.
- **Accuracy record:** 97.86% (D=64, step89) — accuracy track PAUSED (FLOPs budget).
- **Efficiency config:** N=2048, D=16, K_hh=2, K_iter=5, α_AH=1.0, α_reflect=0.5, α_turing=0.0.

Active focus: paper validation (ablations, cross-dataset, baselines), CUDA optimization to realize the 116× FLOPs advantage, K_iter reduction.

## Primary Goal — Paper First
Write a paper demonstrating SGNNET as a general-purpose DL architecture more parameter-efficient than transformers. Near-term proxy for viability: replace VGG16's FC layer at ≤1% params AND ≤1% FLOPs — **BOTH MET (step199).** Next milestones: cross-dataset generalization (CIFAR-10), baseline comparisons (pruned VGG, random projection), N-scaling law.

## Core Tenet — Every Claim Needs Evidence
We are writing a paper. **Every hypothesis, speculation, or design choice must be validated by an experiment or cited research paper** — never intuition alone. Create many hypotheses; each hypothesis owes a planned experiment or a reference.

Tag every causal claim:
- **CONFIRMED** — clean ablation, exactly one variable changed, control present
- **HYPOTHESIS** — post-hoc explanation, confounded experiment, or cited-only
- **STALE** — true on a prior arch/scale that has since changed significantly

Rules:
1. Post-hoc explanations for failures are `HYPOTHESIS`, not facts.
2. "KILLED" requires `CONFIRMED` evidence. Otherwise mark `KILLED (unvalidated)` and note the missing control.
3. When the base changes (arch patch, new N, new defaults), old conclusions become `STALE`. Retest load-bearing ones first.
4. Before closing a research direction, verify the evidence is not confounded. One bad confounded experiment does not kill a direction.

## Session Start
1. `mcp__graphiti__search_memory_facts("recent experiments results running", group_ids=["dhiraj"])`
2. Read `learnings/EXPERIMENT_QUEUE.md`
3. `scripts/slot_status.sh` — verify queue matches real tmux state; investigate ghost `RUNNING` entries before launching replacements.
4. Create training monitor cron (20 min, recurring, background Agent — see Training Monitor below).

## Session End
Before closing a session with completed experiments:
1. Mark `DONE` in `learnings/EXPERIMENT_QUEUE.md`.
2. `mcp__graphiti__add_memory` — one episode per completed experiment (step, config, result, verdict).
3. Update relevant `learnings/concepts/*.md` page(s) if findings touch them.
4. `jj describe` with a meaningful message; push if milestone-worthy.

## Training Infrastructure (5 slots, 3 machines)
Full protocol (multi-user safety, lock files, orphan cleanup) in `.claude/skills/sgnnet-research/SKILL.md`.

| Slot | Machine | Device | Working dir | Python |
|---|---|---|---|---|
| `mini_mps`    | Mac Mini (local)    | MPS  | `/Volumes/T9/IndraAstra/dhiraj/neuro_graph`    | `d_env/bin/python3` |
| `mini_cpu`    | Mac Mini (local)    | CPU  | same                                           | same |
| `studio_mps`  | Mac Studio (ssh `mac-studio`) | MPS | `/Users/admin/ml/dhiraj/qwen2_omni/testing` | same |
| `studio_cpu`  | Mac Studio          | CPU  | same                                           | same |
| `5060ti_cuda` | RTX 5060 Ti (ssh `5060ti`) | CUDA | `/home/indra/sgnnet_bench`              | `venv/bin/python3` |

RAM threshold before launch: Mac Studio ≥50 GB, Mac Mini ≥20 GB, 5060ti ≥16 GB. `vm_stat | grep -E 'free|inactive'` → (Pages free + Pages inactive) × 16384 / 1073741824.

**STRICT:** All launches through `scripts/launch_slot.sh <slot> <script> [args]`. The wrapper enforces lock files, cleans stale sessions, prevents collisions between teammates. No bare `tmux new-session`, no `nohup`, no `&`.

**Session prefix:** `${SGNNET_USER:-$(whoami)}-<step_name>`. Undocumented (non-prefixed) sessions may belong to a colleague using a legacy launch — leave them if alive, clean them up if dead.

## Script Hygiene
- Start every new experiment from `scripts/TEMPLATE_experiment.py`. Copy, rename, fill config dict.
- **Smoke-test before launch:** `d_env/bin/python3 scripts/SCRIPT.py --help`. If this fails, fix before launching. Mandatory — catches import bugs, kwarg typos, bad class hierarchies. (Root cause of step66/222/232/306 first-run crashes.)
- **Before writing:** confirm Ref baseline, scale (N/D/K_hh/K_iter), tier (0/1/2), ablation axis (one variable changes vs Ref), device target. If ambiguous, ask — don't infer silently.
- **Design discussions always end with a script + queue entry** in the same session. "We'll script it later" = later never happens.
- **Broken scripts:** log root cause in `learnings/LEARNINGS_ops.md` "Known Failure Modes". Create a follow-up step with the fix. Do not reuse the broken script.

## Tier Protocol (STRICT)
| Tier | Budget | Data | Purpose | Advance rule |
|---|---|---|---|---|
| Tier 0 Scout | 20ep | 50% | Rejection filter | Configs NOT clearly failing → Tier 1 |
| Tier 1 Calibration | 75ep | 50% | Reliable comparison vs Ref | Winner ≥+0.5pp → defaults |
| Tier 2 Validation | 150ep | 100% | Paper-bound only | Only for publication |

Never skip Tier 0. Tier 0 is a **rejection filter, not a top-N selector** — advance every config with positive/neutral delta. Tier 0 predicts winner ~80% of the time (ρ=0.80 over 46 early experiments). 30ep raises to 91%.

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

## Compounding Rule
Mechanisms on the same signal path tend to cancel (gate-death, co-adaptation). Compounding is safe only when mechanisms are **orthogonal** (e.g., topology + signal modification; different W_* matrices). Before compounding, name the signal path of each mechanism; if shared, do an isolation ablation first.

## Training Diagnostics (MANDATORY)
Every training script integrates `src.training.diagnostics.TrainingDiagnostics(model, device, log_every=5)`. Metrics: effective rank of Z, neuron utilization %, W_pos cosine similarity, separability ratio, per-group gradient norms.

Loss/accuracy are lagging indicators. Diagnostics reveal why training works or fails — enabling data-driven next-experiment design.

## Training Monitor
Cron every 20 min, recurring. Created at session start. The cron prompt MUST launch a **background Agent** (`run_in_background=true`) for all tmux/ssh checks — never run these inline in the main context (pollutes conversation). Agent returns `SILENT` when idle; main context sees only notable events.

## Maintaining This File
- Current truth — git tracks history. Do not append logs.
- Target ≤150 lines. Every addition must remove something of equal or lesser value. If over, compress or split (legacy reference → `CLAUDE_REFERENCE.md`).
- Confirm structural changes with the user before editing. Wording/query tweaks don't need confirmation.
- Observe workflow friction across sessions; synthesize before proposing.
