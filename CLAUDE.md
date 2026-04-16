# Project: SGNNET / neuro_graph — Dhiraj (group_id: "dhiraj")

## Project State (2026-04-16)
SGNNET: sparse O(N·K) graph neural network, Fourier encoding on S^{D-1}, Imagenette testbed.

- **Efficiency champion:** step605 K=1 soft-KD student — 95.95% @ 0.20M FLOPs (0.16% of VGG16 FC), 34,976 params (0.029%), 12.7µs B=32 wall-time (**5.26× faster than VGG_FC** on RTX 5060 Ti, bench_step608). Pareto-dominates VGG_FC on 4/5 dimensions (loses only peak memory).
- **D=16 ceiling (no distillation):** 97.30% T2 @ 0.98M FLOPs — step291 D / step235 Aug.
- **Standard seed:** SGNNET_SmallWorld with spatial precomputation is the enforced base. Legacy variants (`model.py`, `model_wave.py`, `model_proximity_wave.py`) emit DeprecationWarning. 16× seed FLOP reduction, bit-exact.
- **Current defaults:** N=2048, D=16, K_hh=2, K_iter=5, α_AH=1.0, α_reflect=0.5 (for K=1 student: distil from K=5 ΔW teacher). K_in=15 for N≥4096, K_in=25 at N=2048.

Active focus: (1) N=16384 record: step297 K_in=10+aug T2 running (watching >96.89%); (2) text gap confirmed negative (step410 SST-2, step407 AG News — step411 AG News config sweep running); (3) paper results largely complete — params=34,976 (CONFIRMED, not 67K which was STALE pre-refactor), 0.029% of VGG_FC.

## Primary Goal
**Long-term goal (stubborn):** improve memory footprint + energy efficiency of deep learning in general.
**First milestone (stubborn):** ship a paper. Current paper vehicle is SGNNET as an FC-replacement for VGG16 (≤1% params + ≤5% FLOPs + ≥95% accuracy — all met via step199→step605).
**Vehicle (flexible):** the amount of focus on SGNNET specifically is open for discussion — if a different approach serves the milestone better, pivot. **Evaluate on the full Pareto table (accuracy + params + FLOPs + wall-time + memory), never on accuracy alone.** When SGNNET trails a baseline, first ask *"what closes the gap so SGNNET wins on efficiency?"* — not *"narrow the scope."*

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
4. **Meditation check** — read `learnings/meditations/TRACKER.md`. If DONE count since last meditation ≥ 25, surface a one-line notice ("Meditation due — run now or defer?"). Don't auto-run; user decides. Protocol: `.claude/skills/sgnnet-meditation/SKILL.md`.
5. Create training monitor cron (20 min, recurring, background Agent — see Training Monitor below).

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
