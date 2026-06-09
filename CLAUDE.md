# Project: SGNNET / neuro_graph — Dhiraj (group_id: "dhiraj")

## Project State (2026-04-22)
SGNNET: sparse O(N·K) GNN, Fourier encoding on S^{D-1}, Imagenette testbed.

- **Efficiency champion:** step605 K=1 soft-KD student — 95.95% @ 0.20M FLOPs (0.16% of VGG16 FC), 34,976 params (0.029%), 12.7µs B=32 wall-time (**5.26× faster than VGG_FC** on RTX 5060 Ti, bench_step608). Pareto-dominates VGG_FC on 4/5 dimensions (loses only peak memory).
- **D=16 ceiling (no distillation):** 97.30% T2 @ 0.98M FLOPs — step291 D / step235 Aug.
- **Standard seed:** SGNNET_SmallWorld with spatial precomputation enforced base. Legacy variants (`model.py`, `model_wave.py`, `model_proximity_wave.py`) emit DeprecationWarning. 16× seed FLOP reduction, bit-exact.
- **Current defaults:** N=2048, D=16, K_hh=2, K_iter=5, α_AH=1.0, α_reflect=0.5 (K=1 student: distil from K=5 ΔW teacher). K_in=15 for N≥4096, K_in=25 at N=2048.
- **CIFAR-10 paper claim (CONFIRMED):** 80.57% ±0.12pp (step980 T2, 3 seeds), gap −5.67pp vs Linear 86.24%. T2 tighter than T1 (±0.12 vs ±0.31pp).
- **Paper scope:** vision + audio (negative result) + TS + scaling law — multimodal, Paper 1 decided 2026-04-21.

Active focus: (1) CIFAR-10 aug pipeline — hflip feature extraction running, step929 T1 queued on 5060ti; (2) CIFAR-100 N=8192 T0 running (step927, studio_mps); (3) Audio gap confirmed structural (step926/928) — ΔW-proj NOT close audio gap, paper = honest negative result; step962 isolation (VGG mean-pool) pending; (4) TS experiments: step010–030 series; (5) CNN distiller: step003/004 running; (6) Dynamic routing: gate-death theorem confirmed, all multiplicative-gating variants killed (steps 873–916).

## Primary Goal
**Long-term goal (stubborn):** improve memory footprint + energy efficiency of deep learning.
**First milestone (stubborn):** ship paper. Vehicle = SGNNET as FC-replacement for VGG16 (≤1% params + ≤5% FLOPs + ≥95% accuracy — all met via step199→step605).
**Vehicle (flexible):** pivot if better approach serves milestone. **Evaluate on full Pareto table (accuracy + params + FLOPs + wall-time + memory), never accuracy alone.** When SGNNET trails baseline, ask *"what closes gap so SGNNET wins on efficiency?"* — not *"narrow scope."*

## Core Tenet — Every Claim Needs Evidence
Writing paper. **Every hypothesis, speculation, design choice must be validated by experiment or cited research** — never intuition alone.

Tag every causal claim:
- **CONFIRMED** — clean ablation, one variable changed, control present
- **HYPOTHESIS** — post-hoc explanation, confounded experiment, or cited-only
- **STALE** — true on prior arch/scale that changed significantly

Rules: (1) post-hoc failure explanations = `HYPOTHESIS`, not facts. (2) "KILLED" requires `CONFIRMED` evidence; else `KILLED (unvalidated)`. (3) Base changes → old conclusions become `STALE` — retest load-bearing ones first. (4) Before closing direction, verify evidence not confounded.

## Session Start
1. `mcp__graphiti__search_memory_facts("recent experiments results running", group_ids=["dhiraj"])`
2. Read `learnings/EXPERIMENT_QUEUE.md`
3. `scripts/slot_status.sh` — verify queue matches real tmux state; investigate ghost `RUNNING` entries before launching replacements.
4. **Meditation check** — read `learnings/meditations/TRACKER.md`. If DONE count since last meditation ≥ 25, surface one-line notice. Don't auto-run; user decides. Protocol: `.claude/skills/sgnnet-meditation/SKILL.md`.
5. Create training monitor cron (5 min, recurring, background Agent — see CLAUDE_reference.md "Training Monitor"). 5060ti completes T1 in ~2-4 min; 20 min missed most completions.

## Session End
1. Mark `DONE` in `learnings/EXPERIMENT_QUEUE.md`.
2. `mcp__graphiti__add_memory` — one episode per completed experiment (step, config, result, verdict).
3. Update relevant `learnings/concepts/*.md` if findings touch them.
4. `jj describe` with meaningful message; push if milestone-worthy.

## Tier Protocol (STRICT)
| Tier | Budget | Data | Purpose | Advance rule |
|---|---|---|---|---|
| Tier 0 Scout | 20ep | 50% | Rejection filter | Configs NOT clearly failing → Tier 1 |
| Tier 1 Calibration | 75ep | 50% | Reliable comparison vs Ref | Winner ≥+0.5pp → defaults |
| Tier 2 Validation | 150ep | 100% | Paper-bound only | Only for publication |

Never skip Tier 0. Tier 0 = **rejection filter, not top-N selector** — advance every config with positive/neutral delta. Tier 0 predicts winner ~80% (ρ=0.80 over 46 early experiments); 30ep raises to 91%.

## Compounding Rule
Mechanisms on same signal path tend to cancel (gate-death, co-adaptation). Compounding safe only when mechanisms **orthogonal** (e.g., topology + signal modification; different W_* matrices). Before compounding, name signal path of each mechanism; if shared, isolation ablation first.

## Cold-path protocols
Training infrastructure (slot table, RAM thresholds, launch script), script hygiene, knowledge persistence (triple-write), training diagnostics, training monitor details → read `CLAUDE_reference.md`.

## Maintaining This File
- Current truth — git tracks history. No logs.
- Target ≤60 lines in root, cold-path content in `CLAUDE_reference.md`.
- Confirm structural changes with user before editing. Wording/query tweaks no confirmation needed.
- Observe workflow friction across sessions; synthesize before proposing.