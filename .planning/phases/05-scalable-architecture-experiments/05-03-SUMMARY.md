---
phase: 05-scalable-architecture-experiments
plan: 03
subsystem: architecture
tags: [sgnnet, reflection-routing, arch-comparison, dead-neurons, smallworld, antihebb]

# Dependency graph
requires:
  - phase: 05-01
    provides: "N-scaling sweep script synced to Mac Studio"
  - phase: 05-02
    provides: "ProximityWave dispatched to Mac Studio"
provides:
  - "SGNNET_Reflection model (alpha_reflect, theta params, dead neuron tracking)"
  - "train_exp4_reflection.py with 4 configs (A=ref, B=leaky, C=hard, D=medium)"
  - "results/arch_comparison.json (12 rows, 3 complete)"
  - "results/arch_comparison.md (partial table with dispatch instructions)"
  - "GA_BEST_D64 in experiment_config.py"
affects: [05-04, 05-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dead neuron tracking per routing step: (Z.norm(dim=-1) < 1e-6).float().mean()"
    - "Sign-conditional routing: Z_struct - alpha_reflect * relu(-Z - theta)"
    - "Partial result tables with pending/queued status in JSON"

key-files:
  created:
    - src/sgnnet/model_reflection.py
    - scripts/train_exp4_reflection.py
    - results/exp4_reflection.json
    - results/arch_comparison.json
    - results/arch_comparison.md
    - learnings/LEARNINGS_phase5_p13_reflection.md
  modified:
    - src/training/experiment_config.py

key-decisions:
  - "Reflection as standalone routing (not wrapped around AntiHebb): cleaner ablation of the mechanism alone"
  - "Z_struct - Z_reflect (minus sign): self-inhibitory direction prevents cascading amplification"
  - "exp4_reflection queued (not dispatched): Mac Studio at 3 concurrent > cap of 2"
  - "arch_comparison.md partial: 3 rows complete, 9 pending step56/exp3/exp4 experiments"
  - "GA_BEST_D64 added to experiment_config.py: was missing despite Plan 01 stating it was added"

# Metrics
duration: 25min
completed: 2026-04-03
---

# Phase 5 Plan 03: Reflection Routing + Architecture Comparison Summary

**Reflection routing mechanism implemented and synced to Mac Studio; architecture comparison table assembled with all available data (3 complete rows, 9 pending)**

## Performance

- **Duration:** 25 min
- **Completed:** 2026-04-03
- **Tasks:** 2
- **Files created/modified:** 7

## Accomplishments

- Implemented `src/sgnnet/model_reflection.py` with `SGNNET_Reflection` class:
  - Wraps any `SGNNET_SmallWorld` base, overrides routing loop
  - Sign-conditional routing: positive activations propagate, strongly negative bounce back
  - Per-step dead neuron tracking with `warnings.warn` if dead_frac > 5%
  - `dead_neuron_report()` method for per-epoch diagnostics
- Wrote `scripts/train_exp4_reflection.py` with 4 configs:
  - Config A: SmallWorld + AntiHebb(0.7) reference (no reflection)
  - Config B: leaky-reflect (alpha=0.1, theta=0.0) — low risk
  - Config C: hard-reflect (alpha=1.0, theta=0.5) — high risk, watch dead_frac
  - Config D: medium-reflect (alpha=0.3, theta=0.0) — intermediate
- Synced all scripts to Mac Studio (`model_reflection.py`, `train_exp4_reflection.py`, `experiment_config.py`)
- Queued dispatch: Mac Studio at 3 concurrent experiments (step29c, step48, step54), cap is 2
- Assembled `results/arch_comparison.json` (12 rows) and `results/arch_comparison.md` with:
  - 3 complete rows from existing step22b/step29 data
  - 9 pending rows with dispatch commands for when experiments complete
- Written `learnings/LEARNINGS_phase5_p13_reflection.md` with hypothesis, design rationale, dead neuron risk analysis, and dispatch instructions

## Task Commits

1. **Task 1: Implement reflection routing and training script** — `89a8dfb` (feat)
2. **Task 2: Assemble architecture comparison table and queue dispatch** — `c8ece6f` (chore)

## Files Created/Modified

- `src/sgnnet/model_reflection.py` — SGNNET_Reflection class: alpha_reflect, theta, dead neuron tracking
- `scripts/train_exp4_reflection.py` — 4-config reflection ablation script
- `src/training/experiment_config.py` — Added GA_BEST_D64 (step29 Config C: 75.24%)
- `results/exp4_reflection.json` — Placeholder (queued; null top1 values)
- `results/arch_comparison.json` — 12-row comparison (3 complete, 9 pending)
- `results/arch_comparison.md` — Markdown table with dispatch instructions
- `learnings/LEARNINGS_phase5_p13_reflection.md` — Hypothesis, design, dead neuron risk, dispatch guide

## Decisions Made

1. **Reflection as standalone routing:** Configs B/C/D use `SGNNET_Reflection` wrapping bare SmallWorld (no Resonant, no AntiHebb). This gives a clean ablation of reflection alone vs AntiHebb alone (Config A). A "reflection + AntiHebb combined" experiment is a natural next step if results warrant it.

2. **Minus sign for reflection:** `Z_new = Z_struct - Z_reflect` — the negative sign makes reflection self-inhibitory rather than additive. Additive reflection would amplify negative activations (destabilising). Self-inhibitory suppresses the source neuron, creating contrast enhancement.

3. **Queued not dispatched:** Mac Studio running step29c, step48, step54 (count=3 > cap=2). Scripts are synced and verified parseable. Launch command documented in LEARNINGS file and arch_comparison.md.

4. **Partial arch_comparison.md is valid:** The plan explicitly allows partial tables noting "pending experiment completion." The 3 confirmed rows (baseline, AntiHebb 0.5, AntiHebb 0.7) provide the anchor points for comparison.

5. **GA_BEST_D64 added:** Plan 01 SUMMARY stated this was added but the worktree experiment_config.py was missing it. Added as a blocking fix (Rule 3) since train_step56_n_scaling.py imports it.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] GA_BEST_D64 missing from experiment_config.py**
- **Found during:** Task 1 (when checking imports for train_exp4_reflection.py)
- **Issue:** Plan 01 SUMMARY claimed GA_BEST_D64 was added to experiment_config.py, but both the worktree and main repo were missing it. train_step56_n_scaling.py imports it directly.
- **Fix:** Added GA_BEST_D64 dict to experiment_config.py with step29 Config C values (top1=0.7524)
- **Files modified:** src/training/experiment_config.py
- **Committed in:** 89a8dfb (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)

## Issues Encountered

- Mac Studio at 3 concurrent experiments when dispatch was attempted. exp4_reflection queued. This matches the plan's key_context guidance ("dispatch or queue").
- exp3_proxwave results not yet available (experiment running). arch_comparison.md has pending rows.
- step56 N-scaling results not yet available (queued, not launched). arch_comparison.md has pending rows.

## Known Stubs

- `results/exp4_reflection.json`: all top1 values are null (queued, not yet run)
- `results/arch_comparison.json`: 9 of 12 rows have null top1 (pending experiments)
- `results/arch_comparison.md`: 9 rows marked PENDING or QUEUED

These are intentional per plan guidance: "partial OK if experiment data pending."

The partial table is usable as a comparison scaffold — the 3 complete rows establish the SmallWorld + AntiHebb performance curve. Remaining rows fill in when step56, exp3_proxwave, and exp4_reflection complete.

## Next Steps When Experiments Complete

1. **When step54 or step29c finishes (freeing a Mac Studio slot):**
   ```bash
   ssh mac-studio 'cd /Users/admin/ml/dhiraj/qwen2_omni/testing && \
     /opt/homebrew/bin/tmux new-session -d -s exp4_ref \
     "d_env/bin/python3 -u scripts/train_exp4_reflection.py --device mps 2>&1 | tee logs/train_exp4_reflection.log"'
   ```

2. **When exp3_proxwave completes:**
   ```bash
   rsync -avz mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/results/exp3_proxwave.json results/
   ```

3. **When step56 N-scaling completes:**
   ```bash
   rsync -avz mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/results/train_step56_n_scaling.json results/
   ```

4. Fill arch_comparison.json and arch_comparison.md with actual numbers.

## Self-Check: PASSED

- `src/sgnnet/model_reflection.py` — exists, syntax verified
- `scripts/train_exp4_reflection.py` — exists, syntax verified
- `results/arch_comparison.json` — 12 rows (>= 8 required), verified via python3
- `results/arch_comparison.md` — contains 75.24%, verified
- `learnings/LEARNINGS_phase5_p13_reflection.md` — exists
- `results/exp4_reflection.json` — exists (queued placeholder)
- Commits 89a8dfb and c8ece6f verified in git log
- GA_BEST_D64 import verified: `from src.training.experiment_config import GA_BEST_D64` returns 0.7524

---
*Phase: 05-scalable-architecture-experiments*
*Completed: 2026-04-03*
