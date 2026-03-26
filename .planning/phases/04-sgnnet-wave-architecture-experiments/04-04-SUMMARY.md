---
phase: 04-sgnnet-wave-architecture-experiments
plan: 04
subsystem: training
tags: [pytorch, mps, ga-search, phasor-routing, wave-architecture, proximity]

# Dependency graph
requires:
  - phase: 04-sgnnet-wave-architecture-experiments (plan 02)
    provides: Trainer class, GASearch harness, SEARCH_SPACE_AB/C
  - phase: 04-sgnnet-wave-architecture-experiments (plan 03)
    provides: Stage A baseline results (top1=0.1027, mAP=0.1103)
provides:
  - Exp 1 (Stage B) GA results and full training metrics with spatial phase routing
  - Exp 2 (Stage C) GA results and full training metrics with spatial phase + W_phase
  - Model checkpoints for both experiments
affects: [04-05-wave-comparison, phase-06-comparative-analysis]

# Tech tracking
tech-stack:
  added: []
  patterns: [GA-then-full-training pipeline for wave experiments]

key-files:
  created:
    - scripts/train_exp1.py
    - scripts/train_exp2.py
    - results/exp1_ga_results.json
    - results/exp1_full.json
    - results/exp2_ga_results.json
    - results/exp2_full.json
    - checkpoints/exp1_best.pt
    - checkpoints/exp2_best.pt
  modified: []

key-decisions:
  - "Exp 1 GA selected K=2, N_hidden=256, lr=0.0024, lambda_safety=0.69 as best config"
  - "Exp 2 GA selected K=2, N_hidden=256, lr=0.01, lr_Wphase=0.01, lambda_safety=0.90"
  - "Exp 1 shows marginal improvement over Stage A (0.1197 vs 0.1027 top1), proximity routing adds signal"
  - "Exp 2 W_phase norm=13.36 confirms W_phase receives gradients but accuracy lower than Exp 1"

patterns-established:
  - "Same 3-phase pipeline (GA search + full train + eval) across all stages"
  - "SEARCH_SPACE_C extends SEARCH_SPACE_AB with lr_Wphase for Stage C experiments"

requirements-completed: [TRAIN-06]

# Metrics
duration: 60min
completed: 2026-03-26
---

# Phase 04 Plan 04: Stage B/C Wave Experiments Summary

**Exp 1 (spatial phase) at 0.1197 top1 and Exp 2 (spatial + W_phase) at 0.1052 top1, both trained via GA search + 50-epoch full training with per-class precision/recall/f1 (TRAIN-06)**

## Performance

- **Duration:** 60 min
- **Started:** 2026-03-26T07:46:16Z
- **Completed:** 2026-03-26T08:46:56Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Created Exp 1 and Exp 2 training scripts following exact train_stageA.py pattern
- Exp 1 (spatial phase only): top1=0.1197, mAP=0.1160 -- marginal improvement over Stage A baseline (0.1027)
- Exp 2 (spatial + W_phase): top1=0.1052, mAP=0.1047, W_phase norm=13.36 -- W_phase trained but did not improve over Exp 1
- Both experiments produce full per-class metrics with precision/recall/f1 for all 10 classes (TRAIN-06)

## Results Comparison

| Stage | Experiment | top1_accuracy | mAP | params | Key Config |
|-------|-----------|---------------|-----|--------|------------|
| A | Static baseline | 0.1027 | 0.1103 | 1,064 | K=2, N_hidden=256, lr=0.008 |
| B | Spatial phase (Exp 1) | 0.1197 | 0.1160 | 1,064 | K=2, N_hidden=256, lr=0.0024 |
| C | Spatial + W_phase (Exp 2) | 0.1052 | 0.1047 | 2,128 | K=2, N_hidden=256, lr=0.01, lr_Wphase=0.01 |

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Exp 1 and Exp 2 training scripts** - `5b3ee60` (feat)
2. **Task 2: Execute Exp 1 and Exp 2 training** - `7a14bf2` (feat)

## Files Created/Modified
- `scripts/train_exp1.py` - Stage B training: GA + full train with spatial phase routing
- `scripts/train_exp2.py` - Stage C training: GA + full train with spatial phase + W_phase operator
- `results/exp1_ga_results.json` - Exp 1 GA search results (best config, 10 generations)
- `results/exp1_full.json` - Exp 1 full training metrics (top1, mAP, per-class)
- `results/exp2_ga_results.json` - Exp 2 GA search results (best config with lr_Wphase)
- `results/exp2_full.json` - Exp 2 full training metrics (top1, mAP, per-class, w_phase_norm)
- `checkpoints/exp1_best.pt` - Exp 1 best model checkpoint
- `checkpoints/exp2_best.pt` - Exp 2 best model checkpoint

## Decisions Made
- Exp 1 GA selected K=2, N_hidden=256, lr=0.0024, lambda_safety=0.69 -- same architecture as Stage A but lower learning rate suits phasor routing
- Exp 2 GA selected K=2, N_hidden=256, lr=0.01, lr_Wphase=0.01 -- maximum learning rates suggest W_phase needs aggressive training
- Both experiments converge to K=2, N_hidden=256 (same as Stage A) -- architecture size not the bottleneck

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Python output buffering masked GA progress for ~15 minutes per experiment (stdout buffered when piped)
- MPS cdist backward fallback to CPU makes phasor routing experiments significantly slower than Stage A
- Checkpoints directory is gitignored; used `git add -f` to force-add experiment checkpoints

## Known Stubs

None - all data flows are complete.

## Next Phase Readiness
- All three stages (A, B, C) have results JSONs ready for wave_comparison.md (Plan 04-05)
- Exp 1 shows proximity routing adds marginal benefit; Exp 2 W_phase does not improve further
- Loss remains near ~3.2 (KL divergence plateau) across all stages -- binary C expressiveness ceiling persists

## Self-Check: PASSED

All 8 created files verified present. Both task commits (5b3ee60, 7a14bf2) verified in git log.

---
*Phase: 04-sgnnet-wave-architecture-experiments*
*Completed: 2026-03-26*
