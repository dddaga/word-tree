---
phase: 04-sgnnet-wave-architecture-experiments
plan: 03
subsystem: training
tags: [ga-search, stage-a, static-baseline, sgnnet-wave, mps]

requires:
  - phase: 04-01
    provides: SGNNET_Wave model with phasor architecture
  - phase: 04-02
    provides: Trainer loop and GASearch harness
provides:
  - Stage A baseline results (top1=10.27%, mAP=11.03%)
  - Best hyperparams from GA search (K=2, N_hidden=256)
  - Stage A checkpoint for comparison with Stages B/C
affects: [04-04, 04-05, 06]

tech-stack:
  added: []
  patterns: [ga-search-then-full-training, three-phase-script-structure]

key-files:
  created:
    - scripts/train_stageA.py
    - results/stageA_ga_results.json
    - results/stageA_full.json
    - checkpoints/stageA_best.pt
  modified: []

key-decisions:
  - "Stage A with 1064 learnable params (only W_pos) achieves ~10% accuracy -- binary C masks alone insufficient for distillation"
  - "GA selected K=2, N_hidden=256, lr=0.008, lambda_safety=0.52 as best config"
  - "Loss converges quickly to ~3.22 plateau -- static binary wiring hits expressiveness ceiling"

patterns-established:
  - "Three-phase training script: GA search -> full training -> final eval with compute_all_metrics"

requirements-completed: [TRAIN-04, TRAIN-05, TRAIN-06]

duration: 22min
completed: 2026-03-26
---

# Phase 04 Plan 03: Stage A Static Baseline Summary

**GA search (20x10) found K=2/N_hidden=256, full 50-epoch training yields 10.27% top-1 with 1064 params (binary C masks only, no dynamic routing)**

## Performance

- **Duration:** 22 min
- **Started:** 2026-03-26T07:20:50Z
- **Completed:** 2026-03-26T07:43:26Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- GA hyperparameter search completed: 20 population x 10 generations on 15% data
- Full Stage A training converged: loss decreased from 8.24 to 3.22 over 50 epochs
- All result artifacts produced: GA results JSON, full metrics JSON (with per-class precision/recall/F1 per TRAIN-06), checkpoint
- Parameter budget verified: 1064 params = 0.00% of VGG16 FC (well within 1% limit, TRAIN-04)
- Sparsity maintained: C_input=90.0%, C_hh=90.3%, C_ho=86.4% (TRAIN-05)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Stage A training script** - `8229175` (feat)
2. **Task 2: Execute Stage A training** - `51a2889` (feat)

## Files Created/Modified
- `scripts/train_stageA.py` - End-to-end Stage A training: GA search then full training with eval
- `results/stageA_ga_results.json` - Best hyperparams from GA: K=2, N_hidden=256, lr=0.008
- `results/stageA_full.json` - Full training metrics: top1=10.27%, mAP=11.03%, per-class with P/R/F1
- `checkpoints/stageA_best.pt` - Best Stage A model checkpoint (26MB, gitignored)

## Decisions Made
- Stage A achieves only ~10% accuracy (random baseline) with 1064 learnable params. The binary C masks alone cannot distill VGG16 knowledge effectively. This establishes the baseline that dynamic routing (Stages B/C) must beat.
- GA search converged to K=2, N_hidden=256 as the best config, with lr_Wpos=0.008 and lambda_safety=0.52. Higher K values (3, 4) did not improve fitness.
- Loss plateaus quickly at ~3.22 after epoch 8, indicating the model hits an expressiveness ceiling without dynamic routing.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Python output buffering in background execution mode made progress monitoring difficult (resolved by waiting for completion and checking output files directly)
- The training script needed PYTHONPATH set to find `src` package when running from main repo directory

## Known Stubs

None - all metrics and outputs are real training results.

## Next Phase Readiness
- Stage A baseline established: top1=10.27%, mAP=11.03% with 1064 params
- Stage B (Plan 04-04) can now compare dynamic proximity routing against this baseline
- Stage C (Plan 04-05) can further compare learned W_phase against both A and B
- The near-random accuracy confirms that binary C alone is insufficient, validating the need for wave-based dynamic routing

---
*Phase: 04-sgnnet-wave-architecture-experiments*
*Completed: 2026-03-26*
