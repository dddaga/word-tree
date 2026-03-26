---
phase: 04-sgnnet-wave-architecture-experiments
plan: 02
subsystem: training
tags: [fp16, mps, ga-search, hyperparameter-optimization, adam, autocast]

requires:
  - phase: 04-01
    provides: "SGNNET_Wave model with phasor activations, binary C masks, wave routing"
provides:
  - "Trainer class with FP16 AMP, position clamping, gradient zeroing"
  - "GASearch class with efficiency-ratio fitness and NaN disqualification"
  - "SEARCH_SPACE_AB and SEARCH_SPACE_C definitions"
  - "CLI script for running GA search by experiment name"
affects: [04-03, 04-04, 04-05]

tech-stack:
  added: [torch.autocast, torch.amp.GradScaler]
  patterns: [separate-param-groups, efficiency-ratio-fitness, partial-data-evaluation]

key-files:
  created:
    - src/training/__init__.py
    - src/training/trainer.py
    - src/training/ga_search.py
    - scripts/run_ga_search.py
  modified: []

key-decisions:
  - "load_balance_loss uses abs sum of scores as proxy for neuron selection frequency"
  - "GradScaler support detected at runtime via PyTorch version check (>= 2.3)"
  - "Partial data evaluation creates fresh random subset per candidate per generation"

patterns-established:
  - "Trainer accepts model+loaders, handles FP16/clamping/loss automatically"
  - "GA fitness = -final_loss * (params_min/model_params)^0.2 with NaN -> -1e6"

requirements-completed: [TRAIN-01, TRAIN-02, TRAIN-03]

duration: 13min
completed: 2026-03-26
---

# Phase 04 Plan 02: Training Infrastructure Summary

**Shared Trainer with FP16 AMP on MPS, position clamping, KL+safety+load_balance loss, and GA hyperparameter search with efficiency-ratio fitness**

## Performance

- **Duration:** 13 min
- **Started:** 2026-03-26T06:56:50Z
- **Completed:** 2026-03-26T07:09:49Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Trainer runs FP16 training epochs on MPS with torch.autocast (not torch.cuda.amp)
- Position clamping W_pos to [0, box_size] after each optimizer step (TRAIN-03)
- Total loss = KL-div + lambda_safety * safety_valve + lambda_lb * load_balance (TRAIN-01)
- GA search evaluates candidates with efficiency-ratio fitness, disqualifies NaN with -1e6
- CLI script for running GA search by experiment name (stageA/exp1/exp2)

## Task Commits

Each task was committed atomically:

1. **Task 1: Create training loop with FP16 AMP and position clamping** - `eb0bf6c` (feat)
2. **Task 2: Create GA hyperparameter search harness and CLI script** - `6a80f78` (feat)

## Files Created/Modified

- `src/training/__init__.py` - Training package init
- `src/training/trainer.py` - Shared training loop with FP16 AMP, position clamping, gradient zeroing
- `src/training/ga_search.py` - Population-based GA hyperparameter search with efficiency-ratio fitness
- `scripts/run_ga_search.py` - CLI for running GA search by experiment name

## Decisions Made

- Used `scores.abs().sum(dim=0)` as proxy for load_balance_loss input (neuron activation frequency)
- GradScaler support detected via PyTorch version parsing (major > 2 or major == 2 and minor >= 3)
- Each GA candidate gets a fresh random partial-data subset (not shared across candidates)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

- `aten::_cdist_backward` not implemented on MPS device -- requires `PYTORCH_ENABLE_MPS_FALLBACK=1` environment variable for safety_valve_loss backward pass. This is a known MPS limitation; the cdist backward falls back to CPU automatically with the env var set.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Trainer and GASearch are ready for Plans 03, 04, 05 (Stage A, Exp 1, Exp 2)
- Plans 03-05 will import Trainer and GASearch directly
- PYTORCH_ENABLE_MPS_FALLBACK=1 should be set in training scripts for MPS compatibility

## Self-Check: PASSED

---
*Phase: 04-sgnnet-wave-architecture-experiments*
*Completed: 2026-03-26*
