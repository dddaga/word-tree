---
phase: 04-sgnnet-wave-architecture-experiments
plan: 05
subsystem: analysis
tags: [comparison, metrics, wave-architecture, phasor]

requires:
  - phase: 04-sgnnet-wave-architecture-experiments/plan-03
    provides: Stage A baseline results (stageA_full.json)
  - phase: 04-sgnnet-wave-architecture-experiments/plan-04
    provides: Exp 1 and Exp 2 results (exp1_full.json, exp2_full.json)
provides:
  - wave_comparison.json with side-by-side metrics for all stages
  - wave_comparison.md with human-readable comparison tables
  - eval_comparison.py script for reproducible comparison generation
affects: [phase-06-comparative-analysis]

tech-stack:
  added: []
  patterns: [baseline-class-name-mapping, delta-computation]

key-files:
  created:
    - scripts/eval_comparison.py
    - results/wave_comparison.json
    - results/wave_comparison.md
  modified: []

key-decisions:
  - "Baseline class name mapping needed (VGG16 uses 'English springer' vs experiment 'english_springer')"
  - "Phase 3 config included as reference row (no training results, only param count)"

patterns-established:
  - "Comparison script pattern: load all result JSONs, compute deltas, emit JSON + markdown"

requirements-completed: [TRAIN-06]

duration: 2min
completed: 2026-03-26
---

# Phase 04 Plan 05: Wave Comparison Report Summary

**Side-by-side comparison of all Phase 4 stages: Stage A (static), Exp 1 (spatial phase), Exp 2 (W_phase) with contribution deltas and per-class breakdown**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-26T08:49:57Z
- **Completed:** 2026-03-26T08:52:00Z
- **Tasks:** 1
- **Files modified:** 3

## Accomplishments

- Created eval_comparison.py (229 lines) that loads all stage results and produces structured comparison
- Generated wave_comparison.json with aggregate metrics, per-class data, and deltas for each stage transition
- Generated wave_comparison.md with formatted tables for aggregate, per-class, contribution analysis, and hyperparameters
- Documented that Exp 1 provides marginal improvement over Stage A (top1 +0.017, mAP +0.006) while Exp 2 with W_phase does not improve over Exp 1 (top1 -0.015, mAP -0.011)

## Key Findings

- **Stage A** (static binary C): top1=0.1027, mAP=0.1103 -- 1,064 params (W_pos only)
- **Exp 1** (spatial path-length phase): top1=0.1197, mAP=0.1160 -- 1,064 params
- **Exp 2** (spatial + W_phase): top1=0.1052, mAP=0.1047 -- 2,128 params (+1,064 for W_phase)
- Proximity routing with phasor interference adds marginal benefit (+1.7% top1) over static wiring
- Learned phase operator W_phase adds parameters but degrades performance (loss exploded at epoch 14 to 506k)
- All stages far below VGG16 baseline (99.54% top1) -- binary C masks with only position learning are insufficient for distillation

## Task Commits

Each task was committed atomically:

1. **Task 1: Create comparison script and generate wave_comparison outputs** - `904e287` (feat)

## Files Created/Modified

- `scripts/eval_comparison.py` - Loads Stage A, Exp 1, Exp 2, baseline; generates JSON and markdown comparison
- `results/wave_comparison.json` - Structured comparison with stageA, exp1, exp2, deltas sections
- `results/wave_comparison.md` - Human-readable tables: aggregate metrics, contribution analysis, per-class, hyperparameters

## Decisions Made

- Handled class name casing mismatch between baseline ("English springer") and experiments ("english_springer") via explicit mapping dictionary
- Included Phase 3 SGNNET config as reference row showing param count (649K with learned C) vs Phase 4 binary-C experiments (1K-2K params)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed baseline class name casing mismatch**
- **Found during:** Task 1
- **Issue:** VGG16 baseline per_class uses "English springer", "chain saw" etc., while experiments use "english_springer", "chain_saw"
- **Fix:** Added BASELINE_CLASS_MAP dictionary for explicit per-class lookups against baseline
- **Files modified:** scripts/eval_comparison.py
- **Verification:** All 10 classes appear correctly in wave_comparison.md per-class table
- **Committed in:** 904e287

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Essential fix for correct per-class VGG16 accuracy display. No scope creep.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Wave architecture experiments complete: all 5 plans in Phase 4 done
- Results show binary C masks with position-only learning insufficient for distillation (~10% accuracy vs 99.5% VGG16)
- Phase 3 amplitude-based SGNNET (649K params with learned C values) remains the better architecture path
- Ready for Phase 5 (PCA compression) or Phase 6 (comparative analysis)

---
*Phase: 04-sgnnet-wave-architecture-experiments*
*Completed: 2026-03-26*
