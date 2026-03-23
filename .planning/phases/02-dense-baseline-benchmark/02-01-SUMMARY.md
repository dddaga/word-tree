---
phase: 02-dense-baseline-benchmark
plan: 01
subsystem: evaluation
tags: [vgg16, metrics, sklearn, thop, imagenette, baseline]

requires:
  - phase: 01-data-pipeline
    provides: VGGExtractor, get_dataloader, IMAGENETTE_CLASSES, Imagenette dataset
provides:
  - src/utils/metrics.py with compute_all_metrics, count_params, count_flops
  - scripts/eval_baseline.py for VGG16 baseline evaluation
  - results/baseline_vgg16.json with dense baseline benchmark numbers
  - tests/test_metrics.py with 4 unit tests for metrics module
affects: [04-sgnnet-training, 06-comparative-analysis]

tech-stack:
  added: [thop]
  patterns: [functional metrics API, FC-only profiling, one-vs-rest mAP]

key-files:
  created:
    - src/utils/__init__.py
    - src/utils/metrics.py
    - scripts/eval_baseline.py
    - results/baseline_vgg16.json
    - tests/__init__.py
    - tests/test_metrics.py
  modified:
    - requirements.txt
    - .gitignore

key-decisions:
  - "thop returns MACs not FLOPs; stored as flops_fc_per_inference with flops_note clarifying"
  - "Reused VGGExtractor from Phase 1 for eval pass (no duplicate model loading)"
  - "Track results/*.json in git (modified .gitignore) since baseline JSON is a key deliverable consumed by Phases 4 and 6"
  - "count_flops detects model device to create dummy tensor on same device (MPS-safe)"

patterns-established:
  - "Functional metrics API: compute_all_metrics takes numpy arrays, returns plain dict"
  - "FC-only profiling: count_params and count_flops operate on model.classifier, not full VGG16"
  - "MPS script pattern: os.environ.setdefault + num_workers=0 + sys.path setup"

requirements-completed: [BASE-01, BASE-02, BASE-03, BASE-04, BASE-05]

duration: 4min
completed: 2026-03-23
---

# Phase 2 Plan 01: Frozen VGG16 Evaluation Summary

**Reusable metrics module (sklearn mAP/P/R/F1, thop MACs) and VGG16 baseline: 99.5% top-1, 99.97% mAP on Imagenette val, 123.6M FC params**

## Performance

- **Duration:** 4 min
- **Started:** 2026-03-23T17:37:11Z
- **Completed:** 2026-03-23T17:41:15Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments
- Created reusable metrics module with compute_all_metrics (top-1, mAP, per-class P/R/F1/AP), count_params (FC-only), and count_flops (FC-only MACs via thop)
- Evaluated frozen VGG16 on Imagenette val (3925 images): top-1 accuracy 99.54%, mAP 99.97%
- Recorded FC parameter count (123,642,856) and FC MACs (123,633,664) in baseline JSON
- All 10 Imagenette classes have per-class accuracy, precision, recall, F1, and AP in results JSON

## Task Commits

Each task was committed atomically:

1. **Task 1: Create metrics module with tests** - `79d84d2d` (feat)
2. **Task 2: Evaluate frozen VGG16 and produce baseline JSON** - `06308baa` (feat)

## Files Created/Modified
- `src/utils/__init__.py` - Package marker for utils module
- `src/utils/metrics.py` - compute_all_metrics, count_params, count_flops (reused in Phases 4, 6)
- `tests/__init__.py` - Package marker for tests
- `tests/test_metrics.py` - 4 unit tests: perfect predictions, mixed predictions, VGG16 param count, FC FLOPs
- `scripts/eval_baseline.py` - VGG16 baseline evaluation script (loads VGGExtractor, runs val inference, writes JSON)
- `results/baseline_vgg16.json` - Dense baseline benchmark: top1=0.9954, mAP=0.9997, fc_params=123642856
- `requirements.txt` - Added thop>=0.1.1
- `.gitignore` - Changed results/ ignore to allow *.json tracking

## Baseline Results

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 99.54% |
| mAP | 99.97% |
| FC Parameters | 123,642,856 |
| FC MACs | 123,633,664 |
| Val Images | 3,925 |
| Classes | 10 |

## Decisions Made
- thop returns MACs (multiply-accumulate ops), not 2x FLOPs; documented in JSON with `flops_note` field
- Reused VGGExtractor from Phase 1 for eval pass rather than loading VGG16 separately
- Modified .gitignore to track results/*.json since baseline_vgg16.json is consumed by Phases 4 and 6
- Added device detection in count_flops to create dummy tensor on model's device (MPS compatibility)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed undefined device variable in count_flops**
- **Found during:** Task 1 (metrics module)
- **Issue:** count_flops referenced `device` variable without defining it; would fail on MPS models
- **Fix:** Added `device = next(model.classifier.parameters()).device` before creating dummy tensor
- **Files modified:** src/utils/metrics.py
- **Verification:** test_count_flops_vgg16 passes
- **Committed in:** 79d84d2d (squashed into Task 1 commit)

**2. [Rule 2 - Missing Critical] Track results JSON in git**
- **Found during:** Task 2 (baseline evaluation)
- **Issue:** results/ was fully gitignored but baseline_vgg16.json is a key deliverable for Phases 4 and 6
- **Fix:** Changed .gitignore from blanket results/ ignore to results/*.pt, results/*.h5, etc., allowing *.json
- **Files modified:** .gitignore
- **Verification:** jj status shows results/baseline_vgg16.json tracked
- **Committed in:** 06308baa (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 missing critical)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
None - plan executed as specified.

## Known Stubs
None - all data flows are wired to real VGG16 inference output.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Baseline benchmark complete with all required metrics in results/baseline_vgg16.json
- metrics.py ready for reuse in Phase 4 (SGNNET evaluation) and Phase 6 (comparative analysis)
- Plan 02-02 (soft label quality verification) can proceed independently

## Self-Check: PASSED

All 8 files verified present on disk. Both commit hashes (79d84d2d, 06308baa) verified in jj log. SUMMARY.md created at expected path. 4 pytest tests pass. Baseline JSON passes schema validation.

---
*Phase: 02-dense-baseline-benchmark*
*Completed: 2026-03-23*
