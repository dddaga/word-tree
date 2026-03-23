---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: In progress
stopped_at: Completed 02-01-PLAN.md
last_updated: "2026-03-23T17:41:15.000Z"
progress:
  total_phases: 2
  completed_phases: 1
  total_plans: 4
  completed_plans: 3
---

# Project State: neuro_graph

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-23)

**Core value:** SGNNET matches VGG16 FC accuracy at ≤1% of its parameters
**Current focus:** Phase 02 — dense-baseline-benchmark

## Current Phase

**Phase 2 — Dense Baseline Benchmark**
Status: In progress (1/2 plans complete)
Next action: Execute Plan 02-02 (soft label quality verification)

## Phase Progress

| Phase | Name | Status |
|-------|------|--------|
| 1 | Data Pipeline | Complete (2/2 plans, UAT passed 7/7) |
| 2 | Dense Baseline Benchmark | In progress (1/2 plans) |
| 3 | SGNNET Core Architecture | Not started |
| 4 | SGNNET Training & Evaluation | Not started |
| 5 | PCA Compression | Not started |
| 6 | Comparative Analysis & Report | Not started |

## Decisions

- **>= version constraints in requirements.txt**: Python 3.14 may need latest wheels; pinning exact versions risks incompatibility (Phase 1, Plan 01-01)
- **Root-relative /data/ in .gitignore**: Prevents accidentally ignoring src/data/ module directory (Phase 1, Plan 01-01)
- **PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0**: Removes 50% MPS memory cap for full-batch extraction (Phase 1, Plan 01-02)
- **num_workers=0 for extraction**: macOS MPS + Python 3.14 multiprocessing spawn incompatible (Phase 1, Plan 01-02)
- **pin_memory disabled on MPS**: Auto-detected in get_dataloader (Phase 1, Plan 01-02)
- **thop MACs not FLOPs**: thop.profile returns MACs; documented with flops_note in JSON (Phase 2, Plan 02-01)
- **Reuse VGGExtractor for eval**: No duplicate VGG16 loading; extractor handles frozen/eval/MPS (Phase 2, Plan 02-01)
- **Track results/*.json in git**: Baseline JSON is a key deliverable consumed by Phases 4 and 6 (Phase 2, Plan 02-01)

## Open Decisions

- **N_in strategy for SGNNET**: N_in=25088 (large C matrix, borderline feasible) vs. input adapter 25088->48 (adapter dominates params). Recommended: adapter 25088->48 to stay within 1% budget. Decide during Phase 3, Plan 3.2.

## Key Files

- Architecture spec: `sparse_geometric_network_report.md`
- Requirements: `requirements.txt`
- Dataset module: `src/data/dataset.py`
- Feature extractor: `src/data/extractor.py`
- Tensor store: `data/store.h5` (13,394 records)
- Manifest: `data/manifest.csv` (13,394 rows)
- Metrics module: `src/utils/metrics.py` (compute_all_metrics, count_params, count_flops)
- Baseline eval script: `scripts/eval_baseline.py`
- Baseline results: `results/baseline_vgg16.json` (top1=0.9954, mAP=0.9997)
- Metrics tests: `tests/test_metrics.py` (4 tests)

## Performance Metrics

| Phase-Plan | Duration | Tasks | Files |
|------------|----------|-------|-------|
| 01-01      | 6min     | 2     | 5     |
| 01-02      | 3min     | 2     | 5     |
| 02-01      | 4min     | 2     | 8     |

## Last Session

- **Stopped at:** Completed 02-01-PLAN.md
- **Timestamp:** 2026-03-23T17:41:15Z

---
*State initialized: 2026-03-23*
*Last updated: 2026-03-23 after Plan 02-01 completion*
