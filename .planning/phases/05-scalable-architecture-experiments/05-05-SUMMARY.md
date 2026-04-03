---
phase: 05-scalable-architecture-experiments
plan: 05
subsystem: experiments
tags: [dynamic-connectivity, input-architecture, PCA, spatial-grouped, ARM-3, ARM-5]

requires:
  - phase: 05-scalable-architecture-experiments
    provides: "SmallWorld model, Resonant wrapper, AntiHebb mechanism, step36/31 results"
provides:
  - "step36 full results analyzed (6 configs, input-gated adjacency at D=64)"
  - "ARM 3 preliminary verdict: PARTIAL-GO (soft input-gating +2.34pp over ceiling)"
  - "PCA input sweep script (k=256,512,1024) ready for dispatch"
  - "step49/50/51/55 scripts synced to Mac Studio, queued for dispatch"
  - "LEARNINGS documenting ARM 3 + ARM 5 progress"
affects: [05-06-synthesis, phase-6-pca-compression]

tech-stack:
  added: [sklearn.decomposition.PCA]
  patterns: [PCA-transformed input pipeline, spatial grouped learned projection]

key-files:
  created:
    - scripts/pca_input_sweep.py
    - learnings/LEARNINGS_phase5_p9_arm3_arm5.md
    - src/sgnnet/model_spatial_grouped.py
    - scripts/train_step55_spatial_grouped_input.py
  modified:
    - results/train_step36_input_gated.json
    - results/train_step29_antihebb_d64.json
    - results/train_step31_dynamic_zknn.json

key-decisions:
  - "step36 soft gate tau=1.0 is the sweet spot; hard gating destroys signal at D=64"
  - "ARM 3 preliminary verdict: PARTIAL-GO -- input-gated adjacency adds +2.34pp but falls far short of AntiHebb 75.24%"
  - "PCA sweep uses K_in=min(50,k) and n_groups=max(8,k//8) to handle small input dims"
  - "step49/50/51 dispatch deferred -- 3 experiments running on Mac Studio (cap=2)"

patterns-established:
  - "PCA input pipeline: fit on training, transform both splits, feed to SmallWorld with adjusted N_in"
  - "ARM verdict framework: GO/PARTIAL-GO/NO-GO with N-squared recovery metric"

requirements-completed: []

duration: 5min
completed: 2026-04-03
---

# Phase 5 Plan 5: ARM 3 Dynamic Connectivity + ARM 5 Input Architecture Summary

**step36 input-gated adjacency analyzed (soft gate +2.34pp, best dynamic signal at D=64); PCA sweep + spatial grouped + step49/50/51 scripts written, synced, queued for Mac Studio dispatch**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-03T12:05:24Z
- **Completed:** 2026-04-03T12:10:24Z
- **Tasks:** 2
- **Files modified:** 10

## Accomplishments

- Analyzed step36 input-gated adjacency: ALL 6 configs (Ref, A-E) complete. Config A (soft gate tau=1.0) achieves 58.62% (+2.34pp over 56.28% static ceiling), recovering 21.4% of the N-squared gap. Hard gating is catastrophic (41% / 11%).
- Wrote `scripts/pca_input_sweep.py`: PCA compression at k=256,512,1024 with full SmallWorld+Resonant+AntiHebb training pipeline, sklearn PCA fitting, explained variance tracking.
- Prepared and synced all ARM 3 experiment scripts (step49/50/51) and ARM 5 scripts (step55, PCA sweep, model_spatial_grouped.py) to Mac Studio. Ready for immediate dispatch when slots open.
- Wrote `learnings/LEARNINGS_phase5_p9_arm3_arm5.md` documenting ARM 3 progress (step36 analyzed, step31 cross-referenced, step49/50/51 queued) and ARM 5 experiment queue.
- ARM 3 preliminary verdict: **PARTIAL-GO** -- soft input-gating works (+2.34pp) but the gain is modest vs AntiHebb alone (75.24%). Compounding test needed.

## Task Commits

1. **Task 1: Sync step36 + ARM 3 scripts** -- `61566dd` (feat)
2. **Task 2: PCA sweep + spatial grouped + LEARNINGS** -- `6790d6d` (feat)

## Files Created/Modified

- `results/train_step36_input_gated.json` -- Complete step36 results (6 configs: Ref, A-E)
- `results/train_step29_antihebb_d64.json` -- AntiHebb reference (75.24% all-time best)
- `results/train_step31_dynamic_zknn.json` -- Z-KNN reference (negative result)
- `scripts/train_step49_signed_kiter_threshold.py` -- Signed coupling K_iter sweep (40ep calibration)
- `scripts/train_step50_spatial_dynamic_conn.py` -- Spatial W_pos K-NN dynamic connectivity
- `scripts/train_step51_spatial_phase_gating.py` -- W_pos K-NN + W_phase strength gating
- `scripts/train_step55_spatial_grouped_input.py` -- Spatial grouped learned input projection (6 configs)
- `scripts/pca_input_sweep.py` -- PCA compression at k=256,512,1024
- `src/sgnnet/model_spatial_grouped.py` -- Spatial grouped input model
- `learnings/LEARNINGS_phase5_p9_arm3_arm5.md` -- ARM 3 + ARM 5 analysis

## Decisions Made

1. **step36 soft gate tau=1.0 is optimal:** Default temperature (tau=1.0) gives best input-gated adjacency. Sharper (0.5) or smoother (2.0) both degrade. Hard gating destroys the signal entirely at D=64 because initial cosine similarities on S^63 are near-zero.

2. **ARM 3 PARTIAL-GO (provisional):** Input-gated adjacency (step36 A) adds +2.34pp over static ceiling, proving O(N*K) dynamic connectivity CAN work at D=64. But the gain is only 21.4% of N-squared recovery and far below AntiHebb (75.24%). Final verdict requires step49/50/51 results.

3. **Dispatch deferred, not blocked:** 3 experiments currently running on Mac Studio (step29c, step48, step54). Concurrency cap = 2. All scripts synced and ready; will dispatch as slots open. This is normal experiment scheduling, not a blocker.

4. **PCA sweep K_in clamping:** For small PCA dimensions (k=256), K_in is clamped to min(50, k) since PCA output may be smaller than the default 50-feature fan-in.

## Deviations from Plan

### Dispatch Blocked (Not a deviation -- experiment scheduling)

**step49/50/51/55/PCA dispatch deferred:** Mac Studio running 3 experiments (step29c PID 89895, step48 PID 91447, step54 PID 95059). Concurrency cap = 2 per project rules. All scripts synced and ready for immediate dispatch when count drops to 1.

**Impact:** Results for step49/50/51/55/PCA not yet available. LEARNINGS documents partial results with step36 analysis. ARM 3 verdict is PARTIAL-GO pending remaining experiments.

No other deviations. Plan executed as written for all locally-completable work.

## Known Stubs

- `learnings/LEARNINGS_phase5_p9_arm3_arm5.md` has placeholder sections for step49, step50, step51, step55, and PCA results (marked "QUEUED" / "PENDING"). These will be filled when experiments complete.
- `results/pca_input_sweep.json` does not yet exist (PCA sweep not dispatched)
- `results/train_step49_signed_kiter_threshold.json` does not yet exist
- `results/train_step50_spatial_dynamic_conn.json` does not yet exist
- `results/train_step51_spatial_phase_gating.json` does not yet exist
- `results/train_step55_spatial_grouped_input.json` does not yet exist

These are experiment outputs, not code stubs. They will be produced when the queued experiments run.

## ARM 3 Analysis Summary

| Step | Mechanism | Best top1 | vs Ceiling | Status |
|------|-----------|-----------|------------|--------|
| step31 | Z-KNN (activation K-NN) | 52.74% | -3.54pp | DONE -- hurts |
| **step36** | **Input-gated adjacency (soft)** | **58.62%** | **+2.34pp** | **DONE -- helps** |
| step49 | Signed coupling K_iter sweep | -- | -- | QUEUED |
| step50 | W_pos K-NN per epoch | -- | -- | QUEUED |
| step51 | W_pos K-NN + W_phase gate | -- | -- | QUEUED |

## Mac Studio Dispatch Queue

When slots open (count <= 1 AND RAM >= 50GB), dispatch in this order:
1. **step49** (40ep calibration -- fastest, resolves signed coupling question)
2. **step55** (150ep -- ARM 5 spatial grouped input)
3. **pca_input_sweep** (150ep x 3 configs -- ARM 5 PCA compression)
4. **step50** (150ep x 5 configs -- ARM 3 spatial K-NN)
5. **step51** (150ep x 5 configs -- ARM 3 spatial + phase gating)

## Self-Check: PASSED

- All 13 expected files found
- Both task commits verified (61566dd, 6790d6d)
- PCA script syntax verified via ast.parse

---
*Phase: 05-scalable-architecture-experiments*
*Completed: 2026-04-03*
