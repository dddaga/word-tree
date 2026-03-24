---
phase: 03-sgnnet-core-architecture
plan: 01
subsystem: architecture
tags: [pytorch, geometry, cdist, spatial-encoding, sgnnet]

requires:
  - phase: 02-dense-baseline-benchmark
    provides: VGG16 baseline metrics and feature dimensions (25088-dim, 512x7x7)
provides:
  - "compute_spatial_encoding: maps flat VGG16 index to [channel_norm, h_norm, w_norm]"
  - "personal_volume_radius: r* = R / N^(1/D)"
  - "dynamic_connectivity_hh: hidden-hidden Gaussian-gated routing with batched cdist"
  - "dynamic_connectivity_ho: hidden-to-output routing with batched cdist"
affects: [03-02-PLAN (SGNNET model uses geometry+encoding), 03-03-PLAN (loss functions use gate)]

tech-stack:
  added: []
  patterns:
    - "Batched cdist for pairwise distance computation"
    - "Gaussian kernel with hard gate for dynamic connectivity"
    - "Column-wise normalization for routing strength"

key-files:
  created:
    - src/sgnnet/__init__.py
    - src/sgnnet/encoding.py
    - src/sgnnet/geometry.py
    - tests/test_encoding.py
    - tests/test_geometry.py
  modified: []

key-decisions:
  - "Encoding returns [channel_norm, h_norm, w_norm] (3 coords, not 4) — value dimension added dynamically per-sample in model forward"
  - "r* for ho uses N_hidden (not N_out) consistent with D-10"
  - "Normalization is per-target (dim=1) matching report Section 3.5"

patterns-established:
  - "TDD: tests written before implementation for all SGNNET modules"
  - "Batched cdist pattern: expand W to [batch, N, D] before cdist"

requirements-completed: [ARCH-02, ARCH-07]

duration: 2min
completed: 2026-03-24
---

# Phase 3 Plan 1: Geometry Primitives and Spatial Encoding Summary

**Personal volume radius r*, Gaussian-gated dynamic connectivity (hh + ho), and VGG16 spatial encoding with 16 passing tests**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-24T10:36:58Z
- **Completed:** 2026-03-24T10:38:57Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 5

## Accomplishments
- Spatial encoding maps 25088-dim flat indices to normalized [channel, h, w] coordinates in [0,1]^3
- personal_volume_radius computes r* = R / N^(1/D) with guaranteed-in-box property
- dynamic_connectivity_hh returns (contribution, gate) using batched cdist, Gaussian kernel, hard gate, zero diagonal
- dynamic_connectivity_ho returns contribution from hidden to output neurons
- 16 tests covering shape, values, edge cases (far-apart neurons, proximity gating)

## Task Commits

Each task was committed atomically (TDD):

1. **Task 1 RED: Failing tests** - `opmrssnl` (test)
2. **Task 1 GREEN: Implementation** - `zmuyrlzy` (feat)

## Files Created/Modified
- `src/sgnnet/__init__.py` - Package marker
- `src/sgnnet/encoding.py` - compute_spatial_encoding: flat index to [channel_norm, h_norm, w_norm]
- `src/sgnnet/geometry.py` - personal_volume_radius, dynamic_connectivity_hh, dynamic_connectivity_ho
- `tests/test_encoding.py` - 7 tests for spatial coordinate mapping
- `tests/test_geometry.py` - 9 tests for r* and dynamic connectivity

## Decisions Made
- Encoding returns 3 coordinates (not 4): the value dimension is added dynamically per sample in the model forward pass, matching D-02 context
- r* for hidden-to-output uses N_hidden (not N_out), consistent with decision D-10
- Normalization uses dim=1 (per-target neuron), matching report Section 3.5 column-wise normalization

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functions are fully implemented with real computation.

## Next Phase Readiness
- Geometry and encoding modules ready for consumption by Plan 03-02 (SGNNET model)
- dynamic_connectivity_hh gate output ready for Plan 03-03 (load balance loss)
- compute_spatial_encoding ready for register_buffer in SGNNET.__init__

---
*Phase: 03-sgnnet-core-architecture*
*Completed: 2026-03-24*
