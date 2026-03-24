---
phase: 03-sgnnet-core-architecture
plan: 02
subsystem: model
tags: [pytorch, nn.module, sparse-connectivity, geometric-network, self-projection]

requires:
  - phase: 03-sgnnet-core-architecture/plan-01
    provides: "geometry.py (dynamic_connectivity_hh/ho), encoding.py (compute_spatial_encoding)"
provides:
  - "SGNNET nn.Module with three-phase forward pass"
  - "src/sgnnet/model.py exporting SGNNET class"
affects: [03-sgnnet-core-architecture/plan-03, 04-sgnnet-training]

tech-stack:
  added: []
  patterns: ["three-phase forward (seed/iterate/readout)", "sparse C with mask buffer + value parameter", "_last_gate for optional load balance loss"]

key-files:
  created: [src/sgnnet/model.py, tests/test_model.py]
  modified: []

key-decisions:
  - "Sparsity test threshold 0.88 for small N_hidden due to guaranteed-connectivity row fix"
  - "Split forward into _seed, _iterate_hidden, _output_readout private methods for readability"

patterns-established:
  - "_make_sparse_c helper for reusable sparse matrix creation with guaranteed row connectivity"
  - "Phase methods (_seed, _iterate_hidden, _output_readout) keep forward() readable"

requirements-completed: [ARCH-01, ARCH-03]

duration: 2min
completed: 2026-03-24
---

# Phase 3 Plan 02: SGNNET nn.Module Summary

**SGNNET nn.Module with three-phase forward pass (seeding via C_input, K-1 hidden iterations with dynamic connectivity, output injection) and self-projection readout producing [batch, 10] scores**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-24T10:40:40Z
- **Completed:** 2026-03-24T10:42:43Z
- **Tasks:** 1 (TDD: RED + GREEN)
- **Files modified:** 2

## Accomplishments
- SGNNET forward pass accepts [batch, 25088] and produces [batch, 10] scores
- Three sparse C matrices (C_input, C_hh, C_ho) with mask buffers and learnable value parameters
- Backward pass produces valid gradients on W and all C_values parameters
- _last_gate correctly None when K=1, tensor when K>1 (for optional load balance loss in Plan 03-03)
- 16/16 tests passing

## Task Commits

Each task was committed atomically (TDD):

1. **Task 1 RED: SGNNET tests** - `dad14c59` (test)
2. **Task 1 GREEN: SGNNET implementation** - `139b2b2e` (feat)

## Files Created/Modified
- `src/sgnnet/model.py` - SGNNET nn.Module with three-phase forward pass and self-projection readout (~170 lines)
- `tests/test_model.py` - 16 tests covering construction, forward shape, backward gradients, C matrix shapes/sparsity, spatial buffer, epsilon safety, _last_gate behavior

## Decisions Made
- Relaxed sparsity test threshold from 0.89 to 0.88 for small N_hidden=16 -- the guaranteed-one-connection-per-row fix adds proportionally more entries with few columns
- Split forward() into three private methods (_seed, _iterate_hidden, _output_readout) for readability and staying under 250-line file limit

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Sparsity threshold too tight for small N_hidden**
- **Found during:** Task 1 GREEN (test execution)
- **Issue:** With N_hidden=16, the guaranteed-connectivity row fix pushes zero fraction from 0.90 to 0.888, failing the 0.89 threshold
- **Fix:** Lowered test threshold to 0.88 with explanatory comment
- **Files modified:** tests/test_model.py
- **Verification:** All 16 tests pass
- **Committed in:** 139b2b2e (Task 1 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Threshold adjustment is correct behavior -- small N_hidden naturally has slightly lower sparsity due to row guarantees. No scope creep.

## Issues Encountered
None

## Known Stubs
None -- all functionality is fully wired.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- SGNNET module ready for Plan 03-03 (loss functions: KL divergence, safety valve, load balance)
- _last_gate interface ready for load_balance_loss consumption
- W parameter ready for safety_valve_loss (Coulomb repulsion)
- scores output ready for KL divergence task loss

## Self-Check: PASSED

- [x] src/sgnnet/model.py exists
- [x] tests/test_model.py exists
- [x] Commit dad14c59 (RED) found
- [x] Commit 139b2b2e (GREEN) found

---
*Phase: 03-sgnnet-core-architecture*
*Completed: 2026-03-24*
