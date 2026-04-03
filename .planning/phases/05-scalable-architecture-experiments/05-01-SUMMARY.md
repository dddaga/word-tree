---
phase: 05-scalable-architecture-experiments
plan: 01
subsystem: training
tags: [sgnnet, scaling, n-sweep, antihebb, smallworld, mps]

# Dependency graph
requires:
  - phase: 04-wave-architecture
    provides: "SmallWorld + Resonant + AntiHebb model stack"
provides:
  - "N-scaling sweep script (N=512..10000) at D=64 with all confirmed winners"
  - "experiment_config.py with GA_BEST_D64 and run_metadata"
  - "LEARNINGS stub with hypothesis, config, and dispatch status"
affects: [05-03, 05-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Incremental JSON save (survives mid-sweep crashes)"
    - "GA_BEST_D64 dict for Phase 5 best config tracking"
    - "run_metadata() for reproducibility embedding in result JSONs"

key-files:
  created:
    - scripts/train_step56_n_scaling.py
    - learnings/LEARNINGS_phase5_p11_n_scaling.md
  modified:
    - src/training/experiment_config.py
    - learnings/EXPERIMENT_QUEUE.md

key-decisions:
  - "losses.py bugs (fill_diagonal_, boolean indexing) already fixed in prior work -- no changes needed"
  - "Step56 queued rather than launched: 3 experiments running on Mac Studio (concurrency cap = 2)"
  - "n_groups capped at min(128, N//8) to keep topology stable at large N"

patterns-established:
  - "Incremental JSON saves: write results after each N completes so partial data survives crashes"
  - "GA_BEST_D64: Phase 5 best config reference dict alongside original Phase 4 GA_BEST"

requirements-completed: [SCALE-01, SCALE-02]

# Metrics
duration: 5min
completed: 2026-04-03
---

# Phase 5 Plan 01: N-Scaling Sweep Summary

**N-scaling sweep script for N=[512,1024,2048,4096,10000] at D=64 K_iter=8 AntiHebb alpha=0.7 -- synced to Mac Studio, queued for dispatch**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-03T12:04:06Z
- **Completed:** 2026-04-03T12:09:00Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Written train_step56_n_scaling.py: sweeps N=[512,1024,2048,4096,10000] with full model stack (SmallWorld + Resonant dynamic_z_geo + AntiHebb wpos alpha=0.7)
- Updated experiment_config.py: added GA_BEST_D64 (step29 Config C = 75.24%), added run_metadata() for reproducibility
- Verified losses.py is already clean (no fill_diagonal_, no boolean indexing, OOM guard at N=5000)
- Script synced to Mac Studio and verified parseable remotely
- LEARNINGS stub documents hypothesis, config, and ready-to-launch command

## Task Commits

Each task was committed atomically:

1. **Task 1: Write N-scaling sweep script with bug fixes** - `6993c4a` (feat)
2. **Task 2: Dispatch N-scaling sweep to Mac Studio and monitor** - `03d4cda` (docs)

## Files Created/Modified
- `scripts/train_step56_n_scaling.py` - N-scaling sweep: 5 N values, sequential, incremental JSON save
- `src/training/experiment_config.py` - Added GA_BEST_D64, run_metadata(), preserved all existing functions
- `learnings/LEARNINGS_phase5_p11_n_scaling.md` - Hypothesis, config table, dispatch status, results template
- `learnings/EXPERIMENT_QUEUE.md` - Added step56 entry to Currently Running section

## Decisions Made
- **losses.py unchanged:** Plan specified "fix if present" -- bugs (fill_diagonal_, boolean indexing) were already fixed in prior work; OOM guard at N>5000 already present
- **Queued, not launched:** 3 experiments on Mac Studio (step29c at e30, step48 at e70, step54 in Config B). Concurrency cap of 2 means we cannot launch. Script is synced and ready.
- **n_groups = min(128, N//8):** Caps group count at 128 to prevent excessive topology fragmentation at large N while maintaining block-local input structure
- **Sequential N sweep:** Each N runs to completion before next starts -- large N (especially 10000) needs full GPU memory

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added run_metadata() to worktree experiment_config.py**
- **Found during:** Task 1
- **Issue:** Worktree had older experiment_config.py missing run_metadata() function; script imports it
- **Fix:** Synced run_metadata() from main repo version into worktree experiment_config.py
- **Files modified:** src/training/experiment_config.py
- **Verification:** python3 syntax check passes; import chain verified
- **Committed in:** 6993c4a (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary for script to import correctly. No scope creep.

## Issues Encountered
- Mac Studio has 3 running experiments, exceeding 2-experiment concurrency cap. Script queued for dispatch when a slot opens. This is expected behavior per the plan's key_context guidance.

## Known Stubs
- `learnings/LEARNINGS_phase5_p11_n_scaling.md` results table is a stub (placeholder dashes). This is intentional: results will be filled after the 150-epoch sweep completes on Mac Studio. The stub documents the experiment design and dispatch readiness, which is the deliverable for this plan.
- `results/train_step56_n_scaling.json` does not yet exist (will be created by the script on Mac Studio).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Script ready on Mac Studio at `/Users/admin/ml/dhiraj/qwen2_omni/testing/scripts/train_step56_n_scaling.py`
- Launch command documented in LEARNINGS file
- Results will feed into Plan 05-03 (architecture comparison) and Plan 05-06 (synthesis)
- When step54 or step29c completes, step56 can be launched immediately

## Self-Check: PASSED

- All 5 expected files found on disk
- Both task commits verified in git log (6993c4a, 03d4cda)
- Script syntax verified locally and on Mac Studio
- No untracked generated files

---
*Phase: 05-scalable-architecture-experiments*
*Completed: 2026-04-03*
