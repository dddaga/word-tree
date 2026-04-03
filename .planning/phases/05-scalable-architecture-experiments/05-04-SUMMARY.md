---
phase: 05-scalable-architecture-experiments
plan: 04
subsystem: experiment-infrastructure
tags: [arm1, arm2, gen4, antihebb, lr-schedule, kiter, lowrank-mixing, compound-config]
dependency_graph:
  requires: [step22b-calibration, step29-antihebb, step29b-mechanisms]
  provides: [gen4-compound-config, step53-lowrank-script, arm1-arm2-learnings]
  affects: [experiment_config.py, gen4-baseline]
tech_stack:
  added: []
  patterns: [gen4-compound-stacking, structured-sub-D-mixing, frequency-pair-mixing]
key_files:
  created:
    - scripts/train_step53_lowrank_mixing.py
    - scripts/train_step32_gen4_compound.py
    - learnings/LEARNINGS_phase5_p8_arm1_arm2.md
    - results/train_step48_kiter_sweep_d64.json
    - results/train_step54_warm_restart_lr.json
    - results/train_step29c_mechanisms_calibrated.json
  modified:
    - src/training/experiment_config.py
decisions:
  - "AntiHebb alpha 0.7 -> 1.0: monotonic scaling confirmed at D=64 (step29c Phase 1)"
  - "Plateau LR retained: 70.78% vs 66.98% warm restarts (step54)"
  - "K_iter stays at 8: K_iter=12 loses 3.16pp without AntiHebb (step48)"
  - "Fast W_phase excluded from Gen4: 49% < 56% baseline (D x D interference pattern)"
  - "Phase excitatory alpha=0.1 included as optional compound test (marginal at 57.91%)"
metrics:
  duration: 10min
  completed: "2026-04-03T12:15:00Z"
---

# Phase 5 Plan 04: ARM 1+2 Sync, Gen4 Compound, and Low-Rank Mixing Summary

AntiHebb alpha=1.0 confirmed as Gen4 primary mechanism (monotonic 40ep scaling to 70.98%); plateau LR beats warm restarts by 3.80pp; K_iter>8 hurts; step53 low-rank mixing and step32 Gen4 compound scripts written and synced.

## Task Completion

| Task | Name | Commit | Status |
|------|------|--------|--------|
| 1 | Sync ARM 2 results + write step53 | 4703b5d | Complete |
| 2 | Sync ARM 1 + compose Gen4 + LEARNINGS | c991d58 | Complete |

## Key Results

### ARM 1 -- step29c Mechanism Calibration (Partial: Phase 1 done, Phase 2 pending)

**AntiHebb** (COMPLETE): Monotonic alpha scaling at D=64 --
- alpha=0.1: 48.56% | alpha=0.3: 54.42% | alpha=0.5: 59.82% | alpha=0.7: 64.69% | alpha=1.0: **70.98%**
- Full surround suppression is optimal. Adopted for Gen4.

**Phase excitatory** (COMPLETE): Marginal at D=64 --
- alpha=0.1: 57.91% | alpha=0.3: 59.34% | alpha=0.5: 45.83% | alpha=1.0: 33.43%
- Low-dose OK, high-dose catastrophic. Optional compound test.

**Fast W_phase** (2/4 done): Dead at D=64 --
- alpha=0.1/tau=0.25: 49.45% | alpha=0.3/tau=0.25: 49.30%
- D x D cross-dim mixing confirmed incompatible with Fourier encoding.

### ARM 2 -- New Mechanisms

**step48 K_iter sweep** (2/8 complete): K_iter>8 hurts without AntiHebb --
- Ref (K_iter=8): 58.24% | A (K_iter=12): 55.08% | B (K_iter=16): ~51% at e70
- Critical AntiHebb + deep K_iter configs (E/F/G) still pending.

**step54 LR schedule** (2/4 complete): Plateau wins --
- Plateau: 70.78% | WarmRestarts T_mult=1: 66.98%
- LR never decayed (plateau patience never triggered). Effectively constant-LR.

**step52 High-D routing**: Script synced. Dispatch blocked by concurrency (3 running).

**step53 Low-rank mixing**: Script written (7 configs: Ref + A-F). Synced. Dispatch blocked.

### Gen4 Adopted Configuration

| Parameter | Previous | Gen4 | Source |
|-----------|----------|------|--------|
| AntiHebb alpha | 0.7 | **1.0** | step29c P1 calibration |
| alpha_reflect | 0.3 | **0.5** | step22b calibration |
| LR schedule | plateau | plateau | step54 confirmation |
| K_iter | 8 | 8 | step48 (no gain at >8) |
| fast W_phase | -- | excluded | step29c (49% < 56%) |
| D x D mixing | -- | excluded | step30/37/29c pattern |

## Scripts Written

1. **train_step53_lowrank_mixing.py**: 7 configs testing structured sub-D mixing (low-rank rank=4/8, frequency-pair 2x2, group 8x8). All preserve Fourier encoding structure.

2. **train_step32_gen4_compound.py**: 8 configs testing Gen4 compound stacking (AntiHebb alpha=1.0 + phase_exc + centering diversity).

Both scripts synced to Mac Studio via rsync. Dispatch blocked by 3-process concurrency.

## Deviations from Plan

### Auto-adjusted scope: partial results instead of full sync

**Found during:** Tasks 1 and 2
**Issue:** All three experiments (step29c, step48, step54) are still running on Mac Studio. Only partial results available (completed configs, not full JSON outputs).
**Adjustment:** Created partial result JSONs from log parsing with `_partial: true` flags. Documented status of each config. Analysis based on available data.
**Impact:** Gen4 compound config is preliminary -- some ARM 2 results (step48 E/F/G with AntiHebb, step54 B/C) will refine the final configuration.

### Auto-adjusted scope: dispatch blocked by concurrency

**Found during:** Task 1 step 6
**Issue:** 3 experiments running on Mac Studio (concurrency cap = 2). Cannot dispatch step52, step53, or step32.
**Adjustment:** Scripts written, validated, and synced to Mac Studio. Will dispatch when slot opens (~6-12h as step54 configs complete).
**Impact:** None -- scripts are ready; only dispatch timing affected.

## Known Stubs

None. All scripts are complete and self-contained. Result JSONs are partial but contain real data from experiment logs.

## Self-Check: PASSED

All 8 deliverable files found. Both task commits (4703b5d, c991d58) verified in git log.
