---
phase: 05-scalable-architecture-experiments
plan: 02
subsystem: architecture
tags: [sgnnet, proximity-wave, knn-topology, phasor-routing, anti-hebbian, d64, scaling]

# Dependency graph
requires:
  - phase: 04-sgnnet-wave-architecture-experiments
    provides: SmallWorld model, Resonant wrapper, AntiHebbian mechanism
provides:
  - SGNNET_ProximityWave model with D=64 fourier encoding, l2 norm, inline anti-Hebbian
  - Training script for N=1024 and N=4096 with phasor routing benchmarks
  - Dispatched experiment on Mac Studio (exp3_pw session)
affects: [05-03, 05-04, phase-6-pca-compression]

# Tech tracking
tech-stack:
  added: []
  patterns: [inline anti-Hebbian suppression in routing weights, k-NN topology rebuild logging]

key-files:
  created:
    - scripts/train_exp3_proxwave.py
    - results/exp3_proxwave.json
    - learnings/LEARNINGS_phase5_p12_proxwave.md
  modified:
    - src/sgnnet/model_proximity_wave.py

key-decisions:
  - "Inline anti-Hebbian suppression in phasor routing weights rather than separate wrapper (avoids interface mismatch with phasor Z_re/Z_im)"
  - "Fourier encoding mode added for D=64 compatibility (original model only supported D=4 linear encoding)"

patterns-established:
  - "ProximityWave inline anti-Hebbian: multiply routing strength weights by (1 - alpha * cos_sim(W_pos)) before phasor aggregation"
  - "Topology change logging in tick_epoch: print changed_rows/N_hidden for monitoring adaptation"

requirements-completed: [SCALE-03, SCALE-04]

# Metrics
duration: 9min
completed: 2026-04-03
---

# Phase 5 Plan 02: ProximityWave at Scale Summary

**SGNNET_ProximityWave with D=64 fourier encoding, inline anti-Hebbian (alpha=0.7), k-NN topology rebuild, dispatched for N=1024 and N=4096 on Mac Studio**

## Performance

- **Duration:** 9 min
- **Started:** 2026-04-03T12:04:10Z
- **Completed:** 2026-04-03T12:13:23Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- Enhanced SGNNET_ProximityWave model with fourier encoding (D=64), l2 normalization, inline anti-Hebbian suppression (alpha=0.7 wpos), and topology change logging
- Wrote training script with N=1024 and N=4096 configs matching confirmed winning hyperparameters (D=64, K_iter=8, plateau LR)
- Verified O(N*K) forward pass: cdist only in build_knn_conn (epoch-level k-NN rebuild), not in per-batch forward
- Successfully dispatched experiment to Mac Studio (exp3_pw session, 0 concurrent experiments, 175GB RAM free)
- Initial benchmark: N=1024 forward pass = 595ms/batch (phasor routing overhead vs SmallWorld ~300ms)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement SGNNET_ProximityWave model and training script** - `3ef9073` (feat)
2. **Task 2: Dispatch ProximityWave experiments and analyze results** - `42fd692` (chore)

## Files Created/Modified
- `src/sgnnet/model_proximity_wave.py` - Enhanced with fourier encoding, l2 norm, inline anti-Hebbian, topology logging
- `scripts/train_exp3_proxwave.py` - Training script for N=1024 and N=4096 with D=64 K_iter=8
- `results/exp3_proxwave.json` - Placeholder results (experiment dispatched, awaiting completion)
- `learnings/LEARNINGS_phase5_p12_proxwave.md` - Hypothesis, configs, baselines, initial observations

## Decisions Made

1. **Inline anti-Hebbian instead of wrapper chain:** The standard AntiHebbian wrapper expects SGNNET_Resonant's real-valued routing interface, which is incompatible with ProximityWave's complex phasor (Z_re/Z_im) routing. Instead of creating a separate AntiHebbian-for-phasor wrapper, the suppression is computed from W_pos cosine similarity and applied directly to routing strength weights within sparse_phasor_route. This keeps the phasor routing pure O(N*K) and the mechanism identical (wpos-similarity decorrelation).

2. **Fourier encoding for D=64:** The original model_proximity_wave.py used compute_spatial_encoding (3 dims, D=4 only). Added encoding_mode='fourier' parameter to use compute_fourier_encoding which produces D-1 spatial dims for any D>=4.

3. **l2 norm mode:** Validated in Phase 5 Step 1 as the winning normalization (23.9% vs masked=10.2% vs relu=8.2%). Added norm_mode parameter to ProximityWave (was previously hardcoded to masked).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Remote experiment_config.py missing run_metadata**
- **Found during:** Task 2 (initial dispatch)
- **Issue:** Mac Studio's experiment_config.py was outdated (missing run_metadata function), causing ImportError on launch
- **Fix:** Force-rsync'd experiment_config.py from main repo to Mac Studio using --checksum flag
- **Files modified:** remote experiment_config.py (Mac Studio)
- **Verification:** re-launched experiment, confirmed successful import
- **Committed in:** N/A (remote file only)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Minimal -- stale remote dependency resolved by sync.

## Issues Encountered
- tmux not in default PATH on Mac Studio SSH session -- resolved by using full path `/opt/homebrew/bin/tmux`
- N=1024 forward pass at 595ms/batch is slower than expected; phasor routing with 8 iterations at D=64 involves per-edge distance computation, phase rotation, and complex-number gather-sum that is more expensive than SmallWorld's simple real-valued gather-sum

## Known Stubs

- `results/exp3_proxwave.json` has null values for top1, ms_per_epoch, topology_rebuilds (experiment dispatched but not yet complete)
- LEARNINGS file has TBD for ProximityWave accuracy results

These are intentional: the experiment requires multi-hour execution on Mac Studio. Results will be synced when complete per the instructions in LEARNINGS_phase5_p12_proxwave.md.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- ProximityWave experiment is running on Mac Studio (exp3_pw session)
- Results will be available for Plan 05-03 (architecture comparison table) when training completes
- Sync command: `rsync -avz mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/results/exp3_proxwave.json results/`

---
*Phase: 05-scalable-architecture-experiments*
*Completed: 2026-04-03*
