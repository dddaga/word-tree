# Phase 5 Plan 12: ProximityWave at N=1024 and N=4096

**Date:** 2026-04-03
**Status:** DISPATCHED (running on Mac Studio in `exp3_pw` tmux session)
**Log:** `mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/logs/train_exp3_proxwave.log`

## Hypothesis

Geometry-based k-NN topology (no fixed index-based groups) + dynamic phasor routing
+ periodic reconnection outperforms SmallWorld's fixed-group approach at the same N.

Key differences from SmallWorld+Resonant+AntiHebb:
1. **conn_hh from W_pos k-NN** (not index-based Watts-Strogatz groups)
2. **Phasor routing** (Z_re/Z_im with distance-based phase rotation, not real-valued)
3. **Periodic reconnection** (topology rebuilds from W_pos every 10 epochs)
4. **Inline anti-Hebbian** (wpos-similarity suppression in routing weights, alpha=0.7)

## Architecture

```
SGNNET_ProximityWave(
  N_hidden=1024/4096, D=64, K_iter=8,
  K_local=4, K_random=2, K_in=50,
  reconnect_every=10, encoding_mode='fourier',
  norm_mode='l2', anti_hebb_alpha=0.7
)
```

Forward pass: `_seed -> _route (8x sparse_phasor_route) -> _readout`
- All operations O(N*K*B*D) -- no O(N^2) in forward
- k-NN rebuild O(N^2) at epoch level only (every 10 epochs)

## Configs

| Config | N_hidden | D | K_iter | K_local | K_random | Params | Status |
|--------|----------|---|--------|---------|----------|--------|--------|
| N1024 | 1024 | 64 | 8 | 4 | 2 | 66,176 | Running |
| N4096 | 4096 | 64 | 8 | 4 | 2 | 262,720 | Pending (after N1024) |

Both trained for 150 epochs, batch=128, plateau LR scheduler.

## Initial Observations

- **N=1024 forward pass: 595 ms/batch** (128 samples)
  - This is above the 500ms target for N=4096
  - Phasor routing at D=64 with 8 iterations is more expensive than SmallWorld's
    real-valued gather-sum routing
  - The cost comes from per-edge distance computation, phase rotation, and
    complex-number operations in sparse_phasor_route

## Comparison Baselines (at N=1024)

| Model | Config | top-1 | Params | Note |
|-------|--------|-------|--------|------|
| SmallWorld+Resonant (no AntiHebb) | D=64 K_iter=8 | 56.41% | ~66K | step29 Ref |
| SmallWorld+Resonant+AntiHebb(0.7) | D=64 K_iter=8 | **75.24%** | ~66K | step29 Config C (ALL-TIME BEST) |
| ProximityWave+AntiHebb(0.7) | D=64 K_iter=8 | TBD | 66,176 | This experiment |

## Expected Outcomes

1. **Optimistic:** k-NN topology + phasor routing captures richer geometric
   structure than SmallWorld, potentially matching or exceeding 75.24%.
2. **Likely:** Some gain over plain SmallWorld (56.41%) from topology adaptation,
   but may not match the full Resonant+AntiHebb stack because ProximityWave's
   phasor routing lacks the beam-based phase inhibition of SGNNET_Resonant.
3. **Pessimistic:** Phasor routing at D=64 may show the same issues as signed
   coupling (cos-sim noise on S^63), resulting in near-random phase rotations.

## Known Limitations

- **No phase inhibition:** ProximityWave uses phasor routing (excitatory+phase)
  but lacks the Turing two-scale inhibitory signal that SGNNET_Resonant provides.
  The anti-Hebbian suppression partially compensates but through a different mechanism.
- **Forward pass speed:** 595ms at N=1024 is slower than SmallWorld (~300ms).
  At N=4096 this will be even more expensive due to the per-edge distance computation.

## Sync Commands (for when results are ready)

```bash
rsync -avz mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/results/exp3_proxwave.json results/
ssh mac-studio "tail -50 /Users/admin/ml/dhiraj/qwen2_omni/testing/logs/train_exp3_proxwave.log"
```

---

*Updated: 2026-04-03 -- experiment dispatched, awaiting results*
