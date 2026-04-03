# Phase 5 Architecture Comparison — Track A

**Generated:** 2026-04-03
**Status:** Partial — 3 of 12 rows complete; 9 pending experiment completion

| Architecture | N | D | K_iter | top-1 | Params | Key Mechanism | Source | Status |
|---|---|---|---|---|---|---|---|---|
| SmallWorld baseline | 1024 | 64 | 8 | 56.28% | — | none | step29 Ref | DONE |
| SmallWorld + AntiHebb(0.5) | 1024 | 64 | 8 | 70.14% | — | AntiHebb α=0.5 wpos | step29 A | DONE |
| **SmallWorld + AntiHebb(0.7)** | **1024** | **64** | **8** | **75.24%** | **—** | **AntiHebb α=0.7 wpos** | **step29 C** | **DONE — ALL-TIME BEST** |
| SmallWorld + AntiHebb(0.7) | 512 | 64 | 8 | pending | — | AntiHebb α=0.7 wpos | step56 | PENDING |
| SmallWorld + AntiHebb(0.7) | 2048 | 64 | 8 | pending | — | AntiHebb α=0.7 wpos | step56 | PENDING |
| SmallWorld + AntiHebb(0.7) | 4096 | 64 | 8 | pending | — | AntiHebb α=0.7 wpos | step56 | PENDING |
| SmallWorld + AntiHebb(0.7) | 10000 | 64 | 8 | pending | — | AntiHebb α=0.7 wpos | step56 | PENDING |
| ProximityWave + AntiHebb(0.7) | 1024 | 64 | 8 | pending | 66,176 | k-NN + phasor + AntiHebb inline | exp3 N1024 | PENDING |
| ProximityWave + AntiHebb(0.7) | 4096 | 64 | 8 | pending | 262,720 | k-NN + phasor + AntiHebb inline | exp3 N4096 | PENDING |
| SmallWorld + reflect(leaky a=0.1) | 1024 | 64 | 8 | pending | — | Reflection α=0.1 θ=0.0 | exp4 B | QUEUED |
| SmallWorld + reflect(hard a=1.0) | 1024 | 64 | 8 | pending | — | Reflection α=1.0 θ=0.5 | exp4 C | QUEUED |
| SmallWorld + reflect(medium a=0.3) | 1024 | 64 | 8 | pending | — | Reflection α=0.3 θ=0.0 | exp4 D | QUEUED |

## Confirmed Results

### SmallWorld + AntiHebb Sweep (N=1024, D=64, K_iter=8)

| Config | alpha_ahebb | top-1 | Delta vs baseline |
|---|---|---|---|
| Baseline (no AntiHebb) | — | 56.28% | reference |
| AntiHebb alpha=0.3 | 0.3 | 65.58% | +9.30pp |
| AntiHebb alpha=0.5 | 0.5 | 70.14% | +13.86pp |
| **AntiHebb alpha=0.7** | **0.7** | **75.24%** | **+18.96pp** |

**Takeaway:** AntiHebb suppression shows monotonic gains with alpha up to 0.7.
All-time best is 75.24% (step29 Config C). GEN4_BEST in experiment_config.py uses alpha=1.0
(step29c pending) but 0.7 is the confirmed data point.

### D=64 Routing Calibration (40ep, step22b)

| alpha_reflect | top-1 (40ep) |
|---|---|
| 0.0 | 49.20% |
| 0.1 | 50.19% |
| 0.3 | 50.19% |
| **0.5** | **52.94%** |

**Takeaway:** alpha_reflect=0.5 beats lower values by +2.75pp at 40ep.
This is the SmallWorld + Resonant baseline before AntiHebb was added.

## Pending Data — How to Fill

### Step 56 (N-scaling sweep)
```bash
# On Mac Studio — launch when slot opens (needs count <= 1 AND RAM >= 50GB):
ssh mac-studio 'cd /Users/admin/ml/dhiraj/qwen2_omni/testing && \
  /opt/homebrew/bin/tmux new-session -d -s step56 \
  "d_env/bin/python3 -u scripts/train_step56_n_scaling.py --device mps 2>&1 | tee logs/train_step56_n_scaling.log"'

# Sync results when done:
rsync -avz mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/results/train_step56_n_scaling.json results/
```

### Exp3 ProximityWave
```bash
# Currently running as exp3_pw session — sync when done:
rsync -avz mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/results/exp3_proxwave.json results/
```

### Exp4 Reflection
```bash
# Scripts synced to Mac Studio — launch when slot opens:
ssh mac-studio 'cd /Users/admin/ml/dhiraj/qwen2_omni/testing && \
  /opt/homebrew/bin/tmux new-session -d -s exp4_ref \
  "d_env/bin/python3 -u scripts/train_exp4_reflection.py --device mps 2>&1 | tee logs/train_exp4_reflection.log"'

# Sync results when done:
rsync -avz mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/results/exp4_reflection.json results/
```

## Architecture Notes

### SmallWorld (O(N·K) routing)
- Fixed fan-in index tables: conn_hh [N, K_hh] built at init from Watts-Strogatz graph
- Real-valued activations, K_iter gather-sum steps
- No topology adaptation during training
- Fast: ~300ms/batch at N=1024, D=64

### ProximityWave (O(N·K) routing, periodic topology rebuild)
- K-NN based conn_hh rebuilt from W_pos every 10 epochs (O(N²) rebuild, not per-batch)
- Complex phasor routing: Z_re + Z_im with distance-based phase rotation
- Inline anti-Hebbian suppression on routing weights
- Slower: ~595ms/batch at N=1024, D=64 (phasor overhead)

### Reflection (O(N·K) routing, sign-conditional)
- Wraps SmallWorld base, overrides routing loop
- Positive activations: propagate forward via gather-sum (standard)
- Negative activations: bounce back as self-inhibitory signal
- Leaky (alpha=0.1): low risk, all negatives participate
- Hard (alpha=1.0, theta=0.5): full strength, only strongly negative

---
*Updated when pending experiments complete. Track A deliverable for Phase 5 Plan 03.*
