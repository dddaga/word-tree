# Phase 5 Part 11: Neuron Scaling Sweep (N=[512,1024,2048,4096,10000])

**Status:** QUEUED -- script synced to Mac Studio, awaiting slot (3 experiments running)
**Script:** `scripts/train_step56_n_scaling.py`
**Log:** `logs/train_step56_n_scaling.log` (will exist after launch)

## Hypothesis

At D=64 with AntiHebb alpha=0.7 wpos, increasing N beyond 1024 will improve accuracy
because:
1. More neurons = more diverse W_pos directions on S^63 (not crowded at D=64)
2. Denser small-world graph = richer routing paths
3. K_in=50 fixed means each neuron samples 50/25088 = 0.2% of inputs; more neurons
   means better coverage of the 25088-dim input space

Counter-hypothesis: N=1024 is sufficient because S^63 is vast, and the bottleneck
is the routing mechanism (K_local=4, K_random=2 = 6 neighbours per neuron), not
the number of neurons.

## Config

| Param | Value |
|-------|-------|
| D | 64 |
| K_iter | 8 |
| K_in | 50 |
| K_local | 4 |
| K_random | 2 |
| AntiHebb alpha | 0.7 (wpos) |
| Resonant mode | dynamic_z_geo |
| beam_size | 32 |
| K_phase | 8 |
| Epochs | 150 |
| Batch | 128 |
| LR | 2.364e-3 (plateau, patience=10, factor=0.5) |
| lambda_safety | scaled: 0 for N>5000 |
| n_groups | min(128, N//8) |

## N Values

| N | Expected params | Safety valve | Notes |
|---|----------------|--------------|-------|
| 512 | ~35K | active (scaled lambda) | Below step29 N -- reference floor |
| 1024 | ~70K | active (scaled lambda) | Step29 replication -- expect ~75% |
| 2048 | ~140K | active (scaled lambda) | 2x scale -- key test |
| 4096 | ~280K | active (scaled lambda) | 4x scale -- diminishing returns? |
| 10000 | ~670K | DISABLED (N>5000) | Max scale -- OOM/time ceiling |

## Dispatch Status

- **2026-04-03:** Script synced to Mac Studio via rsync
- Currently blocked: 3 experiments running (step29c, step48, step54)
- Concurrency rule: launch when running count <= 1 AND RAM >= 50 GB
- Launch command:
  ```
  ssh mac-studio "cd /Users/admin/ml/dhiraj/qwen2_omni/testing && \
    tmux new-session -d -s step56_nscale \
    'd_env/bin/python3 -u scripts/train_step56_n_scaling.py --device mps 2>&1 | tee logs/train_step56_n_scaling.log'"
  ```

## Results

**PENDING** -- will be filled after experiment completes.

| N | top-1 | params | wall_time (min) | best_epoch | vs N=1024 |
|---|-------|--------|-----------------|------------|-----------|
| 512 | -- | -- | -- | -- | -- |
| 1024 | -- | -- | -- | -- | ref |
| 2048 | -- | -- | -- | -- | -- |
| 4096 | -- | -- | -- | -- | -- |
| 10000 | -- | -- | -- | -- | -- |

## Analysis

**PENDING** -- analysis will cover:
1. Accuracy vs N curve shape (linear, sublinear, plateau?)
2. Where diminishing returns begin
3. Compute efficiency: accuracy / wall_time ratio
4. Whether N=10000 successfully completes without OOM
5. Recommendation for optimal N going forward

## Key Questions This Resolves

- Is N=1024 already at the ceiling for this architecture at D=64?
- Does the O(N*K) cost scaling remain practical at N=10000?
- Should future experiments use a larger N, or invest in mechanism improvements instead?
