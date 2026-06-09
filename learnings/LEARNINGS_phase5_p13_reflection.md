# Phase 5 Plan 13: Reflection Routing Ablation

**Date:** 2026-04-03
**Step:** exp4_reflection
**Status:** Script implemented, synced; dispatch queued (Mac Studio over concurrency cap)

## Hypothesis

Signal reflection routing — strongly negative activations "bounce back" as self-inhibitory signal — creates input-dependent routing that could improve classification accuracy over AntiHebb baseline.

Motivation: standard relu gating discards below-threshold activations entirely. Reflection reclaims that signal. Negative activation tells which neurons actively *suppress* feature; routing suppression back to source creates contrast-enhancement effect (winner-take-all at neuron level).

## Architecture Design

SGNNET_Reflection wraps SGNNET_SmallWorld, overrides routing loop:

```python
Z_prop    = relu(Z)                           # positive propagates
Z_reflect = alpha_reflect * relu(-Z - theta)  # negative bounces back
Z_struct  = Z_prop[:, conn_hh, :].sum(dim=2)  # gather-sum neighbours
Z_new     = Z_struct - Z_reflect              # excite + self-inhibit
```

Key design choices:
1. `Z_struct - Z_reflect`: minus sign makes reflection self-inhibitory (not additive noise)
2. `relu(-Z - theta)`: only activations below -theta reflect; theta controls selectivity
3. Dead neuron tracking: `(Z.norm(dim=-1) < 1e-6).float().mean()` per step
4. Warning if dead_frac > 5% (dying neuron cascade risk)

## Ablation Configs (N=1024, D=64, K_iter=8, 100 epochs)

| Config | Mechanism | alpha_reflect | theta | Expected behavior |
|---|---|---|---|---|
| A | AntiHebb(0.7) baseline | — | — | Reference: 75.24% (150ep); expect ~70% at 100ep |
| B | Leaky reflect | 0.1 | 0.0 | Low-risk: weak bounce-back, all negatives |
| C | Hard reflect | 1.0 | 0.5 | High-risk: strong bounce-back, only strongly negative |
| D | Medium reflect | 0.3 | 0.0 | Intermediate: moderate bounce-back |

Note: Configs B, C, D use SmallWorld directly (no Resonant wrapper or AntiHebb). Config A uses full SmallWorld + Resonant + AntiHebb stack. Ablation of reflection as alternative to AntiHebb, not additive.

## Dead Neuron Risk Analysis

Leaky reflect (alpha=0.1, theta=0.0):
- Every negative activation contributes small self-inhibitory signal
- Expected: dead_frac < 1% (gentle, global effect)
- Low cascade risk

Hard reflect (alpha=1.0, theta=0.5):
- Only activations < -0.5 trigger full-strength bounce-back
- Can create strong localized suppression
- Risk: if many neurons end up < -0.5, self-inhibition cascade kills them
- Monitor dead_frac carefully

Medium reflect (alpha=0.3, theta=0.0):
- Intermediate; safer than hard-reflect
- Expected dead_frac: 1-3%

## Results (pending — queued for dispatch)

Mac Studio state at 2026-04-03:
- Running: step29c (pid 89895), step48 (pid 91447), step54 (pid 95059) — 3 experiments
- Concurrency cap: 2 concurrent (both conditions: count <= 1 AND RAM >= 50 GB)
- Scripts synced to Mac Studio: train_exp4_reflection.py, model_reflection.py

Launch when slot opens:
```bash
ssh mac-studio 'cd /Users/admin/ml/dhiraj/qwen2_omni/testing && \
  /opt/homebrew/bin/tmux new-session -d -s exp4_ref \
  "d_env/bin/python3 -u scripts/train_exp4_reflection.py --device mps 2>&1 | tee logs/train_exp4_reflection.log"'
```

Sync when complete:
```bash
rsync -avz mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/results/exp4_reflection.json results/
```

## Results Table (fill when available)

| Config | top-1 | vs A baseline | dead_max | dead_warning | Status |
|---|---|---|---|---|---|
| A: AntiHebb(0.7) ref | TBD | — | 0% | N/A | QUEUED |
| B: leaky-reflect(0.1) | TBD | TBD | TBD | TBD | QUEUED |
| C: hard-reflect(1.0) | TBD | TBD | TBD | TBD | QUEUED |
| D: medium-reflect(0.3) | TBD | TBD | TBD | TBD | QUEUED |

## Architecture Comparison Context

From existing experiments (complete data):
- SmallWorld baseline (no mechanisms): 56.28% (step29 Ref, 150ep)
- SmallWorld + AntiHebb(0.7): **75.24%** (step29 C, 150ep) — all-time best
- Routing calibration (step22b, 40ep): alpha_reflect=0.5 gave 52.94% vs 49.20% baseline

Pending (to fill arch_comparison.md):
- step56: N-scaling [512, 2048, 4096, 10000] — queued, script synced
- exp3_proxwave: ProximityWave N=1024/4096 — running on Mac Studio
- exp4_reflection: all 4 configs — queued (this experiment)

## Interpretation Guide (for when results arrive)

**If B > A (leaky-reflect beats AntiHebb):**
- Reflection viable alternative to lateral inhibition
- Mechanism: self-correction via negative activation feedback
- Next: combine reflection + AntiHebb; test at multiple N

**If B ≈ A:**
- Reflection provides different path to same top-1
- Could be useful as regularizer at larger N
- Next: test at N=2048/4096 (may scale differently)

**If B < A:**
- Reflection routing insufficient without AntiHebb lateral suppression
- Next: add AntiHebb on top of reflection, test combined

**If dead_frac > 5% for C (hard-reflect):**
- Strong reflection threshold at 0.5 too aggressive
- Self-inhibitory cascade kills marginally negative neurons
- Fix: reduce theta or alpha_reflect; use leaky variant instead

## Recommendation for arch_comparison.md (preliminary, before results)

Based on mechanism analysis:
1. **SmallWorld + AntiHebb(0.7)** remains most likely winner at N=1024
2. Reflection cheaper alternative (no complex phasor math, no wrapper overhead)
3. ProximityWave phasor routing adds 2x compute overhead vs SmallWorld — accuracy improvement would need to be substantial to justify

Final recommendation updated when all three Track A experiments complete.