---
name: project_cnn_ga_evgg_line
description: CNN EfficientVGG GA experiment line status — regularization phase CLOSED, now pivoting to architecture widening (GA5)
metadata:
  type: project
---

GA line status (updated 2026-06-22):

**T0/T1/T2 progression DONE:**
- step016-019: GA search + T1 + T2 complete. GA2=77.25% (seed42), mean=76.80%±0.92% EFF-PARETO.
- step020: GA2 multi-seed T2 mean=76.80%±0.92%.
- step021: GA2+Mixup(α=0.2) T1 baseline = 73.61%. Peak shifted ep25→ep65 (Mixup CONFIRMED).
- step022: GA2+Mixup T2 mean=77.32%±0.56% EFF-PARETO (gap −0.29pp to STRONG=77.61%).

**Regularization sweep CLOSED (steps 021-028) — 6 experiments, all within ±0.60pp of 73.61%:**
- step023: α sweep — α=0.3 best (74.24%), α=0.4/0.5 NO-GAIN
- step024: α=0.3 T2 — 77.17%±0.82% (WORSE than α=0.2 77.32%)
- step025: LabelSmoothing CLOSED — double-softens CE, conflicts with DKD
- step026: Loss weight ablation CLOSED — feat_cos@0.50 load-bearing, irreplaceable
- step027: CutMix(α=0.2) — 73.81%, Δ=+0.20pp NO-GAIN
- step028: MixCutMix(α=0.2) — 73.94%, Δ=+0.33pp NO-GAIN (best but still below 74.11%)

**Architecture bottleneck CONFIRMED:** Cannot break past ~74% T1 with GA2 arch via regularization.

**Current pivot: GA5 width experiment (step029):**
- GA5: C=(48,96,384) k=3 exp=2 crelu=True, side=False
- 523K params / 141.3M MACs (under Ref 583K/215.3M — EFF-PARETO eligible)
- HYPOTHESIS: GA2 bottleneck at C1=32/C2=64 early feature extraction (too narrow for VGG 25088-dim feat_cos target)
- Advance threshold: ≥74.11% vs GA2 baseline 73.61%
- Script: `scripts/cnn_distiller/train_cnn_step029_ga5_t1.py`
- Status: QUEUED (mini_mps FREE)

**Key confirmed findings:**
- CReLU in block3 (early placement) = +2-3pp vs late (CONFIRMED, step008)
- dil=4 KILLED (42% slower, −0.74pp, step008)
- exp=2 (inverted bottleneck) differentiator for GA2 over GA1
- feat_cos weight 0.50 is load-bearing (monotonic degradation when reduced, step026)
- Mixup α=0.2 best regularizer; α=0.3 T2 actually worse than α=0.2

**Why:** EfficientVGG targets efficiency over MultiScaleCNN (capped at ~42%).
**How to apply:** Architecture capacity is now the lever — regularization exhausted. Width > depth hypothesis based on early-stage representation bottleneck.

See [[cnn_step018_findings]] for T1 confirmed mechanisms.
