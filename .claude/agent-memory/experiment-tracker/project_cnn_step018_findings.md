---
name: cnn_step018_findings
description: Confirmed mechanisms from cnn_step018 EfficientVGG GA T1 — ep35 dip, GA3 late convergence, CReLU effect
metadata:
  type: project
---

Findings from cnn_step018 T1 (2026-06-21). All 4 configs completed.

(1) ep35 LR-dip CONFIRMED systematic: All GA1/GA2/GA3/GA4 show val drop when LR≈1.70e-4, which occurs at ep35 of 75ep cosine schedule. Expect equivalent dip at ep70 in T2 (T_max=150). Do not read ep70 T2 val as convergence — check ep75+ or final.

(2) GA3 late convergence: C2=96 wide bottleneck slows early training (ep40=69.71%) but wins final acc (72.97% vs GA1 72.61%). T0 rank-order (GA2>GA3) reversed at T1. Lesson: wide bottleneck needs more epochs to saturate — T0 rank not reliable for C2-wide configs.

(3) GA4 no-CReLU hypothesis CONFIRMED: T0 gap vs GA1 was 1.52pp; T1 gap narrows to 0.15pp (72.76% vs 72.61%). CReLU accelerates early training but does not determine final accuracy. T0 penalizes no-CReLU unfairly for short-run configs.

(4) Best config: GA3 — C=(32,96,384), exp=2, crelu=True. Best efficiency config: GA4 (62.9M MACs / 304K params, 72.76%).

**Why:** These findings inform T2 monitoring strategy and config weighting.
**How to apply:**
- Expect ep70 dip in T2; use ep80+ for final acc reads.
- GA3 is the primary bet for T2 STRONG threshold; GA4 is efficiency bet.
- For future T0 ablations involving CReLU or wide bottleneck: extend T0 to 30ep or interpret T0 rank with caution.

See [[project_cnn_ga_evgg_line]] for full step status.
