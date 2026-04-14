# run35 — Notes

## Config delta from run29
- `iterations: 4` (was 5)
- All else identical (uniform routing, N=4146, C=200, V=8)

## Result
- **val_best: 88.56% @ep40** (still improving — val loss still declining)
- FLOPs/fwd: ~0.72B (3/4 of run29's 0.96B; (I-1)=3 vs 4)
- vs run29 (I=5): -0.59pp at 75% FLOPs
- vs run33 (I=3): +1.40pp at 1.5x FLOPs
- vs run17 (softmax, I=5): +1.71pp at 75% FLOPs

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 22.55%  |
| ep5   | 62.37%  |
| ep10  | 75.62%  |
| ep15  | 81.15%  |
| ep20  | 84.03%  |
| ep25  | 85.95%  |
| ep30  | 87.21%  |
| ep35  | 87.67%  |
| ep40  | 88.56%  |

## Analysis
- **Iteration curve is smooth** between I=3 and I=5 (no surprises):
  - I=2: 72.05%  → below floor
  - I=3: 87.16%  → +15pp from I=2 (huge jump)
  - I=4: 88.56%  → +1.40pp from I=3 (moderate gain)
  - I=5: 89.15%  → +0.59pp from I=4 (diminishing returns)
- **Diminishing returns above I=3.** Each additional iteration gives less accuracy for the same FLOPs cost.
- **Best FLOPs efficiency at I=3:** run33 gets 87.16% at half FLOPs of run29 — 89.2% of accuracy at 50% cost.
- Still improving at ep40 — could reach 89%+ with more epochs.
- Train acc 84.57% ≈ run29 (85.52%) — similar generalization.

## Verdict
DONE. Smooth iteration curve confirmed between I=3 and I=5. Diminishing returns clear.
**Pareto frontier (accuracy vs FLOPs):**
- 0.24B: run31 (C=50, I=5) → 79.54%
- 0.48B: run33 (C=200, I=3) → 87.16% ★ best value
- 0.72B: run35 (C=200, I=4) → 88.56%
- 0.96B: run29 (C=200, I=5) → 89.15%
Hitting a ceiling around 89%. Next: test D=16 (vector_dim) to see if larger feature dim helps.
