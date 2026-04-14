# run33 — Notes

## Config delta from run29
- `iterations: 3` (was 5)
- All else identical to run29 (uniform routing, N=4146, C=200, V=8)

## Result
- **val_best: 87.16% @ep40** (still improving — val loss still declining)
- FLOPs/fwd: ~0.48B (2x savings from iterations, (I-1)=(3-1)=2 vs (5-1)=4)
- vs run17 (softmax, I=5): **+0.31pp at HALF the FLOPs** — beats original baseline
- vs run29 (uniform, I=5): -1.99pp at half the FLOPs
- vs run32 (uniform, C=100, I=5): **+1.27pp at same FLOPs (0.48B)**

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 21.02%  |
| ep5   | 62.34%  |
| ep10  | 74.70%  |
| ep15  | 79.95%  |
| ep20  | 82.96%  |
| ep25  | 84.94%  |
| ep30  | 86.20%  |
| ep35  | 86.88%  |
| ep40  | 87.16%  |

## Analysis
- **Iterations are a BETTER FLOPs lever than cardinality.** At the same 0.48B FLOPs budget:
  - Reduce C (run32, C=100, I=5): 85.89%
  - Reduce I (run33, C=200, I=3): 87.16% → **+1.27pp better**
- **Intuition:** Reducing iterations preserves the full connectivity graph; the model propagates fewer times but uses all its connections. Reducing cardinality permanently removes edges, losing expressive capacity.
- **Surprisingly competitive with run29 (I=5, full budget):** Only -1.99pp for half the FLOPs. Uniform routing converges rapidly — much of the work is done in the first few iterations.
- **Beats softmax run17 (86.85%)** with half the FLOPs. Combined with uniform routing's accuracy gain, iteration reduction is now strictly better than the original architecture.
- Still improving at ep40 (val_loss: 0.4146→0.4115) — likely reaches 87-88% with more epochs.
- Train acc 83.13% vs run29's 85.52% — some underfitting from fewer iterations, but not catastrophic.

## Pareto frontier updated
| Config | Val | FLOPs | Pareto? |
|--------|-----|-------|---------|
| run31 (uniform, C=50, I=5) | 79.54% | 0.24B | YES (best at 0.24B) |
| **run33 (uniform, C=200, I=3)** | **87.16%** | **0.48B** | **YES (best at 0.48B) — beats run32** |
| run29 (uniform, C=200, I=5) | 89.15% | 0.96B | YES (best at 0.96B) |

## Verdict
DONE. **Iterations beat cardinality as a FLOPs reduction mechanism.** run33 at I=3 achieves 87.16% — beats softmax baseline (86.85%) at half the FLOPs. Combined with uniform routing gain, we now have 2-3x FLOPs efficiency over run17.
Next: push iterations lower (I=2) to test the floor of iteration reduction.
