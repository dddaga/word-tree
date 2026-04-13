# run19 — Notes

## Config delta from run17
- `cardinality: 100` (was 200)
- All else identical to run17

## Result
- **val_best: 80.66% @ep40** (still improving at final epoch)
- FLOPs/fwd: 0.48B (2x reduction vs run17)
- Baseline: run17 → 86.85% @ep38

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep10  | 65.04%  |
| ep20  | 73.96%  |
| ep30  | 78.01%  |
| ep35  | 78.93%  |
| ep38  | 80.64%  |
| ep39  | 80.56%  |
| ep40  | 80.66%  |

## Analysis
- **-6.19pp vs run17** at half the FLOPs. This is a substantial drop.
- Convergence is slow but run is still improving at ep40 — underfitting. C=100 may need more epochs.
- However: the accuracy gap is large enough that C=100 alone is not an acceptable trade-off at 2x savings.
- The sweep continues with C=50 (run20) to map the full curve.

## Verdict
DONE. C=100 → -6.19pp vs run17. Cardinality matters more than expected. FLOPs-accuracy trade-off is unfavorable at 2x reduction. Curve continues: run20 (C=50), run21 (C=25), run22 (C=4).
