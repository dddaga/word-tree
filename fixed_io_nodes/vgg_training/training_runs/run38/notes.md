# run38 — Notes

## Config delta from run36
- `cardinality: 100` (was 200)
- All else identical (uniform routing, N=2048, V=16, I=3)

## Result
- **val_best: 85.91% @ep39** (still improving, val_loss still declining)
- FLOPs/fwd: ~0.23B (half of run36's 0.45B)
- vs run36 (C=200): -3.64pp at HALF the FLOPs
- vs run32 (D=8, C=100, 0.48B): **+0.02pp at HALF the FLOPs** — D=16 doubles efficiency
- vs run31 (D=8, C=50, 0.24B — prior Pareto best): **+6.37pp at similar FLOPs**

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep5   | 66.23%  |
| ep10  | 75.25%  |
| ep15  | 79.88%  |
| ep20  | 82.52%  |
| ep25  | 83.54%  |
| ep30  | 84.56%  |
| ep35  | 85.45%  |
| ep39  | 85.91%  |
| ep40  | 85.86%  |

## Analysis
- **D=16 + C=100 = D=8 + C=100 (matched accuracy at HALF the FLOPs).** The D=16 expressivity gain offsets the halving FLOPs.
- **Cardinality cost softened at D=16:** Halving C from 200→100 costs 3.64pp at D=16 vs ~6pp at D=8. D=16 is more robust to low C.
- **New Pareto-optimal at 0.23B FLOPs:** run38 (85.91%) replaces run31 (79.54%) as the best operating point at this budget.
- Still improving at ep40 (val_loss declining). Could reach ~86.5% with more epochs.

## Updated Pareto Frontier
| FLOPs | Config | Val |
|-------|--------|-----|
| 0.23B | **run38 (D=16, C=100, I=3)** | **85.91%** ★ (was 79.54%) |
| 0.45B | run36 (D=16, C=200, I=3) | 89.55% ★ |
| 0.72B | run35 (D=8, C=200, I=4) | 88.56% |
| 0.90B | run37 (D=16, C=200, I=5) | 90.45% ★ best absolute |

## Verdict
DONE. **D=16 massively expands Pareto frontier.** At 0.23B FLOPs we now get 85.91% (was 79.54%).
At same FLOPs, D=16 + C=100 matches D=8 + C=200 accuracy.
Next: test D=16 + C=50 (further FLOPs reduction to ~0.12B).
