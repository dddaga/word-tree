# run37 — Notes

## Config delta from run36
- `iterations: 5` (was 3)
- All else identical (uniform routing, N=2048, V=16, C=200)

## Result
- **val_best: 90.45% @ep40** — **BROKE 90%! NEW ALL-TIME BEST**
- FLOPs/fwd: ~0.9B (similar to run29)
- vs run36 (I=3): **+0.90pp at 2x FLOPs**
- vs run29 (D=8, I=5, same FLOPs): **+1.30pp at same FLOPs**
- vs run17 (softmax, D=8, I=5): **+3.60pp**

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 27.62%  |
| ep5   | 68.84%  |
| ep10  | 79.80%  |
| ep15  | 83.80%  |
| ep20  | 86.27%  |
| ep25  | 87.75%  |
| ep30  | 88.82%  |
| ep35  | 89.43%  |
| ep38  | 90.01% ← first epoch past 90% |
| ep39  | 90.37%  |
| ep40  | 90.45%  |

## Analysis
- **First run to break 90%.** Combination of uniform routing + D=16 + I=5 pushes past the ceiling.
- **D=16 + I=5 + uniform routing = 90.45%** — combines all the gains:
  - Uniform routing (+2.30pp vs softmax)
  - D=16 (+1.30pp at same FLOPs vs D=8)
  - I=5 full iterations (+0.90pp vs I=3)
- **Per-FLOP efficiency worse than run36:** +0.90pp for 2x FLOPs. Diminishing returns from iterations.
- **Still improving at ep40** (val_loss: 0.3259→0.3208 @ep40). Could hit 91%+ with more epochs.
- **Train acc 87.59% vs run36's 87.00%** — similar generalization, slight improvement.

## Updated Pareto Frontier
| FLOPs | Best Config | Val |
|-------|-------------|-----|
| 0.24B | run31 (uniform, D=8, C=50, I=5) | 79.54% |
| **0.45B** | **run36 (uniform, D=16, C=200, I=3)** | **89.55%** ★ best efficiency |
| 0.72B | run35 (D=8, I=4) | 88.56% |
| **0.90B** | **run37 (uniform, D=16, C=200, I=5)** | **90.45%** ★ best absolute |

## Verdict
DONE. **MILESTONE: 90.45% accuracy broken.** Compound gain from uniform routing + D=16 + I=5.
Gap to linear baseline (~95%): 4.55pp. Gap to team member's SGNNET (95.52%): 5.07pp.
Next: push FLOPs lower at current accuracy — test D=16 with lower C (C=100, C=50) to map Pareto frontier.
