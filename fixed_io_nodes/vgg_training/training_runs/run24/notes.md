# run24 — Notes

## Config delta from run17
- `beam_width: 1024` (was 0 = disabled)
- All else identical to run17

## Result
- **val_best: 86.68% @ep40** (still improving at final epoch)
- FLOPs/fwd: ~0.37B (~2.6x savings vs run17's 0.96B)
- vs run17 (beam=off): **-0.17pp** — essentially lossless
- vs run23 (beam=2048): **+0.18pp** — run24 is *better* than run23!
- Baseline: run17 → 86.85% @ep38, run23 → 86.50% @ep40

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 26.24%  |
| ep5   | 64.33%  |
| ep10  | 74.01%  |
| ep15  | 78.09%  |
| ep20  | 80.97%  |
| ep25  | 83.97%  |
| ep30  | 84.92%  |
| ep34  | 85.86%  |
| ep37  | 86.32%  |
| ep38  | 86.50%  |
| ep39  | 86.52%  |
| ep40  | 86.68%  |

## Analysis
- **beam=1024 (N/4) is better than beam=2048 (N/2)**: run24 = 86.68% > run23 = 86.50%. Both are within 0.35pp of run17 (86.85%).
- **2.6x FLOPs savings at -0.17pp cost** — practically lossless. A strong result.
- Tracking identically to run23 in early epochs (ep20: 80.97% vs run23's 80.99%), then slightly exceeds it in final epochs.
- Still improving at ep40 — no sign of convergence plateau.
- Train acc 81.56% (comparable to run17's 81.7%) — similar generalization gap.
- The beam sweep so far: beam=0 → 86.85%, beam=2048 → 86.50% (-0.35pp), beam=1024 → 86.68% (-0.17pp). The progression is non-monotone — beam=1024 is slightly better than beam=2048. This may reflect beam acting as slight regularization at N/4 without over-filtering signal.

## Verdict
DONE. beam=1024 (N/4) delivers 2.6x FLOPs savings at only -0.17pp accuracy cost — better than beam=2048.
CONFIRMED: Beam width down to N/4 is essentially lossless. Sweep continues: run25 (512=N/8), run26 (256=N/16).
