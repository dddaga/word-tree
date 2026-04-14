# run26 — Notes

## Config delta from run17
- `beam_width: 256` (was 0 = disabled)
- All else identical to run17

## Result
- **val_best: 84.20% @ep39** (ep40 dipped to 84.10%)
- FLOPs/fwd: ~0.22B (~4.4x savings vs run17's 0.96B)
- vs run17 (beam=off): **-2.65pp**
- vs run25 (beam=512): **-1.58pp**
- vs run24 (beam=1024): **-2.48pp**

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 26.32%  |
| ep5   | 63.41%  |
| ep10  | 72.18%  |
| ep15  | 76.41%  |
| ep20  | 79.36%  |
| ep25  | 80.71%  |
| ep30  | 81.81%  |
| ep35  | 83.13%  |
| ep37  | 83.69%  |
| ep38  | 84.00%  |
| ep39  | 84.20%  |
| ep40  | 84.10%  |

## Analysis
- **beam=256 (N/16) costs -2.65pp vs run17** at 4.4x FLOPs savings. Steeper drop than beam=512 (-1.07pp), confirming the curve accelerates below N/8.
- **Beam sweep complete.** Full curve: beam=0→86.85%, beam=2048→86.50% (-0.35pp), beam=1024→86.68% (-0.17pp), beam=512→85.78% (-1.07pp), beam=256→84.20% (-2.65pp).
- **Cost per halving accelerates:** 2048→1024 = +0.18pp (improved!), 1024→512 = -0.90pp, 512→256 = -1.58pp. The elbow is at N/4–N/8.
- Train acc 79.10% at ep40 (vs run17's 81.7%) — slightly more underfitting.
- Still improving at ep39–40 — not fully converged.
- **Optimal operating point: beam=1024 (N/4)** — 2.6x FLOPs savings at only -0.17pp. beam=512 (N/8) is the aggressive choice at -1.07pp for 3.6x.

## Verdict
DONE. beam=256 (N/16) delivers 4.4x FLOPs savings at -2.65pp cost. Curve steepens below N/8.
BEAM SWEEP COMPLETE. Winner: **beam=1024 (N/4)** — 2.6x FLOPs at -0.17pp (essentially lossless).
