# run25 — Notes

## Config delta from run17
- `beam_width: 512` (was 0 = disabled)
- All else identical to run17

## Result
- **val_best: 85.78% @ep40** (still improving at final epoch)
- FLOPs/fwd: ~0.27B (~3.6x savings vs run17's 0.96B)
- vs run17 (beam=off): **-1.07pp**
- vs run24 (beam=1024): **-0.90pp**
- vs run23 (beam=2048): **-1.28pp** (run25 worse than both larger beams)

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 26.19%  |
| ep5   | 65.38%  |
| ep10  | 73.73%  |
| ep15  | 77.63%  |
| ep20  | 80.46%  |
| ep25  | 82.75%  |
| ep30  | 83.21%  |
| ep35  | 84.28%  |
| ep37  | 85.07%  |
| ep38  | 85.02%  |
| ep39  | 84.94%  |
| ep40  | 85.78%  |

## Analysis
- **beam=512 (N/8) costs -1.07pp vs run17** at 3.6x FLOPs savings. First meaningful accuracy drop in the beam sweep.
- **Beam sweep curve:** beam=0→86.85%, beam=2048→86.50% (-0.35pp), beam=1024→86.68% (-0.17pp), beam=512→85.78% (-1.07pp). The elbow is between beam=1024 and beam=512 — a ~0.9pp jump in cost when halving from N/4 to N/8.
- **Late-epoch dip then recovery:** ep37=85.07%, ep38=85.02%, ep39=84.94% (3-epoch dip), ep40=85.78% (new best). Suggests training wasn't fully converged — likely 1-2pp more with more epochs.
- Train acc 81.25% — similar to run17 (81.7%) and run24 (81.56%). Same generalization gap.
- Early convergence close to run24: ep5=65.38% vs run24's 64.33%, ep10=73.73% vs run24's 74.01%. The runs track each other until mid-training, then beam=512 falls slightly behind.
- Still improving at ep40 — plateau not reached. Estimated ceiling ~86.5-87% with more epochs (may close the gap further).

## Verdict
DONE. beam=512 (N/8) delivers 3.6x FLOPs savings at -1.07pp cost. The curve has an elbow between N/4 and N/8.
Sweep continues: run26 (256=N/16) — expected to probe the cliff.
