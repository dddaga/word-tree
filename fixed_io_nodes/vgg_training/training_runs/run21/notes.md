# run21 — Notes

## Config delta from run17
- `cardinality: 25` (was 200)
- All else identical to run17

## Result
- **val_best: 64.54% @ep40** (still improving at final epoch)
- FLOPs/fwd: 0.12B (8x savings vs run17's 0.96B)
- Gap from run17: **-22.31pp**
- Gap from run20 (C=50): -7.10pp for halving cardinality again
- Gap from run19 (C=100): -16.12pp

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 11.41%  |
| ep5   | 28.56%  |
| ep10  | 42.14%  |
| ep15  | 49.50%  |
| ep20  | 53.40%  |
| ep25  | 58.01%  |
| ep30  | 60.43%  |
| ep35  | 62.90%  |
| ep38  | 63.49%  |
| ep39  | 64.43%  |
| ep40  | 64.54%  |

## Analysis
- **-22.31pp vs run17** at 8x FLOPs savings. Cardinality is the dominant cost.
- **Cardinality trend:** C=200→86.85%, C=100→80.66% (-6.19pp), C=50→71.64% (-9.02pp), C=25→64.54% (-7.10pp). Each halving costs ~7-9pp.
- Very slow early convergence: ep1=11.41% (vs run17's ep1=27.4%). C=25 graph is too sparse to propagate signal quickly in early epochs.
- Still improving at ep40 — likely reaches 66-67% with more epochs, but gap remains large.
- Train acc only 49.31% at ep40 (vs run17's 81.7%) — underfitting at this cardinality; the GNN doesn't have enough connectivity to learn the task well.
- **Cliff prediction:** The cardinality drop curve is approximately constant per-halving (~7-9pp). C=4 (run22) would project to ~56-58%, but the actual expected behavior is collapse — C=4 is far below where gradients can flow meaningfully through softmax routing. Actual failure likely much worse than linear extrapolation.

## Verdict
DONE. C=25 delivers 8x FLOPs savings but at -22.31pp accuracy cost. Not viable for production use.
Run22 (C=4) is the floor probe — expected to fail badly via gradient starvation.
