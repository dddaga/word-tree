# run20 — Notes

## Config delta from run17
- `cardinality: 50` (was 200)
- All else identical to run17

## Result
- **val_best: 71.64% @ep40** (still improving at final epoch)
- FLOPs/fwd: 0.24B (4x savings vs run17's 0.96B)
- Gap from run17: **-15.21pp**
- Gap from run19 (C=100): -9.02pp for halving cardinality again

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 17.04%  |
| ep5   | 41.73%  |
| ep10  | 51.90%  |
| ep15  | 58.06%  |
| ep20  | 61.76%  |
| ep25  | 64.71%  |
| ep30  | 67.13%  |
| ep35  | 69.71%  |
| ep38  | 70.80%  |
| ep39  | 70.96%  |
| ep40  | 71.64%  |

## Analysis
- **-15.21pp vs run17** at 4x FLOPs savings. Cardinality is not a free lever — accuracy drops are steep.
- **Accelerating drop:** C=200→100 cost -6.19pp; C=100→50 cost -9.02pp. Trend is super-linear.
- Still improving at ep40 — estimated ~72-73% with more epochs, but gap remains large.
- Early convergence very slow: ep1=17.04%, ep10=51.90% (vs run17: ep10=73.99%). C=50 learns much more slowly.
- Cardinality trajectory so far: C=200=86.85%, C=100=80.66%, C=50=71.64%.
- **FLOPs are not the binding constraint** — cardinality matters much more than expected. Going from C=200 to C=50 (4x FLOPs savings) costs 15pp, which is too steep to be practically useful.

## Verdict
DONE. C=50 delivers 4x FLOPs savings but at -15.21pp accuracy cost — not a viable operating point.
Cardinality trend steepening: run21 (C=25) next to locate the cliff.
