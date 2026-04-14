# run41 — Notes

## Config delta from run40
- `cardinality: 10` (was 50)
- All else identical (uniform routing, N=2048, V=16, I=5)

## Result
- **val_best: 27.97% @ep39** (plateau — ~2.8x random chance)
- FLOPs/fwd: ~0.057B (4x cheaper than run40)
- vs run40 (C=50): -55.09pp — cliff confirmed

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 9.55%   |
| ep5   | 17.40%  |
| ep10  | 20.66%  |
| ep15  | 23.12%  |
| ep20  | 24.87%  |
| ep25  | 26.27%  |
| ep30  | 26.75%  |
| ep35  | 27.80%  |
| ep39  | 27.97%  |
| ep40  | 27.87%  |

## Analysis
- **C=10 is below the useful floor.** Even with D=16 + I=5 (best supporting config), accuracy plateaus at ~28% (2.8x random).
- **Steep cardinality cliff identified:** C=50→83% → C=10→28%. Between C=10 and C=50 lies a viability threshold.
- **Compared to softmax variants:** run22 (D=8, C=4, softmax) got 25.83%, run30 (D=8, C=2, uniform) got 18.39%, run41 (D=16, C=10, uniform) gets 27.97%. All fall in 18-30% range — extreme sparsity kills both routing mechanisms.
- **Graph connectivity is the issue:** With C=10 per node × 5 iterations, maximum reachable nodes per output is 10^5 = 100K, but our graph has only 2048 nodes — so in theory full coverage is possible. The failure is in *learnable* signal propagation, not just topology.

## Verdict
DONE. **C=10 below the viability floor.** Plateaus at 2.8x random chance.
Next: test C=25 to precisely locate the cliff (halfway between C=10 and C=50).
