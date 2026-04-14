# run34 — Notes

## Config delta from run29
- `iterations: 2` (was 5)
- All else identical (uniform routing, N=4146, C=200, V=8)

## Result
- **val_best: 72.05% @ep40** (still improving slowly)
- FLOPs/fwd: ~0.24B (4x savings vs run29)
- vs run29 (I=5): -17.10pp at 4x FLOPs savings
- vs run33 (I=3): -15.11pp at 2x FLOPs savings
- vs run31 (C=50, I=5, same FLOPs): **-7.49pp** — reducing cardinality is better than I=2

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 14.75%  |
| ep5   | 38.72%  |
| ep10  | 53.21%  |
| ep15  | 59.68%  |
| ep20  | 63.72%  |
| ep25  | 66.99%  |
| ep30  | 70.04%  |
| ep35  | 71.08%  |
| ep40  | 72.05%  |

## Analysis
- **I=2 is below the minimum useful iteration count.** With only 1 propagation step after input injection, the graph cannot effectively aggregate information from multiple hops. Each output node sees only nodes 1 hop away.
- **Iterations > cardinality is NOT universal** — it reverses at extreme reductions:
  - Moderate reduction (I=5→3, 0.96B→0.48B): iterations win by +1.27pp (run33 vs run32)
  - Extreme reduction (I=5→2, 0.96B→0.24B): iterations LOSE by -7.49pp (run34 vs run31)
- **Floor on iterations:** below I=3, the GNN doesn't have enough propagation steps.
- **Pareto frontier unchanged at 0.24B:** run31 (C=50, I=5) still wins at 79.54%.
- Still improving at ep40 — could reach 73-74% with more epochs, but won't match run31.

## Verdict
DONE. **I=2 is below the iteration floor** — insufficient propagation for the GNN.
Best operating points:
- 0.24B FLOPs: reduce cardinality (run31, C=50, I=5) → 79.54%
- 0.48B FLOPs: reduce iterations (run33, C=200, I=3) → 87.16%
- 0.96B FLOPs: full config (run29) → 89.15%
Next: fill in the iteration curve (I=4) or push accuracy ceiling (more epochs).
