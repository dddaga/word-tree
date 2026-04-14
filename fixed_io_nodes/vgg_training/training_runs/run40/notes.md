# run40 — Notes

## Config delta from run39
- `iterations: 5` (was 3)
- All else identical (uniform routing, N=2048, V=16, C=50)

## Result
- **val_best: 83.06% @ep39** (still improving)
- FLOPs/fwd: ~0.23B (2x of run39's 0.12B)
- vs run39 (I=3): **+7.57pp from I=5 alone** — confirms I matters at low C
- vs run31 (D=8, C=50, I=5): **+3.52pp from D=16** — D=16 helps on top of I=5
- vs run38 (D=16, C=100, I=3, same FLOPs): -2.85pp

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep5   | 58.65%  |
| ep10  | 69.05%  |
| ep15  | 73.68%  |
| ep20  | 77.07%  |
| ep25  | 79.60%  |
| ep30  | 80.87%  |
| ep35  | 82.17%  |
| ep39  | 83.06%  |
| ep40  | 82.85%  |

## Analysis
- **Isolated contributions at C=50:**
  - D: 8→16 gives +3.52pp (run40 vs run31)
  - I: 3→5 gives +7.57pp (run40 vs run39)
  - I dominates at low C, but D still helps
- **C-I coupling confirmed:** Low C requires high I for effective propagation. At C=50, I=5 is the sweet spot.
- **Same FLOPs comparison at 0.23B:**
  - run38 (D=16, C=100, I=3): 85.91%
  - run40 (D=16, C=50, I=5): 83.06%
  - **High C + low I beats low C + high I at same FLOPs** by +2.85pp
- **Per-FLOP ranking confirmed:** C reduction is worse than I reduction as a lever (consistent with earlier finding).

## Verdict
DONE. **Best operating strategy: keep C high (100-200), reduce I as needed.**
run38 (D=16, C=100, I=3) remains Pareto-best at 0.23B — +2.85pp over run40.
Next: explore further FLOPs reduction or push accuracy ceiling.
