# run39 — Notes

## Config delta from run38
- `cardinality: 50` (was 100)
- All else identical (uniform routing, N=2048, V=16, I=3)

## Result
- **val_best: 75.49% @ep40** (still improving)
- FLOPs/fwd: ~0.12B (4x cheaper than run36, 2x cheaper than run38)
- vs run38 (C=100): -10.42pp at HALF the FLOPs
- vs run31 (D=8, C=50, I=5): **-4.05pp** — surprising, D=16 doesn't help here
- vs run36 (D=16, C=200, I=3): -14.06pp

## Analysis
- **Surprising finding: D=16 + C=50 + I=3 < D=8 + C=50 + I=5.** Three confounded variables:
  - D: 8→16 (helps)
  - I: 5→3 (hurts at low C)
  - N: 4146→2048 (hurts)
  - Net effect at low C: the combined reductions eat the D=16 gain.
- **At low cardinality, iterations matter more.** The graph needs more propagation steps to cover a sparse graph. I=3 is too few at C=50.
- **At high C (run38, C=100), I=3 was fine** because the graph is denser — fewer hops to cover.
- **Implication:** There's a coupling between C and I. Low C needs high I. Cannot reduce both simultaneously.

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep5   | 48.94%  |
| ep10  | 63.39%  |
| ep15  | 67.83%  |
| ep20  | 70.28%  |
| ep25  | 71.90%  |
| ep30  | 73.60%  |
| ep35  | 74.70%  |
| ep40  | 75.49%  |

## Verdict
DONE. **At C=50, I=3 is too few iterations.** D=16 doesn't compensate for reduced iterations + smaller N.
Next: test D=16 + C=50 + I=5 to isolate whether iterations or N is the culprit.
