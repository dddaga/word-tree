# run31 — Notes

## Config delta from run29
- `cardinality: 50` (was 200)
- All else identical to run29 (uniform routing, N=4146, V=8, I=5)

## Result
- **val_best: 79.54% @ep40** (still improving — val loss still declining)
- FLOPs/fwd: ~0.24B (4x savings vs run17's 0.96B)
- vs run20 (softmax, C=50): **+7.90pp** — massive gain at moderate cardinality
- vs run29 (uniform, C=200): -9.61pp — accuracy drops with less cardinality

## Val trajectory (selected)
| Epoch | Val Acc | run20 @same ep |
|-------|---------|----------------|
| ep1   | 17.94%  | 17.04% |
| ep5   | 45.58%  | 41.73% |
| ep10  | 59.69%  | 51.90% |
| ep15  | 67.69%  | 58.06% |
| ep20  | 71.49%  | 61.76% |
| ep25  | 74.62%  | 64.71% |
| ep30  | 76.56%  | 67.13% |
| ep35  | 78.47%  | 69.71% |
| ep40  | 79.54%  | 71.64% |

## Analysis
- **Uniform routing at C=50 is MASSIVELY better than softmax at C=50:** +7.90pp (79.54 vs 71.64). Much bigger gain than at C=200 (+2.30pp).
- **Peak uniform routing benefit is at moderate cardinality.** The gain vs softmax:
  - C=2: **-10.65pp** (uniform worse)
  - C=50: **+7.90pp** (uniform much better)
  - C=200: **+2.30pp** (uniform better)
- **Explanation:** At moderate C, softmax's gradient concentration is maximally harmful. C=200 has enough sources that softmax spreads signal OK; C=2 has too few sources for uniform to work. C=50 is the sweet spot where softmax's weakness is most exposed.
- **Per-FLOP efficiency:**
  - run29 (C=200): 89.15% @ 0.96B FLOPs
  - run31 (C=50): 79.54% @ 0.24B FLOPs — **3.3x cheaper per forward pass, -9.6pp**
- **Still improving at ep40** (val_loss: 0.6905→0.6823 from ep39→ep40). Likely +1-2pp with more epochs.
- **Gains are consistent across training** — not just final epochs. Uniform converges much faster at moderate C.

## Verdict
DONE. **Uniform routing gain is cardinality-dependent: peaks at moderate C (~50), declines at C=2 and C=200.**
Pareto frontier expanded: 79.54% at 0.24B FLOPs (4x savings vs run17) — still above run14's 45.48% at same FLOPs.
Next: test uniform routing at C=100 (halfway to run29) to find the accuracy peak per-FLOP.
