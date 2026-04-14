# run30 — Notes

## Config delta from run29
- `cardinality: 2` (was 200)
- All else identical to run29 (uniform routing, N=4146, V=8, I=5)

## Result
- **val_best: 18.39% @ep37** (~1.84x random chance)
- vs run27 (softmax, C=2): **-10.65pp** — uniform is WORSE than softmax at C=2
- vs run29 (uniform, C=200): **-70.76pp** — extreme sparsity kills performance

## Val trajectory (selected)
| Epoch | Val Acc |
|-------|---------|
| ep1   | 10.42%  |
| ep5   | 12.89%  |
| ep10  | 16.05%  |
| ep15  | 16.82%  |
| ep20  | 17.35%  |
| ep25  | 17.86%  |
| ep30  | 17.96%  |
| ep37  | 18.39%  |
| ep40  | 18.17%  |

## Analysis
- **Uniform routing is WORSE than softmax at C=2.** run27 (softmax, C=2) got 29.04% vs run30 (uniform, C=2) at 18.39% — a 10.65pp gap. At extreme sparsity, the ability to weight informative sources higher (softmax) is more valuable than even gradient distribution (uniform).
- **Routing mechanism interaction with cardinality:**
  - C=200: uniform BETTER (+2.30pp) — many sources, even gradient wins
  - C=2: uniform WORSE (-10.65pp) — few sources, selectivity wins
  - Crossover exists somewhere between C=2 and C=200
- **The C=2 problem is structural, not routing:** Only 1-2 incoming edges + self-loop per node. After 4 iterations, information from any input node can reach at most 2^4 = 16 other nodes. At N=4146, that's <0.4% coverage. Neither routing mechanism can fix this topology limitation.
- Plateau reached by ep25 (~18%). Train acc 17.20% — extreme underfitting.

## Verdict
DONE. C=2 fails with BOTH routing mechanisms. The problem is graph sparsity, not routing.
**Key finding: Uniform routing trades selectivity for even gradient distribution.**
- At high C: even distribution wins (many sources dilute selectivity's value)
- At low C: selectivity wins (few sources make every routing decision critical)
Next: test uniform routing at moderate C (C=50) — where softmax scored 71.64%.
