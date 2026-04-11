# run11 — Notes

**Config delta from run10:** total_nodes=15454→4146 (3136 input + 1000 intermediate + 10 output). ~12K→~1K intermediates. 40ep (extended from initial 20).
**Hypothesis:** Fewer intermediate nodes may help gradient reach the 10 output nodes more effectively.
**Status:** DONE (40/40 epochs, val_best=61.32% @ep37).

## Results

| Metric | Value |
|--------|-------|
| Val best | **61.32%** @ ep37 |
| Val final (ep40) | 61.25% |
| Train acc final | 56.20% |
| Val loss final | 1.2101 |
| Parameters | 66,352 (3.7x fewer than run10's 247,280) |
| FLOPs/fwd | 0.96B (3.7x fewer than run10's 3.56B) |

## Verdict vs run10

**run11 massively outperforms run10** with 3.7x fewer parameters and FLOPs:
- run11 ep20: **53.61%** vs run10 ep20: **37.63%** (+15.98pp)
- run11 ep23: **53.81%** vs run10 ep23: **40.33%** (+13.48pp)
- run11 final: **61.32%** (run10 stopped at ep23/40.33%)

**CONFIRMED**: Fewer intermediate nodes (1K vs 12K) dramatically improve FFN-free GNN learning. Gradient reaches output nodes much more effectively with a smaller graph. Still 24.49pp below run6 target (85.81%) — more work needed.

## Val acc curve
ep1:20.4 → ep5:37.0 → ep10:47.2 → ep15:52.3 → ep20:53.6 → ep25:56.3 → ep30:58.8 → ep35:60.5 → ep37:61.3 → ep40:61.3
