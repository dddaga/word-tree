# run12 — Notes

**Config delta from run10:** model.routing_temperature=4.0 (was 1.0). 50% training data (stratified). All else identical to run10 (N=15454, ~12K intermediates, no FFN). 40ep.
**Hypothesis:** Softmax routing concentration (0.4% of nodes carry 50% gradient) is the bottleneck. Temperature=4.0 softens the distribution, spreading gradient more evenly across nodes.
**Status:** DONE (40/40 epochs, val_best=82.19% @ep39).

## Results

| Metric | Value |
|--------|-------|
| Val best | **82.19%** @ ep39 |
| Val final (ep40) | 82.04% |
| Train acc final | 76.02% |
| Val loss final | 0.5743 |
| Parameters | 247,280 |
| FLOPs/fwd | 3.56B |

## Verdict vs run10 (T=1.0)

**T=4.0 is a massive improvement — CONFIRMED as key lever:**
- run12 ep20: **76.00%** vs run10 ep20: **37.63%** (+38.37pp)
- run12 ep23: **77.43%** vs run10 ep23: **40.33%** (+37.10pp)
- run12 final: **82.19%** vs run10 final: **40.33%** (+41.86pp)

**Surpassed run1 (76.05% with FFN) at ep19, run3 (81.38% with FFN) at ep35.**
Only 3.62pp below run6 target (85.81% with FFN) — GNN without FFN is approaching FFN-aided performance.

Part of temperature sweep: run10(T=1.0, 40.33%) → run12(T=4.0, 82.19%) → run13(T=7.0, pending).

## Val acc curve
ep1:22.9 → ep5:53.0 → ep10:67.1 → ep15:72.3 → ep20:76.0 → ep25:78.3 → ep30:79.9 → ep35:81.4 → ep39:82.2 → ep40:82.0
