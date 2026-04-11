# run13 — Notes

**Config delta from run10:** model.routing_temperature=7.0 (was 1.0). 50% training data (stratified). All else identical to run10 (N=15454, flat, no FFN, 40ep).
**Hypothesis:** Near-uniform routing (T=7.0) distributes gradient more broadly than T=4.0. Part of sweep: run10(T=1.0), run12(T=4.0), run13(T=7.0).
**Status:** DONE (40/40, val_best=85.40% @ep40)
**Comparison targets:** run10(T=1.0, 40.33%), run12(T=4.0, 82.19%).

## Results

| Epoch | Val Acc |
|-------|---------|
| 10 | 72.69% |
| 20 | 80.92% |
| 30 | 83.18% |
| 37 | 84.84% |
| 39 | 85.07% |
| **40** | **85.40%** ← best |

## Analysis

- **T=7.0 (run13) > T=4.0 (run12) > T=1.0 (run10)**: 85.40% > 82.19% > 40.33%. Higher temperature continues to improve within 40-epoch window.
- **Near-surpassed run6 target (85.81%)**: gap is only 0.41pp without FFN, with 50% data. FFN-free GNN is competitive.
- **Convergence pattern**: Best at final epoch — T=7.0 hadn't converged at 40ep, suggesting more epochs or full data may close the remaining gap.
- **Temperature sweep verdict (HYPOTHESIS)**: Higher T is better at 40ep scale. Optimal T ≥ 7.0.
- **See:** `concepts/gradient_starvation.md`, `gradient_starvation_analysis.py`.
