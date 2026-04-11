# run14 — Notes

**Config delta from run11:** topology="layered", data_fraction=0.5 (50% data). All else identical to run11 (N=4146, T=1.0, no FFN, 40ep).
**Hypothesis:** Layered topology (output←intermediate only, intermediate←any) enforces directed signal flow and may improve gradient propagation vs flat random graph.
**Status:** DONE (40/40, val_best=45.48% @ep40)
**Comparison target:** run11 (flat, N=4146, T=1.0, 100% data, 61.32% @ep37).

## Results

| Epoch | Val Acc |
|-------|---------|
| 10 | 23.85% |
| 20 | 33.55% |
| 30 | 40.23% |
| 36 | 43.08% |
| 39 | 43.90% |
| **40** | **45.48%** ← best |

## Analysis

- **Layered (run14, 45.48%) < Flat (run11, 61.32%)**: -15.84pp gap.
- **CONFOUNDED: data fraction differs.** run14 used 50% data vs run11's 100%. Data difference may explain part of the gap — not a clean ablation.
- **Convergence pattern**: Best at final epoch, still improving at ep40. Layered topology learns but significantly slower/worse than flat.
- **Verdict (HYPOTHESIS):** Layered topology appears worse than flat at same N/T, but data fraction confound prevents CONFIRMED status. A clean ablation requires run14 repeated with 100% data.
- **Slow start**: ep1-3 val acc 11-14% (vs run11's faster early ramp), suggesting layered connectivity hampers early gradient flow.
