# run16 — Notes

**Config delta from run12:** total_nodes=4146 (was 15454), data_fraction=1.0/100% (was 50%). All else identical to run12 (T=4.0, flat, no FFN, 40ep).
**Hypothesis:** run11 (N=4146) and run12 (T=4.0) were independent improvements — combining them should compound. Full data used because fewer nodes = faster epochs and 66K params benefits from more samples.
**Status:** DONE (40/40, val_best=86.55% @ep40)
**Comparison targets:**
  - run12 (T=4.0, N=15454, 50% data → 82.19%): isolates N=4146 + full data effect
  - run11 (T=1.0, N=4146, 100% data → 61.32%): isolates T=4.0 effect
  - run13 (T=7.0, N=15454, 50% data → 85.40%): best prior FFN-free result
  - Target: 85.81% (run6 with FFN)

## Results

| Epoch | Val Acc |
|-------|---------|
| 10 | 76.20% |
| 20 | 81.61% |
| 30 | 84.61% |
| 33 | 85.30% |
| 35 | 85.71% |
| 37 | 85.78% |
| 38 | 86.06% |
| 39 | 86.17% |
| **40** | **86.55%** ← best |

## Analysis

- **Exceeded FFN target:** 86.55% > 85.81% (run6 with FFN). +0.74pp above target. **Proof-of-concept complete.**
- **Exceeded best prior FFN-free (run13):** 86.55% vs 85.40%, +1.15pp. With N=4146+T=4.0 vs N=15454+T=7.0.
- **vs run12 (T=4.0, N=15454, 50% data):** +4.36pp. Smaller graph + full data outperforms larger graph + half data at same T.
- **vs run11 (N=4146, T=1.0, 100% data):** +25.23pp. T=4.0 delivers massive gains even with same N and data fraction.
- **Convergence:** Best at final epoch, still improving (+0.38pp ep39→ep40). Did not converge — more epochs likely push higher.
- **Evidence tag (HYPOTHESIS):** Two variables changed vs each baseline (N and data_fraction vs run12; T and nothing else vs run11 — actually this is CONFIRMED for T effect). The compound effect cannot be cleanly attributed to either variable alone vs a common baseline. A clean ablation would require run11 repeated at T=4.0 only, or run12 repeated at N=4146 only (which is essentially run16 minus the data fraction change, confounded).

## Verdict

**HYPOTHESIS:** N=4146 + T=4.0 improvements compound. Primary driver likely T=4.0 (CONFIRMED from run12 single-variable ablation), with N=4146 + full data providing additional benefit. Proof-of-concept ACHIEVED: GNN without FFN exceeds the FFN-based target (85.81%).
