# run16 — Notes

**Config delta from run12:** total_nodes=4146 (was 15454), data_fraction=1.0/100% (was 50%). All else identical to run12 (T=4.0, flat, no FFN, 40ep).
**Hypothesis:** run11 (N=4146) and run12 (T=4.0) were independent improvements — combining them should compound. Full data used because fewer nodes = faster epochs and 66K params benefits from more samples vs 4.5K at 50%.
**Status:** PENDING (after run15).
**Comparison targets:**
  - run12 (T=4.0, N=15454, 50% data → 82.19%): isolates N=4146 + full data effect
  - run11 (T=1.0, N=4146, 100% data → 61.32%): isolates T=4.0 effect
