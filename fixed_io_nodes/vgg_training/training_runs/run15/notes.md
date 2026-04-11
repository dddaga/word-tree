# run15 — Notes

**Config delta from run13:** routing_temperature annealed linearly 7.0→1.0 per-step over all training steps. All else identical to run13 (N=15454, flat, no FFN, 50% data, 40ep).
**Mechanism:** `current_temp = 7.0 - 6.0 * (global_step / total_steps)` set before each forward pass.
**Status:** DONE (40/40, val_best=76.97% @ep27)
**Comparison targets:**
  - run13 (T=7.0 fixed, N=15454, 50% data → 85.40% @ep40): single-variable vs run15 — only T schedule differs
  - run12 (T=4.0 fixed, N=15454, 50% data → 82.19% @ep39)

## Results

| Epoch | Val Acc | T_end |
|-------|---------|-------|
| 1 | 21.17% | 6.85 |
| 5 | 56.99% | 6.25 |
| 10 | 69.43% | 5.50 |
| 15 | 74.55% | 4.75 |
| 20 | 76.61% | 4.00 |
| **27** | **76.97%** ← best | **2.95** |
| 30 | 75.54% | 2.50 |
| 35 | 71.13% | 1.75 |
| 40 | 52.66% | 1.00 |

## Analysis

- **val_best = 76.97% @ep27 (T_end=2.95)** — peak occurred at mid-T range (~3), not during high-T or low-T phase.
- **Annealing CONFIRMED worse than T=7.0 fixed (CONFIRMED):** 76.97% vs 85.40% (-8.43pp). Single-variable ablation vs run13. Annealing to T=1.0 is clearly harmful.
- **Catastrophic decline in low-T phase:** ep27(76.97%) → ep40(52.66%), -24.31pp as T drops below 3. This is gradient starvation returning — as T anneals toward 1.0, routing collapses back to winner-take-all and previously learned representations degrade.
- **Early high-T phase was fast:** T=7→5 (ep1-10) ramped from 21% to 69% quickly, suggesting high-T does help early feature learning. But the benefit is wiped out by low-T degradation.
- **Best epoch at T≈3:** coincides with the T=4.0 fixed run12's regime. Suggests T=3-4 may be near the sweet spot for this architecture.
- **Evidence tag: CONFIRMED** — clean single-variable ablation vs run13 (only T schedule changed).

## Verdict

T annealing 7→1 is **HARMFUL**. The end-state T=1.0 causes gradient starvation that undoes all gains from high-T early training. T=7.0 fixed (run13, 85.40%) remains the best FFN-free result. Do NOT anneal temperature downward.
