# run18 — Notes

**Config delta from run17:** `routing_temperature: 1.0` (was 4.0). All else identical. Clean single-variable ablation of T under new post-update LN.
**Hypothesis:** Does new post-update LN (learnable γ/β) rescue T=1.0 from gradient starvation?
**Status:** DONE (40/40, val_best=**61.99% @ep40**)
**Comparison targets:**
  - run11 (T=1.0, old LN, N=4146, 100% data → 61.32% @ep37)
  - run17 (T=4.0, new LN, N=4146, 100% data → 86.85% @ep38)

## Results

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 5 | 22.26% | 27.87% |
| 10 | 32.34% | 40.20% |
| 15 | 39.72% | 46.70% |
| 20 | 43.69% | 51.52% |
| 25 | 48.94% | 56.99% |
| 30 | 51.25% | 58.17% |
| 35 | 52.97% | 60.15% |
| 38 | 55.52% | 61.76% |
| 39 | 54.67% | 61.86% |
| **40** | 55.50% | **61.99%** ← best |

## Analysis

- **run18 ≈ run11** (61.99% vs 61.32%, +0.67pp). Well within run-to-run variance.
- **~25pp gap vs run17** (T=4.0 under same new LN regime). Gradient starvation at T=1.0 persists under new LN.
- **Still monotonically improving at ep40** — curve shape typical of T=1.0 runs. No collapse.
- **Train/val gap similar to run11** — no generalization regime change.
- **Decision gate → "run18 ≈ run11" branch taken.** No T sweep follow-up needed.

## Verdict

**CONFIRMED (clean single-variable ablation vs run17):** Under new post-update LN, T=1.0 still causes severe gradient starvation. The +0.67pp improvement over run11 is within variance.

**CONFIRMED interpretation:** The LN placement fix (commit f11f768) does NOT change the gradient starvation regime at T=1.0. Routing temperature remains the dominant lever for breaking out of starvation, independent of normalization scheme. The T scale is not shifted under new LN.

**Practical conclusion:** Continue using T=4.0 as the best-known setting. Focus next experiments on orthogonal levers (FLOPs reduction, topology, etc.).
