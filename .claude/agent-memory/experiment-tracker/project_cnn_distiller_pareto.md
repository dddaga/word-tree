---
name: project_cnn_distiller_pareto
description: CNN distiller EfficientVGG Pareto table — T2 4-seed results for paper
metadata:
  type: project
---

cnn_step004 T2 DONE (2026-06-21): CNN distiller 3-anchor Pareto table, 4 seeds each.

| Config | T2 mean | MACs | Params | δ vs Ref |
|---|---|---|---|---|
| F_wide (128/256/256 k=7) | 79.03% ±0.41pp | 559.8M | 825K | +1.42pp |
| Ref (64/128/256 k=7) | 77.61% ±0.45pp | 183.2M | 419K | — |
| D_small_s (32/64/128 k=7) | 75.82% ±1.02pp | 57.5M | 150K | −1.79pp |

D_small_s: 3.2× fewer MACs, 2.8× fewer params vs Ref at −1.79pp cost — EFF-PARETO.
F_wide: accuracy ceiling at 2× MACs and 2× params.
D_small_s variance ±1.02pp driven by seed=42 low outlier (74.52%); other seeds 75.64–77.02%.

**Why:** Paper needs authoritative multi-seed Pareto table for CNN distiller comparison.
**How to apply:** Table is complete. Report all 3 configs with means ±std. D_small_s is the efficiency story for paper. cnn_step019 T2 (EfficientVGG GA line) will add a 4th anchor if STRONG.
