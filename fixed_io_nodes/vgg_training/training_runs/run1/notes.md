# run1 — Notes

**Config:** flat, total_nodes=15454, cardinality=200, iterations=5, vector_dim=8, no layernorm, no dropout, no radiation, lr=0.0001, 20ep.
**Result:** val best=76.05% (ep17), final=75.64% (ep19). Train acc=94.39%.
**Notes:** First complete run. Established config schema and checkpoint format. High train/val gap (94% vs 76%) — overfitting without regularization.
