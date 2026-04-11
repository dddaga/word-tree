# Design Notes

Cross-cutting findings and decisions. Per-run details live in `training_runs/runN/notes.md`. Run overview in `EXPERIMENT_QUEUE.md`.

---

**LayerNorm (CONFIRMED):** run1 vs run3 clean ablation — exactly one variable, +5.33pp. Default ever since.

**Radiation (HYPOTHESIS):** run4→5→6 progressively reduced radiation until disabled, each step improving. But run4→5 and run5→6 both changed multiple variables simultaneously. No clean radiation-only ablation exists.

**Dropout (HYPOTHESIS):** Added in run6 alongside radiation removal. The train < val pattern in run6 (81% vs 85%) confirms strong regularization, but the gain cannot be attributed to dropout alone without a run6 clone with dropout=0.

**Strategic pivot (2026-04-09):** All experiments from run10 onwards drop the nn.Linear FFN head. GNN must classify directly (output_nodes=10). Reason: `diagnose.ipynb` showed the FFN was doing most classification work — intermediate/input weights barely changed during training. Keeping the FFN defeats the project goal of proving the GNN can replace VGG16's FC layers. Target: match run6's 85.81% without FFN.
