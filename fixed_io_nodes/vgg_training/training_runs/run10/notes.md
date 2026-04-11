# run10 — Notes

**Config delta from run6:** output_nodes=256→10, nn.Linear head removed. GNN act_strength for 10 output nodes fed directly into CrossEntropyLoss. total_nodes=15454 (~12K intermediates). 40ep (stopped at 23).
**Result:** val best=40.33% (ep23, stopped). Still improving at stop — no plateau.

| Epoch | Train Acc | Val Acc | Val Loss |
|-------|-----------|---------|----------|
| 1     | 12.49%    | 13.89%  | 2.713    |
| 5     | 16.22%    | 18.39%  | 2.254    |
| 10    | 21.04%    | 27.18%  | 2.072    |
| 13    | 23.74%    | 30.80%  | 1.978    |
| 20    | 28.48%    | 37.63%  | 1.830    |
| 23    | 30.63%    | 40.33%  | 1.777    |

**Notes:** GNN learns without FFN but convergence dramatically slower than with FFN (run6 hit ~80% by ep13). Diagnostic (`gradient_starvation_analysis.py`) at ep13 checkpoint showed 0.4% of nodes carrying 50% of gradient — softmax routing concentration confirmed as bottleneck. FFN-free baseline for run11+.
**Finding: CONFIRMED (within FFN-free runs) — GNN can classify directly. Gradient starvation is the active bottleneck.**
**Diagnostic:** `gradient_starvation_analysis.py`, `diagnose.ipynb`. See `concepts/gradient_starvation.md`.
