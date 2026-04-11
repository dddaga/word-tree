# Baselines and Experiments Needed for Publication

## Critical (blocks submission)

### 1. Baseline comparisons at equivalent param count
- [ ] **MLP** at 67K params (same as step199 efficiency config) — what does a standard MLP achieve?
- [ ] **Pruned VGG16 FC** — prune to 67K params, report accuracy
- [ ] **Random features + linear** — same random projections as SGNNET but NO iterative routing. Just project → linear classifier. Isolates the contribution of routing.
- [ ] **Standard GNN (GCN/GAT)** at 529K params on same task — SGNNET vs established GNN baselines

### 2. Second dataset (generalization)
- [ ] **CIFAR-10** via VGG16 features (same pipeline, drop-in replacement)
- [ ] **Tabular dataset** — something non-vision to show architecture-agnostic benefit (e.g., Forest Cover Type, or a Kaggle classification task)

### 3. FLOPs efficiency demonstration
- [ ] Show competitive accuracy at ≤5% VGG16 FLOPs (≤6.18M)
- [ ] step140 (N×K tradeoff) results will inform this
- [ ] Pareto frontier: accuracy vs FLOPs curve with multiple N/D/K configurations

## Important (strengthens paper)

### 4. Ablation table (comprehensive)
- [ ] Full ablation of each component: conn_in random vs learned, K_iter sweep, K_hh sweep, AH on/off, reflect on/off, F.normalize on/off, encoding type
- [ ] Most data exists across experiments — needs to be consolidated into one table

### 5. N-scaling curve
- [ ] Accuracy vs N for N={256, 512, 1024, 2048, 4096, 8192} at fixed D, K_hh, K_iter
- [ ] step72 script exists but hasn't been run yet

### 6. Convergence analysis
- [ ] Training curves showing how SGNNET converges — does it need fewer epochs?
- [ ] Learning rate sensitivity

## Nice to have

### 7. Theoretical analysis
- [ ] Connection to random feature theory (Rahimi & Recht)
- [ ] Connection to graph signal processing (diffusion on random graphs)
- [ ] Why F.normalize is essential — connection to spherical representations

### 8. Visualization
- [ ] t-SNE/UMAP of neuron activations before/after routing
- [ ] Which neurons are "important" — activation magnitude heatmap
- [ ] Connectivity pattern visualization
