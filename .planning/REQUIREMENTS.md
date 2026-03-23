# Requirements: neuro_graph

**Defined:** 2026-03-23
**Core Value:** SGNNET matches VGG16 FC accuracy at ≤1% of its parameters

## v1 Requirements

Requirements for the 3-day experimental sprint (March 23–25, 2026).

### Data Pipeline

- [ ] **DATA-01**: Imagenette dataset downloaded and split into train/val sets (~9.5k / ~3.9k images)
- [ ] **DATA-02**: VGG16 pretrained (ImageNet weights) loaded in eval mode, backbone frozen
- [ ] **DATA-03**: All images passed through VGG16 CNN layers; pre-FC activations (25088-dim) extracted via forward hook
- [ ] **DATA-04**: Soft labels computed: VGG16 FC output → logits for 10 Imagenette classes → softmax
- [ ] **DATA-05**: Feature vectors and soft labels stored in HDF5 tensor store
- [ ] **DATA-06**: CSV manifest maps each record to tensor index, split (train/val), and ground-truth class

### Dense Baseline (Frozen VGG16)

- [ ] **BASE-01**: Pretrained VGG16 (frozen, eval mode) run on Imagenette val set — no training
- [ ] **BASE-02**: Top-1 accuracy on Imagenette val set recorded as the benchmark to beat
- [ ] **BASE-03**: VGG16 FC parameter count (~123.6M) and FLOPs per inference recorded

### SGNNET Architecture

- [ ] **ARCH-01**: SGNNET core module implemented: W positions [N, D], C sparse matrix [N, N], K-iteration loop
- [ ] **ARCH-02**: Dynamic connectivity function with personal volume radius r* = (box_size/2) / N^(1/D)
- [ ] **ARCH-03**: Self-projection readout: score_i = dot(A_i, W_i) / ||W_i||
- [ ] **ARCH-04**: Safety valve loss (dead-zone Coulomb repulsion) implemented
- [ ] **ARCH-05**: Load balance loss implemented (variance of per-neuron selection frequency)
- [ ] **ARCH-06**: K-means initialization for hidden and output neuron positions
- [ ] **ARCH-07**: N_in strategy resolved: either accept large N (25088+) or add input adapter projection

### SGNNET Training

- [ ] **TRAIN-01**: Total loss = KL divergence + λ_safety × safety_valve + λ_lb × load_balance
- [ ] **TRAIN-02**: Input neuron gradients zeroed (fixed positions); hidden + output positions learned
- [ ] **TRAIN-03**: Position clamping enforced after each optimizer step (keep inside [0, box_size])
- [ ] **TRAIN-04**: SGNNET achieves ≤1.24M trainable parameters (1% of VGG16 FC ~123.6M)
- [ ] **TRAIN-05**: Static C matrix maintains ≥90% sparsity throughout training
- [ ] **TRAIN-06**: Top-1 accuracy on Imagenette val set measured and recorded

### PCA Compression

- [ ] **PCA-01**: PCA fitted on training feature vectors (25088-dim); explained variance curve produced
- [ ] **PCA-02**: Compression sweep over k ∈ {64, 128, 256, 512, 1024, 2048} principal components
- [ ] **PCA-03**: SGNNET retrained with PCA-compressed input for each k
- [ ] **PCA-04**: Accuracy vs. compression ratio curve produced
- [ ] **PCA-05**: Optimal k identified (highest compression with <2% accuracy drop vs. full-dim SGNNET)
- [ ] **PCA-06**: PCA transformation overhead (FLOPs) accounted for in compute comparison

### Comparative Analysis

- [ ] **ANAL-01**: Parameter count table: VGG16 FC / Dense MLP / SGNNET full / SGNNET+PCA(k*)
- [ ] **ANAL-02**: FLOPs per inference computed for all four variants (including PCA transform overhead)
- [ ] **ANAL-03**: Top-1 accuracy on Imagenette val set for all four variants
- [ ] **ANAL-04**: Accuracy vs. parameter count scatter plot
- [ ] **ANAL-05**: Accuracy vs. compression ratio curve (for PCA sweep)
- [ ] **ANAL-06**: Summary report (`results/report.md`) with tables and key findings

## v2 Requirements

Deferred to future milestone.

### Dynamic Topology

- **TOPO-01**: Co-activation tracking across training steps
- **TOPO-02**: Hebbian prune-and-grow topology updates (every M steps)
- **TOPO-03**: Visualization of connectivity graph evolution over training

### Differentiable Routing

- **ROUT-01**: Gumbel-softmax replacement for hard gate (dists < r*)
- **ROUT-02**: Comparison: differentiable routing vs. hard gate, accuracy and gradient flow

### Broader Application

- **APP-01**: SGNNET applied to BERT FFN layer (hidden_dim=768 N_in)
- **APP-02**: Ablation: static C only vs. dynamic connectivity only vs. both
- **APP-03**: K ablation: K = 1, 3, 5, 10 — performance vs. compute

## Out of Scope

| Feature | Reason |
|---------|--------|
| Full ImageNet (1.2M images) | Storage/compute prohibitive for 3-day sprint; Imagenette is sufficient |
| VGG16 fine-tuning end-to-end | Backbone frozen by design; distillation only in v1 |
| CUDA GPU | User is on Apple Silicon (MPS); CUDA path not needed |
| Dynamic topology (v2) | Adds complexity; validate core architecture first |
| Attention comparison (Section 10.2 of report) | Research question, deferred to post-v1 analysis |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 through DATA-06 | Phase 1 | Pending |
| BASE-01 through BASE-03 | Phase 2 | Pending |
| ARCH-01 through ARCH-07 | Phase 3 | Pending |
| TRAIN-01 through TRAIN-06 | Phase 4 | Pending |
| PCA-01 through PCA-06 | Phase 5 | Pending |
| ANAL-01 through ANAL-06 | Phase 6 | Pending |

**Coverage:**
- v1 requirements: 35 total
- Mapped to phases: 36
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-23*
*Last updated: 2026-03-23 after initial definition*
