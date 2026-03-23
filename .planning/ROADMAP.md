# Roadmap: neuro_graph

**Milestone:** v1 — SGNNET VGG16 Distillation Experiment
**Timeline:** March 23–25, 2026 (3 days)
**Status:** Not started

---

## Phase 1 — Data Pipeline
**Day:** March 23 (morning)
**Goal:** Download Imagenette, extract VGG16 pre-FC feature vectors and soft labels, persist to HDF5 tensor store.
**Requirements:** DATA-01 through DATA-06
**Done when:** HDF5 store exists with 13.4k records, each containing a 25088-dim feature vector and a 10-dim soft probability vector; CSV manifest maps every record.

### Plan 1.1 — Environment Setup
Set up Python environment with all dependencies: torch, torchvision, h5py, scikit-learn, pandas, matplotlib, tqdm.

**Deliverables:**
- `requirements.txt` with pinned versions
- `setup.py` or confirmed working `d_env` venv
- Smoke test: `import torch; print(torch.backends.mps.is_available())`

**Verification:** `python -c "import torch, torchvision, h5py, sklearn; print('OK')"` exits 0

---

### Plan 1.2 — Imagenette Dataset Download
Download Imagenette 320px variant via torchvision or direct URL. Verify class split and image counts.

**Imagenette classes (and their ImageNet indices):**
| Class | ImageNet idx |
|-------|-------------|
| tench | 0 |
| English springer | 217 |
| cassette player | 482 |
| chain saw | 491 |
| church | 497 |
| French horn | 566 |
| garbage truck | 569 |
| gas pump | 571 |
| golf ball | 574 |
| parachute | 701 |

**Deliverables:**
- `data/` directory with train/val split
- `src/data/dataset.py`: ImagenetteDataset class with standard VGG16 preprocessing (resize 224, normalize ImageNet mean/std)

**Verification:** Dataset loads 9469 train + 3925 val images; all 10 classes present in each split.

---

### Plan 1.3 — VGG16 Feature Extractor
Load pretrained VGG16, register a forward hook on `features[-1]` (pool5 output), run all images through to collect the 25088-dim flattened activations.

**Architecture note:**
- VGG16 pool5 output: [batch, 512, 7, 7] → flatten → [batch, 25088]
- Hook target: `model.features` (AdaptiveAvgPool2d or MaxPool2d final layer)
- Soft labels: forward through `model.classifier` → logits[1000] → select 10 Imagenette indices → softmax

**Deliverables:**
- `src/data/extractor.py`: VGGExtractor class
  - `extract_features(dataloader) -> (features: Tensor[N, 25088], soft_labels: Tensor[N, 10])`
  - Runs on MPS if available, CPU fallback

**Verification:** features.shape == (13394, 25088); soft_labels.sum(dim=1) ≈ 1.0 for all records.

---

### Plan 1.4 — Tensor Store & CSV Manifest
Persist extracted features and soft labels to HDF5. Write CSV manifest with metadata.

**HDF5 schema:**
```
store.h5
  /train/features    [9469, 25088]  float32
  /train/soft_labels [9469, 10]     float32
  /train/labels      [9469]         int64
  /val/features      [3925, 25088]  float32
  /val/soft_labels   [3925, 10]     float32
  /val/labels        [3925]         int64
```

**CSV schema:** `index, split, class_name, class_idx, h5_idx`

**Deliverables:**
- `data/store.h5`
- `data/manifest.csv`
- `src/data/store.py`: TensorStore class (read/write, indexed access)

**Verification:** Random sample: `store[42]` returns (feature[25088], soft_label[10], label); CSV has 13394 rows.

---

## Phase 2 — Dense Baseline Distillation
**Day:** March 23 (afternoon)
**Goal:** Implement and train a dense MLP that replicates VGG16 FC behavior via distillation. Establishes accuracy ceiling and parameter reference point.
**Requirements:** BASE-01 through BASE-04
**Done when:** Dense MLP trained, top-1 accuracy ≥85% on val set, parameter count and FLOPs recorded.

### Plan 2.1 — Dense MLP Architecture
Implement the dense MLP matching VGG16 FC layer structure.

**Architecture:**
- FC1: 25088 → 4096, ReLU, Dropout(0.5)
- FC2: 4096 → 4096, ReLU, Dropout(0.5)
- FC3: 4096 → 10, (no activation — logits)
- Parameter count: 25088×4096 + 4096×4096 + 4096×10 + biases ≈ 123.6M

**Deliverables:**
- `src/models/dense_mlp.py`: DenseMLP class
- `src/models/__init__.py`

**Verification:** `count_params(model)` returns ~123.6M; forward pass on random input [B, 25088] produces [B, 10].

---

### Plan 2.2 — Distillation Training Loop
Training loop using soft KL divergence loss against VGG16 soft labels. Loads data from HDF5 tensor store.

**Loss:** `F.kl_div(F.log_softmax(logits/T), soft_labels, reduction='batchmean') * T²`
where T = temperature (default 4.0)

**Deliverables:**
- `src/training/trainer.py`: Trainer class
  - `train_epoch(model, dataloader, optimizer) -> avg_loss`
  - `evaluate(model, dataloader) -> top1_accuracy`
- `src/training/config.py`: TrainConfig dataclass (lr, epochs, batch_size, temperature)
- `scripts/train_baseline.py`: CLI entry point

**Verification:** Training loss decreases over first 5 epochs; no NaN/Inf in loss.

---

### Plan 2.3 — Baseline Evaluation & Metrics
Evaluate trained dense MLP, record all baseline metrics.

**Deliverables:**
- `results/baseline_dense.json`:
  ```json
  {
    "model": "DenseMLP",
    "params": 123646952,
    "top1_accuracy": ...,
    "flops_per_inference": ...,
    "training_epochs": ...
  }
  ```
- `src/utils/metrics.py`: `count_params(model)`, `count_flops(model, input_shape)`

**Verification:** `results/baseline_dense.json` exists; top1_accuracy ≥ 0.85.

---

## Phase 3 — SGNNET Core Architecture
**Day:** March 24 (morning)
**Goal:** Implement SGNNET from the spec in `sparse_geometric_network_report.md`. All components modular, tested independently.
**Requirements:** ARCH-01 through ARCH-07
**Done when:** SGNNET forward pass runs without error, produces valid gradients, neuron positions move during a toy training loop.

### Plan 3.1 — Geometric Primitives
Implement the mathematical primitives: personal volume radius, dynamic connectivity, safety valve.

**Deliverables:**
- `src/sgnnet/geometry.py`:
  - `personal_volume_radius(N, D, box_size=1.0) -> float`
  - `dynamic_connectivity(A, W, N, D, box_size) -> Tensor[N, D]`
    (Gaussian-weighted routing with hard gate at r*)

**Math:**
```
r* = (box_size / 2) / N^(1/D)
strength[i,j] = exp(-||A_i - W_j||² / (r*² + ε)) * (||A_i - W_j|| < r*)
output[j] = Σ_i (strength[i,j] / Σ_k strength[k,j] + ε) * A_i
```

**Verification:**
- `r*` is always ≤ box_size/2
- Expected neighbors per neuron ≈ 1 (empirically verify on random W, A)
- `dynamic_connectivity` output shape == [N, D]; diagonal is zero (no self-routing)

---

### Plan 3.2 — SGNNET Module
Implement the full SGNNET nn.Module: W positions, C sparse matrix, K-iteration loop, self-projection readout.

**Deliverables:**
- `src/sgnnet/model.py`: SGNNET class
  - `__init__(N, D, N_in, N_out, sparsity=0.90, K=3, box_size=1.0)`
  - `forward(x: Tensor[B, N_in, D]) -> Tensor[B, N_out]`
  - Inner loop: `A = F.relu(norm(A @ C + dynamic_connectivity(A, W, ...)))`
  - Self-projection readout: `score_i = (A_out * W_norm).sum(dim=-1)`

**N_in design decision (resolve here):**
- Option A: N_in = 25088 (N ≥ 25088; C matrix memory ≈ N² × 0.1 × 4 bytes)
  - At N=25100: C = 25100² × 0.1 = 63M entries = 252MB ← borderline feasible
- Option B: Input adapter Linear(25088 → N_in) before SGNNET, N_in = 512
  - Adapter params: 512 × 25088 = 12.8M — exceeds 1% budget alone
- Option C: Input adapter Linear(25088 → N_in) before SGNNET, N_in = 48
  - Adapter params: 48 × 25088 = 1.2M — fits within 1% budget

**Recommended default:** Option C (adapter 25088→48, SGNNET N_in=48)
Record this decision in PROJECT.md Key Decisions during execution.

**Verification:**
- `model.forward(x)` produces [B, N_out] tensor
- `loss.backward()` runs cleanly; no NaN gradients
- `model.W.grad[:N_in]` is zeroed after backward (input neurons fixed)

---

### Plan 3.3 — Loss Functions
Implement the three-component loss and K-means initialization.

**Deliverables:**
- `src/sgnnet/losses.py`:
  - `safety_valve_loss(W, box_size, N, D) -> scalar`
  - `load_balance_loss(selection_counts) -> scalar`
  - `total_loss(scores, targets, W, ...) -> scalar`
- `src/sgnnet/init.py`:
  - `kmeans_init(data, n_clusters, D) -> Tensor[n_clusters, D]`
  - `initialize_sgnnet(model, X_train, Y_train)`

**Verification:**
- `safety_valve_loss` returns exactly 0.0 when all neurons are far apart (> r_repel)
- `safety_valve_loss` returns > 0.0 when two neurons are closer than r_repel
- `load_balance_loss` is minimal when all neurons selected equally

---

### Plan 3.4 — Parameter Count & Sparsity Verification
Verify SGNNET meets the 1% parameter target before training.

**Parameter budget:**
- VGG16 FC: ~123.6M params → 1% = 1.236M
- SGNNET (Option C, N_in=48, N=48+512+10=570, D=64, sparsity=0.90):
  - C_values: 570² × 0.10 = 32,490 params
  - W (hidden + output only): (512+10) × 64 = 33,408 params
  - Input adapter: 48 × 25088 = 1,204,224 params
  - Norm: 2 × 64 = 128 params
  - **Total: ~1.27M params (1.02%)** ✓

**Deliverables:**
- Verified param count in `results/sgnnet_config.json`
- Sparsity verified: `(C_mask == 0).float().mean()` ≥ 0.90

---

## Phase 4 — SGNNET Training & Evaluation
**Day:** March 24 (afternoon)
**Goal:** Train SGNNET with full loss (task + safety valve + load balance), evaluate on val set, compare to dense baseline.
**Requirements:** TRAIN-01 through TRAIN-06
**Done when:** SGNNET trained, val accuracy recorded, comparison with dense baseline documented.

### Plan 4.1 — SGNNET Training Loop
Extend Trainer to support SGNNET-specific loss and gradient zeroing.

**Deliverables:**
- `src/training/sgnnet_trainer.py`: SGNNETTrainer (extends or wraps Trainer)
  - `train_epoch`: compute total_loss, zero W.grad[:N_in], clamp W inside box
  - `evaluate`: top-1 accuracy on val set
  - `log_epoch`: log loss components separately (task / safety / load_balance)
- `scripts/train_sgnnet.py`: CLI entry point

**Training config:**
- Optimizer: Adam, lr=1e-3
- Epochs: 50 (adjust based on convergence)
- Batch size: 256
- λ_safety: 0.5
- λ_lb: 0.01
- Temperature T: 4.0

**Verification:** Training loss decreases; safety_valve_loss stays near 0 during normal training; load_balance_loss decreases over time.

---

### Plan 4.2 — Ablation: K and N Sweep
Quick sweep over key hyperparameters to find best config within time budget.

**Sweep plan (parallel, background jobs):**
- K ∈ {1, 3, 5} — recursive iteration count
- N_hidden ∈ {256, 512, 1024} — hidden neuron count
- 10 epochs each (fast ablation)

**Deliverables:**
- `results/ablation_K.json`: accuracy vs. K
- `results/ablation_N.json`: accuracy vs. N_hidden
- Best config selected for final training run

**Verification:** At least one config achieves val accuracy > 60% (sanity threshold for soft-label distillation).

---

### Plan 4.3 — Final SGNNET Evaluation
Train best config for full epochs, record all metrics.

**Deliverables:**
- Trained model checkpoint: `checkpoints/sgnnet_best.pt`
- `results/sgnnet_full.json`:
  ```json
  {
    "model": "SGNNET",
    "params": ...,
    "top1_accuracy": ...,
    "flops_per_inference": ...,
    "sparsity": 0.90,
    "K": 3,
    "N": ...,
    "D": 64
  }
  ```

**Verification:** `sgnnet_full.json` exists; params ≤ 1.24M; top1_accuracy recorded.

---

## Phase 5 — PCA Compression
**Day:** March 25 (morning)
**Goal:** Apply PCA to 25088-dim features, sweep compression ratios, retrain SGNNET for each k, find optimal compression point.
**Requirements:** PCA-01 through PCA-06
**Done when:** Accuracy vs. k curve produced; optimal k identified.

### Plan 5.1 — PCA Fitting & Analysis
Fit PCA on training features, produce explained variance curve.

**Deliverables:**
- `src/data/pca.py`: PCATransform class
  - `fit(X_train)` — fit on 25088-dim training features
  - `transform(X, k) -> Tensor[N, k]` — project to k components
  - `explained_variance_ratio(ks) -> array` — fraction of variance explained
- `results/pca_explained_variance.json`: explained variance for k ∈ {32, 64, 128, 256, 512, 1024, 2048}
- `results/pca_explained_variance.png`: curve plot

**Verification:** PCA fitted; explained variance at k=2048 ≥ 95%.

---

### Plan 5.2 — Compression Sweep
Retrain SGNNET with PCA-compressed input for each k. Run in parallel (one job per k).

**Sweep:** k ∈ {64, 128, 256, 512, 1024, 2048}

For each k:
- SGNNET: N_in = k (no adapter needed — PCA handles projection)
- N_hidden = 512, D = 64, K = 3, sparsity = 0.90
- Train 50 epochs, record accuracy

**Compute overhead per inference (for comparison):**
- PCA transform: k × 25088 multiply-adds = k × 25088 FLOPs
- SGNNET inference: depends on N, K

**Deliverables:**
- `results/pca_sweep.json`:
  ```json
  {
    "k64":  {"accuracy": ..., "params": ..., "total_flops": ...},
    "k128": {"accuracy": ..., "params": ..., "total_flops": ...},
    ...
  }
  ```
- `src/scripts/train_pca_sweep.py`: runs all k values

**Verification:** All 6 k values trained; `pca_sweep.json` has entries for all.

---

### Plan 5.3 — Optimal Compression Point
Identify the highest compression (smallest k) with accuracy drop < 2% vs. full-dim SGNNET.

**Deliverables:**
- `results/pca_optimal.json`: `{"k_optimal": ..., "accuracy": ..., "compression_ratio": ...}`
- `results/accuracy_vs_k.png`: curve plot with optimal k marked

**Verification:** `k_optimal` identified; `compression_ratio` = 25088 / k_optimal computed.

---

## Phase 6 — Comparative Analysis & Report
**Day:** March 25 (afternoon)
**Goal:** Aggregate all results into a clean comparison table. Produce final report.
**Requirements:** ANAL-01 through ANAL-06
**Done when:** `results/report.md` exists with all tables, plots, and key findings.

### Plan 6.1 — Parameter & FLOPs Accounting
Compute params and FLOPs per inference for all four variants with full accounting.

**Four variants:**
1. VGG16 FC (original): 123.6M params, dense matmul FLOPs
2. Dense MLP (distilled): same architecture, same FLOPs, distilled accuracy
3. SGNNET full (25088 input via adapter): adapter + SGNNET params
4. SGNNET + PCA (k* input): PCA FLOPs + SGNNET params (no adapter needed)

**FLOPs accounting:**
- Dense FC: 2 × (25088×4096 + 4096×4096 + 4096×10) multiply-adds
- SGNNET: K × [sparse_matmul(N²×0.1, D) + cdist(N²×D) + einsum] + readout
- PCA transform: k × 25088 multiply-adds (one-time per inference)

**Deliverables:**
- `src/utils/flop_counter.py`: `count_flops_dense(...)`, `count_flops_sgnnet(...)`, `count_flops_pca(...)`
- `results/compute_accounting.json`: full breakdown for all variants

---

### Plan 6.2 — Comparison Table
Aggregate all results into a single comparison table.

**Deliverables:**
- `results/comparison_table.md`:

```
| Model              | Params   | % of VGG FC | Top-1 Acc | FLOPs (inf) |
|--------------------|----------|-------------|-----------|-------------|
| VGG16 FC           | 123.6M   | 100%        | ~xx%      | xxx M       |
| Dense MLP (dist.)  | 123.6M   | 100%        | xx%       | xxx M       |
| SGNNET (full dim)  | ~1.24M   | ~1%         | xx%       | xxx M       |
| SGNNET + PCA (k*)  | ~xxx K   | <1%         | xx%       | xxx M       |
```

---

### Plan 6.3 — Visualizations
Produce all plots: accuracy vs. params, accuracy vs. compression ratio, neuron position evolution.

**Deliverables:**
- `results/accuracy_vs_params.png`: scatter plot
- `results/accuracy_vs_k.png`: PCA compression curve
- `results/neuron_positions_before_after.png`: W position visualization (2D PCA projection of W before/after training)
- `results/safety_valve_firing_rate.png`: safety valve loss over training epochs

---

### Plan 6.4 — Summary Report
Write final `results/report.md` documenting the experiment, methodology, results, and open questions.

**Deliverables:**
- `results/report.md` with sections:
  1. Experiment setup (dataset, VGG16 extraction, distillation approach)
  2. Dense baseline results
  3. SGNNET architecture summary (N, D, K, sparsity, N_in approach chosen)
  4. SGNNET results and comparison
  5. PCA compression results
  6. Key findings and open questions (routing differentiability, K sensitivity, N_in strategy)

**Verification:** All 6 ANAL requirements checked off; report is readable stand-alone document.

---

## Timeline Summary

| Day | Phases | Target Outcome |
|-----|--------|----------------|
| March 23 | 1 + 2 | Tensor store ready; dense baseline trained and evaluated |
| March 24 | 3 + 4 | SGNNET implemented and trained; vs. dense baseline comparison |
| March 25 | 5 + 6 | PCA sweep complete; final report with all comparisons |

---
*Roadmap created: 2026-03-23*
*Last updated: 2026-03-23 after initialization*
