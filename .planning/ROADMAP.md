# Roadmap: neuro_graph

**Milestone:** v1 — SGNNET VGG16 Distillation Experiment
**Timeline:** March 23–25, 2026 (3 days)
**Status:** In progress

---

## Phase 1 — Data Pipeline
**Day:** March 23 (morning)
**Goal:** Download Imagenette, extract VGG16 pre-FC feature vectors and soft labels, persist to HDF5 tensor store.
**Requirements:** DATA-01 through DATA-06
**Done when:** HDF5 store exists with 13.4k records, each containing a 25088-dim feature vector and a 10-dim soft probability vector; CSV manifest maps every record.
**Plans:** 2 plans

Plans:
- [x] 01-01-PLAN.md — Environment setup, Imagenette download, and ImagenetteDataset class
- [ ] 01-02-PLAN.md — VGG16 feature extraction, HDF5 tensor store, and CSV manifest

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

## Phase 2 — Dense Baseline Benchmark
**Day:** March 23 (afternoon)
**Goal:** Evaluate frozen pretrained VGG16 on Imagenette val set. No training — this is the benchmark accuracy that SGNNET must approach.
**Requirements:** BASE-01 through BASE-05
**Done when:** `results/baseline_vgg16.json` exists with top-1 accuracy, FC parameter count, and FLOPs.
**Plans:** 2 plans

Plans:
- [x] 02-01-PLAN.md — Metrics module, frozen VGG16 eval, baseline JSON
- [ ] 02-02-PLAN.md — Soft label quality verification

### Plan 2.1 — Frozen VGG16 Evaluation
Run pretrained VGG16 (frozen, eval mode) on Imagenette val set. Compute full per-class metrics.

**What this measures:**
- Per-class accuracy, precision, recall, F1 = the class-level benchmark for each of the 10 Imagenette classes
- mAP (mean Average Precision) = primary aggregate comparison metric
- VGG16 FC parameter count = the 100% parameter reference (SGNNET targets ≤1% of this)

**mAP computation:** For each class c, treat it as a binary problem (class c vs. rest). Rank val samples by softmax score for class c. Compute AP = area under precision-recall curve. Average across 10 classes = mAP.

**Deliverables:**
- `src/utils/metrics.py`:
  - `compute_all_metrics(scores, labels, class_names) -> MetricsDict`
    Returns: top1_acc, per_class_acc, per_class_precision, per_class_recall, per_class_f1, mAP
  - `count_params(model)`, `count_flops(model, input_shape)`
- `scripts/eval_baseline.py`: load VGG16 pretrained, run on val set, write results
- `results/baseline_vgg16.json`:
  ```json
  {
    "model": "VGG16 (frozen)",
    "fc_params": 123646952,
    "top1_accuracy": ...,
    "mAP": ...,
    "per_class": {
      "tench":          {"accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "AP": ...},
      "english_springer": {...},
      ...
    },
    "flops_fc_per_inference": ...
  }
  ```

**Verification:** `results/baseline_vgg16.json` exists; mAP typically ~0.97–0.99 for VGG16 on Imagenette.

---

### Plan 2.2 — Soft Label Quality Check
Verify that the soft labels saved in the tensor store in Phase 1 faithfully reflect VGG16's output distribution.

**What to check:**
- Hard prediction from soft labels (argmax) matches VGG16 direct eval accuracy
- Soft label entropy is reasonable (not collapsed to one-hot)
- Class distribution is balanced across Imagenette's 10 classes

**Deliverables:**
- `scripts/verify_soft_labels.py`: loads HDF5, computes argmax top-1, compares to baseline_vgg16.json
- Adds `"soft_label_accuracy_check": true/false` to `results/baseline_vgg16.json`

**Verification:** Argmax accuracy from soft labels matches VGG16 direct eval accuracy within 0.1%.

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
Train best config for full epochs, record full per-class metrics using the same `compute_all_metrics` from Phase 2.

**Deliverables:**
- Trained model checkpoint: `checkpoints/sgnnet_best.pt`
- `results/sgnnet_full.json`:
  ```json
  {
    "model": "SGNNET",
    "params": ...,
    "top1_accuracy": ...,
    "mAP": ...,
    "per_class": {
      "tench":          {"accuracy": ..., "precision": ..., "recall": ..., "f1": ..., "AP": ...},
      ...
    },
    "flops_per_inference": ...,
    "sparsity": 0.90,
    "K": 3,
    "N": ...,
    "D": 64
  }
  ```

**Verification:** `sgnnet_full.json` exists; params ≤ 1.24M; mAP and per-class metrics recorded for all 10 classes.

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
- Train 50 epochs, record full per-class metrics

**Compute overhead per inference (for comparison):**
- PCA transform: k × 25088 multiply-adds = k × 25088 FLOPs
- SGNNET inference: depends on N, K

**Deliverables:**
- `results/pca_sweep.json`:
  ```json
  {
    "k64":  {
      "top1_accuracy": ..., "mAP": ..., "params": ..., "total_flops": ...,
      "per_class": {"tench": {...}, ...}
    },
    "k128": {...},
    ...
  }
  ```
- `src/scripts/train_pca_sweep.py`: runs all k values

**Verification:** All 6 k values trained; `pca_sweep.json` has per-class metrics for all.

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

### Plan 6.2 — Comparison Tables
Aggregate all results. One aggregate table + one per-class table.

**Deliverables:**
- `results/comparison_aggregate.md`:

```
| Model              | Params   | % of VGG FC | Top-1 Acc | mAP    | FLOPs (inf) |
|--------------------|----------|-------------|-----------|--------|-------------|
| VGG16 FC (frozen)  | 123.6M   | 100%        | xx%       | x.xxx  | xxx M       |
| SGNNET (full dim)  | ~1.24M   | ~1%         | xx%       | x.xxx  | xxx M       |
| SGNNET + PCA (k*)  | ~xxx K   | <1%         | xx%       | x.xxx  | xxx M       |
```

- `results/comparison_per_class.md`:

```
| Class              | VGG16 Acc | VGG16 AP | SGNNET Acc | SGNNET AP | SGNNET+PCA Acc | SGNNET+PCA AP |
|--------------------|-----------|----------|------------|-----------|----------------|---------------|
| tench              | xx%       | x.xxx    | xx%        | x.xxx     | xx%            | x.xxx         |
| english_springer   | ...       | ...      | ...        | ...       | ...            | ...           |
| ... (all 10)       |           |          |            |           |                |               |
```

---

### Plan 6.3 — Visualizations
Produce plots for aggregate and per-class comparisons.

**Deliverables:**
- `results/map_vs_params.png`: mAP vs. parameter count scatter (all 3 variants)
- `results/map_vs_k.png`: mAP vs. PCA k (compression sweep)
- `results/per_class_accuracy_delta.png`: heatmap of per-class accuracy delta (SGNNET − VGG16 and SGNNET+PCA − VGG16) — shows which classes the sparse model gains or loses on
- `results/per_class_ap_comparison.png`: grouped bar chart, AP per class for all 3 variants
- `results/neuron_positions_before_after.png`: 2D PCA projection of W before/after training
- `results/safety_valve_firing_rate.png`: safety valve loss over training epochs

---

### Plan 6.4 — Summary Report
Write final `results/report.md`.

**Deliverables:**
- `results/report.md` with sections:
  1. Experiment setup (dataset, VGG16 extraction, distillation approach)
  2. VGG16 baseline — aggregate and per-class metrics
  3. SGNNET architecture summary (N, D, K, sparsity, N_in approach chosen)
  4. SGNNET results — aggregate and per-class comparison vs. VGG16
  5. PCA compression results — mAP and per-class curves vs. k
  6. Key findings: which classes SGNNET recovers well, which it struggles on, optimal PCA k
  7. Open questions (routing differentiability, K sensitivity, N_in strategy)

**Verification:** All 8 ANAL requirements checked off; report is a readable stand-alone document.

---

## Timeline Summary

| Day | Phases | Target Outcome |
|-----|--------|----------------|
| March 23 | 1 + 2 | Tensor store ready; dense baseline trained and evaluated |
| March 24 | 3 + 4 | SGNNET implemented and trained; vs. dense baseline comparison |
| March 25 | 5 + 6 | PCA sweep complete; final report with all comparisons |

---
*Roadmap created: 2026-03-23*
*Last updated: 2026-03-23 after Phase 2 planning*
