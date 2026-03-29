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
- [x] 02-02-PLAN.md — Soft label quality verification

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
**Goal:** Implement SGNNET with D=4 geometric space, split C matrices, and three-phase forward pass. Architecture diverges from report per CONTEXT.md decisions.
**Requirements:** ARCH-01 through ARCH-05, ARCH-07 (ARCH-06 K-means init deferred to Phase 4)
**Done when:** SGNNET forward pass runs without error, produces valid gradients, neuron positions move during a toy training loop.
**Plans:** 2/3 plans executed

Plans:
- [x] 03-01-PLAN.md — Geometric primitives (r*, dynamic connectivity) and input encoding
- [x] 03-02-PLAN.md — SGNNET nn.Module with three-phase forward pass and self-projection readout
- [x] 03-03-PLAN.md — Loss functions, initialization, parameter budget verification, and integration test

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

## Phase 4 — SGNNET Wave Architecture & Experiments
**Goal:** Progressively validate distillation of VGG16 FC into a smaller sparse network. Three stages in sequence: (1) sparse static connectivity only as the baseline, then (2) add dynamic signal propagation in two variants. Each stage benchmarked independently before the next begins.
**Requirements:** TRAIN-01 through TRAIN-06
**Done when:** All three stages trained and benchmarked; wave_comparison.md shows the contribution of each addition.
**Plans:** 3/5 plans executed

Plans:
- [x] 04-01-PLAN.md — SGNNET_Wave model architecture (norm_masked, wave_routing, model_wave, tests)
- [x] 04-02-PLAN.md — Training infrastructure (trainer with FP16 AMP, GA search harness)
- [x] 04-03-PLAN.md — Stage A static connectivity baseline (GA search + full training)
- [x] 04-04-PLAN.md — Exp 1 + Exp 2 wave experiments (GA search + full training)
- [x] 04-05-PLAN.md — Evaluation and wave comparison


**Experimental progression:**
```
Stage A — Static connectivity only
          Binary C matrices, real activations, no proximity routing
          → establishes: can sparse wiring alone distill VGG16?

Stage B — Static + dynamic propagation (Exp 1: spatial phase)
          Adds proximity routing with path-length phase, λ = r*/2
          → establishes: does geometric wave propagation improve over static?

Stage C — Static + dynamic propagation (Exp 2: spatial phase + W_phase)
          Adds learned per-dimension phase operator on top of Exp 1
          → establishes: does learned phase identity improve over geometry alone?
```

Exp 1 and Exp 2 are variants of Stage B/C — they are the dynamic connectivity experiments. Stage A is the prerequisite benchmark.

**Architecture changes from Phase 3 baseline:**

| Component | Phase 3 | Phase 4 |
|---|---|---|
| C matrix values | Learned scalars | Binary 0/1 only, immutable |
| Activation | Real D-vector | Complex phasor Z ∈ ℂ^D (D real + D imag) |
| Normalization | LayerNorm over all neurons | Masked — active neurons only (|Z_j| > ε) |
| Proximity routing | Gaussian amplitude only | Gaussian(d) × Z × exp(i×2πd/λ), λ = r*/2 |
| Dynamic topology | Supported | Deferred to Generation 2 |
| Training precision | FP32 | FP16 mixed precision (autocast + GradScaler) |

**λ = r*/2 = r_repel:** one full oscillation in active zone [r_repel, r*]. Both boundaries in-phase (φ=0). Single inhibitory ring at d = 3r*/4.

**FP16 training (MPS-correct AMP):** All training uses `torch.autocast('mps', dtype=torch.float16)` + `torch.amp.GradScaler('mps')` (PyTorch 2.3+). Do NOT use `torch.cuda.amp.*` — that is CUDA-only and silently no-ops or errors on MPS. Model weights stored in FP32; forward and backward compute in FP16. Apple Silicon GPU processes FP16 natively (~1.5–2× throughput for large matmuls). `cdist` and norm ops may auto-promote to FP32 internally — that is expected and correct. If `torch.__version__ < 2.3`, fall back to `autocast` alone (skip GradScaler — MPS scaler had inf-detection bugs before 2.3).

---

### Plan 4.1 — Stage A: Static Connectivity Baseline

Train and benchmark SGNNET with binary static C matrices and no dynamic proximity routing. This is the reference point that all subsequent dynamic experiments are compared against.

**Architecture (Stage A):**
- C matrices: binary 0/1 mask, immutable — no learned values
- Activations: real D-vectors (no phasor, no phase)
- No proximity routing — forward pass is pure sparse static recurrence
- Selective normalization: only neurons with |A_j| > ε participate

**Forward pass:**
```
A_hidden = masked_norm(C_input_mask @ A_input)           # seeding
for k in K-1:
    A_hidden = masked_norm(C_hh_mask @ A_hidden)         # static recurrence
A_out = C_ho_mask @ A_hidden                             # output
scores = (A_out × W_norm).sum(dim=-1)                    # self-projection readout
```

**Stage 1 — GA hyperparameter search (partial data):**
- Search: K ∈ {1,2,3,4}, N_hidden ∈ {64,128,256}, D ∈ {2,4,8}, lr, λ_safety
- Fitness: convergence quality after 15 epochs on 15% of data

**Stage 2 — Full training with found hyperparams (FP16 + GradScaler).**

**Deliverables:**
- `src/sgnnet/model_wave.py`: SGNNET_Wave with `use_proximity=False` flag for Stage A
- `src/sgnnet/norm_masked.py`: `masked_normalize`
- `src/training/trainer.py`: shared training loop with `autocast('mps', float16)` + `GradScaler`
- `results/stageA_ga_results.json`, `results/stageA_full.json`
- `checkpoints/stageA_best.pt`

**Verification:** Loss converges; `stageA_full.json` has top1, mAP, per-class metrics for all 10 classes.

---

### Plan 4.2 — Wave Architecture: Proximity + Phase Infrastructure

Extend SGNNET_Wave to support proximity-based routing with path-length phase. Shared infrastructure for Stage B (Exp 1) and Stage C (Exp 2).

**New components added to `model_wave.py`:**
- `use_proximity=True` flag enables the proximity path
- Phasor activation state: `(Z_re, Z_im)` each `[batch, N_hidden, D]`
- Seeding: `Z_re = C_input_mask @ A_input`, `Z_im = zeros`
- C path: `C_hh_mask @ Z_re`, `C_hh_mask @ Z_im` (binary, no phase)
- Proximity path with path-length phase (λ = r*/2):
```
d_ij     = ||W_pos_i − W_pos_j||
strength = exp(−d²/r*²) × (d < r*)
phase_ij = 2π × d_ij / λ

Z_prox_re_j = Σ_i strength_ij × (Z_re_i × cos(phase_ij) − Z_im_i × sin(phase_ij))
Z_prox_im_j = Σ_i strength_ij × (Z_re_i × sin(phase_ij) + Z_im_i × cos(phase_ij))
```
- Combined: `Z_new = C_path + proximity_path`
- Readout: `A_out = sqrt(Z_re² + Z_im²)` (magnitude), self-projection as before

**Also delivers:**
- `src/sgnnet/norm_masked.py`: `masked_normalize` — mean/var over neurons with `|Z_j| > eps` only
- `tests/test_model_wave.py`: forward shapes, Z_im non-zero after first proximity step, selective norm

**Verification:**
- `Z_im` is all-zeros after seeding, non-zero after first proximity step
- Gradients flow through both re and im paths back to W_pos
- Selective norm excludes zero-activation neurons from statistics

---

### Plan 4.3 — GA Hyperparameter Search Harness

Build the search infrastructure used by all three stages.

**Deliverables:**
- `src/training/ga_search.py`: GASearch class
  - Population-based search, configurable search space
  - Fitness function: train N_epochs on partial data (10-20%), score = convergence quality
    - Score = −final_loss if loss decreased monotonically; large penalty if loss explodes or oscillates
  - Mutation: Gaussian perturbation on continuous params, random resample on discrete
  - Selection: top-k tournament
- `scripts/run_ga_search.py`: CLI — takes experiment name, writes `results/{exp}_ga_results.json`

**Search space (both experiments):**

| Hyperparameter | Type | Range |
|---|---|---|
| K | discrete | {1, 2, 3, 4} |
| N_hidden | discrete | {64, 128, 256} |
| D | discrete | {2, 4, 8} |
| lr_Wpos | continuous (log) | [1e-5, 1e-2] |
| λ_safety | continuous | [0.0, 1.0] |
| batch_size | discrete | {64, 128, 256} |

**GA config:** population=20, generations=10, top_k=5, partial_data_fraction=0.15, epochs_per_eval=15

**Verification:**
- GA runs to completion without error on partial data
- `ga_results.json` contains best config with fitness score and convergence curve

---

### Plan 4.4 — Stage B: Experiment 1 — Spatial Path-Length Phase Only

Phase derived entirely from geometric distance between neuron positions. No learned phase operator.

**What trains:** W_pos (shapes proximity topology and path-length phase), safety valve loss.

**Forward pass (Exp1):**
```
Z_j_new = masked_norm(C_path(Z) + proximity_path_with_phase(Z, W_pos))
          ↑ no W_phase — phase comes only from ||W_pos_i − W_pos_j||
```

**Stage 1 — GA search:**
- Run GASearch on Exp1 architecture with partial data
- Output: `results/exp1_ga_results.json` — best (K, N_hidden, D, lr_Wpos, λ_safety)

**Stage 2 — Full training:**
- Train with GA-found hyperparams on full dataset
- Log per-epoch: task loss, safety loss, |Z| distribution, W_pos spread

**Deliverables:**
- `scripts/train_exp1.py`
- `checkpoints/exp1_best.pt`
- `results/exp1_full.json`:
  ```json
  {
    "experiment": "spatial_phase_only",
    "hyperparams": {"K": ..., "N_hidden": ..., "D": ..., "lambda": ...},
    "top1_accuracy": ..., "mAP": ...,
    "per_class": {"tench": {...}, ...},
    "params": ..., "percent_of_vgg16_fc": ...
  }
  ```

**Verification:** Loss converges; `top1_accuracy` and `mAP` recorded for all 10 classes.

---

### Plan 4.5 — Stage C: Experiment 2 — Spatial Phase + W_phase Operator

Adds a learned per-neuron per-dimension phase operator on top of Experiment 1.

**Additional component:**
- `W_phase_j ∈ ℝ^D` per neuron — one learned phase rotation per activation dimension
- Shape: `[N_hidden + N_out, D]` — same shape as W_pos
- Applied at each receiving node after the phasor sum:

```
# After accumulating Z_incoming at node j (same as Exp1):
Z_re_new_j[k] = Z_incoming_re_j[k] × cos(W_phase_j[k]) − Z_incoming_im_j[k] × sin(W_phase_j[k])
Z_im_new_j[k] = Z_incoming_re_j[k] × sin(W_phase_j[k]) + Z_incoming_im_j[k] × cos(W_phase_j[k])
```

**What trains:** W_pos, W_phase (separate learning rates from GA search).

**Additional search dimension vs Exp1:**

| Hyperparameter | Type | Range |
|---|---|---|
| lr_Wphase | continuous (log) | [1e-5, 1e-2] |

**Stage 1 — GA search:** same harness as Plan 4.3, extended search space.

**Stage 2 — Full training:** same pipeline as Exp1.

**Deliverables:**
- `scripts/train_exp2.py`
- `checkpoints/exp2_best.pt`
- `results/exp2_full.json` (same schema as exp1_full.json, experiment = "spatial_phase_plus_operator")

**Verification:** W_phase parameters receive non-zero gradients; results comparable to Exp1 for fair ablation.

---

### Plan 4.6 — Evaluation & Comparison

Compare Stage A (static only), Stage B (Exp1), Stage C (Exp2), and Phase 3 amplitude baseline.

**What each comparison answers:**
- Stage A vs Phase 3: does removing dynamic connectivity hurt, and do binary C matrices work at all?
- Stage B vs Stage A: does geometric wave propagation (path-length phase) improve over static wiring?
- Stage C vs Stage B: does learned W_phase improve over geometry-only phase?
- Parameter cost: W_phase adds N×D params — is the improvement worth the cost?

**Deliverables:**
- `results/wave_comparison.json`: all models side by side
- `results/wave_comparison.md`:

```
| Model                  | Params | Top-1 | mAP  | Key addition vs. prior     |
|------------------------|--------|-------|------|----------------------------|
| SGNNET v1 (Phase 3)    | 649k   | xx%   | x.xx | Amplitude, learned C values|
| Stage A (static only)  | ~Nk    | xx%   | x.xx | Binary C, no dynamic       |
| Stage B / Exp1 (φ geo) | ~Nk    | xx%   | x.xx | + path-length phase        |
| Stage C / Exp2 (φ+Wφ)  | ~Nk    | xx%   | x.xx | + learned phase operator   |
```

**Verification:** All models evaluated on same val set with `compute_all_metrics`; each stage's contribution isolated.

---

## Phase 5 — Scalable Architecture Experiments
**Goal:** Extend SGNNET to large neuron counts (N ≤ 20 000) by replacing O(N²) operations with O(N·K) sparse topology. Three architectural variants tested and compared. Signal reflection routing introduced as a candidate mechanism.
**Requirements:** SCALE-01 through SCALE-06
**Done when:** All three architectures swept across N=[256,512,1024,2048,4096,10000,20000]; comparison table produced; signal reflection experiment completed.

Plans:
- [ ] 05-01-PLAN.md — Bug fixes for SGNNET_Wave at large N (fill_diagonal_ in-place, safety loss boolean indexing), neuron scaling sweep N≤10000
- [ ] 05-02-PLAN.md — SGNNET_SmallWorld: fixed fan-in index tables replacing dense C_hh einsum; no phases, index-based groups
- [ ] 05-03-PLAN.md — SGNNET_ProximityWave: sparse k-NN topology + dynamic phasor routing; O(N·K) per batch; periodic W_pos-based reconnection
- [ ] 05-04-PLAN.md — Signal reflection routing experiment
- [ ] 05-05-PLAN.md — Architecture comparison: SmallWorld vs ProximityWave vs SGNNET_Wave across all N values

### Plan 5.1 — SGNNET_Wave Bug Fixes & Neuron Scaling Sweep

**Bugs fixed:**
- `wave_routing.py`: `strength.fill_diagonal_(0)` in-place on tracked tensor → replaced with `strength * (1 - eye)` (out-of-place)
- `losses.py`: `dists[mask]` boolean indexing creates backward shape mismatch at large N on MPS → replaced with element-wise masking; safety loss disabled above N=5000 (O(N²) OOM guard)
- `lambda_safety` scaled by `(256/N)^(1/D)` to compensate for denser neuron packing at large N

**Sweep:** N_hidden = [512, 1024, 2048, 4096, 10000]

**Deliverables:**
- Fixed `src/sgnnet/losses.py`, `src/sgnnet/wave_routing.py`
- `scripts/train_exp1_scale_neurons.py`
- `results/exp1_scale_{N}.json` per run; `results/exp1_neuron_scaling.json` summary

**Verification:** All N values complete 150 epochs without crash; loss stays bounded.

---

### Plan 5.2 — SGNNET_SmallWorld

Fixed fan-in topology with no phases. Replaces O(N²) C_hh dense einsum and cdist with O(N·K) gather+sum.

**Architecture:**
- `conn_in [N_hidden, K_in]`: block-local input → hidden fan-in (K_in=50)
- `conn_hh [N_hidden, K_hh]`: Watts-Strogatz small-world graph built at init; K_local=4 within-group + K_random=2 long-range shortcuts
- No cdist, no phases — purely structural routing

**Key property:** K_random≥1 achieves ~100% graph connectivity (graph diameter ≈ O(log N)); K_random=0 leaves groups isolated (empirically 3% reachability at N=256).

**Deliverables:**
- `src/sgnnet/model_smallworld.py`
- `scripts/train_exp2_smallworld.py`
- `results/exp2_sw_{N}.json` per run; `results/exp2_smallworld.json` summary

**Verification:** N=20000 completes without OOM; backward OK at all N.

---

### Plan 5.3 — SGNNET_ProximityWave

Sparse topology with dynamic phasor routing. `conn_hh` built from W_pos k-NN (geometry-based, no hard group boundaries). Phase and strength computed only over the K edges per neuron (O(N·K) not O(N²)).

**Architecture:**
- `build_knn_conn(W_pos, K_local, K_random)` — k-NN in W_pos space + random shortcuts; no index-based groups → overlap is natural from geometry
- `sparse_phasor_route` — per-edge distance → Gaussian strength + 2π·d/λ phase; O(N·K·B·D) per forward pass
- `tick_epoch()` — called by Trainer each epoch; rebuilds conn_hh from current W_pos every `reconnect_every` epochs (adaptive topology without per-batch O(N²) cost)

**Deliverables:**
- `src/sgnnet/model_proximity_wave.py`
- `scripts/train_exp3_proxwave.py`
- `results/exp3_pw_{N}.json` per run; `results/exp3_proxwave.json` summary

**Verification:** At N=10000 each forward pass stays under 500ms; topology changes logged at reconnect points.

---

### Plan 5.4 — Signal Reflection Routing Experiment

**Idea:** In the routing step, activations propagate conditionally based on sign and magnitude. Strongly negative activations are reflected back to the originating neuron rather than propagating.

**Rule:**
```
Z_prop[h]    = relu(Z[h])             # positive part — travels to neighbours
Z_reflect[h] = relu(-Z[h] - θ)        # only strongly negative values bounce back
Z_new[h]     = -Z_reflect[h] + Σ_k weight[h,k] * Z_prop[k]
```
Where θ is a threshold hyperparameter (default 0.0 = any negative reflects; >0 = only strongly negative).

**Properties expected:**
- Sparse activation propagation: only positive neurons transmit each step
- Self-inhibition: strongly suppressed neurons actively dampen themselves, creating routing "dead zones" that information flows around
- Input-dependent information channels: active path through the graph shifts per input
- Asymmetric gradient flow: gradients only propagate through connections where source was positive

**Risk:** Dying neuron cascade — once negative a neuron may never recover. Mitigation: leaky reflection `α·relu(-Z - θ)` with α=0.1 lets negative signal drain rather than accumulate.

**Phasor extension:** gate on real component (in-phase = propagate, out-of-phase = reflect); imaginary component tracks phase direction.

**Implementation:** Add `reflective: bool` and `reflect_threshold: float` flags to `SGNNET_ProximityWave._route()`. Compare N=1024 with/without at 100 epochs.

**Deliverables:**
- `reflective` flag in `src/sgnnet/model_proximity_wave.py`
- `scripts/train_exp4_reflection.py` — ablation at N=1024: standard vs leaky-reflect (α=0.1, θ=0.0) vs hard-reflect (α=1.0, θ=0.5)
- `results/exp4_reflection.json` — side-by-side metrics

**Verification:** No NaN gradients; leaky variant does not produce dead neurons (monitor fraction of neurons with |Z|<ε per epoch).

---

### Plan 5.5 — Architecture Comparison

Aggregate all three architectures across the N sweep. Identify accuracy vs. compute trade-off.

**Comparison axes:**
- top-1 accuracy and mAP at each N
- Training time per epoch (ms/epoch) vs N
- Forward pass memory footprint vs N
- Whether phase routing adds measurable benefit over flat SmallWorld gather

**Deliverables:**
- `results/arch_comparison.json` — all three models at all N values
- `results/arch_comparison.md` — human-readable table

---

## Phase 6 — PCA Compression
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

## Phase 7 — Comparative Analysis & Report
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
| March 24 | 3 | SGNNET core architecture implemented (amplitude baseline) |
| March 25+ | 4 | Wave architecture refactor; Exp1 + Exp2 GA search + training |
| March 26+ | 5 | Scaling experiments: SmallWorld, ProximityWave, signal reflection, N≤20000 |
| TBD | 6 + 7 | PCA sweep; final comparative report |

**Note:** Phase 4 redesigned on March 25 to implement wave-based phasor architecture with two experimental variants. Phase 5 and 6 timeline adjusted accordingly.

---
*Roadmap created: 2026-03-23*
*Last updated: 2026-03-26 — Phase 5 added (scalable architecture experiments + signal reflection); PCA → Phase 6; Report → Phase 7*
