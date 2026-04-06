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

## Phase 5 — Scalable Architecture Experiments & Mechanism Discovery
**Goal:** Two parallel tracks: (A) architectural exploration — neuron scaling sweep, SGNNET_ProximityWave, and signal reflection routing at large N; (B) mechanism discovery — systematic ARM experiments to maximize accuracy at D=64/N=1024/K_iter=8. Both tracks feed the final architecture comparison. Infrastructure (SGNNET_SmallWorld, SGNNET_Resonant, SGNNET_AntiHebbian, Trainer) is complete.
**Confirmed so far:** D=64 Fourier encoding ceiling = 56.28%; AntiHebb α=1.0 = 80.08% (+23.80pp, all-time best, step29c Config A); K_iter=3→8 gap = 15pp; signed coupling dead at D=64 (5 experiments); input-gated adjacency +2.34pp over ceiling.
**Requirements:** SCALE-01 through SCALE-06, ARM-01 through ARM-05
**Done when:** Neuron scaling sweep complete (N=[512,1024,2048,4096,10000]); ProximityWave trained at N=1024 and N=4096; signal reflection experiment complete; Gen4 compound result exists; dynamic connectivity question closed; spatial grouped input tested; architecture comparison table produced; final Phase 5 ceiling documented.

**Primary goal (added 2026-04-04):** SGNNET as parameter-efficient FFN replacement for transformer models. O(N×K) hard constraint. N-scaling law hypothesis: increasing N incorporates higher complexity. Steps 57-61 are additional ablation experiments within Phase 5 to test new architectural mechanisms before synthesis.

Plans:
- [x] 05-01-PLAN.md — Neuron scaling sweep (N=[512,1024,2048,4096,10000]) using SGNNET_SmallWorld baseline
- [x] 05-02-PLAN.md — SGNNET_ProximityWave: sparse W_pos k-NN topology + dynamic phasor routing + periodic reconnection
- [x] 05-03-PLAN.md — Signal reflection routing experiment + architecture comparison (SmallWorld vs ProximityWave)
- [ ] 05-04-PLAN.md — ARM 1+2: Gen4 compound (step29c → step32) + new mechanisms (step48 K_iter, step52/53 high-D routing, step54 LR schedule)
- [ ] 05-05-PLAN.md — ARM 3+4: Dynamic connectivity closure (step36/49/50/51) + input architecture (step55 spatial grouped, PCA input)
- [ ] 05-06-PLAN.md — Phase 5 synthesis: best config identified; REQUIREMENTS updated; LEARNINGS consolidated

**New architecture experiments (2026-04-04, additional work within Phase 5):**
- Steps 57-61 are a fast ablation wave (50% data, 75ep) testing: benchmark anchor (57), resonance-excitatory (58), beam unification (59), phase-distance routing options A/B (60), hub interneurons (61).
- Winners from steps 58-61 compound in Wave 2 (full data, 150ep) before Phase 5 synthesis (05-06).
- Updated by new architecture experiments step57-61 (2026-04-04)

### Infrastructure Already Complete (done in earlier Phase 5 iterations)

- `src/sgnnet/model_smallworld.py` — O(N·K) sparse gather-sum routing (conn_in, conn_hh); K_in=50, K_local=4, K_random=2, n_groups=128
- `src/sgnnet/model_resonant.py` — Beam + dynamic_z_geo routing wrapper (SGNNET_Resonant); K_phase=8, beam_size=32
- `src/sgnnet/mechanisms_inhibitory.py` — AntiHebbian inhibition wrapper (SGNNET_AntiHebbian); variant=wpos
- `src/sgnnet/model_spatial_grouped.py` — Learned per-group input projections + bridge neurons (step55, ready to dispatch)
- `src/training/trainer.py` — Trainer with MPS/CUDA, FP16, plateau LR, safety valve
- `src/training/experiment_config.py` — topology_kwargs, trainer_kwargs, run_metadata

### Plan 5.1 — Neuron Scaling Sweep

Sweep SGNNET_SmallWorld (with best AntiHebb α=0.7 mechanism stack) across N=[512, 1024, 2048, 4096, 10000] to establish the accuracy vs. compute trade-off at scale.

**Note:** Also fixes any remaining SGNNET_Wave bugs (fill_diagonal_ in-place, safety loss boolean indexing at large N, lambda_safety scaling).

**Bug fixes to carry forward:**
- `wave_routing.py`: `strength.fill_diagonal_(0)` → `strength * (1 - eye)` (out-of-place)
- `losses.py`: `dists[mask]` boolean indexing → element-wise masking; safety loss disabled above N=5000 (OOM guard)
- `lambda_safety` scaled by `(256/N)^(1/D)` to compensate for denser neuron packing at large N

**Sweep:** N_hidden = [512, 1024, 2048, 4096, 10000]; all at D=64, K_iter=8, AntiHebb α=0.7 wpos

**Deliverables:**
- Fixed `src/sgnnet/losses.py` (if needed), updated safety scaling in `experiment_config.py`
- `scripts/train_step56_n_scaling.py` — runs all N values in sequence on Mac Studio
- `results/train_step56_n_scaling.json` — accuracy, params, time per N
- LEARNINGS entry: accuracy vs N curve; where does performance plateau?

**Verification:** All N values complete 150 epochs; JSON records top-1 per N; no OOM at N=10000.

---

### Plan 5.2 — SGNNET_ProximityWave at Scale

Sparse topology with dynamic phasor routing. `conn_hh` built from W_pos k-NN (geometry-based, no hard group boundaries). Phase and strength computed only over K edges per neuron (O(N·K) not O(N²)).

**Architecture:**
- `build_knn_conn(W_pos, K_local, K_random)` — k-NN in W_pos space + random shortcuts; overlap natural from geometry
- `sparse_phasor_route` — per-edge distance → Gaussian strength + 2π·d/λ phase; O(N·K·B·D) per forward
- `tick_epoch()` — rebuilds conn_hh from current W_pos every `reconnect_every` epochs (adaptive topology without per-batch O(N²) cost)

**Sweep:** N=[1024, 4096]; compare SmallWorld vs ProximityWave at each N

**Deliverables:**
- `src/sgnnet/model_proximity_wave.py` (implement or verify existing)
- `scripts/train_exp3_proxwave.py` — N=1024 and N=4096 runs
- `results/exp3_proxwave.json` — per-N accuracy, time per epoch, topology change logs

**Verification:** At N=4096 each forward pass stays under 500ms; topology changes logged at reconnect points; no NaN gradients.

---

### Plan 5.3 — Signal Reflection Routing + Architecture Comparison

Signal reflection: activations propagate conditionally based on sign and magnitude. Strongly negative activations are reflected back to the originating neuron rather than propagating.

**Reflection rule:**
```
Z_prop[h]    = relu(Z[h])              # positive → travels to neighbours
Z_reflect[h] = relu(-Z[h] - θ)         # strongly negative → bounces back
Z_new[h]     = -Z_reflect[h] + Σ_k weight[h,k] * Z_prop[k]
```
θ = threshold (default 0.0 = any negative reflects; >0 = only strongly negative).

**Properties expected:**
- Sparse activation propagation: only positive neurons transmit each step
- Self-inhibition: strongly suppressed neurons dampen themselves → routing "dead zones"
- Input-dependent information channels: active path shifts per input
- Risk: dying neuron cascade. Mitigation: leaky reflection α=0.1 (`α·relu(-Z-θ)`)

**Ablation at N=1024 (100 epochs):**
- standard routing (SmallWorld Ref)
- leaky-reflect (α=0.1, θ=0.0)
- hard-reflect (α=1.0, θ=0.5)
- ProximityWave + leaky-reflect

**Architecture comparison table (Phase 5 deliverable):**

| Architecture | N | top-1 | ms/epoch | Notes |
|---|---|---|---|---|
| SmallWorld baseline | 1024 | 56.28% | — | D=64 ceiling |
| SmallWorld + AntiHebb | 1024 | 75.24% | — | best mech |
| SmallWorld + AntiHebb | 4096 | ? | — | from Plan 5.1 |
| ProximityWave | 1024 | ? | — | from Plan 5.2 |
| SmallWorld + reflection | 1024 | ? | — | this plan |

**Deliverables:**
- `reflective` flag added to `src/sgnnet/model_proximity_wave.py` (or new `model_reflection.py`)
- `scripts/train_exp4_reflection.py`
- `results/exp4_reflection.json`
- `results/arch_comparison.md` — full architecture comparison table

**Verification:** No NaN gradients; leaky variant has <5% dead neurons; arch_comparison.md produced.

---

### Plan 5.4 — ARM 1 + ARM 2: Gen4 Compound & New Mechanisms

Sync and analyze the generational compounding results (ARM 1) and new mechanism sweeps (ARM 2). Adopt winners into the definitive base configuration.

**ARM 1 — Gen4 compound (step29c → step32):**
- Monitor + sync `results/train_step29c_mechanisms_calibrated.json` from Mac Studio
- Analyze: which of (AntiHebb, phase_exc, interneurons, fast_W_phase) survive on calibrated base?
- Write `scripts/train_step32_gen4_compound.py` with confirmed winners stacked
- Dispatch step32; sync JSON when complete
- Update LEARNINGS: definitive Gen4 ceiling

**ARM 2 — New mechanisms (step48, step52, step53, step54):**
- Sync step48 (K_iter={8,12,16,24,32} ± AntiHebb) and step54 (LR schedule: plateau vs CosineWarmRestarts vs cosine)
- Dispatch step52 (high-D Z-subspace / W_pos-subspace / projection / centering routing) when slot opens
- Dispatch step53 (low-rank + frequency-group cross-dim mixing rank-4/8) when slot opens
- Analyze each: does K_iter > 8 help? Does CosineWarmRestarts improve convergence? Does subspace gating compound?

**Decision rules:**
- Any ARM 2 winner → adopt into Gen4 base before step32
- If K_iter > 8 wins: update topology_kwargs default
- If CosineWarmRestarts wins: update trainer_kwargs default

**Deliverables:**
- `results/train_step29c_mechanisms_calibrated.json` (synced)
- `results/train_step32_gen4_compound.json`
- `results/train_step48_kiter_sweep_d64.json`, `results/train_step54_warm_restart_lr.json` (synced)
- `results/train_step52_*.json`, `results/train_step53_*.json`
- `learnings/LEARNINGS_phase5_p8_arm1_arm2.md` — findings and adopted changes

**Verification:** step32 JSON exists; step48/54/52/53 JSONs exist; LEARNINGS documents winners vs losers.

---

### Plan 5.5 — ARM 3 + ARM 5: Dynamic Connectivity & Input Architecture

Resolve the two remaining architectural questions: (1) can O(N·K) input-dependent topology match O(N²) signed coupling? (2) does learned spatial projection beat random K_in=50?

**ARM 3 — Dynamic connectivity closure:**
- Sync step36 final result (all gated configs); analyze vs static ceiling
- Dispatch step49 (signed coupling K_iter threshold 3-7 sweet spot) and step50 (spatial W_pos K-NN, epoch-level rebuild) and step51 (W_pos K-NN + W_phase strength gating)
- Render verdict: is any O(N·K) input-dependent topology viable at D=64?

**ARM 5 — Input architecture (step55 spatial grouped + PCA):**
- Rsync `src/sgnnet/model_spatial_grouped.py` + `scripts/train_step55_spatial_grouped_input.py` to Mac Studio
- Dispatch step55 (6 configs: Ref / 7 rows 0% / 7 rows 20% / 7 rows 40% / 49 pos 20% / 8 flat 20%)
- Separately: fit PCA on training features; test SGNNET with PCA-compressed input at k={256,512,1024}; compare vs spatial grouped
- Best input mechanism: random K_in=50 vs learned spatial vs PCA?

**Deliverables:**
- `results/train_step36_input_gated.json` (all configs final)
- `results/train_step49_*.json`, `results/train_step50_*.json`, `results/train_step51_*.json`
- `results/train_step55_spatial_grouped_input.json`
- `results/pca_input_sweep.json` (k sweep)
- `learnings/LEARNINGS_phase5_p9_arm3_arm5.md` with dynamic topology verdict and best input mechanism

**Verification:** All dynamic connectivity experiments have results; step55 JSON exists; PCA sweep exists; a clear best input mechanism is documented.

---

### Plan 5.6 — Phase 5 Synthesis

Consolidate all Phase 5 findings. Identify the definitive best SGNNET_SmallWorld configuration. Produce architecture comparison. Update REQUIREMENTS.

**What this plan does:**
- Read all Phase 5 result JSONs; assemble unified comparison table:
  - Mechanisms ladder (D=64 ceiling → AntiHebb → Gen4 compound)
  - Architecture comparison (SmallWorld vs ProximityWave vs Reflection at N=1024 and N=4096)
  - Input architecture (random K_in vs spatial grouped vs PCA)
- Identify the single best configuration for handoff to Phase 6 report
- Update `src/training/experiment_config.py::GA_BEST` to confirmed Phase 5 ceiling
- Write `learnings/LEARNINGS_phase5_FINAL.md` — findings, dead ends, best config
- Update REQUIREMENTS.md to mark all Phase 5 requirements as done; record achieved accuracy

**Deliverables:**
- `learnings/LEARNINGS_phase5_FINAL.md`
- `results/phase5_summary.json` — all experiment best results in one table
- Updated `src/training/experiment_config.py` with correct GA_BEST

**Verification:** `phase5_summary.json` has ≥15 experiment rows; LEARNINGS_phase5_FINAL.md records definitive best config.

---

### Plan 5.2 — ARM 2: New Mechanisms Analysis

Sync and analyze the ARM 2 sweep: K_iter scaling (step48), LR schedule (step54), high-D routing (step52), low-rank mixing (step53). Adopt winners into the base config.

**Active experiments:** step48 (K_iter sweep), step54 (LR schedule) running on Mac Studio.

**What this plan does:**
- Sync step48/54 JSONs when complete; extract winners
- Dispatch step52 (high-D subspace routing) and step53 (low-rank mixing) to Mac Studio when slots open
- Analyze: does K_iter > 8 help? Does CosineWarmRestarts beat plateau? Does subspace routing compound with AntiHebb?
- Update `learnings/EXPERIMENT_QUEUE.md` with results; mark closed questions

**Deliverables:**
- `results/train_step48_kiter_sweep_d64.json`, `results/train_step54_warm_restart_lr.json` (synced)
- `results/train_step52_*.json`, `results/train_step53_*.json` (when complete)
- `learnings/LEARNINGS_phase5_p8_*.md` entry documenting ARM 2 findings

**Verification:** All 4 steps have result JSONs and LEARNINGS entries.

---

### Plan 5.3 — ARM 3: Close Dynamic Connectivity Question

Resolve whether O(N·K) input-dependent topology can recover the gains of O(N²) signed coupling. Decision must be reached before Phase 6.

**Context:** step31 (Z-KNN per step) = 44-53%, failed. step36 (input-gated adjacency) = 58.62% for Ref config, gated configs pending. step49/50/51 queued.

**What this plan does:**
- Sync step36 final result; analyze gated configs
- Dispatch step49 (signed coupling K_iter threshold), step50 (spatial W_pos K-NN), step51 (W_pos K-NN + W_phase gate) when slots open
- Render final verdict: is there a viable input-dependent topology at O(N·K)?
- Document in `learnings/LEARNINGS_phase5_p9_dynamic_connectivity.md`

**Deliverables:**
- `results/train_step36_input_gated.json` (final, all configs)
- `results/train_step49_*.json`, `results/train_step50_*.json`, `results/train_step51_*.json`
- `learnings/LEARNINGS_phase5_p9_dynamic_connectivity.md` with verdict

**Verification:** All dynamic connectivity experiments have results; a go/no-go decision is documented.

---

### Plan 5.4 — Input Architecture: Spatial Grouped + PCA

Test whether the input mechanism (currently random sparse gather K_in=50) can be improved by learned spatial projections or PCA compression.

**Context:** `src/sgnnet/model_spatial_grouped.py` implemented locally; `scripts/train_step55_spatial_grouped_input.py` written. VGG16 pool5 = [512,7,7] = 25088 → spatial structure exploitable.

**What this plan does:**
- Rsync step55 files to Mac Studio; dispatch when slot opens
- Sync result when complete; analyze: does learned per-group proj outperform random K_in=50?
- Separately: fit PCA on training features; test SGNNET with PCA-compressed input (k=256/512/1024) as lightweight alternative to raw 25088-dim sparse gather
- Document input architecture winner

**Deliverables:**
- `results/train_step55_spatial_grouped_input.json` (synced)
- `results/pca_input_sweep.json` (PCA sweep across k values)
- `learnings/LEARNINGS_phase5_p10_input_architecture.md` with best input mechanism

**Verification:** step55 JSON exists; PCA sweep exists; a best input mechanism is identified.

---

### Plan 5.5 — Phase 5 Synthesis

Consolidate all Phase 5 findings. Identify the definitive best SGNNET_SmallWorld configuration. Update REQUIREMENTS with the achieved accuracy. Write the Phase 5 LEARNINGS summary.

**What this plan does:**
- Read all Phase 5 result JSONs; assemble comparison table (Ref vs each ARM winner vs Gen4 compound)
- Identify the single best configuration: base + K_iter + AntiHebb α + LR schedule + input architecture + any dynamic topology winner
- Update `src/training/experiment_config.py::GA_BEST` to final Phase 5 ceiling
- Write `learnings/LEARNINGS_phase5_FINAL.md` summarizing findings, dead ends, and the best config
- Update REQUIREMENTS.md: mark ARM-01 through ARM-05 as done; record final top-1 achieved

**Deliverables:**
- `learnings/LEARNINGS_phase5_FINAL.md`
- Updated `src/training/experiment_config.py` with correct GA_BEST
- `results/phase5_summary.json` — all experiment best results in one table

**Verification:** `phase5_summary.json` exists with ≥10 experiment rows; LEARNINGS_phase5_FINAL.md records the definitive best config and ceiling.

---

## Phase 6 — Comparative Analysis & Final Report
**Goal:** Aggregate all Phase 5 results into a clean comparison table. Compute parameter and FLOP counts for all variants. Produce the final report showing SGNNET vs VGG16 FC across accuracy, params, and compute dimensions. The PCA input compression question is folded here from Plan 5.4.
**Requirements:** PCA-01 through PCA-06, ANAL-01 through ANAL-06
**Done when:** `results/report.md` exists with full comparison table, parameter accounting, and key findings from all 5 ARM phases.

### Plan 6.1 — Parameter & FLOPs Accounting

Compute params and FLOPs per inference for all variants produced across Phases 3–5.

**Four variants:**
1. VGG16 FC (original): 123.6M params, dense matmul FLOPs
2. Dense MLP (distilled): same architecture, same FLOPs, distilled accuracy
3. SGNNET_SmallWorld best config (from Phase 5 synthesis): params = N×D (W_pos) + K_hh×N (conn) + proj
4. SGNNET + PCA input (best k from Plan 5.4): PCA FLOPs + SGNNET params

**FLOPs accounting:**
- Dense FC: 2 × (25088×4096 + 4096×4096 + 4096×10) multiply-adds
- SGNNET_SmallWorld: K_iter × [K_hh×N×D (sparse gather-sum)] + readout
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
| Class              | VGG16 Acc | VGG16 AP | SGNNET Best | SGNNET+PCA |
|--------------------|-----------|----------|-------------|------------|
| tench              | xx%       | x.xxx    | xx%         | xx%        |
| english_springer   | ...       | ...      | ...         | ...        |
| ... (all 10)       |           |          |             |            |
```

---

### Plan 6.2 — Visualizations & Key Findings

**Deliverables:**
- `results/accuracy_vs_params.png`: top-1 accuracy vs. parameter count (all variants)
- `results/per_class_accuracy_delta.png`: per-class accuracy delta (SGNNET − VGG16) heatmap
- `results/mechanism_gains.png`: bar chart of each confirmed mechanism's contribution (pp gain vs baseline)

---

### Plan 6.3 — Summary Report

Write final `results/report.md`.

**Deliverables:**
- `results/report.md` with sections:
  1. Experiment setup (dataset, VGG16 extraction, distillation approach)
  2. VGG16 baseline — aggregate and per-class metrics
  3. SGNNET architecture evolution: Phase 3 → Phase 4 → Phase 5 SmallWorld
  4. Mechanism discovery findings: what works, what doesn't, why
  5. Best config: D=64 N=1024 K_iter=8 + AntiHebb α=0.7 + [Gen4 compound] — accuracy vs VGG16
  6. PCA compression: optimal k, accuracy vs compression trade-off
  7. Open questions and future directions

**Verification:** All ANAL + PCA requirements checked off; report readable as a standalone document.

---

## Timeline Summary

| Phase | Name | Status |
|-------|------|--------|
| 1 | Data Pipeline | Complete |
| 2 | Dense Baseline Benchmark | Complete |
| 3 | SGNNET Core Architecture | Complete |
| 4 | SGNNET Wave Architecture & Experiments | Complete |
| 5 | Mechanism Discovery & Architecture Optimization | In progress — 3 experiments running on Mac Studio |
| 6 | Comparative Analysis & Final Report | Not started |

**Phase 5 key findings so far (April 2026):**
- D=64 Fourier encoding + SmallWorld = 56.28% ceiling
- AntiHebb α=0.7 wpos = **75.24%** (+18.96pp) — all-time best
- K_iter=3→8 = +15pp gap (all 8 steps necessary)
- Signed coupling architecturally dead at D=64 (5 experiments confirm)
- Input-gated adjacency (step36): +2.34pp over ceiling — best dynamic connectivity result

---
*Roadmap created: 2026-03-23*
*Last updated: 2026-04-03 — Phase 5 revised to reflect actual ARM-based mechanism discovery work; Phases 6+7 merged into Phase 6 (analysis + report); timeline updated*
