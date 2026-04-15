# SGNNET Historical Specifications (Phases 1-4)

Consolidated from deprecated GSD framework files. Authoritative record for data pipeline,
baseline methodology, and pre-Phase-5 architecture decisions.

---

## Phase 1: Data Pipeline

### Dataset — Imagenette
- Train: 9469 samples, Val: 3925 samples (13394 total)
- HDF5 store: `data/store.h5` — shape [N, 25088] float32 (VGG16 pool5 features)
- Soft labels: T=1 softmax at storage time, no temperature scaling
- Accuracy diff = 0.0 (stored soft labels identical to direct VGG16 eval — Phase 2 verified)
- Manifest: `data/manifest.csv` — columns: index, split, class_name, class_idx, h5_idx (13394 rows)
- 10 Imagenette classes remapped to canonical 0-9 ordering

### UAT Results (7/7 pass)
1. Python environment — all packages import, MPS available: **PASS**
2. Imagenette dataset on disk — 9469 train / 3925 val across 10 class folders: **PASS**
3. ImagenetteDataset class loading — 9469 samples, `ds[0]` = (tensor[3,224,224], int): **PASS**
4. VGG16 feature extraction shapes — [9469,25088] train + [3925,25088] val float32: **PASS**
5. Soft label validity — sum≈1.0, no NaN/Inf, non-negative: **PASS**
6. HDF5 indexed access — `TensorStore.read("train", 42)` returns (feature[25088], soft_label[10], label): **PASS**
7. CSV manifest — 13394 rows, correct splits, all 10 classes: **PASS**

### Operational constraints
- `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` — removes 50% MPS memory cap for full-batch extraction
- `num_workers=0` — macOS MPS + Python 3.14 multiprocessing spawn incompatible
- `pin_memory=False` — auto-detected in get_dataloader on MPS
- `PYTORCH_ENABLE_MPS_FALLBACK=1` — required for cdist backward, segment_reduce on MPS

---

## Phase 2: Dense Baseline Benchmark

### Baselines measured (from `results/baseline_vgg16.json`)
- VGG16 FC top-1: **99.54%** | mAP: **99.97%** on Imagenette val
- FC params: **123,642,856** (25088×4096 + 4096×4096 + 4096×10 + biases)
- FLOPs counted on FC head only via `thop`

### Methodology
- FLOPs counting: `thop.profile(model, inputs=...)` — returns MACs; documented as "FLOPs" throughout (MACs × 2 convention, noted in JSON as `flops_note`)
- `count_flops(model, input_shape)` in `src/utils/metrics.py` wraps `thop.profile`
- Soft label quality thresholds (warn-only): accuracy match within 0.1%, mean entropy ≥ 0.01 nats, all classes ≥ 5% of val
- Metrics API: `compute_all_metrics(scores, labels)` — `scores` = softmax probabilities (float32, shape [N,10])
- mAP via `sklearn.average_precision_score` one-vs-rest, averaged across 10 classes
- `VGGExtractor` reused for Phase 2 eval (no duplicate VGG16 loading)

---

## Phase 3: Core Architecture Decisions

### Rejected approaches (do NOT re-propose)
- **N_in adapters**: Linear(25088→N) before SGNNET. Rejected — spatial encoding preserves spatial structure that adapter destroys. Adapter adds params without benefit.
- **Input neuron recurrence**: Input neurons participating in K-iteration loop. Rejected — seeding-only is sufficient; recurrence introduces instability.
- **Multiple output timing strategies**: Output neurons active across all K iterations. Rejected — one-shot readout at final K_iter step is cleanest (no readout leak).
- **Output neurons in C_hh routing**: Output neurons as sources in hidden dynamics. Rejected — pure sinks only; participating causes readout leak.
- **Soft gate (fully differentiable)**: Gaussian-only routing without hard gate. Deferred; hard gate + load balance is the spec.
- **K-means init for W_pos**: K-means on seeded activations. Deferred; random uniform used for Phase 3.

### Confirmed design decisions
- **Spatial encoding D=4**: `[value, h_norm, w_norm, channel_norm]` — value added per-sample in forward; spatial coords from `compute_spatial_encoding(N_in=25088)` returning [25088, 3]
- **Coulomb repulsion init for W_pos**: safety_valve_loss at init for position diversity
- **Three-matrix C split**: C_input [N_in, N_hidden] seeding | C_hh [N_hidden, N_hidden] iterations | C_ho [N_hidden, N_out] readout at step K only
- **Load balance**: per-neuron selection frequency via `(contribution, gate)` tuple from dynamic_connectivity; gate enables selection_count in trainer
- **Active param counting**: only mask==1 entries count (649K active vs 6.5M dense in Phase 3 reference config)
- **KL divergence** (not MSE) for distillation task loss
- **Sparsity test threshold**: 0.88 for small N_hidden (guaranteed-connectivity row fix raises effective sparsity)
- **C_ho sparsity threshold**: 0.85 (small 256×10 matrix, guaranteed-connectivity effect)

---

## Phase 4: Wave/Phasor Architecture

### Core decisions (load-bearing)
- **Binary immutable C masks (D-05)**: C_hh and C_ho registered as buffers (not nn.Parameter), values 0/1 only, fixed at init, no gradients through C. Isolates dynamic wave routing contribution.
- **Phasor activations Z ∈ ℂ^D (D-06)**: Stored as `(Z_re, Z_im)` each `[batch, N_hidden, D=4]`. Amplitude-only routing broken: LayerNorm output near 0 while W_pos ∈ [0,1]^D → cdist(A_hidden, W_pos) ≈ 1.0 >> r*≈0.125, killing all proximity gates. Phasor decouples routing (always-positive Gaussian amplitude) from interference (complex phase rotation).
- **Wavelength derivation (D-08)**: λ = r*/2 where r* is median inter-neuron distance. One oscillation per active zone [r*/2, r*]; single inhibitory ring at d=3r*/4. All constants derived from N and D via r*; no free wavelength hyperparameter.
- **Masked normalization (D-07)**: Activation criterion |Z_j| > ε; neurons below threshold contribute zero to mean/var statistics. Phasor magnitude: `||Z_j|| = sqrt(sum_d(Z_re_jd² + Z_im_jd²))`.
- **FP16 on MPS**: `torch.autocast('mps', dtype=torch.float16)` + `torch.amp.GradScaler('mps')` (requires PyTorch ≥2.3). Do NOT use `torch.cuda.amp.*` (CUDA-only). If PyTorch <2.3: use autocast alone (GradScaler had MPS inf-detection bugs before 2.3).
- **W_phase (Exp 2 / Stage C only)**: ∈ ℝ^D per neuron, shape [N_hidden + N_out, D=4]; separate lr_Wphase from GA search; separate from W_pos.
- **Phasor amplitude-only routing broken**: Attempted Z ∈ ℂ^D amplitude routing failed — phase info must propagate.

### Stage A/B/C configurations (GA-selected best configs)
| Stage | Config | top-1 | Notes |
|---|---|---|---|
| Stage A (static binary C) | K=2, N_hidden=256, lr=0.008, λ_safety=0.52 | 10.27% | Binary C alone insufficient; loss plateaus ~3.22 after epoch 8 |
| Stage B (spatial phase Exp1) | K=2, N_hidden=256, lr=0.0024 | 11.97% | Proximity routing adds marginal benefit (+1.7pp) |
| Stage C (spatial + W_phase Exp2) | K=2, N_hidden=256, lr=0.01, lr_Wphase=0.01 | 10.52% | W_phase trained (norm=13.36) but did not improve accuracy |

### GA fitness function (Phase 4)
- `score = −final_loss × (params_min / model_params)^0.2`
- `params_min` = smallest config count (N_hidden=64, K=1), computed once before GA run
- β=0.2: 4× larger model needs ~24% lower loss to score equal
- NaN/inf → score = −1e6 (disqualified immediately)
- final_loss = mean loss over last 3 epochs on 15% partial data, 15 epochs/eval
- GA config: population=20, generations=10, top_k=5
- Search space: K ∈ {1,2,3,4}, N_hidden ∈ {64,128,256}, D FIXED=4

### Phase 4 Conclusion
Binary C mask ceiling identified at ~3.2 KL plateau. Static wiring hits expressiveness wall;
dynamic routing required → Phase 5 redesign with learned C values + Fourier D encoding.

---

## Phase 5 Early Dead Mechanisms (killed before step29 baseline)

Already in [[architecture_dead_ends]]. Do not re-test without new evidence:
- Signed coupling variants (Z @ Z^T): cos-sim on S^63 ≈ noise; 5 experiments confirm ≤32%
- D=128 Fourier encoding: routing collapses, all configs ~10%
- Cross-dimensional W_mix (D×D matrices): neutral at D=16, hurts D=64 (−15pp)
- Phase-queried D×D matrix bank: same interference pattern (−5pp)
- MoD adaptive K_iter: 19-20%; all K_iter steps necessary
- Oja's rule routing update: −32pp; PCA compression destroys diversity
- Soft beam routing: gradients through hard selection not the bottleneck

---

## Phase 5 Infrastructure (from 05-CONTEXT.md)

### Mac Studio protocol (current as of early Phase 5)
- SSH alias: `mac-studio`; tmux binary at `/opt/homebrew/bin/tmux` (NOT in default PATH)
- Concurrency cap: 2 experiments per machine; launch only when count ≤ 1 AND RAM ≥ 50GB free+inactive
- RAM check: `vm_stat | grep -E 'free|inactive'` → (Pages free + Pages inactive) × 16384 / 1073741824
- 256GB RAM total — tight with 2× VGG16 + 2× dataset in memory (~2-3GB per N=512 process)
- n_groups: always `max(8, N//8)` — N//32 caused −4.12pp regression (step86)

### Early Phase 5 accuracy timeline
| Step | Config | top-1 |
|---|---|---|
| step22 | D=64 N=1024 K_iter=8 | 56.28% |
| step29A | + AntiHebb α=0.5 | 70.14% |
| step29C | + AntiHebb α=0.7 | 75.24% |

*See `learnings/EXPERIMENT_QUEUE.md` and `learnings/INDEX.md` for full Phase 5 history.*
