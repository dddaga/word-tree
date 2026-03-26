# Phase 4: SGNNET Wave Architecture & Experiments - Context

**Gathered:** 2026-03-26
**Status:** Ready for planning

<domain>
## Phase Boundary

Refactor SGNNET to phasor activations (Z ∈ ℂ^D), implement wave infrastructure with binary/immutable C matrices, then run three sequential stages: Stage A (static connectivity baseline), Stage B (spatial path-length phase — Exp 1), Stage C (spatial phase + learned W_phase operator — Exp 2). Each stage trained with GA hyperparameter search on partial data followed by full training. Final evaluation compares all stages side-by-side in wave_comparison.md.

This phase is experiments-and-training — no new data loading or preprocessing. All architecture changes live in `src/sgnnet/model_wave.py` and `src/training/trainer.py` (new).

</domain>

<decisions>
## Implementation Decisions

### N_in Strategy
- **D-01:** Keep N_in=25088 direct — no adapter. C_input_mask is [25088, N_hidden], binary immutable. Seeding: `Z_re = C_input_mask @ A_input`, `Z_im = zeros`. Consistent with Phase 3 design. Since C is binary (no learned values), C_input carries zero learnable parameters in Phase 4.

### Geometric Dimensionality
- **D-02:** D=4 locked for all Phase 4 experiments. The GA search sweeps only K ∈ {1,2,3,4} and N_hidden ∈ {64,128,256}. D is NOT swept. Phase 3's spatial encoding [feature_val, h_norm, w_norm, channel_norm] applies as-is to A_input [batch, N_in, D=4].

### GA Fitness Function
- **D-03:** Fitness = efficiency ratio scoring:
  ```
  score = −final_loss × (params_min / model_params)^0.2
  ```
  Where `params_min` = parameter count of the smallest config in the current sweep (e.g., N_hidden=64, K=1). `model_params` = total trainable params for the candidate config. Penalizes larger models unless their loss improvement justifies the complexity. `final_loss` = mean loss over last 3 epochs of partial-data training (15% data, 15 epochs).

  **Disqualification:** If loss is NaN or inf at any point during the eval run → score = −1e6.

  **NaN handling:** Disqualify immediately; do not average NaN into the score.

### Code Reuse from Phase 3
- **D-04:** `model_wave.py` imports shared helpers from existing Phase 3 modules:
  - `_make_sparse_c` (or a binary variant) from `src/sgnnet/model.py` — for constructing immutable binary masks
  - `compute_spatial_encoding` from `src/sgnnet/encoding.py` — unchanged
  - `personal_volume_radius` from `src/sgnnet/geometry.py` — unchanged
  Phase 3 files (`model.py`, `geometry.py`, `encoding.py`) are NOT modified. `model_wave.py` is additive only.

### Architecture (locked from prior session)
- **D-05:** C matrices binary/immutable: registered as buffers (not nn.Parameter), values 0/1 only. Pattern fixed at init; no gradients flow through C.
- **D-06:** Activation is phasor Z ∈ ℂ^D stored as `(Z_re, Z_im)` each `[batch, N_hidden, D=4]`.
- **D-07:** Normalization: masked — only neurons with |Z_j| > ε participate in mean/var. Implemented in `src/sgnnet/norm_masked.py` (new file).
- **D-08:** λ = r*/2 (path-length phase wavelength). One full oscillation in active zone [r*/2, r*], single inhibitory ring at d=3r*/4. λ derived from N and D via r*; no free hyperparameter.
- **D-09:** FP16 training: `torch.autocast('mps', dtype=torch.float16)` + `torch.amp.GradScaler('mps')`. Requires PyTorch ≥2.3. If <2.3, use autocast alone (skip GradScaler — MPS scaler had inf-detection bugs before 2.3). Do NOT use `torch.cuda.amp.*` — that is CUDA-only.
- **D-10:** W_phase (Exp 2 only) ∈ ℝ^D per neuron, shape [N_hidden + N_out, D=4]. Separate lr_Wphase from GA search; separate from W_pos.

### Training (GA search)
- **D-11:** GA config: population=20, generations=10, top_k=5, partial_data_fraction=0.15, epochs_per_eval=15. These match ROADMAP specs.
- **D-12:** GA search space (Stage A and Stage B share same space; Stage C adds lr_Wphase):

  | Hyperparameter | Type | Range |
  |---|---|---|
  | K | discrete | {1, 2, 3, 4} |
  | N_hidden | discrete | {64, 128, 256} |
  | D | **FIXED** | 4 (not swept) |
  | lr_Wpos | continuous (log) | [1e-5, 1e-2] |
  | λ_safety | continuous | [0.0, 1.0] |
  | batch_size | discrete | {64, 128, 256} |

### Claude's Discretion
- Binary C mask initialization strategy (e.g., random sparsity vs. spatially-structured for C_input)
- ε threshold for masked normalization (|Z_j| > ε)
- Whether `_make_sparse_c` in model.py is refactored to accept a `binary=True` flag or if model_wave.py defines its own `_make_binary_c`
- Exact mutation/crossover strategy for continuous GA hyperparameters
- Whether Stage B and C reuse Stage A's best K/N_hidden as starting population seed

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Architecture Spec
- `sparse_geometric_network_report.md` — Original SGNNET spec. Phase 4 diverges significantly: phasor activations, binary C, wave routing. Report is background context only; decisions in this file and ROADMAP.md take precedence.

### Phase 4 Requirements & Plans
- `.planning/ROADMAP.md` §Phase 4 — Full plan breakdown (Plans 4.1–4.6), architecture diff table, forward pass formulas for Stage A/B/C, FP16 notes.
- `.planning/REQUIREMENTS.md` §Training — TRAIN-01 through TRAIN-06 acceptance criteria.

### Project Constraints
- `.planning/PROJECT.md` §Constraints — File size limit (250 lines/file), top-down code style, MPS hardware.

### Prior Phase Artifacts (Reusable)
- `src/sgnnet/model.py` — SGNNET class, `_make_sparse_c` helper (D-04: import don't copy).
- `src/sgnnet/geometry.py` — `personal_volume_radius`, `dynamic_connectivity_hh`, `dynamic_connectivity_ho`.
- `src/sgnnet/encoding.py` — `compute_spatial_encoding` (produces [N_in, 3] spatial coords; same as Phase 3).
- `src/sgnnet/losses.py` — `safety_valve_loss`, `load_balance_loss`, `total_loss` — review for phasor-compatible versions.
- `src/utils/metrics.py` — `compute_all_metrics`, `count_params` — used for benchmarking all stages.
- `results/baseline_vgg16.json` — VGG16 baseline (top1=0.9954, mAP=0.9997, fc_params=123,642,856) — reference denominator for all param% calculations.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `src/sgnnet/model.py`: `_make_sparse_c(rows, cols, sparsity, zero_diag)` — create values+mask. Phase 4 needs binary variant (no learned values, just the mask as buffer). Can either add `binary=True` flag or define `_make_binary_c` in model_wave.py.
- `src/sgnnet/encoding.py`: `compute_spatial_encoding(N_in=25088)` returns `[25088, 3]` spatial coords. In Phase 4, A_input is assembled the same way as Phase 3: `[feature_val, spatial_coords]` → D=4.
- `src/sgnnet/geometry.py`: `personal_volume_radius(N, D)` is pure math — import directly. Proximity routing formulas in Phase 4 extend this.
- `src/sgnnet/losses.py`: `safety_valve_loss` and `load_balance_loss` may need phasor-aware versions (|Z| instead of A for detecting neuron activation).
- No `src/training/` directory — trainer.py is entirely new work.

### Established Patterns
- **MPS device:** `device = "mps" if torch.backends.mps.is_available() else "cpu"`.
- **num_workers=0:** Required for MPS on macOS.
- **PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0:** Required for large tensors on MPS.
- **File size ≤250 lines:** model_wave.py will likely need splitting (model_wave_stage_a.py and model_wave_stage_bc.py, or wave_primitives.py + model_wave.py).
- **Top-down code style:** High-level class/function structure first, fill in details progressively.

### Integration Points
- `src/sgnnet/norm_masked.py` — new file for `masked_normalize`. Both Stage A (real) and Stage B/C (phasor) use it.
- `src/training/trainer.py` — new shared training loop consumed by `scripts/train_exp1.py` and `scripts/train_exp2.py`.
- `src/training/ga_search.py` — new GA harness, uses efficiency ratio fitness (D-03).
- Output: `results/stageA_*.json`, `results/exp1_*.json`, `results/exp2_*.json`, `results/wave_comparison.*`.

</code_context>

<specifics>
## Specific Ideas

### GA Fitness — params_min Calculation
`params_min` should be computed once before the GA run starts by evaluating the smallest config in the search space (N_hidden=64, K=1). Store it as a constant for the run so the ratio is stable across generations.

### Efficiency Ratio β=0.2 Rationale
With β=0.2, going from N_hidden=64 (params_min) to N_hidden=256 (4× more params on hidden neurons) applies a penalty factor of (1/4)^0.2 ≈ 0.76. So a 256-neuron model needs to achieve ~24% lower loss than a 64-neuron model to score the same. Reasonable for this scale.

### Masked Normalization — Phasor Magnitude
For Stage B/C, the active neuron condition is `|Z_j| = sqrt(Z_re_j² + Z_im_j²) > ε`. This is computed over the D dimensions: `magnitude_j = ||Z_j|| = sqrt(sum_d(Z_re_jd² + Z_im_jd²))`. Only neurons with magnitude > ε participate in norm statistics.

</specifics>

<deferred>
## Deferred Ideas

- **D sweeping in GA search**: D=4 is locked for Phase 4. Sweeping D could be a Phase 4.5 or v2 experiment.
- **Pareto-front GA selection**: User considered but chose efficiency ratio. Could be revisited if efficiency ratio β needs per-experiment tuning.
- **Time-per-epoch in fitness**: Discussed but not added — adds implementation complexity. If training times diverge significantly between configs, revisit.
- **Stage B/C seeding from Stage A best config**: Mentioned (Claude's discretion). If Stage A results in a clear winner (K, N_hidden), starting Stage B/C GA with that config as a population seed is reasonable.
- **Adaptive K (stop when activations converge)**: Deferred from Phase 3, still deferred.
- **Soft gate (differentiable routing)**: Deferred from Phase 3, still deferred.

</deferred>

---

*Phase: 04-sgnnet-wave-architecture-experiments*
*Context gathered: 2026-03-26*
