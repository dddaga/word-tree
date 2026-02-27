# V3 Dynamic Phasor GNN Experiment -- Implementation Details

## 1. Overview

The V3 Dynamic Phasor experiment applies a phasor-based graph neural network to the UCI Wine Quality Red dataset (11 features, 6 classes). It addresses five engineering issues identified in the GNN Analysis Report:

- **Issue 1**: Complete complex activation (real + imaginary channels)
- **Issue 4**: Signal normalization (magnitude clamping + RMSNorm)
- **Issue 5**: Auxiliary loss for radiation diversity (entropy regularization)
- **Issue 7**: Load balancing for radiation targets (MoE-style capacity limits)
- **Issue 9**: Uncertainty estimation (MC Dropout + null activations + temperature calibration)

**Architecture**: 200 nodes total -- 22 input, 172 intermediate, 6 output. Each node stores a discrete phase index (`[0, 511]`) and magnitude index (`[0, 1023]`) per vector dimension (`vector_dim=5`). Signals are computed via lookup tables: `signal = cos(phase) * exp(sin(mag))`.

**Training**: Discrete gradient descent -- continuous gradients are quantized to `{-1, 0, +1}` parameter updates with threshold-based accumulation. Forward propagation runs up to 30 timesteps with decay factor 0.6.

---

## 2. File Map & Dependency Graph

### 2.1 File Table

| # | File | Purpose |
|---|------|---------|
| 1 | `config.yaml` | Experiment configuration (architecture, training, all 5 issue params) |
| 2 | `main_v3.py` | CLI entry point: load config, optional ablation, run experiment loop |
| 3 | `run_ablation.py` | 7-way ablation study runner (baseline + 5 singles + all) |
| 4 | `__init__.py` | Package init |
| 5 | `model/v3_train_context.py` | Central orchestrator: forward, backward, training loop, evaluation |
| 6 | `model/v3_phase_cell.py` | Phase cell wrapping ModularPhaseCell + complex activation + normalization |
| 7 | `model/v3_forward_engine.py` | Forward engine wrapping VectorizedForwardEngine + V3 post-processing |
| 8 | `model/wine_input_adapter.py` | Wine dataset loader + MLP projection + phase/mag quantization |
| 9 | `components/complex_signal.py` | Issue 1: Full complex phasor computation + dual-channel gradients |
| 10 | `components/signal_normalization.py` | Issue 4: Magnitude index clamping + signal RMSNorm |
| 11 | `components/radiation_diversity.py` | Issue 5: Entropy-based radiation diversity loss |
| 12 | `components/load_balancer.py` | Issue 7: MoE-style load balancing + temperature annealing |
| 13 | `components/uncertainty.py` | Issue 9: MC Dropout + learnable null activation + temperature calibration |
| 14 | `metrics/complexity_tracker.py` | Space/time complexity measurement |
| 15 | `metrics/experiment_logger.py` | Loss/accuracy logging + matplotlib plot generation |
| 16 | `data/winequality-red.csv` | UCI Wine Quality Red dataset (1599 samples, semicolon-separated) |

### 2.2 Import Dependency Tree

```
main_v3.py
├── model/v3_train_context.py
│   ├── core/high_res_tables.py          (HighResolutionLookupTables)
│   ├── core/node_store.py               (NodeStore)
│   ├── core/graph.py                    (build_static_graph)
│   ├── core/modular_forward_engine.py   (VectorizedForwardEngine)
│   ├── core/activation_table.py         (VectorizedActivationTable)
│   ├── modules/class_encoding.py        (generate_fixed_class_encodings)
│   ├── modules/classification_loss.py   (ClassificationLoss)
│   ├── components/complex_signal.py     (ComplexSignalComputer)
│   ├── components/signal_normalization.py (PhasorNormalization)
│   ├── components/radiation_diversity.py (RadiationDiversityTracker)
│   ├── components/load_balancer.py      (RadiationLoadBalancer)
│   ├── components/uncertainty.py        (MCDropoutUncertainty)
│   ├── model/wine_input_adapter.py      (WineInputAdapter)
│   │   └── core/high_res_tables.py
│   ├── model/v3_phase_cell.py           (V3PhaseCell)
│   │   ├── core/modular_cell.py         (ModularPhaseCell)
│   │   ├── components/complex_signal.py
│   │   └── components/signal_normalization.py
│   └── model/v3_forward_engine.py       (V3ForwardEngine)
│       ├── core/modular_forward_engine.py
│       ├── core/node_store.py
│       ├── core/radiation.py
│       ├── components/signal_normalization.py
│       ├── components/radiation_diversity.py
│       └── components/load_balancer.py
├── metrics/complexity_tracker.py        (ComplexityTracker)
└── metrics/experiment_logger.py         (ExperimentLogger)

run_ablation.py
└── main_v3.py (load_config, apply_ablation, run_experiment)
```

---

## 3. Per-File Documentation

### 3.1 `config.yaml`

**Purpose**: YAML configuration file defining all hyperparameters, architecture choices, and feature flags for the V3 experiment. Loaded by `main_v3.py:load_config()`.

**Sections**:
- `dataset`: Wine Quality Red, test_size=0.30, val_fraction=0.50, seed=42
- `architecture`: 200 nodes (22 input, 172 intermediate, 6 output), vector_dim=5, cardinality=4
- `resolution`: phase_bins=512, mag_bins=1024
- `complex_activation`: enabled=true, gamma=1.0
- `normalization`: enabled=true, clamp range [0.1, 0.9]
- `radiation_diversity`: enabled=true, alpha=0.01
- `load_balance`: enabled=true, beta=0.001, capacity_limit=5, temp_init=1.0, temp_min=0.1, anneal_rate=0.995
- `uncertainty`: enabled=true, dropout_rate=0.1, num_mc_samples=10
- `training`: epochs=100, batch_size=64, phase_lr=0.015, mag_lr=0.012
- `forward_pass`: max_timesteps=30, decay=0.6, top_k_neighbors=4

---

### 3.2 `main_v3.py`

**Purpose**: CLI entry point that loads configuration, applies optional ablation settings, and orchestrates the full experiment lifecycle (train → evaluate → calibrate → measure complexity → log/plot).

**Imports**: `V3TrainContext`, `ComplexityTracker`, `ExperimentLogger`

**Functions**:

#### `load_config(config_path, overrides) -> dict` (line 28)
Loads YAML config and applies CLI overrides for `epochs`, `gamma`, `device`.

#### `apply_ablation(config, ablation) -> dict` (line 44)
Disables all 5 issue fixes, then selectively re-enables based on ablation name. Choices: `baseline`, `complex_only`, `normalization_only`, `diversity_only`, `balance_only`, `uncertainty_only`, `all`.

#### `run_experiment(config, experiment_name) -> dict` (line 77)
Main experiment loop:
1. Initializes `ExperimentLogger` and `ComplexityTracker`
2. Builds `V3TrainContext(config)`
3. Trains for N epochs, logging every epoch
4. Evaluates on test set (accuracy, F1, precision, recall)
5. If uncertainty enabled: calibrates temperature on val set
6. Measures inference latency (200 forward passes)
7. Generates complexity report, saves metrics, plots curves

**Known Issues**:
- **BUG 5** (lines 125-130): Duplicate reset code -- `main_v3.py` calls `diversity_tracker.reset()` and `load_balancer.anneal_temperature()` / `reset_epoch()` again after `ctx.train_epoch()` already manages these internally via `train()`. **Severity**: Low. **Impact**: Maintenance hazard; double-reset is harmless but confusing.

---

### 3.3 `run_ablation.py`

**Purpose**: Runs all 7 ablation configurations sequentially and produces a comparison table of test metrics and parameter counts. Results saved as `ablation_comparison.json`.

**Imports**: `load_config`, `apply_ablation`, `run_experiment` from `main_v3.py`

**Configurations** (line 27-35):
```
baseline           → All fixes disabled
complex_only       → Issue 1 only
normalization_only → Issue 4 only
diversity_only     → Issue 5 only
balance_only       → Issue 7 only
uncertainty_only   → Issue 9 only
all                → All fixes enabled
```

**Functions**:

#### `run_ablation_study(base_config_path, epochs, device) -> dict` (line 38)
Iterates through `ABLATION_CONFIGS`, runs each experiment, collects results, prints comparison table (accuracy, F1, precision, recall, params), saves JSON.

**Known Issues**: None specific to this file; inherits all issues from `run_experiment`.

---

### 3.4 `model/v3_train_context.py`

**Purpose**: Central orchestrator for the V3 experiment. Manages graph construction, node storage, input encoding, forward/backward propagation, discrete parameter updates, training loops, and evaluation metrics.

**Imports**: `HighResolutionLookupTables`, `NodeStore`, `build_static_graph`, `VectorizedForwardEngine`, `VectorizedActivationTable`, `generate_fixed_class_encodings`, `ClassificationLoss`, all 5 V3 components, `WineInputAdapter`, `V3PhaseCell`, `V3ForwardEngine`

**Class**: `V3TrainContext`

#### Constructor `__init__(config)` (line 36)
Reads feature flags from config, calls 6 setup methods, initializes loss tracking lists.

#### `_setup_core()` (line 70)
Creates `HighResolutionLookupTables(512, 1024)`, `V3PhaseCell(vector_dim=5, gamma=1.0, ...)`.

#### `_setup_graph()` (line 97)
Calls `build_static_graph(200, 22, 6, 5, 512, 1024, cardinality=4)`, creates `NodeStore`, defines input nodes `n0-n21` and output nodes `n194-n199`.

#### `_setup_input()` (line 116)
Creates `WineInputAdapter(input_dim=11, ...)` which loads Wine Quality data, builds MLP, splits into train/val/test.

#### `_setup_output()` (line 131)
Generates 10 fixed class encodings (random phase/mag pairs), creates `ClassificationLoss(num_classes=6)`, reads learning rates (phase_lr=0.015, mag_lr=0.012).

#### `_setup_v3_components()` (line 149)
Creates `PhasorNormalization(clamp=[102,921])`, `RadiationDiversityTracker(200, alpha=0.01)`, `RadiationLoadBalancer(200, beta=0.001, capacity=5, temp=1.0)`, `MCDropoutUncertainty(dropout=0.1, mc_samples=10)`.

#### `_setup_forward_engine()` (line 185)
Creates `VectorizedForwardEngine(max_timesteps=30, decay=0.6, top_k=4)` then wraps it in `V3ForwardEngine`.

#### `forward_pass(input_context) -> dict` (line 219)
Runs `V3ForwardEngine.forward_pass()`, then for each of 6 output nodes: if active, gets signal via `lookup_tables.get_signal_vector(phase, mag)`; if inactive and uncertainty enabled, uses learnable null activation; else zero signal. Returns `{node_id: signal_tensor[5]}`.

#### `compute_logits(output_signals) -> Tensor` (line 245)
Delegates to `ClassificationLoss.compute_logits_from_signals()` which computes cosine similarity between each output signal and each class encoding, then averages across output nodes. Returns `[1, num_classes]`.

#### `backward_pass(logits, target_label, output_signals) -> (gradients, loss)` (line 252)
1. Computes cross-entropy loss
2. Computes logit gradient: `softmax(logits) - one_hot(target)`
3. Loops over output nodes only (n194-n199)
4. For each output node `i`: uses `class_id = i` to select a single class encoding
5. Computes upstream = `logit_grad[class_id] * (class_signal / (norm_s * norm_c))`
6. Computes phase/mag gradients via `lookup_tables.compute_signal_gradients()`
7. Returns `{node_id: (phase_grad, mag_grad)}` -- **only 6 entries**

**Known Issues**:
- **BUG 1** (lines 270-292, **Critical**): Only loops over `self.output_nodes` (6 nodes). The 172 intermediate nodes (n22-n193) receive ZERO gradient updates. In the original `ModularTrainContext` (lines 649-699), ALL active nodes receive gradients via cosine alignment credit assignment. This means 172/200 nodes are permanently frozen at random initialization.
- **BUG 2** (lines 279-287, **High**): Uses `class_id = i` (node index in output list) rather than summing gradient contributions across all classes. Node n194 only gets gradient from class 0, n195 from class 1, etc. Correct formula: `d(loss)/d(signal_j) = sum_c logit_grad[c] * (1/6) * d(cosine_sim(signal_j, encoding_c))/d(signal_j)`.
- **BUG 8** (**High**): The `WineInputAdapter` MLP is never trained -- no loss backpropagation through the MLP parameters. `self.input_adapter.parameters()` are counted but never have `.grad` computed or optimizers applied. The MLP remains at random initialization, so all wine samples produce random phase/mag indices regardless of input features.

#### `apply_updates(node_gradients)` (line 296)
For each node, calls `quantize_gradients_to_discrete_updates()` (threshold=0.01) then `apply_discrete_updates()` (phase wraps modular, mag clamps [0, M-1]).

#### `train_single_sample(sample_idx, dataset) -> (loss, accuracy)` (line 314)
Gets input context → applies MC dropout → forward → backward → apply updates → compute aux losses → return total loss and accuracy.

#### `train_epoch() -> (avg_loss, div_loss, bal_loss, train_acc)` (line 350)
Samples `batch_size` (64) random training indices, trains each, returns averages.

#### `train(num_epochs) -> dict` (line 376)
Full loop: train_epoch → validate every 5 epochs → anneal temperature → reset diversity/balance counters → print progress. Returns all loss/accuracy histories.

#### `evaluate(dataset, max_samples) -> dict` (line 436)
Loops over all samples in split, runs forward + compute_logits, collects predictions, computes accuracy, weighted F1, precision, recall, confusion matrix via sklearn.

#### `count_parameters() -> int` (line 503)
Sums: node_store params (200 nodes × 5 dims × 2 tables × 1 param = 2,000) + input_adapter MLP (~46,940) + uncertainty (11). Total: ~49,151.

---

### 3.5 `model/v3_phase_cell.py`

**Purpose**: Wraps `ModularPhaseCell` with complex activation (Issue 1) and signal normalization (Issue 4). Drop-in replacement matching the same `forward()` return signature: `(phase_out, mag_out, signal, strength, grad_phase, grad_mag)`.

**Imports**: `HighResolutionLookupTables`, `ModularPhaseCell`, `ComplexSignalComputer`, `PhasorNormalization`

**Class**: `V3PhaseCell(nn.Module)`

#### Constructor (line 21)
Creates `ModularPhaseCell(vector_dim, lookup_tables)`, `ComplexSignalComputer(lookup_tables, gamma)`, `PhasorNormalization(mag_bins)`.

#### `forward(ctx_phase, ctx_mag, self_phase, self_mag) -> tuple` (line 46)
1. Phase transfer: `phase_out = (ctx + self) % phase_bins` (line 60)
2. Magnitude transfer: `mag_out = (ctx + self) % mag_bins` (line 61)
3. If normalization: clamp mag_out to [102, 921] (line 65)
4. If complex enabled:
   - Compute `(real, imag, envelope)` via `ComplexSignalComputer` (line 69)
   - Signal = real part (line 72)
   - Apply RMSNorm if normalization enabled (line 76)
   - Strength = sum(envelope) (line 79)
   - **Compute gradients with hardcoded upstream** (lines 82-87)
5. Else: fallback to base lookup table forward

**Known Issues**:
- **BUG 3** (lines 83-84, **High**): Hardcodes `upstream_real = torch.ones_like(real)` and `upstream_imag = torch.zeros_like(imag)`. These gradients are pre-computed during the forward pass and are completely disconnected from the actual loss. They represent "what if we wanted all real parts to increase uniformly" rather than "what the loss tells us to do". The gradients are stored but never used meaningfully because BUG 1 prevents intermediate nodes from receiving any updates anyway.

#### `compute_routing_strength(ctx_phase, self_phase) -> Tensor` (line 97)
Delegates to `base_cell.compute_routing_strength()`.

---

### 3.6 `model/v3_forward_engine.py`

**Purpose**: Wraps `VectorizedForwardEngine` with V3 post-processing: output magnitude clamping, radiation diversity recording, load balancer step management, and timing.

**Imports**: `VectorizedForwardEngine`, `NodeStore`, `get_radiation_neighbors`, `clear_radiation_cache`, `PhasorNormalization`, `RadiationDiversityTracker`, `RadiationLoadBalancer`

**Class**: `V3ForwardEngine`

#### Constructor (line 23)
Stores base engine, node_store, normalizer, diversity_tracker, load_balancer, feature flags. Initializes timing lists.

#### `forward_pass(input_context) -> ActivationTable` (line 48)
1. Optionally clear radiation cache if load balancer says so (line 60-61)
2. Run `engine.forward_pass_vectorized(input_context)` (line 64)
3. If normalization: clamp output magnitudes (line 67-68)
4. If diversity: record radiation targets (line 72)
5. If balance: call `load_balancer.step_iteration()` (line 76)
6. Record timing

#### `_clamp_output_magnitudes()` (line 84)
For each output node, reads mag from NodeStore, clamps to [102, 921], writes back if changed.

#### `_record_radiation_from_stats()` (line 95)
**Known Issues**:
- **BUG 4** (lines 95-102, **Medium**): Gets "active output nodes" from the engine (at most 6 node IDs like `["n194", "n195", ...]`) and records them as "radiation targets". But radiation targets should be the intermediate nodes that received radiation hits during propagation. The base engine doesn't expose individual radiation targets, so this is a proxy that records the wrong thing entirely. Result: diversity tracker always sees the same ~6 output nodes → entropy is frozen at a constant value (observed: 0.00662 every epoch).

---

### 3.7 `model/wine_input_adapter.py`

**Purpose**: Custom dataset adapter for UCI Wine Quality Red. Loads data, normalizes features via StandardScaler, performs stratified train/val/test split, and projects 11 wine features to discrete phase/magnitude indices for 22 input nodes via a learned MLP.

**Imports**: `torch.nn`, `pandas`, `sklearn.preprocessing.StandardScaler`, `sklearn.model_selection.train_test_split`, `HighResolutionLookupTables`

**Class**: `WineInputAdapter(nn.Module)`

#### Constructor (line 30)
- Builds MLP: `11 → Linear(128) → LayerNorm → ReLU → Dropout(0.1) → Linear(128) → LayerNorm → ReLU → Dropout(0.1) → Linear(220) → Tanh`
- Output dimension: `22 × 5 × 2 = 220` (num_input_nodes × vector_dim × 2 for phase+mag)
- Total MLP parameters: ~46,940
- Calls `_load_dataset()` for data loading and splitting

#### `_load_dataset(test_size, val_fraction, seed, data_path)` (line 82)
1. Loads CSV (semicolon-separated)
2. Remaps quality labels: {3→0, 4→1, 5→2, 6→3, 7→4, 8→5}
3. Stratified split: 70% train (1119), 15% val (240), 15% test (240)
4. StandardScaler fit on train only, transform val/test
5. Converts to torch tensors

#### `forward(x) -> Tensor` (line 141)
Runs `self.projection(x)` → output in `[-1, 1]` via final Tanh.

#### `quantize_to_phase_mag(projected) -> (phase_idx, mag_idx)` (line 147)
- Reshapes `[220]` → `[22, 5, 2]`
- Phase channel: `[-1,1] → [0, 2π) → [0, 511]` via floor quantization
- Mag channel: `[-1,1] → [-3, 3] → [0, 1023]` via linear scaling + floor

#### `get_input_context(sample_idx, input_node_ids, dataset) -> (dict, label)` (line 177)
Selects sample from appropriate split, runs forward + quantize, builds `{node_id: (phase[5], mag[5])}` dict.

**Known Issues**:
- **BUG 8** (**High**): The MLP has ~46,940 parameters but is NEVER trained. `V3TrainContext` uses discrete gradient descent for the NodeStore but never computes gradients for the MLP (no `loss.backward()` through the projection network, no optimizer for `self.input_adapter.parameters()`). Since MLP weights are random (PyTorch default init), ALL wine samples are projected to essentially random phase/mag indices. The GNN receives random input encodings regardless of input features, making it impossible to learn meaningful representations. This accounts for 95.5% of total parameters being completely wasted.

---

### 3.8 `components/complex_signal.py`

**Purpose**: Implements Issue 1 -- complete complex phasor activation with both real and imaginary channels. Extends the original single-channel `signal = cos(phi) * exp(sin(m))` to `signal = exp(gamma*sin(m)) * [cos(phi) + i*sin(phi)]`.

**Imports**: `HighResolutionLookupTables`

**Class**: `ComplexSignalComputer`

#### Constructor (line 19)
Stores `lookup_tables` reference and `gamma` scaling factor (default 1.0).

#### `get_complex_signal(phase_indices, mag_indices) -> (real, imag, envelope)` (line 23)
```
cos_vals = cos_table[phase_indices]           # cos(phi)
sin_vals = sin_table[phase_indices]           # sin(phi)
exp_sin  = mag_exp_sin_table[mag_indices]     # exp(sin(m))
envelope = exp_sin ^ gamma
real = envelope * cos_vals
imag = envelope * sin_vals
```
When `gamma=1.0`, real part exactly matches the original `get_signal_vector()`.

#### `compute_complex_strength(phase, mag) -> Tensor` (line 51)
Returns `sum(envelope)` -- always positive scalar.

#### `compute_complex_gradients(phase, mag, upstream_real, upstream_imag) -> (phase_grad, mag_grad)` (line 60)
Chain rule through both channels:
```
d(real)/d(phi) = -sin(phi) * envelope
d(imag)/d(phi) =  cos(phi) * envelope
d(real)/d(m)   = gamma * cos(m) * exp(gamma*sin(m)) * cos(phi)
d(imag)/d(m)   = gamma * cos(m) * exp(gamma*sin(m)) * sin(phi)

phase_grad = upstream_real * d(real)/d(phi) + upstream_imag * d(imag)/d(phi)
mag_grad   = upstream_real * d(real)/d(m)   + upstream_imag * d(imag)/d(m)
```
Scaled by `phase_grad_scale` and `mag_grad_scale` from lookup tables.

**Known Issues**: The math is correct. However, gradients are wasted because BUG 3 in `V3PhaseCell` passes `upstream_real=ones, upstream_imag=zeros` instead of actual loss-derived upstream gradients, and BUG 1 prevents intermediate nodes from receiving updates.

---

### 3.9 `components/signal_normalization.py`

**Purpose**: Implements Issue 4 -- two-pronged normalization: (a) discrete magnitude index clamping to prevent `exp(sin(m))` extremes, and (b) continuous signal RMSNorm.

**Class**: `PhasorNormalization`

#### Constructor (line 13)
Computes clamp bounds: `clamp_low = int(1024 * 0.1) = 102`, `clamp_high = int(1024 * 0.9) = 921`.

#### `clamp_magnitude_indices(mag_indices) -> Tensor` (line 25)
`torch.clamp(mag_indices, 102, 921)` -- prevents extreme `exp(sin(m))` values at bin boundaries.

#### `rmsnorm_signal(signal) -> Tensor` (line 29)
`signal / sqrt(mean(signal^2) + eps)` -- preserves direction, normalizes scale to RMS ≈ 1.0.

#### `normalize_strengths(strengths) -> Tensor` (line 37)
Optional z-score normalization: `(strengths - mean) / std`. Not used in current pipeline.

**Known Issues**: None. Implementation is correct and functional.

---

### 3.10 `components/radiation_diversity.py`

**Purpose**: Implements Issue 5 -- entropy-based auxiliary loss penalizing concentrated radiation target selection. Tracks which nodes receive radiation hits and computes `L_diversity = alpha * (1 - H_norm)` where H_norm is Shannon entropy normalized to [0, 1].

**Class**: `RadiationDiversityTracker`

#### Constructor (line 14)
`hit_counts = zeros(200)`, `alpha = 0.01`.

#### `record_radiation_targets(target_indices)` (line 20)
Increments hit count for each target. Accepts both string (`"n42"`) and int indices.

#### `compute_entropy_loss() -> Tensor` (line 29)
```
p = hit_counts / total_hits
H = -sum(p * log(p))               # Shannon entropy
H_max = log(200) = 5.298
loss = 0.01 * (1 - H/H_max)        # lower entropy = higher loss
```

#### `get_metrics() -> dict` (line 61)
Returns: entropy, normalized_entropy, gini_coefficient, top10_hit_fraction, unique_targets.

#### `reset()` (line 97)
Zeros hit_counts and total_hits at epoch boundary.

**Known Issues**: The implementation is mathematically correct, but it is fed wrong data due to BUG 4 in `v3_forward_engine.py`. Only output node IDs (~6) are ever recorded, not the actual radiation targets from the propagation engine. With only 6 unique targets out of 200 nodes, the normalized entropy is frozen near `H/H_max = log(6)/log(200) = 1.79/5.30 = 0.338`, giving a constant loss of `0.01 * (1 - 0.338) = 0.00662`.

---

### 3.11 `components/load_balancer.py`

**Purpose**: Implements Issue 7 -- MoE-style load balancing for radiation targets. Applies score penalties based on cumulative hits, enforces capacity limits, and anneals selection temperature.

**Class**: `RadiationLoadBalancer`

#### Constructor (line 14)
`epoch_hit_counts = zeros(200)`, `step_hit_counts = zeros(200)`, beta=0.001, capacity_limit=5, temperature=1.0, anneal_rate=0.995.

#### `adjust_scores_for_balance(scores, candidate_indices) -> Tensor` (line 38)
`adjusted_score[i] = score[i] - beta * log(1 + hit_count[i])` -- penalizes frequently-hit nodes.

#### `check_capacity(target_idx) -> bool` (line 57)
Returns True if `step_hit_counts[idx] < 5`.

#### `record_hit(target_idx)` (line 64)
Increments both `step_hit_counts` and `epoch_hit_counts`.

#### `compute_balance_loss() -> Tensor` (line 71)
`beta * Var(epoch_hit_counts)` if any hits recorded, else 0.0.

#### `anneal_temperature()` (line 78)
`temperature = max(0.1, temperature * 0.995)` -- called once per epoch.

#### `step_iteration()` (line 85)
Resets per-step counters, increments iteration for cache invalidation.

#### `reset_epoch()` (line 97)
Zeros both hit count tensors.

**Known Issues**:
- **BUG 7** (**High**): `record_hit()` is NEVER called from any code path. `adjust_scores_for_balance()` and `check_capacity()` are also never called. The load balancer's integration points (`record_hit`, `adjust_scores`, `check_capacity`) were designed to be called from within the forward engine's radiation selection logic, but the `V3ForwardEngine` wrapper only calls `step_iteration()` and never hooks into the actual radiation selection process. As a result: `epoch_hit_counts` is always all-zeros → `compute_balance_loss()` always returns 0.0 → the balance auxiliary loss is dead code.

---

### 3.12 `components/uncertainty.py`

**Purpose**: Implements Issue 9 -- uncertainty estimation via MC Dropout in discrete index space, learnable null activations for inactive output nodes, and temperature calibration via grid search.

**Class**: `MCDropoutUncertainty(nn.Module)`

#### Constructor (line 16)
- `null_phase`: Parameter, random `[vector_dim]` floats in [0, phase_bins)
- `null_mag`: Parameter, random `[vector_dim]` floats in [0, mag_bins)
- `temperature`: Parameter, initialized to 1.0
- Total parameters: 5 + 5 + 1 = 11

#### `apply_phase_dropout(phase_indices, mag_indices, training) -> tuple` (line 42)
```
mask = Bernoulli(1 - 0.1)      # keep_prob = 0.9
masked_phase = phase_indices * mask
masked_mag   = mag_indices * mask
```

**Known Issues**:
- **BUG 6** (lines 55-61, **Medium**): Multiplying indices by 0 sets them to index 0, not to "null". Index 0 maps to `cos(0) = 1.0` and `exp(sin(0)) = 1.0`, injecting a systematic bias signal. Correct approach: replace dropped dimensions with the learnable null activation (`self.null_phase`, `self.null_mag`) or with a random index, not with the deterministic value at bin 0. With 10% dropout, ~1 of every 10 dimensions across all 22 input nodes injects cos(0)=1.0 into the graph.

#### `estimate_uncertainty(forward_fn, input_context) -> dict` (line 64)
Runs `num_mc_samples` (10) stochastic forward passes with dropout enabled. Returns mean logits, variance, predictive entropy, and confidence.

#### `get_null_activation() -> (phase, mag)` (line 114)
Clamps and converts learnable float parameters to long tensors in valid range.

#### `calibrate_temperature(logits_list, labels_list) -> float` (line 120)
Grid searches T in [0.1, 5.0] (50 values) minimizing NLL of `softmax(logits/T)` on validation set. Writes optimal T to `self.temperature`.

---

### 3.13 `metrics/complexity_tracker.py`

**Purpose**: Measures space complexity (parameter counts, memory) and time complexity (epoch training time, per-sample time, inference latency) for the experiment.

**Class**: `ComplexityTracker`

#### `measure_space_complexity(modules) -> dict` (line 23)
Iterates named modules, sums `numel()` and `numel() * element_size()` for each. Returns breakdown and totals.

#### `record_epoch_time(seconds)` (line 52)
Appends to `epoch_train_times` list.

#### `measure_inference_latency(forward_fn, input_context, num_runs=200) -> float` (line 58)
5 warmup runs, then 200 timed runs. Returns median latency in microseconds.

#### `get_full_report(modules) -> dict` (line 97)
Combines space and time complexity into single report.

**Known Issues**: None.

---

### 3.14 `metrics/experiment_logger.py`

**Purpose**: Logs per-epoch losses, accuracies, and issue-specific metrics. Generates matplotlib plots (loss curves, accuracy curves, issue metric dashboards) and saves all metrics to JSON.

**Class**: `ExperimentLogger`

#### `log_epoch(epoch, primary_loss, diversity_loss, balance_loss, total_loss, train_acc, val_acc, issue_metrics)` (line 48)
Appends all values to internal lists.

#### `save_metrics() -> str` (line 74)
Dumps all data to `{experiment_name}_metrics.json`.

#### `plot_loss_curves() -> str` (line 100)
Two-panel plot: left = loss curves (total, primary, diversity, balance), right = accuracy curves (train, val).

#### `plot_issue_metrics() -> str` (line 139)
Four-panel plot: normalized entropy, hit count variance, temperature, unique targets.

#### `print_summary()` (line 185)
Formatted summary table of final metrics.

**Known Issues**: None.

---

## 4. Configuration Reference

```yaml
# ---- Dataset ----
dataset:
  name: "wine_quality_red"            # UCI Wine Quality Red
  test_size: 0.30                      # 30% held out for val+test
  val_fraction: 0.50                   # 50% of held-out → val, 50% → test
  seed: 42                             # Random seed for reproducibility

# ---- Architecture ----
architecture:
  total_nodes: 200                     # Total graph nodes
  input_nodes: 22                      # Nodes receiving input features (2× features)
  output_nodes: 6                      # One per wine quality class (3-8)
  intermediate_nodes: 172              # Processing nodes between input and output
  vector_dim: 5                        # Phase/magnitude vector dimension per node
  cardinality: 4                       # Number of connections per node in graph

# ---- Resolution ----
resolution:
  phase_bins: 512                      # N: phase discretization [0, 2π) → [0, 511]
  mag_bins: 1024                       # M: magnitude discretization → [0, 1023]

# ---- Issue 1: Complex Activation ----
complex_activation:
  enabled: true                        # Use full exp(iφ + γsin(m)) vs cos(φ)·exp(sin(m))
  gamma: 1.0                           # Magnitude-to-phase coupling strength

# ---- Issue 4: Normalization ----
normalization:
  enabled: true
  magnitude_clamping:
    range: [0.1, 0.9]                 # Clamp mag indices to [102, 921] of [0, 1023]
  signal_rmsnorm:
    enabled: true                      # signal / sqrt(mean(signal²) + eps)

# ---- Issue 5: Radiation Diversity ----
radiation_diversity:
  enabled: true
  alpha: 0.01                          # Weight of diversity loss term

# ---- Issue 7: Load Balancing ----
load_balance:
  enabled: true
  beta: 0.001                          # Weight of balance loss + score penalty scale
  capacity_limit: 5                    # Max radiation hits per node per step
  temperature_init: 1.0                # Initial selection temperature
  temperature_min: 0.1                 # Minimum temperature after annealing
  temperature_anneal_rate: 0.995       # temp *= rate each epoch

# ---- Issue 9: Uncertainty ----
uncertainty:
  enabled: true
  dropout_rate: 0.1                    # Fraction of index dimensions to zero out
  num_mc_samples: 10                   # Stochastic forward passes for uncertainty

# ---- Training ----
training:
  epochs: 100                          # Training epochs
  batch_size: 64                       # Samples per epoch (random subset)
  phase_learning_rate: 0.015           # Discrete update threshold for phase
  magnitude_learning_rate: 0.012       # Discrete update threshold for magnitude

# ---- Forward Pass ----
forward_pass:
  max_timesteps: 30                    # Maximum propagation timesteps
  decay_factor: 0.6                    # Signal decay per timestep
  min_activation_strength: 1.0         # Minimum strength for node activation
  min_output_activation_timesteps: 2   # Minimum timesteps before output nodes activate
  top_k_neighbors: 4                   # Neighbors considered for radiation
  radiation_batch_size: 128            # Batch size for radiation computation
  use_radiation: true                  # Enable radiation-based propagation
```

---

## 5. Data Flow Summary

A single training sample flows through the V3 pipeline in 12 steps:

1. **Feature Loading**: Wine features `x ∈ R^11` (e.g., acidity, pH, alcohol) loaded from `X_train[idx]` (already StandardScaler-normalized).

2. **MLP Projection**: `WineInputAdapter.forward(x)` → `projected ∈ R^220` via `Linear(11→128) → LayerNorm → ReLU → Dropout → Linear(128→128) → LayerNorm → ReLU → Dropout → Linear(128→220) → Tanh`. Output bounded to `[-1, 1]`.

3. **Phase/Mag Quantization**: `quantize_to_phase_mag(projected)` reshapes `[220] → [22, 5, 2]`, quantizes phase channel `[-1,1] → [0, 511]` and mag channel `[-1,1] → [0, 1023]`. Produces `input_context = {n0: (phase[5], mag[5]), ..., n21: (phase[5], mag[5])}`.

4. **MC Dropout** (if enabled): `apply_phase_dropout()` generates Bernoulli(0.9) mask, multiplies indices. Dropped dims → index 0.

5. **Forward Propagation**: `VectorizedForwardEngine.forward_pass_vectorized()` injects input context into activation table, then runs up to 30 timesteps. Each timestep: active nodes radiate to neighbors via `V3PhaseCell.forward()` (modular transfer → magnitude clamping → complex signal → RMSNorm). Signals propagate through intermediate nodes with decay 0.6.

6. **Magnitude Clamping**: Post-propagation, output node magnitudes are clamped to `[102, 921]` via `_clamp_output_magnitudes()`.

7. **Output Signal Extraction**: For each of 6 output nodes (n194-n199): if active, `signal = cos(phase) × exp(sin(mag))` via lookup table; if inactive, use learnable null activation or zero.

8. **Logit Computation**: `compute_logits_from_signals()` computes for each class c: `logit[c] = mean over output_nodes j of cosine_sim(signal_j, class_encoding_c)`. Returns `logits ∈ R^6`.

9. **CE Loss**: `F.cross_entropy(logits, target_label)`. With near-random logits, loss ≈ `ln(6) ≈ 1.79`.

10. **Backward (6 output nodes only)**: Computes `logit_grad = softmax(logits) - one_hot(target)`. For each output node i, approximates upstream gradient using class_id = i, computes phase/mag gradients via lookup table chain rule. **172 intermediate nodes get zero gradient.**

11. **Discrete Parameter Updates**: `quantize_gradients_to_discrete_updates()` maps continuous gradients to `{-1, 0, +1}` with threshold 0.01. `apply_discrete_updates()` applies modular wrap for phase, clamped update for magnitude. Result: at most 6 × 5 = 30 phase + 30 mag = 60 discrete parameter updates.

12. **Auxiliary Loss Computation**: Diversity loss from entropy of recorded radiation targets (frozen ~0.00662). Balance loss from variance of hit counts (always 0.0). Total loss = primary + diversity + balance.

---

## 6. Known Issues Summary

| Bug | Severity | File | Lines | Summary | Impact |
|-----|----------|------|-------|---------|--------|
| BUG 1 | **Critical** | `v3_train_context.py` | 270-292 | Only 6 output nodes get gradient updates | 172 intermediate nodes frozen at random forever; 60/2000 params trainable |
| BUG 2 | **High** | `v3_train_context.py` | 279-287 | 1:1 class-node mapping in gradient | Wrong gradient direction; each node gets gradient from single class instead of all 6 |
| BUG 3 | **High** | `v3_phase_cell.py` | 83-84 | Hardcoded `upstream_real=ones` | Cell gradients disconnected from loss; opportunity cost of complex activation wasted |
| BUG 4 | **Medium** | `v3_forward_engine.py` | 95-102 | Records output nodes as radiation targets | Diversity loss frozen at 0.00662; entropy regularization is dead code |
| BUG 5 | **Low** | `main_v3.py` | 125-130 | Duplicate reset code vs `train()` | Maintenance hazard; double-reset is functionally harmless |
| BUG 6 | **Medium** | `uncertainty.py` | 55-61 | Dropout zeros indices to 0 | Injects systematic cos(0)=1.0 bias into ~10% of input dimensions |
| BUG 7 | **High** | `load_balancer.py` | 64-69 | `record_hit()` never called | Balance loss always 0.0; load balancing is dead code |
| BUG 8 | **High** | `v3_train_context.py` | (systemic) | Input MLP never trained | 46,940 MLP params at random init; all wine samples encoded as random indices |
