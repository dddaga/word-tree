# V3 Dynamic Phasor -- Dry Run Analysis

A complete trace of one training sample through the V3 pipeline, showing exactly where each bug manifests and how it causes training collapse.

---

## 1. Setup

### Dataset
- **Source**: UCI Wine Quality Red (1,599 samples, 11 features, quality 3-8)
- **Label remapping**: quality {3→0, 4→1, 5→2, 6→3, 7→4, 8→5}
- **Split** (seed=42, stratified):
  - Train: 1,119 samples (70%)
  - Validation: 240 samples (15%)
  - Test: 240 samples (15%)
- **Class distribution** (approximate): quality 5 (681, 42.6%) and 6 (638, 39.9%) dominate. Quality 3 (10, 0.6%) and 8 (18, 1.1%) are rare. This imbalance means a model predicting only class 2 (quality 5) achieves ~43% accuracy by chance.

### Architecture
- 200 nodes: n0-n21 (input), n22-n193 (intermediate), n194-n199 (output)
- vector_dim = 5 (each node stores 5 phase indices + 5 magnitude indices)
- Total discrete parameters: 200 × 5 × 2 = 2,000 (phase + magnitude)
- Phase bins N = 512, Magnitude bins M = 1024
- Graph: layered DAG, cardinality = 4 connections per node

### Lookup Tables
```
cos_table[512]:        cos(2π·i/512)  for i in [0, 511]
sin_table[512]:        sin(2π·i/512)  for i in [0, 511]
mag_exp_sin_table[1024]: exp(sin(2π·j/1024))  for j in [0, 1023]
    Range: [exp(-1), exp(1)] ≈ [0.368, 2.718]
```

### Key Config Values
```
phase_lr = 0.015, mag_lr = 0.012
batch_size = 64 samples/epoch
max_timesteps = 30, decay = 0.6
MC dropout_rate = 0.1
diversity alpha = 0.01, balance beta = 0.001
magnitude clamp = [102, 921]
```

---

## 2. Initialization Trace

### Step 2.1: V3TrainContext(config) called

**Lookup Tables** (`_setup_core`, line 81):
```python
HighResolutionLookupTables(phase_bins=512, mag_bins=1024, device="cpu")
# Creates: cos_table[512], sin_table[512], mag_exp_sin_table[1024]
# Also: mag_exp_sin_grad_table[1024] = cos(2π·j/1024) * exp(sin(2π·j/1024))
# phase_grad_scale = 2π/512 ≈ 0.01227
# mag_grad_scale = 2π/1024 ≈ 0.00614
```

**V3PhaseCell** (`_setup_core`, line 88):
```python
V3PhaseCell(vector_dim=5, lookup_tables, gamma=1.0,
            enable_complex=True, enable_normalization=True, mag_bins=1024)
# Contains: ModularPhaseCell + ComplexSignalComputer(gamma=1.0) + PhasorNormalization(clamp=[102,921])
```

**Graph** (`_setup_graph`, line 100):
```python
build_static_graph(200, 22, 6, 5, 512, 1024, cardinality=4, seed=42)
# Creates layered DAG: input layer → intermediate layers → output layer
# Each node connected to ~4 others in the next layer
```

**NodeStore** (`_setup_graph`, line 110):
```python
NodeStore(graph_df, 5, 512, 1024)
# For each of 200 nodes:
#   phase_table[node_id] = LongTensor[5] with random values in [0, 511]
#   mag_table[node_id]   = LongTensor[5] with random values in [0, 1023]
# Total: 200 × 5 × 2 = 2,000 discrete parameters
```

**WineInputAdapter** (`_setup_input`, line 118):
```python
WineInputAdapter(input_dim=11, num_input_nodes=22, vector_dim=5,
                 phase_bins=512, mag_bins=1024, device="cpu",
                 test_size=0.30, val_fraction=0.50, seed=42)
# MLP: 11 → Linear(128) → LN → ReLU → Dropout(0.1)
#       → Linear(128) → LN → ReLU → Dropout(0.1)
#       → Linear(220) → Tanh
# Parameters: 11×128+128 + 128×2 + 128×128+128 + 128×2 + 128×220+220 = ~46,940
# Loads winequality-red.csv, splits, scales
# ⚠️ BUG 8: These 46,940 params will NEVER be updated
```

**Class Encodings** (`_setup_output`, line 132):
```python
generate_fixed_class_encodings(512, 1024, 5, seed=42)
# Returns dict: {0: (phase[5], mag[5]), 1: ..., ..., 9: ...}
# 10 random (phase, mag) pairs; only classes 0-5 used
# Each class has a unique "signature" in phase-magnitude space
```

**V3 Components** (`_setup_v3_components`, line 149):
```python
PhasorNormalization(mag_bins=1024, clamp_low=102, clamp_high=921)
RadiationDiversityTracker(total_nodes=200, alpha=0.01)
RadiationLoadBalancer(total_nodes=200, beta=0.001, capacity=5, temp=1.0)
MCDropoutUncertainty(vector_dim=5, phase_bins=512, mag_bins=1024,
                     dropout_rate=0.1, num_mc_samples=10)
```

---

## 3. Single Sample Forward Pass

**Sample**: index 0 from training set. Suppose `target_label = 2` (wine quality 5).

### 3.1 Input Encoding

```python
# Step 1: Load features
x = X_train[0]  # shape [11], StandardScaler-normalized floats

# Step 2: MLP projection
projected = self.projection(x)  # shape [220], values in [-1, 1] via Tanh
```

**BUG 8 manifests here**: The MLP weights are PyTorch default initialization (Kaiming uniform). Since `loss.backward()` never flows through the MLP and no optimizer updates its parameters, the projection is a **fixed random function**. Two wine samples with identical features would produce identical projections, but the projection carries no meaningful semantic information -- it's equivalent to a random hash.

```python
# Step 3: Quantize
reshaped = projected.view(22, 5, 2)  # [num_input_nodes, vector_dim, phase+mag]
phase_raw = reshaped[:, :, 0]        # [22, 5] in [-1, 1]
mag_raw = reshaped[:, :, 1]          # [22, 5] in [-1, 1]

# Phase: [-1,1] → [0, 2π) → [0, 511]
phase_indices = floor((phase_raw + 1)/2 * 2π mod 2π) / (2π) * 512)
# → LongTensor[22, 5] with values in [0, 511]

# Magnitude: [-1,1] → [-3, 3] → [0, 1] → [0, 1023]
mag_indices = floor((mag_raw * 3 + 3) / 6 * 1024)
# → LongTensor[22, 5] with values in [0, 1023]

# Result: input_context = {
#   "n0": (phase[5], mag[5]),   # e.g., ([256, 103, 412, 78, 330], [512, 823, 145, 667, 901])
#   "n1": (phase[5], mag[5]),
#   ...
#   "n21": (phase[5], mag[5]),
# }
```

Because of BUG 8, these indices are effectively **random** regardless of the wine sample's actual features.

### 3.2 MC Dropout

```python
# uncertainty.apply_phase_dropout(phase, mag, training=True)
mask = Bernoulli(torch.full([5], 0.9)).long()
# Example: mask = [1, 1, 0, 1, 1]  (dim 2 dropped)

masked_phase = phase * mask   # e.g., [256, 103, 0, 78, 330]
masked_mag   = mag * mask     # e.g., [512, 823, 0, 667, 901]
```

**BUG 6 manifests here**: Dimension 2 is dropped by multiplying by 0, setting both indices to 0. This is NOT a null signal -- it maps to:
```
cos(0) = 1.0           (maximum positive phase)
exp(sin(0)) = 1.0      (neutral magnitude)
signal_dim2 = 1.0 × 1.0 = 1.0
```
Instead of representing "no information" (null), index 0 injects a deterministic bias of 1.0 into that dimension. With 10% dropout across 22 input nodes × 5 dimensions = 110 values, approximately 11 dimensions per sample get this 1.0 injection.

### 3.3 Forward Propagation

```python
# V3ForwardEngine.forward_pass(input_context)
# → base_engine.forward_pass_vectorized(input_context)

# Timestep 0: Inject input context
# activation_table.inject(n0..n21, phase_indices, mag_indices, strength)

# Timesteps 1-30: Propagation with decay
for t in range(max_timesteps):
    for each active node:
        # Get radiation neighbors (top-4 by phase alignment)
        neighbors = get_radiation_neighbors(node_id, graph_df, k=4)

        for each neighbor:
            # V3PhaseCell.forward(ctx_phase, ctx_mag, self_phase, self_mag)
            phase_out = (ctx_phase + self_phase) % 512
            mag_out = (ctx_mag + self_mag) % 1024

            # Clamp magnitude (Issue 4)
            mag_out = clamp(mag_out, 102, 921)

            # Complex signal (Issue 1)
            cos_vals = cos_table[phase_out]         # cos(φ)
            sin_vals = sin_table[phase_out]         # sin(φ)
            exp_sin = mag_exp_sin_table[mag_out]    # exp(sin(m))
            real = exp_sin * cos_vals
            imag = exp_sin * sin_vals
            signal = real                           # Use real part
            signal = signal / sqrt(mean(signal²))   # RMSNorm
            strength = sum(exp_sin)
```

**BUG 3 manifests here** (v3_phase_cell.py:83-84):
```python
# During forward(), the cell pre-computes gradients:
upstream_real = torch.ones_like(real)   # ← HARDCODED, not from loss!
upstream_imag = torch.zeros_like(imag)  # ← HARDCODED
grad_phase, grad_mag = complex_signal.compute_complex_gradients(
    phase_out, mag_out, upstream_real, upstream_imag
)
# These gradients are returned but MEANINGLESS because:
# 1. upstream = ones means "increase all real components equally" regardless of loss
# 2. BUG 1 means intermediate nodes never receive updates anyway
```

**Post-propagation** (v3_forward_engine.py:95-102):

**BUG 4 manifests here**:
```python
# _record_radiation_from_stats()
active_outputs = engine.get_active_output_nodes()
# Returns e.g., ["n194", "n195", "n196", "n197", "n198", "n199"]
# These are OUTPUT NODE IDs, not radiation targets!
# Radiation targets = intermediate nodes that received radiation hits
diversity_tracker.record_radiation_targets(active_outputs)
# → hit_counts: only indices 194-199 get incremented
# → max 6 unique targets out of 200 → frozen entropy
```

### 3.4 Output Signal Extraction

```python
# v3_train_context.py:forward_pass() lines 219-243
output_signals = {}
for node_id in ["n194", "n195", "n196", "n197", "n198", "n199"]:
    phase = node_store.get_phase(node_id)  # LongTensor[5]
    mag = node_store.get_mag(node_id)      # LongTensor[5]

    if node_is_active:
        signal = lookup_tables.get_signal_vector(phase, mag)
        # signal[d] = cos(2π·phase[d]/512) * exp(sin(2π·mag[d]/1024))
        # Example: signal = [0.85, -0.42, 1.23, 0.11, -0.67]
    else:
        # Use learnable null activation
        null_p, null_m = uncertainty.get_null_activation()
        signal = lookup_tables.get_signal_vector(null_p, null_m)

    output_signals[node_id] = signal  # Tensor[5]
```

### 3.5 Logit Computation

```python
# classification_loss.compute_logits_from_signals()
logits = torch.zeros(6)
for c in range(6):  # for each class
    class_phase, class_mag = class_encodings[c]
    class_signal = lookup_tables.get_signal_vector(class_phase, class_mag)  # [5]

    for j, node_id in enumerate(output_nodes):  # 6 output nodes
        sim = cosine_similarity(output_signals[node_id], class_signal)
        logits[c] += sim / 6  # average over output nodes

# With random node parameters and random input encoding:
# logits ≈ [0.12, 0.08, 0.15, 0.11, 0.09, 0.13]  (near uniform)
```

---

## 4. Loss Computation

```python
target = torch.tensor([2])  # class 2 (quality 5)
primary_loss = F.cross_entropy(logits.unsqueeze(0), target)
# ≈ -log(softmax(logits)[2])
# ≈ -log(1/6) ≈ 1.79  (random baseline = ln(6))
```

---

## 5. Backward Pass Trace

### 5.1 Logit Gradient

```python
probs = softmax(logits)
# ≈ [0.167, 0.165, 0.170, 0.166, 0.164, 0.168]  (near uniform)

target_one_hot = [0, 0, 1, 0, 0, 0]

logit_grad = probs - target_one_hot
# ≈ [0.167, 0.165, -0.830, 0.166, 0.164, 0.168]
#    ^^^^^^^^^^^^^^^^^^ ^^^^^^ ^^^^^^^^^^^^^^^^^^
#    small positive     large   small positive
#    "push away"        negative "push away"
#                       "pull toward"
```

### 5.2 Per-Output-Node Gradient (BUG 2)

The backward pass loops over the 6 output nodes (v3_train_context.py:270-292):

```python
for i, node_id in enumerate(self.output_nodes):
    # i=0 → n194, i=1 → n195, ..., i=5 → n199

    class_id = i if i < self.num_classes else 0   # ← BUG 2: class_id = node_index!
    # n194 gets gradient ONLY from class 0
    # n195 gets gradient ONLY from class 1
    # n196 gets gradient ONLY from class 2  (the target class)
    # n197 gets gradient ONLY from class 3
    # n198 gets gradient ONLY from class 4
    # n199 gets gradient ONLY from class 5

    class_signal = lookup_tables.get_signal_vector(*class_encodings[class_id])

    upstream = logit_grad[class_id] * (class_signal / (norm_s * norm_c))
    # For n194 (class_id=0): upstream = 0.167 * normalized_class_signal  (small)
    # For n196 (class_id=2): upstream = -0.830 * normalized_class_signal (large, negative)

    pg, mg = lookup_tables.compute_signal_gradients(phase, mag, upstream)
    node_gradients[node_id] = (pg, mg)
```

**What BUG 2 causes**: Each output node is paired with exactly one class. Node n194 only learns "don't look like class 0" (positive logit_grad) while n196 only learns "look more like class 2" (negative logit_grad). The correct formula should sum contributions from ALL classes:

```python
# CORRECT (not implemented):
for j, node_id in enumerate(output_nodes):
    upstream = torch.zeros(vector_dim)
    for c in range(num_classes):
        class_signal = get_signal_vector(*class_encodings[c])
        d_sim_d_signal = (class_signal / (norm_s * norm_c) -
                         cosine_sim * signal / norm_s²) / norm_s
        upstream += logit_grad[c] * (1/6) * d_sim_d_signal
    pg, mg = compute_signal_gradients(phase, mag, upstream)

# BUG 2 (implemented):
for i, node_id in enumerate(output_nodes):
    class_id = i  # ← Wrong! Node index ≠ class contribution
    upstream = logit_grad[class_id] * (class_signal / (norm_s * norm_c))
```

### 5.3 Missing Intermediate Gradients (BUG 1)

After the backward loop completes:

```python
node_gradients = {
    "n194": (phase_grad[5], mag_grad[5]),
    "n195": (phase_grad[5], mag_grad[5]),
    "n196": (phase_grad[5], mag_grad[5]),
    "n197": (phase_grad[5], mag_grad[5]),
    "n198": (phase_grad[5], mag_grad[5]),
    "n199": (phase_grad[5], mag_grad[5]),
}
# Only 6 entries! Nodes n22-n193 (172 intermediate nodes) = ZERO updates
```

**Comparison with original `ModularTrainContext`** (train/modular_train_context.py, lines 649-699):

```python
# ORIGINAL (correct): Updates ALL active nodes
for node_id in all_active_nodes:  # 600+ nodes typically active
    signal = get_signal_vector(phase, mag)
    # Compute "credit" via cosine alignment with output gradient
    for class_c in range(num_classes):
        alignment = cosine_sim(signal, class_signal[c])
        credit = alignment * logit_grad[c]
    # Scale credit by node's position in the graph
    upstream = credit * propagation_weight
    pg, mg = compute_signal_gradients(phase, mag, upstream)
    node_gradients[node_id] = (pg, mg)
```

The original processes ALL active nodes with a credit assignment based on cosine alignment. V3 skips this entirely and only updates the 6 output nodes.

### 5.4 Parameter Update

```python
# For each of the 6 output nodes:
for node_id, (pg, mg) in node_gradients.items():
    # Quantize continuous gradients to discrete updates
    phase_updates, mag_updates = quantize_gradients_to_discrete_updates(
        pg, mg,
        phase_learning_rate=0.015,
        magnitude_learning_rate=0.012,
        node_id=node_id,
    )
    # threshold = 0.01
    # If |pg[d] * 0.015| > 0.01: phase_update[d] = sign(pg[d])  → {-1, +1}
    # Else: phase_update[d] = 0

    # Apply updates
    new_phase = (current_phase + phase_updates) % 512   # modular wrap
    new_mag = clamp(current_mag + mag_updates, 0, 1023) # clamp

    node_store.phase_table[node_id] = new_phase
    node_store.mag_table[node_id] = new_mag
```

**Parameters updated per sample**:
- 6 nodes × 5 dimensions × 2 (phase + mag) = 60 discrete parameters
- Out of 200 × 5 × 2 = 2,000 total = **3% of parameters**
- And ~46,940 MLP parameters are completely frozen

---

## 6. Auxiliary Loss Trace

### 6.1 Diversity Loss

After 64 samples in one epoch:

```python
# diversity_tracker state:
# hit_counts: 200 entries, but only indices 194-199 are nonzero
# Each output node recorded ~64 times (once per sample forward pass)
# total_hits ≈ 64 × 6 = 384

p = hit_counts / total_hits
# p[194] = p[195] = ... = p[199] ≈ 1/6 each
# p[0] = p[1] = ... = p[193] = 0

H = -sum(p * log(p)) = -6 × (1/6 × log(1/6)) = log(6) = 1.7918
H_max = log(200) = 5.2983
normalized_entropy = 1.7918 / 5.2983 = 0.3382

diversity_loss = alpha * (1 - normalized_entropy)
              = 0.01 * (1 - 0.3382)
              = 0.01 * 0.6618
              = 0.006618
```

This value is **constant every epoch** (observed: 0.006618 ± 0.000001 for all 100 epochs) because:
1. BUG 4 always records the same 6 output nodes
2. Each epoch processes 64 samples, each recording ~6 output nodes
3. The distribution is always approximately uniform over 6 nodes

### 6.2 Balance Loss

```python
# load_balancer state:
# epoch_hit_counts: all zeros (200 entries)
# record_hit() is NEVER called ← BUG 7

balance_loss = compute_balance_loss()
# epoch_hit_counts.sum() == 0 → return 0.0
```

Balance loss = **exactly 0.0 for all 100 epochs** (confirmed in metrics).

---

## 7. 100-Epoch Phase Analysis

Using actual data from `logs/v3_experiment/v3_100epochs_metrics.json`:

### Phase 1: Initial Alignment (Epochs 1-13)

| Epoch | Total Loss | Primary Loss | Train Acc | Val Acc |
|-------|-----------|-------------|-----------|---------|
| 1     | 1.9718    | 1.9652      | 1.6%      | -       |
| 5     | 1.7878    | 1.7812      | 43.8%     | 40.0%   |
| 10    | 1.7659    | 1.7593      | 42.2%     | 40.0%   |
| 13    | 1.6962    | 1.6896      | 43.8%     | -       |

- Loss drops from 1.97 → 1.69 (minimum)
- Train accuracy rises to ~43%, val to 40.0%
- The 6 output nodes find coarse alignment where most predictions land on class 2 (quality 5), which happens to be the most frequent class (~43% of data)
- The model is essentially learning a **majority-class predictor** using only 60 parameters

### Phase 2: Plateau & Saturation (Epochs 13-28)

| Epoch | Total Loss | Primary Loss | Train Acc | Val Acc |
|-------|-----------|-------------|-----------|---------|
| 15    | 1.7397    | 1.7331      | 42.2%     | 40.0%   |
| 20    | 1.7756    | 1.7690      | 31.3%     | 40.0%   |
| 24    | 1.7565    | 1.7498      | 50.0%     | -       |
| 25    | 1.7602    | 1.7536      | 45.3%     | 40.0%   |

- Loss plateaus around 1.73-1.78
- The 60 trainable parameters have found their locally optimal configuration
- Train accuracy oscillates between 31-50% (noisy due to small 64-sample batches)
- Val accuracy stuck at 40.0% (majority class fraction)

### Phase 3: Destabilization (Epochs 28-40)

| Epoch | Total Loss | Primary Loss | Train Acc | Val Acc |
|-------|-----------|-------------|-----------|---------|
| 29    | 1.7804    | 1.7737      | 20.3%     | -       |
| 30    | 1.7591    | 1.7525      | 1.6%      | 0.4%    |
| 35    | 1.7904    | 1.7838      | 3.1%      | 0.4%    |
| 40    | 1.8042    | 1.7976      | 1.6%      | 0.4%    |

- Loss starts rising above 1.78
- Train accuracy collapses from 20% → 1.6% at epoch 30
- Val accuracy drops to 0.4% and stays there
- BUG 2 (wrong gradient direction) destabilizes the 60 trainable parameters
- The 1:1 class-node pairing pushes each output node toward its assigned class regardless of the actual training signal, eventually destroying the lucky initial alignment

### Phase 4: Collapsed Random Walk (Epochs 40-100)

| Epoch | Total Loss | Primary Loss | Train Acc | Val Acc |
|-------|-----------|-------------|-----------|---------|
| 50    | 1.8295    | 1.8228      | 3.1%      | 0.4%    |
| 60    | 1.8406    | 1.8340      | 0.0%      | 0.4%    |
| 70    | 1.8536    | 1.8470      | 0.0%      | 0.4%    |
| 80    | 1.8423    | 1.8356      | 0.0%      | 0.4%    |
| 90    | 1.8447    | 1.8380      | 0.0%      | 0.4%    |
| 100   | 1.8335    | 1.8269      | 1.6%      | 0.4%    |

- Loss oscillates between 1.80-1.87 (above random baseline ln(6)=1.79)
- Train accuracy: effectively 0% (occasional 1.6% = 1/64 correct by chance)
- The model predicts a single class for all inputs, and it's NOT the majority class anymore
- Parameters undergo a random walk: BUG 2 pushes them in wrong directions, BUG 1 prevents meaningful correction via intermediate nodes

### Final Test Results

```
Test Accuracy:  0.83%  (2/240 correct)
Test F1:        0.00014
Test Precision: 0.00007
Test Recall:    0.83%

Confusion Matrix:
              Predicted
              c0   c1   c2   c3   c4   c5
Actual c0  [  2,   0,   0,   0,   0,   0]   (2 samples, both predicted c0)
Actual c1  [  8,   0,   0,   0,   0,   0]   (8 samples, all predicted c0)
Actual c2  [103,   0,   0,   0,   0,   0]   (103 samples, all predicted c0)
Actual c3  [ 96,   0,   0,   0,   0,   0]   (96 samples, all predicted c0)
Actual c4  [ 27,   0,   0,   0,   0,   0]   (27 samples, all predicted c0)
Actual c5  [  4,   0,   0,   0,   0,   0]   (4 samples, all predicted c0)
```

**ALL 240 test samples predicted as class 0** (wine quality 3). Only 2 actual class-0 samples exist → 2/240 = 0.83% accuracy. Complete single-class collapse.

### Auxiliary Metrics (constant across all 100 epochs)

| Metric | Value | Explanation |
|--------|-------|-------------|
| Diversity loss | 0.006618 ± 0.000001 | Frozen (BUG 4: always recording same 6 nodes) |
| Balance loss | 0.0 | Dead (BUG 7: record_hit never called) |
| Entropy | 1.79 | = log(6), uniform over 6 output nodes |
| Gini coefficient | 0.97 | Extreme concentration (6/200 nodes) |
| Unique targets | 0 | After reset, before recording |
| Temperature | 1.0 → 0.609 | Decays per epoch but has no effect (dead code) |

### Complexity

```
Total parameters:     49,151
  Node store:          2,200  (200 × 5 × 2 + biases)
  Input adapter MLP:  46,940  (never trained)
  Uncertainty:            11  (null_phase[5] + null_mag[5] + temperature)
Memory:               0.19 MB
Epoch time:           ~12.3s average
Inference latency:    208,000 μs (208 ms) median
Total training time:  2,270s (37.8 minutes)
```

---

## 8. Root Cause Diagnosis

Ranked by contribution to training failure:

### Primary (Fatal): BUG 1 + BUG 8

**BUG 1 (Critical)**: Only 60/2,000 discrete parameters are trainable. The 172 intermediate nodes that should form the computational backbone of the graph are permanently frozen at random initialization. With only 6 output nodes × 5 dimensions × 2 (phase+mag) = 60 free parameters, the model has zero capacity to learn complex decision boundaries for a 6-class problem with 11 features.

**BUG 8 (High)**: The 46,940-parameter MLP is never trained. Since it maps ALL wine feature vectors to random phase/magnitude indices, the GNN receives no meaningful input signal. Even if all 2,000 node parameters were trainable, the random input encoding would make learning nearly impossible.

**Combined impact**: The model has ~0.12% of its parameters actually learning (60 out of 49,151), and those 60 parameters receive random inputs. The system is fundamentally incapable of representing any meaningful function.

### Secondary (Degrading): BUG 2

**BUG 2 (High)**: The 1:1 `class_id = i` mapping provides incorrect gradient signals to even the 60 trainable parameters. Instead of each output node receiving a weighted sum of gradient contributions from all 6 classes (based on how the logit depends on that node's signal), each node receives gradient from only one class. This means:

- n194 is pushed toward/away from class 0 encoding regardless of the target
- n196 is pushed toward class 2 encoding (quality 5) for all class-2 targets, but pushed away for all other targets

This creates conflicting gradient signals that cancel out over batches. In the initial epochs, the majority class (class 2) happens to win, producing ~43% accuracy. But as training continues, the gradient noise from wrong-direction updates on other nodes accumulates and destabilizes the fragile alignment, causing the collapse around epoch 28.

### Contributing (Noise): BUG 6

**BUG 6 (Medium)**: MC dropout multiplying indices by 0 injects cos(0)=1.0 bias into ~10% of input dimensions. While not the primary cause of failure, this systematic noise:
- Distorts the (already random) input encoding further
- Biases output logits via the injected 1.0 values
- Makes the input distribution asymmetric (0 index is over-represented)

### Non-Functional (Inert): BUG 4 + BUG 7

**BUG 4 + BUG 7**: Both auxiliary loss systems are dead code. Diversity loss is frozen at 0.00662 and balance loss is always 0.0. These contribute a constant offset to the total loss but provide zero gradient signal for actual learning. The entire Issue 5 and Issue 7 implementations have no effect on training dynamics.

### Wasted (Opportunity Cost): BUG 3

**BUG 3 (High)**: Complex activation gradients computed with `upstream = ones` are meaningless. Even if intermediate nodes received updates (fixing BUG 1), the gradients would be disconnected from the loss. This wastes the careful Issue 1 implementation of dual-channel gradient computation.

---

## 9. Comparison: Original vs V3 Training

| Aspect | ModularTrainContext (Original) | V3TrainContext |
|--------|-------------------------------|----------------|
| **Nodes updated/sample** | 600+ (all active nodes) | 6 (output only) |
| **Trainable params** | ~2,000 (all node store) | 60 (6 output × 5 dims × 2) |
| **Intermediate credit** | Cosine alignment (lines 649-699) | Missing entirely |
| **Gradient per node** | Sum over ALL classes weighted by alignment | 1:1 class_id = node_index |
| **Input encoding** | Fixed feature projection | Random MLP (never trained) |
| **Upstream gradients** | Loss-derived via chain rule | Hardcoded ones/zeros in cell |
| **Auxiliary losses** | N/A (not implemented) | Dead code (BUG 4 + BUG 7) |
| **Expected accuracy** | 55-65% (Wine Quality ceiling) | 0.83% (single-class collapse) |

---

## 10. Expected Behavior After Fixes

If all 8 bugs were fixed:

| Metric | Current (Broken) | Expected (Fixed) |
|--------|------------------|-------------------|
| Final loss | ~1.83 (above random) | ~1.0-1.2 (below random) |
| Train accuracy | 0-1.6% | 55-65% |
| Val accuracy | 0.4% | 45-55% (within 10% of train) |
| Test accuracy | 0.83% | 45-55% |
| Diversity loss | Frozen 0.00662 | Varies epoch-to-epoch (decreasing) |
| Balance loss | Always 0.0 | Nonzero, decreasing |
| Calibration temp | 5.0 (boundary of grid) | 1.0-2.0 (well-calibrated) |
| Trainable params | 60 (node) + 0 (MLP) | 2,000 (node) + 46,940 (MLP) |

**Key fixes required**:
1. **BUG 1**: Port intermediate credit assignment from `modular_train_context.py:649-699`
2. **BUG 2**: Replace `class_id = i` with sum over all classes
3. **BUG 3**: Pass actual loss-derived upstream gradients to `compute_complex_gradients()`
4. **BUG 4**: Hook into base engine to record actual radiation targets
5. **BUG 6**: Replace dropped indices with null activation instead of 0
6. **BUG 7**: Call `record_hit()` from radiation selection logic
7. **BUG 8**: Add optimizer for MLP parameters, call `loss.backward()` through projection

Note: Wine Quality Red is inherently noisy (inter-annotator agreement ~65%), so accuracy above 60% would indicate the model is learning meaningful patterns. The current 0.83% demonstrates complete failure.
