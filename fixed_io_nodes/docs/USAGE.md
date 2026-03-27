# Usage Guide: NativeNeurographLayer

This guide covers how to use the `native/` module to build and train models with the Neurograph GNN layer. The native module is the current recommended path -- it runs single-process, uses standard PyTorch autograd, and requires no external services (no Qdrant, no shared memory).

---

## Quick Start

```python
import torch
import torch.nn as nn
from native import NativeNeurographLayer, NativeGNNOptimizer
from native.checkpoint import save_full_model, load_full_model

# 1. Define config (or load from YAML)
cfg = {
    "graph": {
        "total_nodes": 150,
        "input_nodes": 8,
        "output_nodes": 10,
        "cardinality": 5,
        "radiation_targets": 5,
    },
    "model": {
        "vector_dim": 8,
        "iterations": 5,
    },
}

# 2. Build model
class MyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_nodes = cfg["graph"]["input_nodes"]
        self.vector_dim = cfg["model"]["vector_dim"]
        self.linear = nn.Linear(4, self.input_nodes * self.vector_dim)
        self.gnn = NativeNeurographLayer(cfg)
        self.out = nn.Linear(cfg["graph"]["output_nodes"], 3)

    def forward(self, x):
        B = x.size(0)
        h = self.linear(x)
        h = h.view(B, self.input_nodes, self.vector_dim)
        return self.out(self.gnn(h))

model = MyModel(cfg)

# 3. Create optimizer (NOT torch.optim.Adam)
optimizer = NativeGNNOptimizer(model, lr=0.001, accumulation_steps=4)

# 4. Standard training loop
criterion = nn.CrossEntropyLoss()
for x, y in dataloader:
    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()

# 5. Save / load
save_full_model(model, "checkpoint.pt")
load_full_model(model, "checkpoint.pt")
```

---

## Configuration Reference

The `NativeNeurographLayer` accepts a config dictionary (or a path to a YAML file). The config is organized into sections.

### Graph Structure (`graph`)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `total_nodes` | int | Yes | Total number of nodes in the graph. Must be sufficiently larger than `input_nodes + output_nodes` (at least 10x the sum). |
| `input_nodes` | int | Yes | Number of input nodes. These receive external input during the forward pass. |
| `output_nodes` | int | Yes | Number of output nodes. Their activation strengths form the layer's output. |
| `cardinality` | int | Yes | Maximum number of incoming connections per non-input node. Each node gets `randint(1, cardinality)` incoming edges from non-output nodes. |
| `radiation_targets` | int | Yes | Number of dynamic edges per active node per iteration (cosine-searched + random). Set to 0 to disable radiation entirely. |

**Sizing rules:**
- The GNN layer expects input of shape `(batch, input_nodes, vector_dim)`. Your preceding layer must produce exactly `input_nodes * vector_dim` features.
- The GNN layer outputs shape `(batch, output_nodes)`. Your subsequent layer (e.g., `nn.Linear`) takes `output_nodes` as input.
- For VGG16 features (512 x 7 x 7 = 25088): set `input_nodes * vector_dim = 25088`. For example, `input_nodes=3136, vector_dim=8`.

### Model Architecture (`model`)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `vector_dim` | int | Yes | -- | Dimensionality of each node's phase/magnitude vectors. |
| `iterations` | int | Yes | -- | Number of message-passing iterations. More iterations allow information to propagate further through the graph. |
| `radiation_similarity_threshold` | float | No | 0.0 | Minimum cosine similarity for radiation targets. Searched targets below this threshold are dropped. Does not affect random targets. |
| `gamma` | float | No | 1.0 | Passed to activation strength function (currently unused in the unquantized path, kept for compatibility). |
| `scattering_prob` | float | No | 0.0 | Base probability that a radiation target is chosen randomly instead of by cosine search. Decays exponentially during training. 0.0 = pure cosine search, 1.0 = pure random. |
| `stochastic_radiation_duration` | float | No | 0.5 | Fraction of total training steps during which scattering is active. After `total_steps * this_value` steps, scattering drops to zero. |

### Training Hyperparameters (`training`)

These are read by the training script, not by the layer itself.

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `lr` | float | Yes | -- | Learning rate for both GNN and head parameters. |
| `batch_size` | int | Yes | -- | Batch size for the DataLoader. |
| `accumulation_steps` | int | Yes | -- | Number of gradient accumulations before a GNN node's weights are updated. See "Per-Node Gradient Accumulation" below. |
| `epochs` | int | Yes | -- | Number of training epochs. |
| `validation_fraction` | float | No | 0.0 | Fraction of data to hold out for validation (0.0 = no validation split). |
| `lr_decay_factor` | float | No | *(none)* | If set, enables `ReduceLROnPlateau` scheduler. Factor by which LR is reduced. |
| `plateau_patience` | int | No | 5 | Number of epochs with no improvement before LR is reduced. Only used when `lr_decay_factor` is set. |
| `min_lr` | float | No | 1e-7 | Minimum learning rate for the scheduler. |

### System (`system`)

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `device` | str | No | "cpu" | PyTorch device: "cpu", "cuda", "mps". |
| `random_seed` | int | No | 42 | Seed for graph topology initialization. Same seed = same graph structure. |
| `use_layer_norm` | bool | No | true | Whether to apply LayerNorm to magnitude activations. Recommended to keep enabled. |

---

## Example Config File (YAML)

```yaml
# --- Graph Structure ---
graph:
  total_nodes: 150
  input_nodes: 8
  output_nodes: 10
  cardinality: 5
  radiation_targets: 5

# --- Model Architecture ---
model:
  vector_dim: 8
  iterations: 5
  radiation_similarity_threshold: 0.1
  scattering_prob: 0.8
  stochastic_radiation_duration: 0.4

# --- Training Hyperparameters ---
training:
  lr: 0.001
  batch_size: 4
  accumulation_steps: 4
  epochs: 16
  validation_fraction: 0.2

# --- System ---
system:
  device: "cpu"
  random_seed: 42
  use_layer_norm: true
```

### Larger Scale Example (VGG features)

```yaml
graph:
  total_nodes: 30000
  input_nodes: 3136       # 25088 / vector_dim = 3136
  output_nodes: 256
  cardinality: 1000
  radiation_targets: 0    # disable for large graphs (expensive)

model:
  vector_dim: 8
  iterations: 5

training:
  lr: 0.0001
  batch_size: 4
  accumulation_steps: 4
  epochs: 5

system:
  device: "mps"
  random_seed: 42
  use_layer_norm: true
```

---

## Key Concepts

### NativeNeurographLayer

The GNN layer. Accepts `(batch, input_nodes, vector_dim)` and returns `(batch, output_nodes)`.

```python
from native import NativeNeurographLayer

# From a config dict:
gnn = NativeNeurographLayer(cfg)

# Or from a YAML file path:
gnn = NativeNeurographLayer("path/to/config.yaml")

# Optional: disable LayerNorm
gnn = NativeNeurographLayer(cfg, use_layer_norm=False)
```

**Important:** The input tensor is mapped to phase space internally via `tanh(x) * pi`. You do not need to preprocess inputs into angles yourself -- just feed raw features (after a linear projection to the right shape).

### NativeGNNOptimizer

A dual-path optimizer that **must** be used instead of `torch.optim.Adam`:

```python
from native import NativeGNNOptimizer

optimizer = NativeGNNOptimizer(
    model,                    # must contain at least one NativeNeurographLayer
    lr=0.001,                 # learning rate
    accumulation_steps=4,     # gradient accumulations before GNN update
    betas=(0.9, 0.999),       # Adam betas (optional)
    eps=1e-8,                 # Adam epsilon (optional)
)
```

**Why not `torch.optim.Adam`?** The GNN's phase/magnitude weights receive sparse gradients -- most nodes get near-zero gradient in any given batch. The optimizer accumulates gradients per-node and only applies Adam when a node has accumulated enough gradient samples (`accumulation_steps`). Standard Adam would apply near-zero updates to most nodes every step, which is wasteful and can destabilize training.

The optimizer has two parameter groups:
- `param_groups[0]` -- head parameters (linear layers), updated every batch with standard Adam
- `param_groups[1]` -- GNN parameters, updated via per-node accumulation

This means LR schedulers work correctly when attached to this optimizer:

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
# ... in training loop:
scheduler.step(val_loss)
```

### Training Progress (Scattering Decay)

If using stochastic radiation (`scattering_prob > 0`), call `set_training_progress` before each batch so the decay schedule works:

```python
total_steps = epochs * len(train_loader)
global_step = 0
for epoch in range(epochs):
    for x, y in train_loader:
        model.gnn.set_training_progress(global_step, total_steps)
        # ... forward, backward, step
        global_step += 1
```

If `scattering_prob` is 0 (default), this call is a no-op and can be skipped.

### Checkpointing

Save and load model weights along with graph topology:

```python
from native.checkpoint import save_full_model, load_full_model

# Save (includes nn.Parameters + graph connections)
save_full_model(model, "checkpoint.pt")

# Load (model must have same config)
load_full_model(model, "checkpoint.pt", map_location="cpu")
```

The checkpoint stores:
- The full `model.state_dict()` (all nn.Parameters including linear layers)
- Graph topology (connections, input/output node IDs) per GNN layer

**Optimizer state** is saved/loaded separately if needed (see `vgg_training/training.py` for the pattern with `_save_optimizer` / `_load_optimizer`).

---

## Building a Model: Step by Step

### 1. Choose your graph dimensions

The GNN layer is a **fixed-size** transformation: `(input_nodes, vector_dim) -> (output_nodes,)`.

- Decide `vector_dim` (typically 4-16; smaller = faster, larger = more expressive).
- Compute `input_nodes` from your feature size: `input_nodes = feature_size / vector_dim`.
- Choose `output_nodes` based on downstream needs (often > num_classes).
- Set `total_nodes` to be at least 10x `(input_nodes + output_nodes)`.

### 2. Adapt your input

The layer expects `(batch, input_nodes, vector_dim)`. Use a linear layer to project:

```python
# For raw features of size F:
self.proj = nn.Linear(F, input_nodes * vector_dim)

# In forward:
h = self.proj(x).view(batch, input_nodes, vector_dim)
out = self.gnn(h)  # (batch, output_nodes)
```

### 3. Add a classification head

The GNN outputs `output_nodes` activation strengths. Add a linear head:

```python
self.classifier = nn.Linear(output_nodes, num_classes)
logits = self.classifier(self.gnn(h))
```

### 4. Complete model examples

**Small (Iris-scale):**

```python
class IrisModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_in = cfg["graph"]["input_nodes"]
        v_dim = cfg["model"]["vector_dim"]
        self.input_nodes = n_in
        self.vector_dim = v_dim
        self.linear = nn.Linear(4, n_in * v_dim)
        self.gnn = NativeNeurographLayer(cfg)
        self.out = nn.Linear(cfg["graph"]["output_nodes"], 3)

    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(1)
        h = self.linear(x).view(B, self.input_nodes, self.vector_dim)
        return self.out(self.gnn(h))
```

**Large (VGG features):**

```python
class VGGModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_nodes = cfg["graph"]["input_nodes"]
        self.vector_dim = cfg["model"]["vector_dim"]
        # No linear projection needed -- VGG features already have the right size
        assert self.input_nodes * self.vector_dim == 512 * 7 * 7
        self.gnn = NativeNeurographLayer(cfg)
        self.out = nn.Linear(cfg["graph"]["output_nodes"], 10)

    def forward(self, x):
        B = x.size(0)
        h = x.view(B, self.input_nodes, self.vector_dim)
        return self.out(self.gnn(h))
```

---

## Loading Config from YAML

```python
from main import load_config

cfg = load_config("path/to/config.yaml")
model = MyModel(cfg)
```

`load_config` is a simple `yaml.safe_load` wrapper.

---

## Per-Node Gradient Accumulation

Because the graph is large but each batch only activates a subset of nodes, most nodes receive negligible gradients on any given step. The optimizer addresses this with per-node accumulation:

1. After `loss.backward()`, nodes with gradient norm > 1e-10 are identified as "active".
2. Their gradients are added to a per-node accumulation buffer.
3. When a node's count reaches `accumulation_steps`, the averaged gradient is applied via Adam, and the buffer is reset.

**Choosing `accumulation_steps`:** This is typically set equal to the batch size. Smaller values mean more frequent updates (faster but noisier); larger values mean smoother gradients but slower convergence. For small datasets (Iris), 4 works well. For larger datasets (ImageNet subsets), match it to your batch size.

---

## Differences from the Distributed Module

The `native/` module replaces the older distributed (`core/`) module. Key differences:

| Feature | `core/` (distributed) | `native/` |
|---------|----------------------|-----------|
| Execution | Multi-process workers + Qdrant | Single-process, no external services |
| Weight storage | Shared memory + Qdrant vector DB | `nn.Parameter` tensors |
| Quantization | Optional lookup-table quantization | Continuous float32 (no quantization) |
| Similarity search | Qdrant HNSW index | On-the-fly cosine similarity (`SimpleCosineSearch`) |
| Gradient flow | Custom `autograd.Function` | Standard PyTorch autograd (clone without detach) |
| Batching | One sample per worker | Full batch processed as mega-graph |
| Optimizer | `GNNAdam` / `GradientAccumulator` | `NativeGNNOptimizer` with `_GradAccumulator` |
| Checkpointing | Manual weight sync | `save_full_model` / `load_full_model` |
