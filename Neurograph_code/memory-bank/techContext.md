# NeuroGraph Technical Context

## Technology Stack

### Core Dependencies
- **Python 3.x**: Primary implementation language
- **PyTorch**: Tensor operations and MNIST dataset loading
- **NumPy**: Numerical computations and array operations
- **Pandas**: Graph topology representation and manipulation
- **Scikit-learn**: PCA dimensionality reduction for MNIST
- **Matplotlib**: Training convergence visualization
- **YAML**: Configuration file management

### Key Libraries Usage
```python
# Core tensor operations (without autograd)
import torch
import torch.nn as nn

# Data processing
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

# Visualization and config
import matplotlib.pyplot as plt
import yaml
```

## Development Environment

### Project Structure
```
neurograph/
├── main.py                 # Entry point and evaluation
├── config/
│   ├── default.yaml        # Hyperparameter configuration
│   └── static_graph.pkl    # Pre-generated graph topology
├── core/                   # Core computation engine
│   ├── cell.py            # PhaseCell signal processing
│   ├── propagation.py     # Single-step propagation
│   ├── forward_engine.py  # Multi-timestep coordinator
│   ├── backward.py        # Manual gradient updates
│   ├── node_store.py      # Parameter storage
│   ├── tables.py          # Lookup table implementations
│   ├── activation_table.py # Temporal signal tracking
│   ├── graph.py           # Graph generation utilities
│   └── radiation.py       # Dynamic neighbor selection
├── modules/               # Input/output interfaces
│   ├── input_adapters.py  # MNIST → graph conversion
│   ├── output_adapters.py # Graph → prediction conversion
│   ├── class_encoding.py  # Target vector generation
│   └── loss.py           # Loss function implementations
├── train/                 # Training orchestration
│   ├── train_context.py   # Main training loop
│   └── data_loader.py     # Data loading utilities
├── utils/                 # Utility functions
│   ├── config.py          # Configuration loading
│   └── ste.py            # Straight-through estimation
└── data/                  # Dataset storage
    └── MNIST/             # MNIST dataset files
```

## Implementation Patterns

### Configuration Management
```python
# utils/config.py
def load_config(path="config/default.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# Usage throughout codebase
cfg = load_config()
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Tensor Operations
```python
# Discrete index operations (no gradients)
phase_out = (ctx_phase_idx + self_phase_idx) % self.N
mag_out = (ctx_mag_idx + self_mag_idx) % self.M

# Lookup table transformations
cos_vals = self.lookup.lookup_phase(phase_out)
exp_vals = self.lookup.lookup_magnitude(mag_out)
signal = cos_vals * exp_vals
```

### Graph Representation
```python
# Pandas DataFrame for graph topology
graph_df = pd.DataFrame({
    'node_id': node_ids,
    'node_type': node_types,
    'input_connections': connection_lists
})

# Dictionary-based parameter storage
phase_table = {node_id: torch.randint(0, phase_bins, (vector_dim,)) 
               for node_id in all_nodes}
```

## Key Technical Constraints

### PyTorch Usage Limitations
- **No Autograd**: Manual gradient computation bypasses automatic differentiation
- **Tensor Operations**: Limited to basic tensor arithmetic and indexing
- **Device Management**: CPU/CUDA compatibility maintained for tensor operations
- **Data Loading**: Uses torchvision for MNIST but processes manually

### Memory Management
- **Lookup Tables**: Pre-computed cos/sin/exp tables stored in memory
- **Activation Tracking**: Temporal storage of active node states
- **Graph Storage**: Static topology loaded once and reused
- **Batch Processing**: Multiple samples merged to reduce memory overhead

### Computational Complexity
- **Forward Pass**: O(T × N × K) where T=timesteps, N=nodes, K=connections
- **Radiation Search**: O(N²) brute-force neighbor selection
- **Backward Pass**: O(output_nodes) parameter updates
- **Loss Computation**: O(batch_size × output_nodes × vector_dim)

## Development Setup

### Installation Requirements
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib pyyaml
```

### Configuration Parameters
```yaml
# Graph structure
total_nodes: 50
num_input_nodes: 5
num_output_nodes: 10
vector_dim: 5
phase_bins: 8
mag_bins: 256
cardinality: 3

# Training parameters
learning_rate: 0.001
num_epochs: 50
batch_size: 5
warmup_epochs: 5

# Propagation parameters
decay_factor: 0.925
max_timesteps: 6
top_k_neighbors: 4
use_radiation: true
```

### Execution Workflow
```bash
# Training and evaluation
python main.py

# Output: loss curve plot and accuracy metrics
# Logs saved to logs/ directory
```

## Technical Innovations

### 1. Discrete Lookup Tables
```python
class ExtendedLookupTableModule:
    def __init__(self, phase_bins, mag_bins):
        # Pre-compute trigonometric and exponential functions
        self.phase_table = torch.cos(torch.linspace(0, 2*π, phase_bins))
        self.mag_table = torch.exp(torch.linspace(-3, 3, mag_bins))
```

### 2. Hybrid Propagation Engine
```python
def propagate_step(active_nodes, node_store, graph_df, use_radiation=True):
    # Static connections from graph topology
    static_targets = get_static_connections(graph_df)
    
    # Dynamic connections via phase alignment
    if use_radiation:
        dynamic_targets = get_radiation_neighbors(...)
    
    return static_targets + dynamic_targets
```

### 3. Manual Gradient Computation
```python
def backward_pass(activation_table, node_store, target_context):
    # Analytical gradients from PhaseCell
    _, _, _, _, grad_phase, grad_mag = phase_cell(...)
    
    # Direct parameter updates
    new_phase = (old_phase - lr * grad_phase) % phase_bins
    new_mag = (old_mag - lr * grad_mag) % mag_bins
```

## Performance Characteristics

### Scalability Limits
- **Node Count**: ~50 nodes practical limit
- **Vector Dimension**: 5D vectors for computational efficiency
- **Batch Size**: 5 samples per batch for memory management
- **Training Time**: ~50 epochs for MNIST convergence

### Hardware Requirements
- **CPU**: Sufficient for current scale (50 nodes)
- **GPU**: Optional CUDA support for tensor operations
- **Memory**: ~100MB for lookup tables and graph storage
- **Storage**: Minimal requirements for MNIST dataset

## Future Technical Considerations

### Potential Optimizations
- **Vectorized Propagation**: Parallel processing across nodes
- **Sparse Representations**: Only store active parameters
- **Efficient Search**: Replace O(N²) radiation with approximate methods
- **Dynamic Graphs**: Runtime topology modification capabilities

### Scaling Challenges
- **Manual Gradients**: May not scale to larger architectures
- **Memory Usage**: Lookup tables grow quadratically with resolution
- **Computation Time**: Brute-force neighbor search becomes bottleneck
- **Graph Generation**: Static topology limits adaptability
