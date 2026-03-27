# Architecture Overview

This document describes the module structure and component responsibilities of the `native/` implementation, which is the current active codebase for the Neurograph model.

---

## Module Map

```
native/
    __init__.py           Exports: NativeNeurographLayer, NativeNodeStore,
                                   NativeGNNOptimizer, save_full_model, load_full_model

    node_store.py         NativeNodeStore -- graph topology, weight storage, similarity search
    layer.py              NativeNeurographLayer -- forward pass, edge construction, batching
    optimizer.py          NativeGNNOptimizer + _GradAccumulator -- dual-path optimization
    checkpoint.py         save_full_model / load_full_model -- serialization

core/
    custom_functions.py   update_activations, activation_strength_forward -- core math kernels
    nodestore.py          SimpleCosineSearch -- cosine similarity search backend
    (other files)         Legacy distributed system (Qdrant-based, multi-process)
```

---

## Component Details

### NativeNodeStore (`native/node_store.py`)

The node store owns the graph's learnable parameters and topology.

**State:**
- `phase_weight`: `nn.Parameter(total_nodes, vector_dim)` -- phase angles per node
- `mag_weight`: `nn.Parameter(total_nodes, vector_dim)` -- log-magnitudes per node
- `edge_indices`: buffer `(2, num_edges)` -- static directed edges + self-loops
- `connections`: dict -- full adjacency list (incoming/outgoing per node)
- `input_nodeids`, `output_nodeids`: sets of designated I/O nodes

**Graph construction** (`_initialize_graph`):
- Deterministic from `seed` (saves/restores RNG state to avoid side effects)
- Input nodes: first `input_nodes` node IDs
- Output nodes: last `output_nodes` node IDs
- For each non-input node: randomly select 1 to `cardinality` incoming edges from non-output nodes
- Self-loops added for every node

**Similarity search** (`search_nodes_batch`):
- Used for radiation target discovery
- Converts phase weights to $[\cos\phi, \sin\phi]$ representation
- Uses conjugate query $[\cos\phi, -\sin\phi]$ so cosine similarity measures phase alignment
- Backed by `SimpleCosineSearch` (brute-force matrix multiply, no index structure)
- Search is non-differentiable (uses `.data` detached from autograd)

### NativeNeurographLayer (`native/layer.py`)

The GNN layer that implements the forward pass. See [forward_pass.md](./forward_pass.md) for the step-by-step walkthrough.

**Key design choices:**
- **Mega-graph batching**: All B samples processed simultaneously by replicating the graph B times. Edge construction happens once (topology is identical across samples), then edges are offset-replicated.
- **Gradient checkpointing**: Each iteration uses `torch.utils.checkpoint.checkpoint` to trade compute for memory.
- **Progressive activation**: Edges are expanded iteration-by-iteration from input nodes outward, until all nodes are reached, at which point the full static edge set is used.
- **Lazy index tensors**: Input/output index tensors are built on first use and cached, auto-migrating to the correct device.

**LayerNorm**: When `use_layer_norm=True` (default), `nn.LayerNorm(vector_dim)` is applied to magnitude activations before each message-passing step. The norm is applied inside the gradient checkpoint closure to avoid retaining intermediates.

### NativeGNNOptimizer (`native/optimizer.py`)

Subclasses `torch.optim.Optimizer` with a dual-path design:

**Path 1 -- Head parameters** (linear layers, classification head):
- Delegated to an internal `torch.optim.Adam` instance
- Updated every batch, standard behavior

**Path 2 -- GNN parameters** (phase_weight, mag_weight):
- Managed by `_GradAccumulator` instances (one per GNN layer)
- Gradients are accumulated per-node across batches
- Adam is applied only when a node reaches `accumulation_steps` gradient samples
- Each node has independent Adam state (step count, first/second moments)

**param_groups layout:**
- `param_groups[0]` = head parameters (shared dict reference with internal Adam)
- `param_groups[1]` = GNN parameters

This layout means LR schedulers modify the dict in-place and both paths see the updated LR.

### _GradAccumulator (`native/optimizer.py`)

Per-node gradient accumulation with vectorized Adam. For each node:

- Maintains separate phase/magnitude gradient buffers and counts
- Maintains separate Adam state (exp_avg, exp_avg_sq, step count)
- `receive_gradients()`: adds gradients for active nodes via `index_add_`
- `step()`: finds nodes that reached the threshold, applies averaged-gradient Adam, resets buffers

### Checkpoint (`native/checkpoint.py`)

**`save_full_model(model, path)`:**
- Saves `model.state_dict()` (all nn.Parameters)
- Also saves graph topology (connections, node IDs) for each NativeNeurographLayer

**`load_full_model(model, path)`:**
- Loads state dict with `strict=False` (allows partial loads)
- Restores graph topology and rebuilds edge indices

The optimizer state is not included in the model checkpoint -- it must be saved/loaded separately if resuming training.

### Core Math Functions (`core/custom_functions.py`)

**`activation_strength_forward(phases, mags, gamma)`:**

Computes activation strength as: $a = \sum_d \exp(\mu_d) \cos(\phi_d)$

This is the real-part projection of the complex signal, used for routing weights and as the final output.

**`update_activations(...)`:**

The message-passing kernel. Given phase/mag activations, weights, activation strengths, and an edge index:
1. Softmax routing weights from activation strengths
2. Weighted complex superposition of source signals
3. Complex multiplication with destination weights
4. Polar decomposition to new phase/mag activations
5. Magnitude mean-centering
6. New activation strength computation

Supports two modes:
- `all_destinations=True`: overwrites all nodes (used when full graph is active)
- `all_destinations=False`: only overwrites nodes that appear as edge destinations (used during progressive activation)

---

## Data Flow Diagram

```
                    ┌─────────────────┐
                    │  Config (YAML)  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼─────────┐       ┌──────────▼──────────┐
    │ NativeNodeStore    │       │ NativeNeurographLayer│
    │  - phase_weight    │◄──────│  - forward()         │
    │  - mag_weight      │       │  - edge construction │
    │  - edge_indices    │       │  - input injection   │
    │  - connections     │       │  - LayerNorm         │
    │  - search_nodes()  │       └──────────┬──────────┘
    └────────────────────┘                  │
                                            │ calls
                               ┌────────────▼────────────┐
                               │ core/custom_functions.py │
                               │  - update_activations()  │
                               │  - activation_strength() │
                               └──────────────────────────┘

    ┌────────────────────────┐
    │  NativeGNNOptimizer    │
    │   ├─ _head_optimizer   │──► torch.optim.Adam (linear layers)
    │   └─ _GradAccumulator  │──► per-node Adam (phase/mag weights)
    └────────────────────────┘

    ┌────────────────────────┐
    │  checkpoint.py         │
    │   - save_full_model()  │──► model.state_dict() + topology
    │   - load_full_model()  │
    └────────────────────────┘
```
