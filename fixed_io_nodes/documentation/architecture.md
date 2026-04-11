# Architecture — NativeNeurographLayer

## What it is

A sparse, iterative graph neural network. Nodes hold complex-valued state vectors (phase + magnitude). In each message-passing iteration, every node aggregates signals from its incoming neighbors via softmax-weighted routing, performs a complex multiplication with its own weight vector, and updates its state. After `iterations` steps, the activation strengths of the `output_nodes` are read out as class logits.

Implemented in `native/layer.py`. Graph structure lives in `native/node_store.py`.

---

## Node types

| Type | Count (run10 config) | Role |
|------|---------------------|------|
| Input | 3136 | Receive VGG16 conv features (512×7×7 = 25088 values → 3136 nodes × 8-dim vectors). Not updated by optimizer. |
| Intermediate | 12308 | Hidden processing nodes. All parameters are `nn.Parameter` and trained. |
| Output | 10 (or 256 with FFN) | Activation strengths read out as logits or fed into a linear head. |

Node IDs are contiguous: `[0 .. input_nodes-1]` = input, `[input_nodes .. total_nodes-output_nodes-1]` = intermediate, `[-output_nodes ..]` = output.

---

## Forward pass (layer.py)

```
x (B, input_nodes, vector_dim)
  │
  ▼
tanh(x) * π           # constrain inputs to phase range [-π, π]
  │
  ├─ initialise phase_act, mag_act from phase_weight, mag_weight (nn.Parameters)
  ├─ inject inputs into input node slots (mini message-pass from virtual sources)
  │
  └─ for iter in range(iterations - 1):
        build edge set (static edges from node_store + optional radiation edges)
        apply edge dropout (training only)
        replicate edges for batch (mega-graph: B*N nodes)
        grad_checkpoint( update_activations(...) )   # saves memory
        [optional temporal decay: mag_act -= log(temporal_decay)]
  │
  ▼
act_strength[output_idx]  →  (B, output_nodes)   # class logits (FFN-free)
```

The mega-graph trick: all B samples are processed as one graph with B×N nodes. Edges are identical across samples (added per-batch via offset). This avoids a Python loop over batch items.

---

## update_activations (core/custom_functions.py)

The core computation for one message-passing step.

**Inputs:** per-node phase activations `(N, V)`, mag activations `(N, V)`, phase/mag weights `(N, V)`, activation strengths `(N,)`, edge_index `(2, E)`, optional pre-computed weight trig.

**Step 1 — routing weights (softmax over incoming sources per destination):**
```
max_s       = scatter_max(act_strength[src], over dest)
exp_s       = exp((act_strength[src] - max_s[dest]) / temperature)
routing_w   = exp_s / scatter_sum(exp_s, over dest)   # (E,) ∈ [0,1]
```
`temperature` (default 1.0) scales the logits before exp. Higher T → softer, more uniform routing. See `documentation/gradient_flow.md`.

**Step 2 — weighted complex superposition:**
```
weighted_input_real = routing_w * exp(mag_src) * cos(phase_src)   # (E, V)
weighted_input_imag = routing_w * exp(mag_src) * sin(phase_src)
dest_real_in  = scatter_sum(weighted_input_real, over dest)        # (N, V)
dest_imag_in  = scatter_sum(weighted_input_imag, over dest)
```

**Step 3 — complex multiply with destination weight:**
```
dest_real_out = dest_real_in * cos(phase_w) - dest_imag_in * sin(phase_w)
dest_imag_out = dest_real_in * sin(phase_w) + dest_imag_in * cos(phase_w)
```

**Step 4 — new phase/mag activation:**
```
new_phase         = atan2(dest_imag_out, dest_real_out)
new_mag           = 0.5 * log(dest_real_out² + dest_imag_out²)   # log-magnitude
new_act_strength  = sum(dest_real_out, dim=V)                    # (N,)
```

`update_activations` no longer touches `new_mag`. When `model.layernorm: true`, the
`NativeNeurographLayer` forward closure applies `nn.LayerNorm` to `new_mag` immediately
after this step (post-update) and recomputes `act_strength` from the normalised mag:
```
new_mag          ← γ · (new_mag − μ) / σ + β              # learned γ, β per dim
new_act_strength = sum(exp(new_mag) · cos(new_phase), dim=V)
```
This replaces the legacy in-function `new_mag -= mean(new_mag, dim=V)` that runs 1–16 used
(hardcoded γ=1, β=0, no learned freedom). See:
`vgg_training/learnings/concepts/mag_normalization.md`.

---

## activation_strength_forward

```python
energy = exp(mag)          # (N, V)
real   = energy * cos(phase)
return real.sum(dim=-1)    # (N,) — signed scalar
```

This is `sum_v(exp(mag_v) * cos(phase_v))`. It can be negative (cos ranges [-1,1]) and has no saturation, making it usable directly as a class logit in the FFN-free setting.

---

## NativeNodeStore (native/node_store.py)

Holds `nn.Parameter` tensors `phase_weight (N, V)` and `mag_weight (N, V)`, the static edge buffer `edge_indices (2, E)`, and the graph topology.

**Initialization:**
- Node IDs 0..N-1 assigned deterministically from `seed`.
- Edges: each non-input node draws `uniform(1, cardinality)` incoming sources.
  - `flat` topology: sources drawn from all non-output, non-self nodes.
  - `layered` topology: output nodes draw from intermediates only; intermediates draw from all non-self.
- Self-loops added for every node.
- Weights initialized: `phase ~ Uniform(0, 2π)`, `mag ~ Normal(1.0, 0.1)`.

---

## NativeGNNOptimizer (native/optimizer.py)

Standard `torch.optim.Adam` cannot be used directly because the GNN has `N×V` parameters but only a sparse subset receives gradient each batch. Using Adam globally would give wrong second-moment estimates for unvisited nodes.

**Solution:** per-node gradient accumulation with individual Adam state.

- `_GradAccumulator`: accumulates gradients for each node. When a node has received `accumulation_steps` gradient contributions, it fires an Adam update on just that node's row.
- Adam state (`exp_avg`, `exp_avg_sq`, `state_steps`) stored as `(N, V)` tensors; updates applied to active rows only.
- Head parameters (linear layers, if any) use standard `torch.optim.Adam`.

**param_groups layout:**
- `[0]` = head parameters (Adam, every batch)
- `[1]` = GNN parameters (`phase_weight`, `mag_weight`, via accumulator)

Scheduler (`ReduceLROnPlateau`) acts on `param_groups[0]["lr"]`, which is shared with the head optimizer via reference — both see the LR change simultaneously.

---

## Config reference

All parameters live under three top-level keys: `graph`, `model`, `training`, `system`.

### graph

| Parameter | Type | Description |
|-----------|------|-------------|
| `total_nodes` | int | Total node count (input + intermediate + output). |
| `input_nodes` | int | Must satisfy `input_nodes × vector_dim = 25088` (VGG16 feature size). |
| `output_nodes` | int | 10 for FFN-free (direct logits), 256 for FFN-assisted runs (historical). |
| `cardinality` | int | Max incoming edges per node (actual count drawn from `uniform(1, cardinality)`). |
| `radiation_targets` | int | Number of dynamically searched/random edges added per active node per iteration. 0 = disabled. |
| `topology` | str | `"flat"` (default) or `"layered"`. See `learnings/concepts/topology.md`. |

### model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vector_dim` | int | — | Node state dimensionality. |
| `iterations` | int | — | Message-passing steps per forward pass. Forward runs `iterations-1` update loops. |
| `layernorm` | bool | false | Apply LayerNorm to mag_act before each `update_activations` call. **CONFIRMED helpful** (run3). |
| `dropout` | float | 0.0 | Edge dropout probability during training. Self-loops always kept. |
| `temporal_decay` | float | 1.0 | Per-iteration log-magnitude decay: `mag_act -= log(temporal_decay)`. 1.0 = no decay. |
| `routing_temperature` | float | 1.0 | Softmax temperature for routing weights. >1.0 = softer routing, more uniform gradient. See `documentation/gradient_flow.md`. |
| `beam_width` | int | 0 | If >0, only the top-K active nodes by act_strength are used as edge sources each iteration. 0 = all nodes. |
| `radiation_similarity_threshold` | float | 0.0 | Cosine similarity threshold for searched radiation targets. 0.0 = no threshold. |
| `scattering_prob` | float | 0.0 | Fraction of radiation edges that are random (vs cosine-searched). |
| `stochastic_radiation_duration` | float | 0.5 | Fraction of total training steps during which stochastic radiation is active (decays from `scattering_prob` to 0). |
| `temporal_decay` | float | 1.0 | Multiplicative decay applied to magnitudes each iteration. |
| `gamma` | float | 1.0 | Unused scaling param in `activation_strength_forward` (kept for call-site compatibility). |

### training

| Parameter | Type | Description |
|-----------|------|-------------|
| `lr` | float | Initial learning rate for Adam. |
| `batch_size` | int | Samples per gradient step. |
| `accumulation_steps` | int | How many gradient contributions a GNN node must accumulate before Adam fires. |
| `epochs` | int | Total training epochs. |
| `lr_decay_factor` | float | Factor for `ReduceLROnPlateau`. |
| `plateau_patience` | int | Patience for LR scheduler. |
| `min_lr` | float | Minimum LR floor. |

### system

| Parameter | Type | Description |
|-----------|------|-------------|
| `device` | str | `"mps"` (Mac Mini), `"cuda"`, or `"cpu"`. |
| `random_seed` | int | Seed for graph initialization and data shuffling. |
| `weights_save_path` | str | Checkpoint path (relative to `vgg_training/`). |
| `scheduler_save_path` | str | LR scheduler state path. |
| `tensorboard_dir` | str | TensorBoard log directory. |

---

## Cross-references

- `core/custom_functions.py` — `update_activations` and `activation_strength_forward` source
- `native/layer.py` — full forward pass
- `native/node_store.py` — graph init and edge building
- `native/optimizer.py` — per-node Adam
- `documentation/gradient_flow.md` — routing temperature, gradient starvation diagnosis
- `vgg_training/learnings/concepts/architecture.md` — experiment-oriented notes (run history, findings)
