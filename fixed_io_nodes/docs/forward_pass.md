# Forward Pass of NativeNeurographLayer

This document walks through the complete forward pass of the `NativeNeurographLayer` as implemented in `native/layer.py`, step by step.

---

## Overview

```
Input: (B, input_nodes, vector_dim) float tensor
                |
                v
        tanh(x) * pi                   [Phase mapping]
                |
                v
        Replicate weights               [Create mega-graph: B*N nodes]
        Initialize activations
                |
                v
        Input injection                  [Blend input into input nodes]
                |
                v
        Iteration 1..K-1:
            Build edges (static + radiation)
            update_activations           [Complex message passing]
            (gradient checkpointed)
                |
                v
        Extract output strengths         [activation_strength / sqrt(D)]
                |
                v
Output: (B, output_nodes) float tensor
```

---

## Step 1: Phase Mapping

```python
x = torch.tanh(x) * torch.pi
```

The input features (arbitrary floats) are squeezed into the range $(-\pi, \pi)$ via `tanh` scaling. This makes them interpretable as phase angles for the complex-valued graph computation.

---

## Step 2: Mega-Graph Construction

The layer processes all $B$ samples simultaneously by replicating the graph $B$ times into a single "mega-graph" with $B \times N$ nodes (where $N$ = `total_nodes`).

```python
phase_weight = self._node_store.phase_weight.clone().repeat(B, 1)   # (B*N, D)
mag_weight = self._node_store.mag_weight.clone().repeat(B, 1)       # (B*N, D)
```

`clone()` (without `detach()`) preserves the gradient connection to the original `nn.Parameter`, so backpropagation flows through to the learnable weights.

Activations are initialized from weights:
- `phase_act = phase_weight.clone()`
- `mag_act = phase_weight.clone()` (with optional LayerNorm)
- `act_strength = activation_strength_forward(phase_act, mag_act)` -- scalar per node

Weight trig values are precomputed once since they don't change across iterations:
```python
w_real = mag_weight * cos(phase_weight)
w_imag = mag_weight * sin(phase_weight)
```

---

## Step 3: Input Injection

Input injection is handled by `_inject_inputs_batched`. For each input node across all $B$ samples:

1. A **virtual source node** is created with:
   - Phase activation = the input value
   - Magnitude activation = zeros
   - Activation strength = computed from the existing node's activations

2. A mini edge set connects each virtual source to its corresponding input node.

3. The standard `update_activations` function runs on these edges, blending the input signal into the existing node state.

This is equivalent to treating external input as a message from a virtual neighbor, processed with the same complex arithmetic as all other messages.

---

## Step 4: Progressive Edge Expansion

Edges are not all active from the start. The layer tracks an `active_mask` over the $N$ single-sample nodes:

1. **Initially:** only input nodes are marked active.
2. **Each iteration:** outgoing edges from active nodes are followed, marking newly reached nodes as active.
3. **Once all nodes are active:** the full static edge set is used directly (no filtering needed).

This progressive activation mirrors how signals physically propagate through a network -- information can only reach nodes that are reachable from the inputs within the given number of iterations.

---

## Step 5: Radiation Edges (Dynamic Connectivity)

If `radiation_targets > 0`, additional edges are added at each iteration:

### Cosine-Searched Targets
Each active node uses its phase weight as a query to find similar nodes:
- Query: $[\cos\phi, -\sin\phi]$ (conjugate form)
- Index: $[\cos\phi, \sin\phi]$ for all nodes
- Top-k results by cosine similarity become edge targets
- Results below `radiation_similarity_threshold` are filtered out

### Random Targets (Scattering)
A fraction of radiation targets are chosen uniformly at random. The fraction (`scattering_prob`) starts at its configured value and decays exponentially to zero over the first `stochastic_radiation_duration` fraction of training.

The split is: `num_random = int(radiation_targets * scattering_prob)`, `num_searched = radiation_targets - num_random`.

---

## Step 6: Message Passing (update_activations)

This is the core computation, run at each iteration. See [maths.md](./maths.md) for the full mathematical details.

In brief, for each destination node $t$ with incoming edges from sources $\{s_1, ..., s_n\}$:

1. **Softmax routing**: Compute attention weights from source activation strengths
2. **Weighted superposition**: Sum source complex vectors weighted by attention, in Cartesian form
3. **Complex multiplication**: Multiply aggregated input by destination's weight vector
4. **Polar decomposition**: Extract new phase (via atan2) and magnitude (via log-norm)
5. **Magnitude centering**: Subtract mean log-magnitude to prevent drift
6. **New activation strength**: Real-part projection normalized by geometric mean

Each iteration is wrapped in `torch.utils.checkpoint.checkpoint` to save memory during backpropagation.

---

## Step 7: Edge Replication for Batching

Single-sample edges are replicated across the $B$ samples by adding node-index offsets:

```python
# For sample b, node i becomes node (b * N + i)
offsets = torch.arange(B) * N
batched_src = single_src + offsets.unsqueeze(1)   # broadcast
batched_dst = single_dst + offsets.unsqueeze(1)
```

This means all $B$ samples share identical graph topology but have independent activations.

---

## Step 8: Output Extraction

After all iterations complete, the activation strengths of the output nodes are gathered:

```python
batched_output_idx = (output_idx + offsets.unsqueeze(1)).reshape(-1)
output = act_strength[batched_output_idx].view(B, -1) / (V ** 0.5)
```

The division by $\sqrt{D}$ (`vector_dim`) normalizes the scale, similar to the scaling in attention mechanisms.

The result is a `(B, output_nodes)` tensor -- one scalar per output node per sample -- which is then typically passed to a linear classification head.
