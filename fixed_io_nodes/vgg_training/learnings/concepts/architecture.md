# Concept: NativeNeurographLayer Architecture

## What it is
A sparse graph neural network layer that replaces VGG16's dense MLP FC layers (123M parameters) for imagenette-10 classification. Implemented in `native/layer.py`.

## Architecture overview
- **Input:** VGG16 conv features reshaped to `(batch, input_nodes, vector_dim)` — e.g. 512×7×7 = 25088 → 3136 nodes × 8-dim vectors
- **Graph:** `total_nodes` nodes (input + intermediate + output), sparse edges with `cardinality` edges per node
- **Iterative propagation:** `iterations` message-passing steps; each step: aggregate → activate → (optional) decay
- **Output (with FFN):** `output_nodes` × `vector_dim` → linear head → 10 classes
- **Output (FFN-free, run10+):** `output_nodes=10`, act_strength used directly as class logits

## Core config parameters

| Parameter | What it controls | Confirmed best (run6) |
|-----------|-----------------|----------------------|
| `total_nodes` | Graph size | 15454 |
| `input_nodes` | VGG feature nodes (must satisfy input_nodes × vector_dim = 25088) | 3136 |
| `output_nodes` | Nodes read out to classifier head | 256 (FFN era); 10 (FFN-free) |
| `cardinality` | Edges per node | 200 |
| `vector_dim` | Node state dimensionality | 8 |
| `iterations` | Message-passing steps per forward pass | 5 |
| `topology` | flat or layered | flat |
| `layernorm` | Normalize magnitudes after each iteration | true |
| `dropout` | Edge dropout probability | 0.2 |
| `temporal_decay` | Log-decay constant subtracted from mag_act each iteration (1.0 = no decay) | 1.0 |
| `beam_width` | Top-K active nodes as sources per iteration (0 = all) | 0 |
| `routing_temperature` | Softmax temperature for routing weights. >1.0 = softer/more uniform routing → more nodes receive gradient. 1.0 = standard softmax (default, backward compatible). See `concepts/gradient_starvation.md`. | 1.0 |

## Parameter budget vs VGG16 FC
VGG16 FC parameters: 512×7×7×4096 + 4096×4096 + 4096×1000 = ~123.6M.
NativeNeurographLayer total connections: total_nodes × cardinality = 15454 × 200 = ~3.1M edges (vectors, not scalars — actual param count depends on vector_dim).

## Regularization history
- **LayerNorm** (run3+): magnitude normalization per iteration. **CONFIRMED helpful** — clean run1 vs run3 ablation, +5.33pp (76.05% → 81.38%). Never disabled after introduction.
- **Radiation** (run4–5): dynamic stochastic edges. run4 collapsed at scattering_prob=0.8. run5 reduced to 0.25 but still underperformed run3 (71.46% vs 81.38% at fewer epochs). Removed in run6. HYPOTHESIS: hurts. No clean ablation isolating radiation from dropout.
- **Dropout** (run6+): edge dropout at 0.2. Added in run6 alongside radiation removal. HYPOTHESIS: helpful. Not individually confirmed — would need a run6 clone with dropout=0.

## Gradient starvation
Active bottleneck in FFN-free runs (run10+). Softmax routing concentration causes 0.4% of nodes to carry 50% of gradient signal. See `concepts/gradient_starvation.md` for root cause analysis and fix.

## Cross-references
- `native/layer.py` — core implementation
- `native/node_store.py` — graph construction + topology
- `core/custom_functions.py` — `activation_strength_forward`, `update_activations`
- `concepts/topology.md` — flat vs. layered topology findings
- `concepts/gradient_starvation.md` — gradient starvation root cause and fixes
- `EXPERIMENT_QUEUE.md` — run overview
