# Gradient Flow & Routing Temperature

## Problem: gradient starvation in the FFN-free setting

**Context:** Runs 10+ remove the `nn.Linear` FFN head. The GNN classifies directly using the activation strength of 10 output nodes as logits. This is correct for the project goal, but it exposes a gradient flow problem that the FFN was previously masking.

**Observed symptom:** `diagnose.ipynb` "Distribution of change of weights" graph (phase 3) shows near-zero weight deltas in intermediate and input nodes throughout training, even while val accuracy slowly improves. The GNN is learning, but only the nodes close to the loss signal are updating meaningfully.

**Quantified by `vgg_training/gradient_starvation_analysis.py` (2026-04-09, run10 checkpoint):**

| Node type | Mean grad norm | Median grad norm | Starved (< 1e-6) |
|-----------|---------------|-----------------|-----------------|
| Output (hop 0) | 2.1e-1 | 1.2e-1 | 0% |
| Hop-1 intermediates (963 nodes, 7.8%) | 4.4e-2 | 2.9e-4 | 7.7% |
| Hop-2 intermediates (11324 nodes, 92%) | 1.1e-2 | 6.4e-5 | 13.3% |
| Input nodes | 5.5e-2 | 7.6e-3 | 0% |

Gradient concentration: **0.4% of all nodes carry 50% of total gradient signal**.

---

## Root cause: softmax routing concentration

`update_activations` computes routing weights as softmax over incoming activation strengths:

```
routing_weight = softmax(act_strength[sources] for each destination)
```

With `act_strength` standard deviation ≈ 3–4 (measured at every iteration), a few nodes with high activation strength get routing weight ≈ 1, and the rest get ≈ 0. In the backward pass, only the "winning" source for each destination receives gradient. The others receive none.

This is a winner-take-all dynamic that compounds over 4 message-passing iterations.

**Why didn't this hurt runs 1–9?** The `nn.Linear` FFN head had its own gradient signal directly to its weight matrix, allowing fast convergence even if GNN intermediate nodes were starved. In the FFN-free setting, all convergence depends on GNN gradients reaching all nodes.

---

## Fix: routing temperature (added 2026-04-09)

### What it does

Adds a `temperature` parameter to `update_activations` in `core/custom_functions.py`:

```python
exp_source_act_strength = torch.exp(
    (source_activation_strengths - max_act_strength[dest]) / temperature
)
```

At `temperature = 1.0` (default), behaviour is identical to before — fully backward compatible.

At `temperature = 2.0`, the logits before exp are halved. This makes the softmax softer: routing weights become more uniform across incoming sources. More sources receive non-zero routing weight → more nodes receive non-zero gradient per step.

### Config parameter

```yaml
model:
  routing_temperature: 2.0   # default 1.0
```

Set in `NativeNeurographLayer.__init__` via:
```python
self._routing_temperature = model.get("routing_temperature", 1.0)
```

Passed into every `update_activations` call via the `_normed_update` / `_default_update` closures in `native/layer.py`.

### Extending to annealing

The temperature is read once at the start of `forward()` from `self._routing_temperature`. To anneal it during training:

```python
# In your training loop, before each forward pass:
model.gnn._routing_temperature = current_temperature(step, total_steps)
```

No code change needed in `layer.py` or `custom_functions.py`. The comment in `layer.py` documents this extension point.

### Experiment: run12

Run12 is a clean single-variable ablation of temperature:
- **Base:** run10 (no FFN, all other config identical)
- **Change:** `routing_temperature: 2.0`
- **Expected:** faster convergence than run10 at same epoch count

If run12 val accuracy at epoch 20 is significantly higher than run10 at epoch 20, temperature scaling is confirmed as a key lever.

---

## Diagnostic tool: gradient_starvation_analysis.py

`vgg_training/gradient_starvation_analysis.py` — standalone script, loads any run checkpoint and reports:

1. Gradient norms by node type (input / intermediate / output)
2. Gradient norms by BFS hop distance from output nodes (how many reverse-edge hops from the loss signal)
3. Activation strength distribution per message-passing iteration
4. Gradient concentration index (% of nodes carrying 50%/90% of signal)

**Usage:**
```bash
cd /Volumes/T9/IndraAstra/sudarshan/word-tree/fixed_io_nodes/vgg_training
/Volumes/T9/IndraAstra/sudarshan/.venv/bin/python gradient_starvation_analysis.py
```

To analyse a different run, change `CFG_PATH` and `WEIGHTS_PATH` at the top of the file.

Runs on CPU (forces `device: cpu` regardless of config) — no GPU required.

---

## Cross-references

- `core/custom_functions.py` — `temperature` param in `update_activations`
- `native/layer.py` — `self._routing_temperature`, `_normed_update` / `_default_update` closures
- `vgg_training/gradient_starvation_analysis.py` — diagnostic script
- `vgg_training/diagnose.ipynb` — weight delta and activation visualisation
- `vgg_training/learnings/concepts/gradient_starvation.md` — experiment-level findings and run history
- `vgg_training/learnings/EXPERIMENT_QUEUE.md` — run12 entry
