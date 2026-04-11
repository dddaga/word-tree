# Concept: Gradient Starvation

## What it is

Gradient starvation occurs when backpropagation fails to deliver meaningful gradient signal to large portions of the graph's intermediate nodes. Affected nodes' weights barely update across thousands of batches, causing extremely slow convergence.

This became the active bottleneck after removing the FFN head (run10+). Per-run results in `training_runs/runN/notes.md`.

---

## Evidence

**diagnose.ipynb** (pre-run10): "Distribution of change of weights" graph (phase 3) showed near-zero weight deltas in intermediate and input node channels across the full training run of run6. While run6 converged to 85.81% val, the convergence was dominated by the output nodes and the nn.Linear head — the GNN internals were barely training.

**gradient_starvation_analysis.py** (2026-04-09, run10 checkpoint at ep13, ~30% val):

| Metric | Value |
|--------|-------|
| Intermediate mean grad norm | 2.6e-3 |
| Intermediate median grad norm | 2.6e-5 (100× less than mean → heavy skew) |
| % intermediates starved (< 1e-6) | 16.2% |
| Nodes carrying 50% of total gradient | 0.4% of all nodes |
| Nodes carrying 90% of total gradient | 7.0% of all nodes |
| Act_strength std (all iterations) | ~3–4 |

**Hop-distance breakdown** (BFS from output nodes via reverse edges):
- Hop 1: 963 nodes (7.8% of intermediates) — mean grad 4.4e-2
- Hop 2: 11,324 nodes (92%) — mean grad 1.1e-2 (20× less than output nodes)
- All nodes reachable within 3 hops (dense flat graph with cardinality=200)

Conclusion: the bottleneck is NOT graph distance (all nodes are reachable). It is **softmax routing concentration**.

---

## Root cause: winner-take-all softmax routing

In `update_activations` (`core/custom_functions.py`):

```
routing_weight[src→dst] = exp(act_strength[src] - max[dst]) / sum_over_sources
```

With act_strength std ≈ 3–4, the source with the highest activation strength for a given destination gets routing weight ≈ 1, all others ≈ 0. In the backward pass, only the "winning" source gets gradient — the rest receive none for that step.

This winner-take-all behaviour compounds over 4 message-passing iterations and across batches, leaving most intermediate nodes with effectively no gradient.

---

## Fix: routing temperature

Added `temperature` parameter to `update_activations` (default 1.0, backward compatible):

```python
exp_s = exp((act_strength[src] - max[dst]) / temperature)
```

Higher temperature → softer softmax → more uniform routing → more nodes receive gradient.

**Config parameter:** `model.routing_temperature` (float, default 1.0)

**Run12** tests temperature=2.0 as a clean one-variable ablation against run10.

For annealing: update `model.gnn._routing_temperature` each training step. No code changes needed.

---

## Results table

| Run | temperature | Val best | Epochs | Notes |
|-----|-------------|----------|--------|-------|
| run10 | 1.0 (default) | 40.33% (ep23, stopped) | DONE | Baseline FFN-free |
| run12 | 2.0 | PENDING | 40 | Temperature fix ablation vs run10 |

---

## Other approaches considered (not yet tried)

| Approach | Mechanism | Status |
|----------|-----------|--------|
| More iterations (5→8+) | Wider receptive field | NOT NEEDED — all nodes already within 3 hops |
| Auxiliary intermediate loss | Direct gradient injection bypassing routing | Possible run13+ |
| Higher cardinality (200→400) | More gradient paths per node | Possible run13+ |
| Mean aggregation (no softmax) | Uniform routing | More disruptive — routing loses semantic weighting |

---

## Cross-references

- `core/custom_functions.py` — temperature param in `update_activations`
- `native/layer.py` — `self._routing_temperature` and update closures
- `vgg_training/gradient_starvation_analysis.py` — full diagnostic script
- `vgg_training/diagnose.ipynb` — weight delta and activation visualisations
- `documentation/gradient_flow.md` — code-level reference for the temperature parameter
- `EXPERIMENT_QUEUE.md` — run12 entry
