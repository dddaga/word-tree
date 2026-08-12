# Readout

**Mechanism:** Final aggregation layer that converts SGNNET's per-neuron activations Z (shape N×D) to class logits.

## Base Architecture Readout

The base SGNNET uses a class-selective readout:
```
logits = (Z @ W_pos.T) → class_head(C_ho)
```
Where C_ho selects activations for class-representative neurons using learned W_pos dot-product. This is the **required** readout when activations are on the unit sphere.

## Critical Finding: Global Mean-Pool Failure (step149)

**Bug discovered:** Replacing the C_ho readout with global mean-pool `fc_out(Z.mean(dim=1))` collapses to near-random accuracy.

| Readout Type | Accuracy | Notes |
|---|---|---|
| C_ho (base, correct) | 82.22% | Ref confirmed after fix |
| Global mean-pool | ~12% | Near-random on FashionMNIST |

**Root cause:** SGNNET activations Z live on the unit sphere (F.normalize per step). Mean-pooling across N neurons averages directional vectors → the result is a near-zero vector with no class signal. fc_out then sees near-zero input → near-uniform logits → random predictions.

**Context:** This bug was discovered during step149 (input de-squashification experiments). A new model variant used `fc_out(Z.mean(dim=1))` for simplicity. The failure was immediate and severe.

## Architectural Constraint [CONFIRMED, step149]

**Hard rule:** Any model with unit-sphere activations MUST use C_ho class-selective readout. Global mean-pool is incompatible with F.normalize() routing.

This applies to:
- All current SGNNET variants
- Any future SGNNET-derived architecture using iterative sphere-normalized routing
- Any model where the final activation space is normalized (L2, unit sphere)

## When Global Mean-Pool Works

Global mean-pool is valid when activations are NOT normalized to unit sphere — e.g., ReLU activations, learned embeddings without sphere constraint. The failure is specific to unit-sphere geometry.

## Compound Note

After fixing the readout bug in step149, Config A (multi_feat K=4) still showed 41.89% vs Ref 82.22%. This 40pp gap is confounded: Config A is a novel architecture without the full Resonant+AH stack. The comparison is not a fair ablation of the readout alone.

## See Also

- [[normalization]] — F.normalize() per step creates the unit-sphere constraint that makes C_ho required
- [[antihebbian]] — AH routing operates in the same unit-sphere space; W_pos diversity maintained by AH
