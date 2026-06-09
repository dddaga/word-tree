# Readout

**Mechanism:** Final aggregation layer. Converts SGNNET per-neuron activations Z (shape N×D) to class logits.

## Base Architecture Readout

Base SGNNET uses class-selective readout:
```
logits = (Z @ W_pos.T) → class_head(C_ho)
```
`C_ho` selects activations for class-representative neurons via learned `W_pos` dot-product. **Required** readout when activations on unit sphere.

## Critical Finding: Global Mean-Pool Failure (step149)

**Bug:** Replacing `C_ho` readout with global mean-pool `fc_out(Z.mean(dim=1))` collapses to near-random accuracy.

| Readout Type | Accuracy | Notes |
|---|---|---|
| C_ho (base, correct) | 82.22% | Ref confirmed after fix |
| Global mean-pool | ~12% | Near-random on FashionMNIST |

**Root cause:** SGNNET activations Z live on unit sphere (`F.normalize` per step). Mean-pooling across N neurons averages directional vectors → near-zero vector, no class signal. `fc_out` sees near-zero input → near-uniform logits → random predictions.

**Context:** Found during step149 (input de-squashification experiments). New model variant used `fc_out(Z.mean(dim=1))` for simplicity. Failure immediate and severe.

## Architectural Constraint [CONFIRMED, step149]

**Hard rule:** Any model with unit-sphere activations MUST use `C_ho` class-selective readout. Global mean-pool incompatible with `F.normalize()` routing.

Applies to:
- All current SGNNET variants
- Any future SGNNET-derived architecture using iterative sphere-normalized routing
- Any model where final activation space normalized (L2, unit sphere)

## When Global Mean-Pool Works

Valid when activations NOT normalized to unit sphere — e.g., ReLU activations, learned embeddings without sphere constraint. Failure specific to unit-sphere geometry.

## Compound Note

After fixing readout bug in step149, Config A (`multi_feat` K=4) still showed 41.89% vs Ref 82.22%. 40pp gap confounded: Config A is novel architecture without full Resonant+AH stack. Not fair ablation of readout alone.

## See Also

- [[normalization]] — `F.normalize()` per step creates unit-sphere constraint making `C_ho` required
- [[antihebbian]] — AH routing operates in same unit-sphere space; `W_pos` diversity maintained by AH