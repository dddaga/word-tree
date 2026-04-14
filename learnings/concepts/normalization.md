# Normalization

**Mechanism:** Per-step output normalization after iterative routing in SGNNET. Controls what information is preserved at each K_iter step.

## Context

SGNNET's base architecture uses `F.normalize(Z, dim=-1)` (L2 sphere norm) at the end of each routing step, projecting activations onto S^{D-1}. This was the original design for maintaining unit-sphere geometry throughout K_iter iterations.

## Results

### step116 — Normalization Ablation (N=1024, D=16, 75ep Tier-1) [CONFIRMED]

| Config | Mechanism | Result | Delta | Verdict |
|--------|-----------|--------|-------|---------|
| Ref | L2 sphere norm (F.normalize) | 82.32% | — | Baseline |
| A | RMSNorm | 70.83% | −11.49pp | KILLED |
| B | pre_route normalize | 75.82% | −6.50pp | KILLED |
| C | LayerNorm (learned affine scale+shift) | **84.56%** | **+2.24pp** | **WINNER** |

**Status: Pending N=4096 D=64 validation before adopting as default.**

## Why LayerNorm Wins

Hard L2 sphere norm discards magnitude information at every step — only direction is preserved. LayerNorm with learned affine (scale γ, shift β per dimension) lets the network:
- Modulate per-dimension scale (routing can emphasize certain Fourier components)
- Preserve magnitude signal across steps
- Break symmetry through learned bias (RMSNorm lacks this, contributing to its −11pp loss)

## Why RMSNorm Fails Most

No learned bias → can't break dimensional symmetry. RMSNorm is strictly a rescaling; without shift, it cannot differentiate dimensions that should carry different signal. The −11.49pp loss is the largest of all normalization variants tested.

## Interaction with AH

The base L2 sphere norm was a key ingredient in AH's signal conservation property: F.normalize() at end of each step restores magnitude, preventing multiplicative decay that kills gating mechanisms ([[gate_death]]). If LayerNorm replaces F.normalize(), the AH signal conservation argument needs re-validation at N=4096. This is why N=4096 validation is required before adopting LayerNorm as default.

## Architectural Constraint: C_ho Readout

A related normalization finding from step149: when activations are on the unit sphere, global mean-pool readout fails catastrophically (12% accuracy). See [[readout]] for the C_ho requirement.

## Open

- [ ] N=4096 D=64 validation of LayerNorm winner (step116 finding is at N=1024 D=16 only)
- [ ] Verify AH signal conservation with LayerNorm (no longer enforcing unit-sphere per step)

## See Also

- [[antihebbian]] — signal conservation depends on F.normalize(); LayerNorm changes this
- [[readout]] — unit-sphere activations require C_ho readout, not global mean-pool
