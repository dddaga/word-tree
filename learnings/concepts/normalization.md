# Normalization

**Mechanism:** Per-step output normalization after iterative routing in SGNNET. Controls what info preserved at each K_iter step.

## Context

SGNNET base architecture uses `F.normalize(Z, dim=-1)` (L2 sphere norm) end of each routing step, projecting activations onto S^{D-1}. Original design for maintaining unit-sphere geometry throughout K_iter iterations.

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

Hard L2 sphere norm discards magnitude info every step — only direction preserved. LayerNorm with learned affine (scale γ, shift β per dimension) lets network:
- Modulate per-dimension scale (routing can emphasize certain Fourier components)
- Preserve magnitude signal across steps
- Break symmetry through learned bias (RMSNorm lacks this, contributing to −11pp loss)

## Why RMSNorm Fails Most

No learned bias → can't break dimensional symmetry. RMSNorm strictly rescaling; without shift, cannot differentiate dimensions that should carry different signal. −11.49pp loss largest of all normalization variants tested.

## Interaction with AH

Base L2 sphere norm key ingredient in AH signal conservation: `F.normalize()` end of each step restores magnitude, preventing multiplicative decay that kills gating mechanisms ([[gate_death]]). If LayerNorm replaces `F.normalize()`, AH signal conservation argument needs re-validation at N=4096. Why N=4096 validation required before adopting LayerNorm as default.

## Architectural Constraint: C_ho Readout

Related normalization finding from step149: when activations on unit sphere, global mean-pool readout fails catastrophically (12% accuracy). See [[readout]] for C_ho requirement.

## Open

- [ ] N=4096 D=64 validation of LayerNorm winner (step116 finding is at N=1024 D=16 only)
- [ ] Verify AH signal conservation with LayerNorm (no longer enforcing unit-sphere per step)

## See Also

- [[antihebbian]] — signal conservation depends on `F.normalize()`; LayerNorm changes this
- [[readout]] — unit-sphere activations require C_ho readout, not global mean-pool