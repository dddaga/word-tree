# Phase 5 Part 16 — Normalization, Diagnostics, Structural Constraints
**Date:** 2026-04-10

---

## step116 — Normalization Ablation (N=1024, D=16, 75ep Tier-1)

| Config | Mechanism | Result | Delta | Verdict |
|--------|-----------|--------|-------|---------|
| Ref | L2 sphere norm (F.normalize) | 82.32% | — | Baseline |
| A | RMSNorm | 70.83% | −11.49pp | KILLED |
| B | pre_route normalize | 75.82% | −6.50pp | KILLED |
| C | LayerNorm (learned affine) | **84.56%** | **+2.24pp** | **WINNER** |

**Interpretation:** Learned affine scale+shift (LayerNorm) beats hard L2 sphere projection. Hard norm discards magnitude signal every step; LayerNorm lets network modulate scale per-dimension. RMSNorm drops most — no learned bias, can't break symmetry between dimensions.

Status: CONFIRMED at N=1024 D=16. **Pending N=4096 D=64 validation** before adopting as default.

---

## step155 — Diagnostics Baseline (20ep)

| Config | N | D | K_hh | K_iter | acc@20ep | eff_rank | neuron_util | wpos_cos | grad_theta | grad_wpos |
|--------|---|---|------|--------|----------|----------|-------------|----------|------------|-----------|
| A (best arch) | 4096 | 64 | 4 | 12 | 95.06% | — | 100% | 0.75 stable | 0.997 | 0.07 |
| B | 1024 | 16 | 8 | 8 | 69.25% | 4.6→rising | 100% | — | — | — |
| D (minimal) | 256 | 16 | 4 | 4 | 47.08% | 9.5/16 | 100% | — | — | — |
| E (AH=0 control) | 1024 | 16 | — | — | 44.92% | 5.2→collapse | 100% | — | — | — |

**Key findings:**

1. **AH is anti-collapse mechanism** [CONFIRMED clean ablation, E vs B]: Without AH, `eff_rank` starts 5.2 and collapses. With AH, starts 4.6 and *increases* during training. AH enforces dimensional diversity — first direct mechanistic confirmation.

2. **W_pos barely learns at N=4096** [CONFIRMED]: `grad_theta`/`grad_wpos` = 14:1. `wpos_norm_mean` = 4.609, static through 20ep. W_pos converges early then freezes — all subsequent learning in θ and `fc_out`. Implication: W_pos init matters; tuning W_pos LR unlikely to help.

3. **Small N uses dimensions more efficiently**: Config D (N=256) `eff_rank`=9.5/16 vs Config B (N=1024) `eff_rank`=4.6/16. Smaller N forces each dimension to carry more signal.

4. **Neuron utilization = 100% across all configs**: No dead neurons. Healthy baseline. Pruning methods (step153) that drop neurons kill non-dead structure.

---

## step152 — Constraint Discovery (N=1024, 75ep)

| Config | Mechanism | Result | Delta | Verdict |
|--------|-----------|--------|-------|---------|
| Ref | Baseline | 33.53% | — | Baseline |
| A | Nuclear norm regularization | 18.42% | −15.11pp | KILLED |
| B | Bottleneck D→8→D | 33.48% | −0.05pp | Neutral |
| C | Dim gate | 30.22% | −3.31pp | KILLED |
| D | L1 sparsity | 30.19% | −3.34pp | KILLED |
| E | Contrastive routing | 20.51% | −13.02pp | KILLED |
| F | Nuclear+bottleneck+dim_gate compound | 19.72% | −13.81pp | KILLED |

**Finding [CONFIRMED]:** Network self-organizes structural constraints. External imposition — rank constraints (nuclear norm), sparsity (L1), routing competition (contrastive), gating (dim_gate) — all destroy performance. Only neutral result: bottleneck (−0.05pp), truly neutral not beneficial.

**Interpretation:** SGNNET emergent structure already near information-theoretic optimum for given N/D/K. External constraints fight learned organization. Closes structural regularization direction.

---

## step153 — Progressive Capacity (N=1024, 75ep)

| Config | Mechanism | Result | Delta | Verdict |
|--------|-----------|--------|-------|---------|
| Ref | Fixed N=1024 D=16 | 32.82% | — | Baseline |
| A | Prune dims (GMP) | 32.20% | −0.62pp | Least harmful |
| B | Nested dropout | 18.06% | −14.76pp | KILLED |
| C | Matformer | 22.85% | −9.97pp | KILLED |
| D | Prune neurons | 20.36% | −12.46pp | KILLED |
| E | Prune both | 20.69% | −12.13pp | KILLED |

**Finding [CONFIRMED]:** All pruning-during-training methods fail catastrophically. Even least-harmful (prune dims GMP) neutral-to-negative. SGNNET needs stable connectivity to converge — routing paths co-adapt during training, removing them mid-training destroys learned structure.

Aligns with step155: 100% neuron utilization = no dead neurons to safely remove. Every neuron load-bearing.

---

## step149 — Input De-squashification Bug Post-Mortem

**Bug:** Global mean-pool readout `fc_out(Z.mean(dim=1))` collapses class signal when activations on unit sphere.

- Symptom: 12% accuracy (near-random on Imagenette)
- Root cause: unit-sphere activations Z direction-only. Mean-pooling across N neurons averages out directional signal → near-zero vector → `fc_out` sees noise.

**Fix:** Replaced with `C_ho` class-selective readout + `W_pos` dot-product (same as base model architecture).

**Verification:** Ref after fix = 82.22% (correct). Config A (`multi_feat` K=4) = 41.89% — still 40pp below Ref, but comparison confounded (Ref uses full Resonant+AH stack; Config A novel architecture without it).

**Lesson [CONFIRMED rule]:** Any model with unit-sphere activations MUST use `C_ho` readout, never global mean-pool. Hard architectural constraint for all future SGNNET variants.

---

## Synthesis — What These Results Tell Us About SGNNET

### Two Confirmed Architectural Insights

1. **LayerNorm > L2 sphere at N=1024 [CONFIRMED, pending N=4096]**: Learned affine norm beats hard L2 sphere by +2.24pp. Magnitude signal carries info that hard norm discards each step.

2. **AH prevents dimensional collapse [CONFIRMED]**: step155 E (AH=0) shows `eff_rank` collapsing; B (AH=1.0) shows rising. First direct mechanistic evidence for *why* AH works — not just that it does.

### Confirmed Dead Ends

- **External structural constraints** (step152): nuclear norm, L1, contrastive routing, dim gates — all KILLED. Network self-organizes; don't impose externally.
- **Pruning during training** (step153): stable connectivity required for convergence. Removing active neurons mid-training destroys co-adapted routing paths.

### Diagnostic Insight: W_pos Learning Dynamics

At N=4096, W_pos barely updates (grad ratio 14:1 theta:wpos). Converges fast, stays static. Means: (a) W_pos init quality matters more than continued learning, (b) Fourier init provides good-enough prior that AH locks in quickly, (c) experiments reshaping W_pos during training face uphill battle — already frozen.

### Readout Rule (Architectural Constraint)

`C_ho` class-selective readout required when activations on unit sphere. Global mean-pool destroys directional signal. Applies to all future SGNNET variants and any architecture using `F.normalize()` at iterative routing output.