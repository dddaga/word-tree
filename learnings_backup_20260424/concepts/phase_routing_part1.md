# Phase Routing

## Concept

SGNNET uses Fourier encoding on S^{D-1} (the unit hypersphere in D dimensions). Phase routing extends this by splitting the activation vector Z into a magnitude channel and phase channels, then applying frequency-dependent phase shifts based on spatial distance between neurons. The goal: input-dependent signal propagation where routing decisions emerge from phase coherence rather than fixed topology.

Core idea: `Z[0]` = activation magnitude (scalar energy), `Z[1:D-1]` = cyclic phase channels. When activation travels from neuron A to neuron B, each phase channel k receives a shift `delta_phi_k = freq_k * ||W_pos[A] - W_pos[B]||`. Multi-frequency interference means the same distance produces D-1 different phase shifts across channels. Neurons that are phase-coherent after shifting constructively interfere; incoherent neurons destructively interfere.

Biological analog: gamma oscillations (high-frequency, local) vs theta oscillations (low-frequency, long-range).

---

## Architecture

### Z Split

- `Z[:,:,0]` = magnitude dimension. Normalized across N neurons (dim=N), not across D.
- `Z[:,:,1:D]` = phase channels. No normalization. Modulo arithmetic after each routing step.

### Phase Shift Mechanics

```
delta_phi_k = freq_k * dist(W_pos[h], W_pos[j])
Z_phase_new = torch.remainder(Z_phase + delta_phi, 2*pi) - pi
```

Gradient of `torch.remainder` is 1 everywhere. Differentiable by construction.

### Phase Range: [-pi, pi]

Chosen over [0, 2*pi]. Rationale: [0, 2*pi] creates a positive bias (~622 baseline for 63 channels). [-pi, pi] gives zero mean baseline: uncorrelated neurons score ~0, aligned score positive, anti-aligned score negative. Single operation: `torch.remainder(Z_phase + delta_phi, 2*pi) - pi`.

### Resonance Metric

```
resonance(A, B) = Z_A[0] * Z_B[0] * sum_k(Z_A[k] * Z_B[k])
```

For fixed source A ranking targets {B_i}: Z_A terms are constant. Ranking is monotonic regardless of phase scale. No normalization needed.

### Magnitude Modes (step60 variants)

| Mode | Formula | Behavior |
|------|---------|----------|
| independent | `Z_mag_new = sum(Z_mag_nb)` | Simple sum, no phase interaction |
| coherent | `gate = cos(arrived - current).mean(-1)`, `Z_mag_new = sum(gate * Z_mag_nb)` | Constructive/destructive interference |
| mag-weighted (B_wt) | coherent gate + magnitude-weighted phase update | Strong activations dominate combined phase |
| decay_coherence | `gate = exp(-lambda * cycles) * coherence` | High-freq attenuates over distance |
| decay_weighted | Same gate + decay-weighted phase update | Full wave propagation model |

### Coherence Variants (step60)

| Variant | Reference | Behavior |
|---------|-----------|----------|
| dynamic-ref | `cos(Z_phase_arrived - Z_phase_h)` | Relative to h's current state (synchronization gradient) |
| anchor-ref | `cos(Z_phase_arrived - W_phase_h)` | Relative to h's static learned anchor (more stable) |
| absolute-ref | `cos(Z_phase_arrived)` | Relative to real axis. Expected to collapse to noise. |

### Decay Mechanism

```
decay_k(j->h) = exp(-lambda * freq_k * dist(j,h) / (2*pi))
```

High-frequency channels attenuate over short distances (local information). Low-frequency channels persist over long distances (global information). Precomputed from existing `delta_phi` tensor; no extra cost.

---

## Experiments and Results

### Direct Phase Routing Experiments

| Step | Mechanism | Configs | Best | vs Ref | Verdict |
|------|-----------|---------|------|--------|---------|
| step60 | Phase-distance routing (magnitude modes + coherence variants) | 13 (Ref + 12 variants) | Ref=73.55% | All 12 others: 10-19% (-55 to -63pp) | KILLED |
| step65 | Distance-phase routing (exp(-gamma*d) on conn_hh) | 7 (Ref + A-F) | A-F: 48-49% | -25pp vs 73.53% REF | KILLED |
| step66 | Phase-target plasticity (W_pos=Key/Value, phase_target=Query) | 5 (Ref + A-D) | Ref=83.18% | A=-15pp, B=-26pp, C=-22pp, D=-43pp | KILLED |

### step60 Full Results

| Config | Description | top1_best | vs Ref (73.55%) |
|--------|-------------|-----------|-----------------|
| Ref | AntiHebb alpha=1.0 static | 73.55% | -- |
| A | independent, no AH, freq=capped_exp | 14.27% | -59pp |
| B | coherent dynamic-ref, no AH, freq=capped_exp | 14.27% | -59pp |
| C | independent, AH=1.0, freq=capped_exp | 18.73% | -55pp |
| D | coherent dynamic-ref, AH=1.0, freq=capped_exp | 13.30% | -60pp |
| A_wt | independent + mag-weighted-phase, no AH | 13.20% | -60pp |
| B_wt | coherent + mag-weighted-phase, no AH | 14.37% | -59pp |
| B_anc | coherent anchor-ref, no AH | 14.14% | -59pp |
| B_abs | coherent absolute-ref, no AH | 14.22% | -59pp |
| E | decay_coherence, no AH, lambda=1.0 | 12.05% | -61pp |
| F | decay_coherence, AH=1.0, lambda=1.0 | 11.92% | -62pp |
| A_prm | independent, harmonic-primes, no AH | 10.80% | -63pp |
| B_prm | coherent, harmonic-primes, no AH | 10.68% | -63pp |

### Phase-Adjacent Experiments

| Step | Mechanism | Best | vs Ref | Verdict |
|------|-----------|------|--------|---------|
| step51 | W_phase spatial gating (phase cos gate on connections) | ~20% | -63pp | KILLED (gate-death: AH + W_phase gate = no signal) |
| step58 | Resonance excitatory (activation gate on Z coherence) | ~55% | -18pp | KILLED |
| step79 | Phase coherence aux loss (lambda 0.01/0.001/0.0001) | 95.54-95.85% | -0.25 to -0.56pp vs 96.10% Ref | KILLED |
| step39 | Mechanism-aware aux loss (phase coherence + inhibition sparsity) | 47.44% | -4.41pp vs 51.85% Ref | KILLED |

### Softmax Redistribution (Phase-Routing Revival)

| Step | Mechanism | Best | vs Ref | Verdict |
|------|-----------|------|--------|---------|
| step73 | Softmax redistribution routing (dot-product score) | D=86.34% | +1.78pp vs 84.56% Ref | WINNER |

step73 config details:

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | AH sum routing (step69 Ref) | 84.56% |
| A | softmax(Z_dot / tau=1.0) -- Z-state only, no AH | 55.77% (collapsed) |
| B | softmax(Z_dot / tau=1.0 + AH_logit) -- Z-state + AH | 84.87% (+0.31pp) |
| C | softmax(AH_logit) -- AH-only softmax redistribution | 85.58% (+1.02pp) |
| D | softmax(Z_dot / tau=0.3 + AH_logit) -- sharper redistribution | **86.34%** (+1.78pp) |

---

## Why It Failed (Wave-1)

Three compounding failure modes destroyed all wave-1 phase routing experiments:

### 1. Gate Death

Every approach added a multiplicative gate g in [0,1] applied per routing step. With K_iter=8:

```
signal ~ product(g_k) for k=1..K_iter
g=0.5 per step -> 0.5^8 = 0.004x original signal
gradient: dL/dZ_0 = product(g_k) * dL/dZ_K ~ 0.004 * dL/dZ_K
```

250x gradient attenuation. The network learns to null the gate (g->0) and route through the residual path only.

### 2. Float32 Overflow

`freq_k = 2^(k//2)` reaches 2^31 at k=62. At D=64, approximately 17 of 63 phase channels are pure numerical noise in float32. Fix: cap at 2^10=1024 via `freq_mode="capped_exp"`. But even with the fix, step60 configs C and D (capped_exp + AH) still died at 13-19%.

### 3. Double Sparsity

AntiHebb removes redundant paths. The phase gate removes coherence-mismatched paths. What survives is near-zero signal. AH already solves diversity/sparsity optimally. Adding a second sparsity mechanism on top creates a compounding signal loss that no hyperparameter tuning can recover from.

step66 Config D (phase-target + AH) = 40.33% demonstrated the worst case: AH suppresses exactly the phase activity that phase-target routing depends on. The two mechanisms are antagonistic.

---


*Continued in [phase_routing_part2.md](phase_routing_part2.md) — Revival via Redistribution, Group-Level Phase Routing, Technical Details, Dependency Chain, See Also.*
