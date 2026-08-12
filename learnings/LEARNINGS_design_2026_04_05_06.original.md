# Design Discussions — 2026-04-05 to 2026-04-06

**Parent:** LEARNINGS_design.md (index)

---

## 2026-04-05 — Phase Routing Architecture Design

### 1. Phase-distance routing (SGNNET_PhaseRouting)
- Z split: `Z[0]` = magnitude scalar, `Z[1:63]` = phase channels
- Phase shift: `delta_phi_k = freq_k × dist(W_pos[h], W_pos[j])`, `freq_k = 2^(k//2)`
- Differentiable modulo: `torch.remainder(Z_phase + delta_phi, 2π) - π`
- **Float32 issue**: `freq_k = 2^(k//2)` reaches `2^31` at k=62 → ~17 of 63 channels pure noise. Fix: cap at `2^10=1024` (`freq_mode="capped_exp"`)

### 2. Resonance metric
- `resonance(A,B) = Z_A[0] × Z_B[0] × Σ_k Z_A[k] × Z_B[k]` (mag_product × phase_dot)
- Phase range `[-π, π]` chosen over `[0, 2π]`: zero-mean baseline, uncorrelated neurons ≈0.

### 3. Coherence calculation — three variants to ablate
- Current: `cos(Z_phase_arr - Z_phase_h).mean(-1)` — relative to h's current state
- Alternative A: `cos(Z_phase_arr - W_phase_h)` — relative to h's static learned anchor (more stable)
- Alternative B: `cos(Z_phase_arr)` — absolute reference, expected to collapse to noise
- Test all three as `coherence_ref` parameter in step60

### 4. Magnitude modes
- **Independent**: `Z_mag_new = Σ Z_mag_nb` — simple sum
- **Coherent**: `coherence = cos(arrived - current).mean(-1)`, `Z_mag_new = Σ coherence × Z_mag_nb`
- **B_wt**: coherent gate + magnitude-weighted phase update
- **decay_coherence**: gate = `exp(-λ × cycles) × coherence`, cycles = `delta_phi/(2π)`

### 5. Decay mechanism
- `decay_k(j→h) = exp(-λ × freq_k × dist(j,h) / (2π))`
- High-freq channels attenuate over short distances (local), low-freq channels persist (global)
- Biological analog: gamma oscillations (local) vs theta oscillations (long-range)

### 6. Prime number frequencies
- `freq_k = 2π / p_k` (k-th prime)
- CRT property: aliasing-free range = product of all primes ≈ 10^90
- Float32 safe: max delta_phi = π × 8 ≈ 25 radians
- Test as `freq_mode="harmonic_primes"` in step60

### 7. Hub interneurons (new design)
- Previous failure: step20/29c only gave K_hh=6 neighbours — no global signal possible
- New design: N_mix = 25% of neurons start blank, receive via `conn_mix [N_mix, fan_in=512]`
- Hubs broadcast through existing conn_hh — global mixing without dedicated inhibition
- Test in step61: fan_in=512 and fan_in=256, with/without AntiHebb

---

## 2026-04-06 — Dynamic Routing Failure Analysis + Next Direction

### Standing Commitment

Dynamic routing of input activations via parameter-efficient mechanisms remains an open problem.
The failures below do NOT close this direction — they reveal *where* prior designs went wrong.
Goal: routing decisions are **input-dependent** while remaining **O(N×K)** — no dense attention.

---

### Failed Experiments: Pattern Analysis

| Step | Mechanism | Best config | vs Ref (83.36%) | Failure mode |
|------|-----------|-------------|-----------------|--------------|
| step58 | Resonance excitatory (activation gate on Z coherence) | ~73% | −10pp | Gate kills signal in early routing steps |
| step59 | Active beam unified (top-K Z magnitude routing) | ~73% | −10pp | Discrete top-K breaks gradient |
| step60 | Phase routing magnitude (freq-shift interference + decay) | ~73% | −10pp | Coherence gate → near-zero signal after 2-3 steps |
| step61 | Hub interneurons (high fan-in mixing nodes) | ~72% | −11pp | Hub connectivity too sparse; AH suppresses hub→hidden |
| step63 | Activation-gated routing (act_gate × Z_nb) | ~72% | −11pp | Multiplicative gate collapses gradients |
| step65 | Distance-phase routing (Euclidean distance freq shift) | ~72% | −11pp | Phase shift accumulates → phase drift |
| step66 | Phase-target plasticity (Z alignment rewires conn_hh) | 83.18% (Ref only) | Any dynamic element: 57–68%, with AH: 40% | — |
| step51 | W_phase spatial gating (phase cos gate on connections) | ~20% | −63pp | Complete gate-death: AH + W_phase gate = no signal |

**Common failure mode:** Every approach adds a GATE that activations must satisfy to propagate.
Combined with AntiHebb's suppression, the result is double sparsity → near-zero signal → gradient collapse.

---

### Root Cause: Gating vs. Redistribution

**Gating** (what failed): `Z_new[h] = gate(Z_h, Z_j) × Z_struct[h]` where gate ∈ [0,1]
- Each routing step attenuates signal by the gate value
- After K=8 steps: if gate=0.7 per step → 0.7^8 ≈ 0.06× original signal

**Redistribution** (what works): `Z_new[h] = Σ_j w_j(Z) × Z_nb[h,j]` where Σ w_j = 1 (softmax)
- Input-dependent weights steer signal, but total incoming is conserved
- No attenuation per step → gradient flows cleanly through K iterations

---

### Candidate Directions for Next Experiments

**1. Softmax redistribution routing** (conserved flow)
- Replace `Z_nb.sum(dim=2)` with `(softmax(score(Z_h, Z_nb), dim=2) * Z_nb).sum(dim=2)`
- Score options: dot(Z_h, Z_j), dot(W_pos_h, W_pos_j), learned per-group temperature
- Key: weights sum to 1 over K_hh neighbors → no attenuation, fully differentiable
- → step73

**2. Input-modulated temperature** (soft routing intensity, not a gate)
- `temp_h = τ_0 × (1 + σ(W_temp × Z_h))` — per-neuron temperature from current activation
- `w_j = softmax(score_j / temp_h)` — hot neurons route sharply, cold neurons route diffusely
- Dynamic without adding gate: softmax still sums to 1
- → step75

**3. Routing by phase alignment with normalization**
- Core failure of step60: phase coherence gate multiplies signal
- Fix: use coherence as softmax weight (redistribution) not as multiplier (gating)
- `w_j = softmax(coherence(Z_h, Z_j) / τ)` instead of `gate = coherence(Z_h, Z_j)`

**4. Hebbian-routed connectivity update** (topology, not weights)
- Don't gate existing connections; instead periodically swap the weakest conn_hh edges
- AH handles suppression; Hebb handles convergence — orthogonal axes
- → step81

**Shared principle:** the routing mechanism must be **conservative** (preserves total signal
magnitude across each routing step) to avoid gradient collapse over K iterations.

---

### Today's Goal (2026-04-06)

Clear the backlog — record results as they arrive, no new scripts. The dynamic routing direction
stays open. When the analysis above crystallizes into a concrete design, script it properly
following the design-to-script rule.
