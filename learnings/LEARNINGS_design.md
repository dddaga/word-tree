
---

## 2026-04-04 — Phase 5 Architecture Design Discussion

### Fourier/Phase-Distance Routing

**Design intent**: Z[0] = activation magnitude, Z[1:64] = 63 cyclic phase channels.
When activation travels A→B: `Δφ_k = freq_k × ||W_pos[A] - W_pos[B]||` for each channel k.
Multi-frequency interference: same distance produces 63 different phase shifts across channels.

**Differentiable implementation**:
- `torch.remainder(φ + Δφ, 2π)` — gradient is 1 everywhere, clean and simple
- Do NOT L2-normalize across all 64 dims (destroys cyclic phase semantics)
- Normalize magnitude dim (Z[:,:,0]) across N neurons (dim=N), not across D
- Phase channels: no normalization, just modulo after each routing step

**Why Phase 4 phasors were dropped**: Failure confounds at Phase 4 (N=256, K_iter=2, binary C, broken cdist proximity). NOT a fundamental phasor failure. Phasor concept never tested at Phase 5 scale (D=64, N=1024, K_iter=8, small-world). Candidate for future experiment.

---

### Resonance-Gated Phase-Excitatory (per-batch rebuild)

**Key distinction from step29c SGNNET_PhaseExcitatory**:
- step29c: static W_phase K-NN (epoch rebuild) + unconditional excitation → 63.64% alone, -13pp vs AntiHebb alone
- New design: per-batch K-NN rebuild + TWO resonance gates before excitation fires

**Two gates**:
1. Source resonance: `dot(Z[h], W_phase[h]).clamp(0)` — h only broadcasts when its activation aligns with its own phase anchor
2. Structural: `dot(W_phase[h], W_phase[j])` — shared resonance mode (K-NN precomputed)

**Why different from step29c failure**: Dynamic gate is conditioned on current Z, not just structural W_phase. After AntiHebb training, W_pos-similar neurons have diverse Z → low res_gate for AntiHebb-suppressed pairs. Mechanisms on orthogonal axes.

**Per-batch rebuild cost**: FAISS K-NN at N=1024 D=64 ~2ms, ~22s over full training. Acceptable.

---

### AntiHebb Adaptations

**For phase-distance routing**:
- Current AntiHebb: cosine similarity of normalized W_pos (angular)
- Phase-distance routing: raw Euclidean ||W_pos[A] - W_pos[B]||
- Adaptation: replace cosine suppression with `1 - α × exp(-dist/r*)` Gaussian falloff to match routing's distance axis

**For resonance-excitatory**:
- Prediction: AntiHebb α=1.0 compatible (dynamic gate breaks W_pos/W_phase correlation)
- Test unchanged first; reduce alpha only if interference observed

---

### New Experiments Queued

| ID | Description | Key design |
|---|---|---|
| hub-interneurons | N_mix=256, conn_mix [256, 512] fan-in, excitatory, start blank | High fan-in — step47 only had K_hh=6 |
| step56 | N-scaling: 1024→8192 + AntiHebb α=1.0 | Scaling law: capacity vs complexity |
| resonance-exc | Phase-excitatory with per-batch K-NN + activation gate | Distinct from step29c phase_exc |
| phase-dist-routing | Z[0]=magnitude, Z[1:64]=cyclic phases, distance-based rotation | Untested at Phase 5 scale |

---

## 2026-04-04 — Phase Architecture Design Decisions (continued)

### Resonance Metric — Scale Alignment Not Needed

`resonance(A, B) = Z_A[0] × Z_B[0] × Σₖ Z_A[k] × Z_B[k]`

For fixed source A ranking targets {B_i}: Z_A terms are constant. Ranking is determined by `Z_Bi[0] × phase_dot(A, Bi)`. Monotonic scale transformation is identical across all pairs — top-K ordering preserved regardless of phase scale. No normalization needed.

### Phase Range: [-π, π] Canonical Implementation

[0, 2π] creates a positive bias (≈622 baseline for 63 channels). [-π, π] gives zero mean baseline — uncorrelated neurons score ≈0, aligned score positive, anti-aligned score negative.

Single operation, always in [-π, π):
```python
Z_phase = torch.remainder(Z_phase + delta_phi, 2 * math.pi) - math.pi
```
Initialize in [-π, π] from the start. No separate centering step at resonance time.

### Magnitude in Forward Pass — Decision Pending

**Current code**: Z[0] has no special treatment — same gather+sum+L2_norm as all channels.

**Option A — Independent accumulation**:
```
Z_new[h][0] = Σ_j Z[j][0]   # sum neighbor magnitudes
normalize across N neurons
```

**Option B — Phase-coherent interference (wave model)**:
```
coherence[j→h] = cos(Z_phase[j] + Δφ[j→h] - Z_phase[h]).mean(-1)
Z_new[h][0]    = Σ_j Z[j][0] × coherence[j]
```
Constructive interference (aligned phases) → magnitude grows.
Destructive interference (anti-aligned) → magnitude suppressed.
Physically motivated: may naturally replace some of AntiHebb's diversity pressure.

**Decision**: pending — user to confirm Option A or B before implementation.

### Ablation Protocol

| Stage | Data | Epochs | Purpose |
|---|---|---|---|
| Fast benchmark | 50% | 75 | Establish new reference (AntiHebb α=1.0 calibrated base) |
| Ablation runs | 50% | 75 | Compare mechanisms internally |
| Final confirmation | 100% | 150 | Winner vs 80.08% existing benchmark |

Confirm top-2 candidates at full scale — not just the single winner from ablations.

---

## 2026-04-05 — Phase Routing Architecture Design

### 1. Phase-distance routing (SGNNET_PhaseRouting)
- Z split: `Z[0]` = magnitude scalar, `Z[1:63]` = phase channels
- Phase shift: `delta_phi_k = freq_k × dist(W_pos[h], W_pos[j])`, `freq_k = 2^(k//2)`
- Differentiable modulo: `torch.remainder(Z_phase + delta_phi, 2π) - π`
- **Float32 issue**: `freq_k = 2^(k//2)` reaches `2^31` at k=62 → ~17 of 63 channels pure noise. Fix: cap at `2^10=1024` (`freq_mode="capped_exp"`)

### 2. Resonance metric
- `resonance(A,B) = Z_A[0] × Z_B[0] × Σ_k Z_A[k] × Z_B[k]` (mag_product × phase_dot)
- Phase range `[-π, π]` chosen over `[0, 2π]`: zero-mean baseline, uncorrelated neurons ≈0. `[0, 2π]` has large positive bias.

### 3. Coherence calculation — three variants to ablate
- Current: `cos(Z_phase_arr - Z_phase_h).mean(-1)` — relative to h's current state, synchronization gradient
- Alternative A: `cos(Z_phase_arr - W_phase_h)` — relative to h's static learned anchor (more stable)
- Alternative B: `cos(Z_phase_arr)` — absolute reference (real axis), expected to collapse to noise
- Test all three as `coherence_ref` parameter in step60

### 4. Magnitude modes
- **Independent**: `Z_mag_new = Σ Z_mag_nb` — simple sum
- **Coherent**: `coherence = cos(arrived - current).mean(-1)`, `Z_mag_new = Σ coherence × Z_mag_nb`
- **B_wt**: coherent gate + magnitude-weighted phase update
- **decay_coherence**: gate = `exp(-λ × cycles) × coherence`, cycles = `delta_phi/(2π)`
- **decay_weighted**: same gate + decay-weighted phase update

### 5. Decay mechanism (key insight)
- `decay_k(j→h) = exp(-λ × freq_k × dist(j,h) / (2π))`
- High-freq channels attenuate over short distances (local), low-freq channels persist (global)
- Biological analog: gamma oscillations (local) vs theta oscillations (long-range)
- Implementation: precompute `decay = torch.exp(-lambda_decay * delta_phi / (2π))` before K_iter loop; reuses existing `delta_phi` tensor

### 6. Prime number frequencies
- `freq_k = 2π / p_k` (k-th prime)
- CRT property: aliasing-free range = product of all primes ≈ 10^90 — two distances never confuse all channels simultaneously
- BUT: individual channels still alias at their period; low-freq channels (high k) barely discriminate nearby neurons
- Float32 safe: max delta_phi = π × 8 ≈ 25 radians
- Test as `freq_mode="harmonic_primes"` in step60

### 7. Magnitude-weighted phase sum
- `Σ (Z_mag_nb / Σ Z_mag_nb) × Z_phase_arr` instead of raw `Σ Z_phase_arr`
- Strong activations dominate combined phase — physically motivated
- Test as `magnitude_mode="weighted_phase"` in step60

### 8. Hub interneurons (new design)
- Previous failure: step20/29c only gave K_hh=6 neighbours — no global signal possible
- New design: N_mix = 25% of neurons start blank, receive via `conn_mix [N_mix, fan_in=512]` from input neurons
- Hubs broadcast through existing conn_hh — global mixing without dedicated inhibition
- Test in step61: fan_in=512 and fan_in=256, with/without AntiHebb
