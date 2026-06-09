# Design Discussions — 2026-04-04

**Parent:** LEARNINGS_design.md (index)

---

## Phase 5 Architecture Design Discussion

### Fourier/Phase-Distance Routing

**Design intent**: Z[0] = activation magnitude, Z[1:64] = 63 cyclic phase channels.
Activation travels A→B: `Δφ_k = freq_k × ||W_pos[A] - W_pos[B]||` per channel k.
Multi-frequency interference: same distance → 63 different phase shifts.

**Differentiable implementation**:
- `torch.remainder(φ + Δφ, 2π)` — gradient 1 everywhere, clean
- Do NOT L2-normalize across all 64 dims (destroys cyclic phase semantics)
- Normalize magnitude dim (Z[:,:,0]) across N neurons (dim=N), not across D
- Phase channels: no normalization, just modulo after each routing step

**Why Phase 4 phasors dropped**: Failure confounds at Phase 4 (N=256, K_iter=2, binary C, broken cdist proximity). NOT fundamental phasor failure. Phasor concept never tested at Phase 5 scale (D=64, N=1024, K_iter=8, small-world). Candidate for future experiment.

---

### Resonance-Gated Phase-Excitatory (per-batch rebuild)

**Key distinction from step29c SGNNET_PhaseExcitatory**:
- step29c: static W_phase K-NN (epoch rebuild) + unconditional excitation → 63.64% alone, −13pp vs AntiHebb alone
- New design: per-batch K-NN rebuild + TWO resonance gates before excitation fires

**Two gates**:
1. Source resonance: `dot(Z[h], W_phase[h]).clamp(0)` — h broadcasts only when activation aligns with own phase anchor
2. Structural: `dot(W_phase[h], W_phase[j])` — shared resonance mode (K-NN precomputed)

**Why different from step29c failure**: Dynamic gate conditioned on current Z, not just structural W_phase. After AntiHebb training, W_pos-similar neurons have diverse Z → low res_gate for AntiHebb-suppressed pairs. Mechanisms on orthogonal axes.

**Per-batch rebuild cost**: FAISS K-NN at N=1024 D=64 ~2ms, ~22s full training. Acceptable.

---

### AntiHebb Adaptations

**For phase-distance routing**:
- Current AntiHebb: cosine similarity of normalized W_pos (angular)
- Phase-distance routing: raw Euclidean ||W_pos[A] - W_pos[B]||
- Adaptation: replace cosine suppression with `1 - α × exp(-dist/r*)` Gaussian falloff to match routing distance axis

**For resonance-excitatory**:
- Prediction: AntiHebb α=1.0 compatible (dynamic gate breaks W_pos/W_phase correlation)
- Test unchanged first; reduce alpha only if interference observed

---

### New Experiments Queued (2026-04-04)

| ID | Description | Key design |
|----|-------------|------------|
| hub-interneurons | N_mix=256, conn_mix [256, 512] fan-in, excitatory, start blank | High fan-in — step47 only had K_hh=6 |
| step56 | N-scaling: 1024→8192 + AntiHebb α=1.0 | Scaling law: capacity vs complexity |
| resonance-exc | Phase-excitatory w/ per-batch K-NN + activation gate | Distinct from step29c phase_exc |
| phase-dist-routing | Z[0]=magnitude, Z[1:64]=cyclic phases, distance-based rotation | Untested at Phase 5 scale |

---

## Phase Architecture Design Decisions (continued)

### Resonance Metric — Scale Alignment Not Needed

`resonance(A, B) = Z_A[0] × Z_B[0] × Σₖ Z_A[k] × Z_B[k]`

For fixed source A ranking targets {B_i}: Z_A terms constant. Ranking determined by `Z_Bi[0] × phase_dot(A, Bi)`. Monotonic scale transform identical across all pairs — top-K ordering preserved regardless of phase scale. No normalization needed.

### Phase Range: [-π, π] Canonical Implementation

[0, 2π] creates positive bias (≈622 baseline for 63 channels). [-π, π] gives zero mean — uncorrelated neurons score ≈0, aligned positive, anti-aligned negative.

Single operation, always in [-π, π):
```python
Z_phase = torch.remainder(Z_phase + delta_phi, 2 * math.pi) - math.pi
```

### Ablation Protocol

| Stage | Data | Epochs | Purpose |
|-------|------|--------|---------|
| Fast benchmark | 50% | 75 | Establish new reference (AntiHebb α=1.0 calibrated base) |
| Ablation runs | 50% | 75 | Compare mechanisms internally |
| Final confirmation | 100% | 150 | Winner vs 80.08% existing benchmark |

Confirm top-2 candidates at full scale — not just single winner from ablations.