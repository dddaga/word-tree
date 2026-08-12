# Design Discussions — 2026-04-09

**Parent:** LEARNINGS_design.md (index)

---

## Phase-Polarized Neurons + Alternating Training (step102)

**Date:** 2026-04-09
**Script:** train_step102_phase_polarizer.py

### Core Concept

Each neuron has two learned properties on S^{D-1}:
- **W_pos**: spatial position → determines WHO is your neighbor (topology)
- **W_phase**: polarization axis → determines HOW MUCH signal passes (filtering)

Three interacting mechanisms:

**1. Malus's Law Polarization**
Signal from j→h filtered by `cos²(angle(W_phase[h], W_phase[j]))`.
- Aligned phases → full signal pass
- Orthogonal phases → zero signal
- NOT a multiplicative gate over K_iter — it's a per-edge coefficient (like AH)
- Gate-death impossible: filter is recomputed from W_phase each forward pass, not compounded

**2. Alternating Training (Block Coordinate Descent)**
- Even epochs: freeze W_phase, train W_pos → topology adjusts with stable filters
- Odd epochs: freeze W_pos, train W_phase → filters adjust with stable topology
- Directly addresses temporal mismatch (step83 root cause): no co-adaptation
- Variant: 2-epoch blocks for more stability per phase

**3. Pauli Exclusion on Combined State**
Diversity penalty on `cat(W_pos, W_phase)` — repels neurons with similar (pos, phase).
- Extends AH (position-only diversity) to full state space
- Two neurons CAN share position if phases differ (functional specialization)
- Two neurons CAN share phase if positions differ (spatial specialization)
- Penalty: `λ * mean(exp(-||state_h - state_j||²))` over connected pairs

### Why D=32

- FLOPs halved → faster iteration
- Phase space more constrained → polarization effects more visible
- step86 F shows D=32 gives ~93% at N=4096 → viable at N=1024
- If this works at D=32, can scale to D=64 later

### Key Differences from Failed Experiments

| Prior failure | Why step102 is different |
|---------------|------------------------|
| step66 (phase-target, −43pp) | Used phase as query/key → multiplicative. step102 uses cos² filter = static per edge |
| step83 (group routing, −6pp) | Simultaneous training → co-adaptation. step102 alternates → stability periods |
| step60 (phase routing, all gate-dead) | Phase coherence × activation = multiplicative. step102: no compound |
| AH wpos (works) | step102 adds orthogonal phase axis — AH handles position diversity, phase handles signal selection |

### Ablation Design (6 configs, N=1024, D=32, 50%/75ep)

| Config | AH | Polar | Alternating | Pauli | Tests |
|--------|-----|-------|-------------|-------|-------|
| Ref | ✓ | ✗ | ✗ | ✗ | D=32 baseline |
| A | ✓ | ✓ | 1-epoch | ✗ | Full mechanism |
| B | ✓ | ✓ | simultaneous | ✗ | Is alternating needed? |
| C | ✓ | ✓ | 1-epoch | λ=0.01 | Does Pauli add diversity? |
| D | ✓ | ✓ | 2-epoch | ✗ | Longer stability periods? |
| E | ✗ | ✓ | 1-epoch | ✗ | Can phase replace AH? |

---

---

## ⚠️ AH Compatibility Rule (from steps 29c, 32, 51, 66)

**Any new mechanism using W_pos as its signal source will ANTAGONIZE AH.**

AH wpos suppresses contributions from W_pos-similar neighbors. If a new mechanism
DEPENDS on those same neighbors' contributions (for routing, gating, etc.), AH
removes the signal it needs → catastrophic double-sparsity.

Evidence:
- step66 D: AH + phase-target(W_pos as Key/Value) = 40.33% (−42.85pp, WORST)
- step51: AH + W_phase gate = ~20% (−63pp) — complete signal death
- step29c/32: AH alone = 80.08%. ANY compound → 67% (−13pp)

**Safe mechanisms** (use separate parameters from W_pos):
- step102 polarizer: uses W_phase (separate param) for cos² filter → orthogonal to AH
- step103 wave: uses W_pos DISTANCE (not cosine similarity) → related but not identical to AH

**Unsafe mechanisms** (use W_pos similarity directly):
- Phase-target routing (W_pos as query/key)
- W_phase spatial gating on W_pos connections
- Any softmax over W_pos-derived scores

**Design rule:** New mechanisms should use either:
1. A separate parameter (W_phase) that AH doesn't touch, OR
2. A different FUNCTION of W_pos (distance, not cosine similarity), OR
3. An entirely different state (Z activations) for routing decisions

---

## Open Design Questions (as of 2026-04-09)

### Redistribution at N=4096 (G4)

step75 Config D gave +3.98pp at N=1024 with redistribution routing (Σw=1).
Never tested at N=4096 where AH achieves 95.87%. Key question:
Does per-neuron redistribution routing add value on top of AH alone at N=4096?

**Script needed.** When scripted: N=4096, K_hh=4, K_iter=12, turing=0.0.

### Group Topology N=4096 (G2)

step82 n_groups=8 won +3.01pp at N=1024. Never tested at N=4096.
With K_hh=4 base (new default from step86), potential combined gain.

**Script:** tweak step82 script (N=4096, K_hh=4).

### Phase Alignment as Softmax Weight

step60 failed because phase coherence was used as a multiplicative gate.
The redistribution fix: `w_j = softmax(coherence(Z_h,Z_j)/τ, dim=2)`.
Σw=1 over K_hh neighbors → no attenuation, gate-death proof.
This directly redeems step60 failure using redistribution principle.

**Script needed.** Distinct from step73 (which uses dot-product score, not phase coherence).
Step number needed (step80 taken by N-scaling).

### Stacked SGNNET Parallel on Patched Arch (step85)

step64 Config F (2-parallel concat-project) = +1.52pp on buggy arch (74.90% vs Ref 73.38%).
best_ep=75/75 — still converging. On patched arch (Ref ~83.36%), headroom unknown.

**Script needed.** 3 configs: Ref, A (same as step64 F), B (2-parallel + K_iter=12 per branch).
N=1024, 50%/75ep, patched arch, turing=0.0, AH=1.0.

### N-Scaling on Patched Arch (step72)

The entire step56 scaling curve is invalid (buggy code). Need N={512,1024,2048,4096,8192} at
patched arch, turing=0.3, K_iter=(calibrated from step71), 100%/150ep.

**Script needed.** step80 only covered N=512 and N=2048 at 50%/75ep (partial).

---


*Continued in [LEARNINGS_design_2026_04_09_part2.md](LEARNINGS_design_2026_04_09_part2.md) — Gemma4/PolarQuant designs (steps 106-109), FLOPs path, 50-experiment gap analysis + transformer-inspired experiments.*
