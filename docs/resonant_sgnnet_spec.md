# ResonantSGNNet — Architecture Specification

Evolved from the SGNNET_Wave / SmallWorld lineage through experimental diagnosis.
This document captures the three-layer architecture and design decisions as of 2026-03-26.

---

## Motivation

Phase 4 diagnosis showed:
- Random C_input (original): top-1 = 10.17%
- Block/channel C_input (structured): top-1 = 17.96%
- The bottleneck is NOT scale or routing depth — it is input connectivity structure

The ceiling is caused by random wiring destroying discriminative signal before routing begins.
The fix requires rethinking what connectivity and routing mean, not just scaling.

---

## Three-Layer Architecture

### Layer 1 — Position Backbone (W_pos)

**What it is:** Each neuron has a learned position in D-dimensional space.

**Two separate lives of W_pos:**
- At rebuild time: W_pos → k-NN → conn_hh (snapshot determines structural wiring)
- Between rebuilds: W_pos trains freely under repulsion loss + readout gradient only
- Connectivity is DECOUPLED from position during the inter-rebuild window

**What it carries:** Excitatory forward signal
```python
Z_fwd = F.relu(Z - theta)    # what is clearly present → propagate forward
```

**Phase shift on structural edges:**
```python
phi = 2 * pi * d_pos(h, k) / lambda    # geometric phase rotation during hop
```
Signal arriving at k from h via structural path is rotated by distance in W_pos space.

**Rebuild:** `tick_epoch()` rebuilds conn_hh from current W_pos every reconnect_every epochs.
Use min-distance guard: exclude edges where d < r_star/4 to prevent 1/d → ∞ in phase.

---

### Layer 2 — Phase Receiver (W_phase)

**What it is:** Each neuron has a learned phase vector — a resonance receiver filter.
W_phase is a separate parameter from W_pos (decouples roles, decouples gradients).

**Receiver semantics (not broadcaster):**
Neuron h does not announce itself. It listens for incoming activations that resonate
with W_phase[h]. The incoming signal is COLLAPSED onto the W_phase direction:

```python
W_h_norm = F.normalize(W_phase[h], dim=-1)
dot = torch.dot(Z[b, k], W_h_norm)          # projection onto receiver direction
Z_received = dot * W_h_norm                  # only aligned component arrives
```

If Z[b,k] is orthogonal to W_phase[h]: nothing arrives (natural blocker).
If Z[b,k] is aligned with W_phase[h]: full signal arrives (resonance).
No manual threshold — the projection handles gating implicitly.

**What it carries:** Inhibitory reflected signal (two-scale Turing mechanism)
```python
Z_ref = -F.relu(-(Z + theta))   # strongly negative → teleport as inhibition
```

**Beam filtering:** Only top-M active neurons transmit (activation-filtered phase).
`beam_size` is a tunable hyperparameter (can be annealed).
```python
top_m = Z.norm(dim=-1).topk(beam_size).indices   # [B, beam_size]
```
Cost: O(M · N · D) per step where M = beam_size << N.

**Phase graph:** conn_phase = k-NN in W_phase space (rebuilt at tick_epoch, separate schedule
from conn_hh). Neurons close in W_phase space are phase partners regardless of W_pos distance.

---

### Layer 3 — Readout

```python
scores = (Z * W_out).mean(dim=1)    # mean aggregation (NOT sum)
```
Mean aggregation fixes fan-in scaling: output scores stay O(1) regardless of N_hidden.
W_out = F.normalize(W_pos[N_hidden:], dim=-1) — output class directions in D-space.

---

## Two-Scale Turing Pattern Mechanism

Turing (1952): short-range activation + long-range inhibition → spontaneous pattern formation.

| Scale | Path | Signal | Effect |
|---|---|---|---|
| Short (structural) | conn_hh, hop-by-hop | Z_fwd = relu(Z - θ) | Local excitation |
| Long (phase) | conn_phase, teleport | Z_ref = -relu(-(Z + θ)) | Global inhibition |

**Why this produces useful representations:**
Features strongly present locally reinforce nearby neurons (Z_fwd via conn_hh).
Features strongly absent/conflicting suppress distant phase-resonant competitors (Z_ref via conn_phase).
Features in competition cancel. Features with a unique phase niche survive.
Result: non-overlapping, diverse feature detectors without explicit diversity loss.

**The theta threshold:**
- Z > +θ: clearly present → propagates forward
- -θ < Z < +θ: ambiguous → neither propagates nor reflects
- Z < -θ: clearly absent → reflects as inhibition
Learnable per-neuron theta gives each neuron its own sensitivity threshold.

---

## Joint Repulsion — Pauli Exclusion in (W_pos × W_phase) Space

No two neurons should occupy the same (position, phase) state.

```python
W_joint = torch.cat([W_pos, W_phase], dim=-1)   # [N, 2D]
d_joint  = torch.cdist(W_joint, W_joint)         # [N, N]
r_star   = 0.5 / (N ** (1.0 / (2 * D)))         # 2D-dimensional personal volume
repulsion = F.relu(1.0 / d_joint.clamp(min=1e-8) - 1.0 / r_star)
loss_repulsion = torch.log1p(repulsion mean over off-diagonal)
```

**Key property:** Neurons can be spatially close if they differ in phase (different channel).
Neurons can share a phase if they are spatially spread. Only penalized for being similar
in BOTH dimensions simultaneously. Effective capacity = position × phase space.

**In 2D-dimensional joint space:** r_star is larger than in D-dimensional space alone.
At N=512, D=4 (2D=8): r_star = 0.5/512^(1/8) = 0.21 vs 0.105 in 4D.
Neurons have more room → safety valve fires less aggressively → N=1024 plateau may resolve.

### Safety valve scale constraint

The repulsion loss is AUXILIARY — it must not overpower the task (CE/KL) loss.
Design target: safety_loss contribution ≤ 10% of total loss in steady state.

```python
lambda_safety * log1p(raw_repulsion) ≤ 0.1 * task_loss   (target invariant)
```

Implemented via:
1. log1p softening (already in place) — compresses large raw values
2. Lambda scaling with N: `lambda_safety = base * (N_ref / N)^(1/D)`
3. Annealing: `lambda_eff(epoch) = lambda_safety * (1 + beta * tau(epoch))`
4. Hard ceiling: if safety_loss > 0.5 * task_loss in any batch, clip safety_loss

---

## Simulated Annealing Schedule

Temperature τ controls three coupled parameters:

```python
tau(epoch) = tau_max * exp(-decay_rate * epoch)   # or cosine schedule

# Coupled to tau:
lambda_eff  = lambda_safety * (1 + beta * tau)     # stronger repulsion early
beam_now    = max(beam_min, int(beam_max * tau))    # shrinking beam width
gate_temp   = tau                                   # routing softmax temperature
```

| Epoch stage | tau | Effect |
|---|---|---|
| Early (hot) | High | Broad routing, strong repulsion, large beam — explore |
| Mid | Decreasing | Routes commit, neurons crystallize, beam focuses |
| Late (cold) | Low | Sharp routing, weak repulsion, small beam — exploit |

Physical analogy: crystal formation from melt. Fast cooling = glass (disordered).
Slow cooling = crystal (ordered, optimal). The schedule controls crystallization rate.

---

## Physics and Biology Grounding

| Concept | Physical parallel | Biological parallel |
|---|---|---|
| Joint repulsion | Pauli exclusion (no two fermions same quantum state) | Cell differentiation (no two cells same fate) |
| W_phase receiver | Radio tuner (bandpass filter) | T-cell receptor (antigen-specific binding) |
| Phase collapse on receive | Quantum measurement collapse | Synaptic specificity (dendritic filtering) |
| Two-scale Turing mechanism | Reaction-diffusion (Turing 1952) | Cortical columns (local) + gamma sync (global) |
| Simulated annealing | Crystal formation from melt | Developmental critical periods |
| SOC at equilibrium | Sandpile criticality (Per Bak) | Neural criticality hypothesis |
| Modern Hopfield step | Activation-query = Hopfield retrieval | Associative memory recall |

---

## Hyperparameters (new in this architecture)

| Name | Role | Typical range | Annealed? |
|---|---|---|---|
| theta | Reflection threshold | 0.1 – 0.5 | Optional (could learn per neuron) |
| beam_size (M) | Active neurons that broadcast in phase channel | 8 – 64 | Yes, with tau |
| K_phase | Phase graph fan-out (phase partners per neuron) | 4 – 16 | No |
| alpha | Weight of phase channel vs structural | 0.1 – 1.0 | Optional |
| tau_max, decay_rate | Annealing schedule | problem-specific | Defines tau |
| lambda_safety | Joint repulsion weight | scaled per N | Yes, with tau |

---

## Connection to Existing Work

- **Modern Hopfield Networks**: activation-query routing = one Hopfield retrieval step
- **Capsule Networks**: phase resonance ≈ routing by agreement (one side is static W_phase)
- **Sparse Transformers**: structural graph = local window, phase graph = global attention
- **Graph Attention Networks**: gated message passing on learned topology
- **HTM/SDR**: joint repulsion → sparse distributed representations naturally

---

## Open Questions

1. Should W_phase be initialized orthogonally (maximally diverse) or randomly?
2. Phase graph rebuild frequency: same schedule as structural, or faster (W_phase changes faster)?
3. Should theta be global, per-neuron, or per-layer?
4. Does the collapse projection (bandpass) on teleportation need a gain term to compensate for projection loss?
5. Can the two-scale mechanism (local excitatory + global inhibitory) replace explicit load-balance loss?
