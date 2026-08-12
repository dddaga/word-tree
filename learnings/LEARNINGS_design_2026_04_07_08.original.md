# Design Discussions — 2026-04-07

**Parent:** LEARNINGS_design.md (index)
**See also:** [LEARNINGS_design_2026_04_08.md](LEARNINGS_design_2026_04_08.md) for 2026-04-08 discussions

---

## 2026-04-07 — Group-Structured Hidden Neuron Topology

### Core Idea

Extend the step55 grouped-input gain into the hidden neuron topology. Currently:
- **Input**: `n_groups` projections, each neuron assigned to one input group
- **Hidden**: small-world topology based on spatial proximity (W_pos positions).

**Proposed**: apply the same group structure to hidden neurons. Neurons assigned to groups **randomly** (not spatially). Connectivity:
- `K_local` wires = same-group neighbours (dense intra-group)
- `K_random` wires = random cross-group wires (sparse inter-group)

Because group assignments are random, a neuron is equally likely to be in the same group as any
other — preserving the small-world property. After K_iter routing steps, near-complete mixing.

### Why This Is Different From Current Design

| Property | Current (W_pos spatial) | Proposed (random group) |
|----------|------------------------|------------------------|
| Local neighbours | Spatially close (W_pos KNN) | Same group (random membership) |
| Group semantics | Implicit (spatial clusters) | Explicit (assignment vector) |
| Specialisation | Spatial regions of the graph | Learned per-group representations |

### Connection to Step55

step55 showed grouped input projections = +5pp over unified projection at matched param count.
The hypothesis: groups allow neurons to specialise on different input subspaces, and K_iter
provides integration. The same logic applies to hidden neurons.

### Design Questions (resolved before scripting)

1. Group size: fixed `N / n_groups` (e.g., N=4096, n_groups=16 → 256 per group)
2. `K_local` wires: random sample from `group_id[h] == group_id[j]` set
3. `K_random` wires: random sample from `group_id[h] != group_id[j]` (cross-group)
4. Group assignment: random at init (fixed), not re-randomised each epoch
5. AntiHebb interaction: keep as-is (suppresses by W_pos cosine similarity)
6. n_groups sweep: 4, 8, 16, 32

### Experiment Design (step82)
- Ref: current W_pos spatial topology (step70 params)
- A: random group topology, n_groups=8
- B: random group topology, n_groups=16
- C: random group topology, n_groups=32
- D: random group topology, n_groups=8 + input-group alignment

N=1024, 50%/75ep. Winning group count → full-scale N=4096 validation.

---

## 2026-04-07 — Group-Level Dynamic Routing + step60 Revival

### Why Group-Level Routing Bypasses Gate Death

The wave-1 gate-death theorem: any multiplicative gate g∈[0,1] applied per-neuron over K_iter
steps → g^K signal attenuation → gradient collapse. This was the failure mode of steps 58-66.

**The group reduction**: instead of routing decisions over N=4096 neurons, make decisions over
n_groups (e.g., 16) groups. Decision space shrinks 256×. Gate applied at group level has a much
shorter product chain, and within-group routing (K_iter steps of static AH) handles local
signal flow unchanged.

### Group State Vector + Inter-Group Dynamic Routing (step83)

**Group state vector**: at each K_iter step:
```
S_g = mean(Z[h] for h in group g)   # [n_groups, D]
```

**Inter-group routing**:
```
w_{g→g'} = softmax(score(S_g, S_{g'}) / τ, dim=1)   # [n_groups, n_groups]
```
Score options: dot(S_g, S_{g'}), learned linear W_route @ [S_g; S_{g'}], or cosine similarity.

**Combined update**:
```
Z_struct[h] = within_group_update(h)
Z_inter[h]  = Σ_{g'} w_{group(h)→g'} × S_{g'}
Z_new[h]    = normalize(Z_struct[h] + β × Z_inter[h])
```

**Why this avoids gate death**: w_{g→g'} sums to 1 (softmax over n_groups), Z_inter is a convex
combination of group states — no attenuation, fully differentiable.

**Experiment design (step83)**: requires step82 winner first.
- Ref: step82 winner (group topology, no inter-group routing)
- A: inter-group routing via dot(S_g, S_{g'}), β=0.5, every step
- B: inter-group routing via dot(S_g, S_{g'}), β=0.5, final step only
- C: inter-group routing, learned W_route score, β=0.5
- D: β=0.1 (weak inter-group mixing)

### Phase-Based Inter-Group Routing (step84)

Combines step82 group topology + phase redistribution at the group level.

**Group phase state**: `P_g = mean(Z_phase[h] for h in group g)`

**Phase coherence between groups**:
```
coherence(g, g') = cos(P_g, P_{g'}).mean(-1)
w_{g→g'} = softmax(coherence / τ, dim=1)        # redistribution, not gate
```

**Why this redeems step60 at group level**:
- step60 failure: per-neuron phase coherence gate → N×K_hh multiplicative gates per step
- step84: per-group phase coherence → n_groups×n_groups softmax (16×16 = 256 values)
- Group phase is more stable: averages over group_size neurons, less noise than per-neuron phase

Experiment design (step84): requires step83 results.

### Dependency Chain

```
step82 (group topology) → if gains confirmed
step83 (group state + inter-group routing)  → if dynamic routing helps
step84 (phase-based inter-group routing)    → if phase routing helps at group level
Full-scale validation at N=4096
```

---

## 2026-04-07 — WHY Dynamic Routing Has Persistently Failed: Synthesis

### Root Cause: Gate-Death Theorem (formal)

Every mechanism except PhaseRouting used a multiplicative gate g ∈ [0,1] applied per routing step:

```
Z_out = g ⊙ Z_in     (per step)
After K_iter steps:  signal ∝ Π g_k
```

At K_iter=8 with g ~ 0.5: signal ∝ 0.5^8 ≈ 0.004. Gradient:
∂L/∂Z_0 = (Π g_k) · ∂L/∂Z_K ≈ 0.004 · ∂L/∂Z_K

**Consequence**: Any per-step multiplicative gate with g < 1 dies in training for K_iter ≥ 4.
This is not a hyperparameter problem. It is structural.

### Why Neuron-Level Routing Is Hard at N=4096

Even with softmax (no gate-death), routing N→N at N=4096 is problematic:
- Each routing weight w_{ij} ≈ 1/K where K is topK sparsity
- Co-adaptation: routing weights and W_pos jointly move, easy to get stuck in a local mode
- At N=4096, neuron-level dynamic routing requires learning ~N² soft weights simultaneously

### What Group-Level Routing Offers

Replacing N-to-N routing with G-to-G routing (G=16 groups):
- Routing matrix: G² = 256 decisions vs N×K ≈ 65,536 at neuron level
- Group state S_g = mean(Z[h]): gradient averages over ~N/G=256 neurons, stable signal
- softmax(score(S_g, S_{g'})) → 16 targets, weights well-separated (1/16 vs 1/4096)
- No multiplicative attenuation: redistribution only

*2026-04-08 discussions moved to [LEARNINGS_design_2026_04_08.md](LEARNINGS_design_2026_04_08.md)*
