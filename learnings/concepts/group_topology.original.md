# Group-Structured Topology

## Concept

Replace W_pos spatial KNN (conn_hh local component) with explicit random group membership. Dense intra-group wiring, sparse inter-group wiring. Topology-only change -- no new parameters, no change to routing loop or readout.

## Motivation

- step55 showed grouped input projection = +5pp over unified projection at matched param count. Same specialisation bias applied to hidden neurons.
- Neuron-level dynamic routing is structurally hard at N=4096: N*K learning problem (65,536 soft weights at K=16). Co-adaptation of routing weights and W_pos causes local-mode traps.
- Group-level reduces to G^2 decisions (G=16 -> 256 vs 65,536). Gradient averages over ~N/G neurons per group -- stable signal.
- Wave-1 experiments (steps 58-66) all failed from gate-death or double-sparsity. Group structure is the first approach that avoids both by construction.

## Design

- **Group assignment**: integer array `group_id[h] in {0, ..., n_groups-1}`, randomly assigned at init, fixed thereafter.
- **K_local**: random sample from neurons with same group_id (within-group neighbours). No KNN needed -- just group membership index.
- **K_random**: random sample from neurons with different group_id (guaranteed cross-group wires).
- **No change** to routing loop (K_iter steps of AH-weighted Z_nb.sum()), AntiHebb, or readout.
- Random group assignment preserves small-world property: any neuron is equally likely to share a group with any other neuron. After K_iter hops, near-complete mixing across all groups.

## Experiment Chain

```
step82 (group topology -- static structure)
    |  if gains confirmed
step83 (group state + inter-group dynamic routing)
    |  if dynamic routing helps
step84 (phase-based inter-group routing)
    |  if phase routing helps at group level
Full-scale validation at N=4096
```

## step82: Group Topology (COMPLETE)

Static group topology ablation. N=1024, 50% data, 75 epochs.

**Mac Studio (50% data, 75ep):**

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | spatial topology n_groups=128 | 83.11% |
| A | random-group n_groups=8 | 85.25% (+2.14pp) |
| B | random-group n_groups=16 | 83.62% (+0.51pp) |
| C | random-group n_groups=32 | 84.05% (+0.94pp) |
| D | random-group n_groups=8 + input-group alignment | 70.04% (DEAD) |

**Mac Mini (50% data, 75ep):**

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | spatial topology n_groups=128 | 82.62% |
| A | random-group n_groups=8 | 85.63% (+3.01pp) |

**Verdict**: n_groups=8 is clear winner on both machines (+2-3pp vs Ref). n_groups=16 and 32 marginal. Input-group alignment (D) catastrophically kills performance -- constraining which groups see which inputs destroys the diversity that makes grouping work.

Scripts: `scripts/train_step82_group_topology.py`

## step83: Group State + Inter-Group Routing (RUNNING)

Dynamic routing at group level. S_g = mean(Z[h] for h in group g). Inter-group weights w_{g->g'} = softmax(dot(S_g, S_{g'}) / tau). Combined update: Z_new[h] = normalize(Z_struct[h] + beta * Z_inter[h]).

N=1024, 50% data, 75 epochs.

**Mac Studio CPU (partial results):**

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | group topology, no inter-group routing (beta=0) | 84.31% |
| A | dot-product score, beta=0.5, every step | 78.96% (-5.35pp) |
| B | dot-product score, beta=0.5, final step only | 82.17% (-2.14pp) |
| C | dot-product score, beta=0.1 (weak mixing) | running |

**Interim verdict**: Inter-group routing via dot-product hurts. Every-step routing (A) is worst (-5.35pp). Final-step-only (B) recovers partially but still below Ref. Structural group topology (step82) is the larger contributor. Waiting on C (weak mixing) to see if near-zero beta preserves the gain.

Scripts: `scripts/train_step83_group_routing.py`

## step84: Phase Inter-Group Routing (PLANNED)

Gated on step83 results. Group phase state P_g = mean(Z_phase[h] for h in group g). Inter-group weights w_{g->g'} = softmax(cos(P_g, P_{g'}) / tau). Ablates tau = {0.5, 1.0, 2.0} + hybrid magnitude-weighted variant.

Rationale: step60 died from per-neuron phase coherence gate (multiplicative, K_iter attenuation). Group-level phase coherence + softmax redistribution removes both failure causes. Group phase is more stable (averages over group_size neurons).

No script yet -- depends on step83 outcome.

## Why This Avoids Gate-Death

Wave-1 gate-death theorem: any multiplicative gate g in [0,1] applied per-neuron over K_iter steps -> g^K signal attenuation -> gradient collapse. At K_iter=8, g=0.7: 0.7^8 = 0.06x original signal.

Group-level routing avoids this in two ways:
1. **Softmax redistribution** over G groups (not N neurons). Weights sum to 1 -- signal is redistributed, not attenuated. No multiplicative gate.
2. **Reduced decision space**: G^2 = 64 routing weights (G=8) vs N*K = 65,536 at neuron level. Each weight is well-separated (1/8 vs 1/4096), gradient signal is strong.
3. **Within-group routing unchanged**: K_iter steps of static AH routing handle local signal flow. Dynamic decisions only at group boundaries.

## Key Findings So Far

- n_groups=8 is optimal (group_size=128 at N=1024). Smaller groups (16, 32) lose the specialisation benefit.
- Input-group alignment is catastrophic (-13pp). Groups must be free to develop their own input preferences.
- Static group topology (+2-3pp) is a stronger signal than dynamic inter-group routing (which hurts so far).
- The gain likely comes from explicit specialisation bias: neurons within a group develop correlated representations, AH provides within-group diversity pressure, K_random wires handle cross-group integration.

## See Also

- `learnings/LEARNINGS_design.md` (2026-04-07 entries: Group-Structured Hidden Neuron Topology, Group-Level Dynamic Routing)
- `learnings/LEARNINGS_phase5_p15_post_wave1.md` (step82/83 results)
- `learnings/EXPERIMENT_QUEUE.md` (step82/83/84 queue entries)
- `scripts/train_step82_group_topology.py`
- `scripts/train_step83_group_routing.py`
