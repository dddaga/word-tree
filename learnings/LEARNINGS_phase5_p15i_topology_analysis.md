# LEARNINGS Phase 5 — Part 15i: Topology Analysis at K_hh=2

**Date:** 2026-04-11

## Finding: Graph Topology at Extreme Sparsity (K_hh=2, N=2048)

**Status: CONFIRMED** (direct computation, step199 connectivity graph)

### Graph Properties

| Metric | Value |
|--------|-------|
| Total neurons | 2,048 |
| Total edges | 4,096 |
| Edge density | 0.098% (99.9% sparse) |
| Graph diameter | 14-15 hops |
| In-degree (gather from) | 2 (fixed, every neuron) |
| Mean out-degree | 2.0 |
| Out-degree std | 1.36 |

### Out-Degree Distribution (Who Reads From Me?)

Every neuron gathers from exactly 2 neighbors (K_hh=2). How many neurons READ from given neuron varies:

| Out-degree | Count | % | Role |
|-----------|-------|---|------|
| 0 | 256 | **12.5%** | Terminal: signal goes nowhere during routing |
| 1 | 545 | 26.6% | Single consumer |
| 2 | 591 | 28.9% | Fair share |
| 3+ | 656 | 32.0% | Hub neurons |
| 5+ | 96 | 4.7% | Super-hubs (up to 8 consumers) |

### Signal Propagation Reach (K_iter=5)

| Start neuron | Reachable neurons | % of network |
|-------------|-------------------|--------------|
| Neuron 0 | 108 | 5.3% |
| Neuron 1024 | 27 | 1.3% |
| Neuron 512 | 1 | 0.05% (isolated!) |

After K_iter=5 steps, signal from one neuron reaches only 1-5% of network. Some neurons completely isolated (zero out-degree → signal never propagates, still contribute at readout).

### Implications

1. **Routing extremely local** — no global mixing. Info stays in small neighborhoods.
2. **12.5% neurons dead-ends** — process but don't propagate. Contribute only at readout.
3. **Hub neurons (32%, out-degree ≥3) disproportionately important** — amplify and broadcast.
4. **Network succeeds despite this** — 95.52% accuracy with 5% signal reach. Suggests readout (reads ALL neurons) does most heavy lifting.

### Design Implication for Heterogeneous K_hh

Natural topology already creates heterogeneous roles: hubs vs terminals. Designed heterogeneous K_hh should ENHANCE this, not fight it: give hubs MORE connections, terminals FEWER. Natural distribution approximately Poisson(λ=2).

## Experiment: step216 — Compound Winners on step199

Testing 5 mechanisms on step199 config (Tier-0, 20ep, 50% data):
- Ref: baseline (homogeneous AH α=1.0)
- A: α_ahebb=1.05 (confirmed +0.79pp at N=4096)
- B: twopop_weight (relay×2 / specialist×0.5) — +6.42pp at N=1024
- C: twopop_theta (aggregator θ=0.05 / filter θ=0.20) — +5.27pp at N=1024
- D: curriculum K_iter 2→5 — +2.85pp at N=1024

**Key question:** do N=1024 gains survive at N=2048 D=16 K_hh=2?

## Experiment: step217 — Polarizer Routing

Each neuron's W_pos acts as polarization axis. Incoming signals filtered by alignment with receiving neuron's polarizer:

```
Z_filtered[j→i] = project(Z[j], W_pos[i])   # what passes through
Z_new[i] = normalize(Σ Z_filtered[j→i])       # aggregate + recover magnitude
```

F.normalize after aggregation prevents gate-death — magnitude always recovered. Polarizer only changes DIRECTION, not intensity.

Testing 5 variants (Tier-0, 20ep, 50% data):
- Ref: standard gather-sum
- A: full polarizer (project onto W_pos)
- B: partial (50% project + 50% original)
- C: soft (30% project + 70% original)
- D: rotation in (Z, W_pos) plane by learned angle

**Key question:** does input-dependent routing improve over static highways?