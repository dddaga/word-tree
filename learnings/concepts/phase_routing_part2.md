<!-- continued from phase_routing_part1.md -->

## Revival via Redistribution

### The Fix: Redistribution Instead of Gating

Critical insight (LEARNINGS_design.md, 2026-04-06): current designs **destroy** signal (gate < 1), while AH **redirects** signal (suppression of one neighbor = more relative weight for others, normalization preserves total).

**Gating** (failed): `Z_new[h] = gate(Z_h, Z_j) * Z_struct[h]` where gate in [0,1]
**Redistribution** (works): `Z_new[h] = sum_j(w_j(Z) * Z_nb[h,j])` where sum(w_j) = 1 (softmax)

Softmax weights sum to 1. No attenuation per step. Gradient flows cleanly through K iterations.

### step73 Results (Softmax Routing)

Config D (softmax(Z_dot/tau=0.3 + AH_logit)) = **86.34%**, +1.78pp over Ref (84.56%). First dynamic routing beating static AH. Config A (Z-state only, no AH) collapsed to 55.77% -- AH remains load-bearing even in redistribution form. Winning formula combines AH structure with sharper Z-state-dependent redistribution.

### step80 (Planned): Phase Coherence as Softmax Weight

Distinct from step73 (dot-product score). step80 uses phase coherence as redistribution weight:

```
w_j = softmax(coherence(Z_h, Z_j) / tau, dim=2)   # [B, N, K_hh]
Z_struct[h] = (w_j.unsqueeze(-1) * Z_nb).sum(dim=2)
```

sum(w_j) = 1 over K_hh neighbors. No attenuation. Directly redeems step60 failure: same phase coherence signal, but redistribution weight instead of multiplicative gate. Script not yet written.

---

## Group-Level Phase Routing

### Hypothesis (step84, gated on step82/83)

Per-neuron phase routing failed because N*K_hh multiplicative gates per step (24,576 at N=4096, K=6) collapse signal. Group-level routing reduces decision space ~256x:

- Group phase state: `P_g = mean(Z_phase[h] for h in group g)` -- [n_groups, D_phase]
- Inter-group coherence: `w_{g->g'} = softmax(cos(P_g, P_{g'}) / tau, dim=1)` -- redistribution, not gate
- 16 groups: 16x16 = 256 routing decisions vs N*K = 65,536

Group phase more stable (averages over ~N/G neurons, less noise). Redistribution (sum w = 1) avoids signal attenuation.

### step82 Results (Group Topology -- Prerequisite)

Group topology confirmed beneficial: n_groups=8 = +2-3pp over spatial KNN reference.

| Config | Description | top1_best (Studio) |
|--------|-------------|-----------|
| Ref | spatial topology n_groups=128 | 83.11% |
| A | random-group n_groups=8 | **85.25%** (+2.14pp) |
| B | random-group n_groups=16 | 83.62% (+0.51pp) |
| C | random-group n_groups=32 | 84.05% (+0.94pp) |
| D | n_groups=8 + input-group alignment | 70.04% (DEAD) |

### step83 Results (Group Dynamic Routing -- Partial)

Inter-group routing via dot-product score did not beat group topology alone:

| Config | Description | top1_best |
|--------|-------------|-----------|
| Ref | group topology, no inter-group routing | 84.31% |
| A | dot-product, beta=0.5, all steps | 78.96% (-5.35pp) |
| B | dot-product, beta=0.5, final step only | 82.17% (-2.14pp) |

### step84 (Planned): Phase-Based Inter-Group Routing

Gated on step83 results. Replaces dot-product inter-group score with phase coherence:

```
coherence(g, g') = cos(P_g, P_{g'}).mean(-1)   # scalar per group pair
w_{g->g'} = softmax(coherence / tau, dim=1)     # redistribution
```

Ablation: tau in {0.5, 1.0, 2.0} + hybrid (coherence * magnitude). Not yet scripted.

---

## Technical Details

### freq_mode Variants

| Mode | Formula | Range at D=64 | Float32 safe? | Tested in |
|------|---------|---------------|---------------|-----------|
| `capped_exp` | `min(2^(k//2), 1024)` | 1 to 1024 | Yes (capped at 2^10) | step60 (A, B, C, D, B_anc) |
| `harmonic_primes` | `2*pi / p_k` (k-th prime) | ~3.14 to ~0.02 | Yes (max delta_phi ~ 25 rad) | step60 (A_prm, B_prm) |
| uncapped exponential | `2^(k//2)` | 1 to 2^31 | No (17/63 channels pure noise) | Never used (design-time catch) |

**harmonic_primes**: CRT property gives aliasing-free range ~ product of all primes (~10^90). Two distances never confuse all channels simultaneously. But individual channels still alias at their period; low-frequency channels barely discriminate nearby neurons. step60 A_prm and B_prm scored lowest (10.68-10.80%), worse than capped_exp variants.

### Fourier Encoding Constraints

D x D transformations scramble Fourier encoding structure. Confirmed dead across three independent experiments:
- step30: cross-dim W_mix (-15pp)
- step37: phase matrix bank (-5pp)
- step29c: fast W_phase (harmful)

Only structured sub-D mixing (e.g., group 8x8 blocks from step53) partially preserves encoding, but even that yields marginal gains (+0.89pp) below AH baseline.

### D=128 Non-Viable

Fourier encoding on S^127 collapses. All LR/schedule combos stuck at ~10% (step33). D=64 confirmed encoding ceiling.

---

## Dependency Chain

```
step60 (phase routing, KILLED)
  |
  +-- diagnosis: gate-death + float32 + double-sparsity
  |
  +-- step73 (softmax redistribution, CONFIRMED +1.78pp)
  |     |
  |     +-- step80 (phase coherence as softmax weight, PLANNED)
  |
  +-- step82 (group topology, CONFIRMED +2-3pp)
        |
        +-- step83 (group dot-product routing, PARTIAL -- underperforming)
              |
              +-- step84 (phase-based inter-group routing, PLANNED)
```

---

## See Also

- [[gate_death]] -- multiplicative gates die over K_iter>=4 steps; structural, not hyperparameter
- [[softmax_routing]] -- step73 redistribution principle; sum(w)=1 avoids attenuation
- [[group_topology]] -- step82 random-group connectivity; n_groups=8 optimal
- [[antihebbian]] -- AH alpha=1.0 load-bearing in all routing designs; removing it collapses performance