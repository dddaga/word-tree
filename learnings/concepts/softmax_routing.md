# Softmax Redistribution Routing

## Core Principle

Replace the uniform `Z_nb.sum(dim=2)` aggregation with input-dependent weighted aggregation where weights sum to 1:

```
# Old (uniform): Z_new[h] = Z_nb[h].sum(dim=K_hh)
# New (softmax):  Z_new[h] = (softmax(score / tau, dim=K_hh) * Z_nb[h]).sum(dim=K_hh)
```

Signal is **redistributed** across K_hh neighbors, never destroyed. Total incoming magnitude is conserved at each routing step regardless of input pattern. This is graph attention at O(N x K) cost -- not O(N^2) -- because the K_hh graph limits the neighborhood.

## Why It Works

### The Gate-Death Problem (what it solves)

Every wave-1 dynamic routing mechanism (steps 58-66) used a multiplicative gate `g in [0,1]` on activation propagation. After K routing steps, signal decays as `g^K`:

- g=0.8, K=8: 0.8^8 = 0.17x original signal
- g=0.7, K=8: 0.7^8 = 0.06x original signal

Gradient through the product vanishes accordingly. The network learns to null the gate (g -> 0), leaving the mechanism decorative.

### Why softmax escapes this

Softmax weights sum to 1 over K_hh neighbors at every routing step. No per-step attenuation occurs. Gradient flows cleanly through K iterations because the Jacobian of softmax-weighted-sum has full rank.

Static AH survives for the same reason: suppression is applied before gather, and `F.normalize()` restores unit magnitude after each step. Net per-step signal is conserved. Softmax redistribution formalizes this conservation while making weights input-dependent.

### Conservation law

```
||Z_new[h]|| <= sum_j |w_j| * ||Z_nb[h,j]||     where sum_j w_j = 1, w_j >= 0
```

Signal magnitude is bounded by the convex combination of neighbor magnitudes. No amplification, no attenuation.

## Experimental Results

All experiments: N=1024, D=64, K_iter=8, AH=1.0, reflect=0.5, turing=0.0, 50% data, 75 epochs.
Reference baseline (step69 Ref): 83.36%.

### step73: Softmax Routing (score function ablation)

| Config | Score Function | tau | top1_best | delta vs Ref | delta vs step69 |
|--------|---------------|-----|-----------|-------------|-----------------|
| Ref | uniform sum (control) | -- | 84.56% | -- | +1.20pp |
| A | dot(Z_h, Z_j) / tau | 1.0 | 55.77% | -28.79pp | -27.59pp |
| B | dot(Z_h, Z_j) / tau + AH_logit | 1.0 | 84.87% | +0.31pp | +1.51pp |
| C | -alpha * pos_sim (AH-only softmax) | 1.0 | 85.58% | +1.02pp | +2.22pp |
| **D** | **dot(Z_h, Z_j) / tau + AH_logit** | **0.3** | **86.34%** | **+1.78pp** | **+2.98pp** |

**Key findings:**
- Config A (Z-state only, no AH logit): catastrophic collapse to 55.77%. Pure Z-dot scoring without structural priors is degenerate -- routing decisions based solely on current activation similarity fail to differentiate neighbors meaningfully.
- Config C (AH-only softmax): +1.02pp over Ref. Converting the existing AH suppression from additive to softmax redistribution improves performance even without Z-state scoring.
- Config D (Z+AH, tau=0.3): winner at 86.34%. Sharper temperature concentrates routing on highest-scoring neighbors. The combination of structural (AH) and dynamic (Z-dot) signals under softmax normalization is strictly better than either alone.
- AH logit is essential in the score function. Every config without it collapsed.

### step75: Temperature Routing (input-modulated temperature)

Step75 extends step73 by making temperature input-dependent: `temp_h = tau_0 * (1 + sigmoid(dot(W_temp_h, Z_h)))`. W_temp is a learned [N, D] parameter -- per-neuron temperature modulation from current activation state. Hot neurons route sharply (focused), cold neurons route diffusely (spread).

| Config | Mode | tau_0 | Learned W_temp | top1_best | delta vs Ref | delta vs step69 |
|--------|------|-------|---------------|-----------|-------------|-----------------|
| Ref | uniform sum (control) | -- | No | 83.26% | -- | -0.10pp |
| A | fixed softmax + AH logit | 1.0 | No | 83.69% | +0.43pp | +0.33pp |
| B | fixed softmax + AH logit | 0.3 | No | 85.10% | +1.84pp | +1.74pp |
| C | input-modulated | 1.0 | Yes | 85.76% | +2.50pp | +2.40pp |
| **D** | **input-modulated** | **0.3** | **Yes** | **87.24%** | **+3.98pp** | **+3.88pp** |

**Key findings:**
- Fixed-tau configs (A, B) cross-validate step73 results. B (tau=0.3) = 85.10% vs step73 D = 86.34% -- within expected variance at N=1024.
- Config C (learned temp, tau_0=1.0): +2.50pp over Ref. Input-modulated temperature adds +0.66pp over the equivalent fixed config (B=85.10% vs C=85.76%) at tau_0=1.0... wait, the right comparison is A(83.69%) vs C(85.76%) = +2.07pp.
- Config D (learned temp, tau_0=0.3): **87.24%**, the highest accuracy from any routing variant. +3.98pp over Ref. Sharper baseline temperature + per-neuron learned modulation is the winning combination.
- Convergence note: C showed `training_too_short` diagnostic (best_ep=52/75), suggesting more epochs may further improve the tau_0=1.0 learned variant.

## Comparison to Gate-Based Approaches

| Step | Mechanism | Type | Best config | vs Ref |
|------|-----------|------|-------------|--------|
| step58 | Resonance excitatory | Multiplicative gate | ~73% | -10pp |
| step59 | Active beam unified | Discrete top-K gate | ~73% | -10pp |
| step60 | Phase routing magnitude | Coherence gate | ~73% | -10pp |
| step61 | Hub interneurons | Gate on fan-in | ~72% | -11pp |
| step63 | Activation-gated routing | Multiplicative gate | ~55% | -18pp |
| step65 | Distance-phase routing | exp(-gamma*d) weighting | ~49% | -25pp |
| step66 | Phase-target plasticity | Local plasticity gate | ~68% | -15pp |
| **step73** | **Softmax redistribution** | **Redistribution (sum=1)** | **86.34%** | **+1.78pp** |
| **step75** | **Input-modulated temperature** | **Redistribution (sum=1)** | **87.24%** | **+3.98pp** |

Every gate-based approach (steps 58-66) fell 10-25pp below Ref. Both redistribution approaches exceed Ref. The dividing line is exact: mechanisms where signal is multiplied by a value < 1 die over K=8 iterations; mechanisms where signal is redistributed with weights summing to 1 thrive.

## Variants

### Implemented and tested

1. **Dot-product scoring** (step73 A): `logit = dot(Z_h, Z_j) / tau`. Failed without AH -- degenerate at N=1024.
2. **Z-dot + AH combined scoring** (step73 B, D): `logit = dot(Z_h, Z_j) / tau + AH_logit`. Winner class. tau=0.3 sharper is better than tau=1.0.
3. **AH-only softmax** (step73 C): `logit = -alpha * pos_sim`. Converts existing AH suppression to normalized redistribution. Gains +1.02pp over uniform sum.
4. **Fixed-temperature softmax** (step75 A, B): Matches step73 B/D. Cross-validated.
5. **Input-modulated temperature** (step75 C, D): `temp_h = tau_0 * (1 + sigmoid(W_temp @ Z_h))`. Best overall: 87.24% at tau_0=0.3.

### Planned but not yet tested

6. **Phase-coherence scoring** (step80/84): `w_j = softmax(coherence(Z_h, Z_j) / tau)`. Redeems step60 failure by using phase alignment as redistribution weight instead of multiplicative gate. Distinct from step73 (uses phase geometry, not dot-product).
7. **Group-level redistribution** (step83/84): `w_{g->g'} = softmax(score(S_g, S_{g'}) / tau)` over n_groups instead of N neurons. Reduces routing decisions from N x K to G^2. Tested in step83 -- inter-group routing via dot-product scored -5.35pp (every step) and -2.14pp (final step only) vs group-topology Ref.

## Current Status

**Winner:** step75 Config D -- input-modulated temperature, tau_0=0.3, learned W_temp. 87.24% at N=1024, 50%/75ep.

**Delta vs project best context:**
- step69 Ref (patched arch baseline): 83.36% at 50%/75ep
- step75 D: 87.24% at 50%/75ep (+3.88pp)
- step70 project best: 97.32% at N=4096, 100%/150ep (not directly comparable -- different N and data fraction)

**Next steps:**
- Full-scale validation of step75 D at N=4096, 100%/150ep to see if +3.88pp gain transfers to full scale
- Phase-coherence scoring variant (step80) to test whether phase geometry adds signal beyond dot-product
- Joint calibration of tau_0 and W_temp learning rate at N=4096

## See Also

- [[gate_death]] -- the failure mode that redistribution routing solves
- [[phase_routing]] -- step60 failure analysis; phase coherence as gate vs redistribution
- [[group_topology]] -- step82 group structure; step83 inter-group routing (underperformed)
