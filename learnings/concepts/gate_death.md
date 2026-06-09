# Gate-Death Theorem

## Statement

Multiplicative gate `g in [0,1]` per routing step over `K_iter` iterations compounds to `g^K_iter` signal attenuation. At `K_iter=8`:

| Per-step gate value | Compound signal | Effective gradient multiplier |
|---------------------|-----------------|-------------------------------|
| g=0.8 | 0.8^8 = 0.17x | ~6x smaller |
| g=0.7 | 0.7^8 = 0.06x | ~17x smaller |
| g=0.5 | 0.5^8 = 0.004x | ~250x smaller |

Gradient through product: `dL/dZ_0 = (prod g_k) * dL/dZ_K`. Network learns to null gate (g -> 0), routes through residual path. Gated mechanism becomes decorative or destructive.

**Structural, not hyperparameter problem.** Per-step multiplicative gate with g < 1 dies in training for K_iter >= 4.

## Evidence

All wave-1 dynamic routing experiments exhibited gate-death. REF_BASELINE = 73.53% (step57, AH=1.0, 50%/75ep).

| Step | Mechanism | Best config | Best acc | vs Ref | Failure mode |
|------|-----------|-------------|----------|--------|--------------|
| step51 | W_phase spatial gating (phase cos gate on connections) | — | ~20% | -53pp | Complete gate-death: AH + W_phase gates = no signal |
| step52 | High-D subspace routing (Z-subspace split) | A (split=16+AH) | 13.99% | -60pp | Gate death; W_pos-subspace also fails (~12%) |
| step58 | Resonance excitatory (Z coherence gate) | — | ~55% | -18pp | Gate collapses signal in early routing steps |
| step59 | Active beam unified (top-K Z magnitude routing) | — | ~55% | -18pp | Discrete top-K breaks gradient, sparse paths collapse |
| step60 | Phase routing magnitude (freq interference + decay) | C (AH=1.0) | 18.73% | -55pp | Coherence gate -> near-zero signal after 2-3 steps; all 12 non-Ref configs gate-dead (10-19%) |
| step61 | Hub interneurons (N_mix=256, fan_in=512) | C (AH=1.0) | 64.05% | -9pp | Hub connectivity too sparse; AH suppresses hub paths |
| step63 | Activation-gated routing (soft-attn mixed candidates) | B (AH=1.0) | 55.41% | -18pp | Multiplicative gate collapses gradients; hop_decay=0.9 marginal (-0.12pp) |
| step65 | Distance-phase routing (exp(-gamma*d_norm)) | A/B (gamma=0.5/1.0) | 48.79% | -25pp | Phase drift -> coherence -> 0 |
| step66 | Phase-target plasticity (attract/repel rewiring) | A | 67.87% | -16pp | Any dynamic element: 57-68%; with AH: 40%. Double sparsity. |
| step36 D/E | Input-gated adjacency (hard gate configs) | D | 41.07% | -32pp | Gate-dead; E=11.21% also gate-dead |
| step47 B/C/D/E | Hub interneurons D=64 (high fan-in configs) | B | 43.24% | -14pp | Gate-dead (B=43%, C=38%, D=34%, E=37%) |

Observable diagnostic: safety valve loss collapses to near-zero (0.002-0.005) on gate-death. Routing diversity collapses — most neurons fire similarly.

## Root Cause Analysis

### Gating vs Redistribution

Static AntiHebbian routing survives because suppression applied **before** gather step. `F.normalize()` at end of each routing step restores magnitude to unit vectors. Net per-step signal conserved — AH **redirects** signal (suppression of one neighbor = more relative weight for others after normalization).

Dynamic gating **destroys** signal:
```
Destroyed:  Z_new = gate(Z) * Z_nb.sum()          gate in [0,1]
Conserved:  Z_new = softmax(score(Z, Z_nb)) * Z_nb   sum(weights) = 1
```

Gating attenuates without restoration. After K_iter steps cumulative attenuation starves both forward signal and backward gradient.

### The Mechanism in Detail

1. Modified routing introduces multiplicative gate `g in [0,1]`
2. Per-step signal decays by factor g
3. After K=8 steps: compound decay `g^8`
4. Safety valve loss collapses to ~0.002-0.005 (threshold gate ReLU-theta regime shifts)
5. Routing diversity collapses — most neurons fire identically
6. Accuracy stuck at 10-55% depending on whether AH can partially compensate
7. Network learns to null gate entirely, routes only through residual path

## The Fix: Redistribution

Replace multiplicative gating with softmax-weighted redistribution where weights sum to 1:

```
Z_new[h] = sum_j( w_j(Z) * Z_nb[h,j] )   where sum(w_j) = 1
```

Signal magnitude redistributed, not destroyed. Gradient flows cleanly across K iterations. Graph attention at O(N*K) cost, not O(N^2).

### step73 Results (softmax redistribution routing, 50%/75ep, N=1024, patched arch)

| Config | Description | top1_best | vs Ref (84.56%) |
|--------|-------------|-----------|-----------------|
| Ref | AH=1.0 static routing | 84.56% | — |
| A | softmax routing variant A | 55.77% | -28.79pp (collapsed) |
| B | softmax routing variant B | 84.87% | +0.31pp |
| C | softmax routing variant C | 85.58% | +1.02pp |
| D | softmax routing variant D | **86.34%** | **+1.78pp** |

Config D = +1.78pp over Ref. First dynamic routing mechanism to beat static AH. Config A collapsed (likely degenerate score function). Redistribution principle confirmed.

### Queued redistribution experiments

- **step75**: Input-modulated temperature routing. `temp_h = tau_0 * sigma(W_temp * Z_h)`, `w_j = softmax(score_j / temp_h)`. Hot neurons route sharply, cold route diffusely. Running.
- **step80**: Phase alignment as softmax weight. `w_j = softmax(coherence(Z_h, Z_j) / tau)`. Direct redemption of step60 failure — coherence as redistribution weight instead of multiplicative gate.
- **step83**: Group-level inter-group routing. `w_{g->g'} = softmax(dot(S_g, S_{g'}))`. G^2=256 routing decisions vs N*K=65536. Gate-death bypassed by construction. Partial results: Ref=84.31%, A=78.96% (-5.35pp), B=82.17% (-2.14pp). Inter-group dot-product routing not beneficial so far.
- **step84**: Phase-based inter-group routing. Group phase coherence + redistribution. Gated on step83.

## Double-Sparsity Interaction

AntiHebbian suppression + any gate creates compounding sparsity:

1. AH removes redundant paths (suppresses structurally similar neighbors via W_pos cosine similarity)
2. New gate removes coherence-mismatched or phase-misaligned paths
3. What survives = near-zero signal — gradient collapses

"Double sparsity" pattern observed in steps 58-66. AH already solves diversity/sparsity optimally. Adding further gate re-solves solved problem, creating redundancy and signal loss.

Concrete evidence:
- step66: AH + phase-target plasticity = 40.33% (Config D). AH alone = 83.18%.
- step51: AH + W_phase spatial gating = ~20%. AH alone = 73.53%.
- step60: AH + phase coherence gate = 18.73% (Config C). AH alone = 73.55%.
- step61: AH + hub mixing = 64.05% (Config C, -9pp). No AH + hub mixing = 40.51%.

Pattern: AH partially compensates for gate damage (+24pp in step63 B vs A), but gate still destroys AH fixed point. Interaction always destructive.

## Experiments That Escaped (Partially)

### step60 PhaseRouting — softmax weights, not pure gate

step60 used softmax weights (sum(w)=1) rather than pure multiplicative gate. Signal magnitude redistributed, not destroyed. Should have avoided gate-death.

Why it still failed:
1. **Double-sparsity**: phase routing + AH sparsification + beam left too few active paths for gradient
2. **Scale confounds**: tested at N=1024, D=64, early Gen4 (no turing/reflect), before arch bugs fixed
3. **Neuron-level softmax at N=4096**: 4096 softmax targets produce vanishingly small per-connection weights (w ~ 1/4096), numerically unstable

### step36 configs A/B/C — soft input-gated adjacency

| Config | Mechanism | top1_best | vs Ref (56.79%) |
|--------|-----------|-----------|-----------------|
| A | Soft input gate | 58.62% | +1.83pp |
| B | Soft input gate variant | 57.99% | +1.20pp |
| C | Soft input gate variant | 57.66% | +0.87pp |
| D | Hard gate | 41.07% | -15.72pp (gate-dead) |
| E | Hard gate | 11.21% | -45.58pp (gate-dead) |

A/B/C used soft (sigmoid) gating with residual paths, avoiding total signal collapse. Small gains (+0.87 to +1.83pp) on pre-Gen4 base. D/E used hard gates and gate-died. Soft gate partially escaped by keeping gate values near 1.0 (minimal attenuation) but also means minimal routing effect.

### step47 configs A/F — hub interneurons with limited fan-in

| Config | top1_best | vs Ref (56.79%) |
|--------|-----------|-----------------|
| F | 57.55% | +0.76pp |
| A | 55.62% | -1.17pp |
| B-E | 33-43% | gate-dead |

Config F marginally escaped (+0.76pp) while B/C/D/E gate-died. F likely had weakest gating interaction, leaving AH routing mostly undisturbed.

### step73 configs B/C/D — softmax redistribution (first clean escape)

Config D = 86.34% (+1.78pp over Ref 84.56%). First mechanism to cleanly beat static AH routing using redistribution (sum(w)=1) instead of gating. Proof gate-death theorem fix works.

## Implications for Future Design

### The constraint

All future dynamic routing mechanisms in SGNNET must be **conservative** (signal-preserving):

1. **No multiplicative gates with g < 1.** Per-step gate kills gradients over K_iter >= 4 iterations.
2. **Weights must sum to 1** (softmax or equivalent normalization). Ensures total incoming signal conserved per routing step.
3. **Do not add conditions activations must satisfy to propagate.** Every condition = implicit gate.
4. **Gradient through routing path must be O(1) per step**, not O(g^K).

### Additional structural constraints at scale

Even with redistribution (no gate-death), neuron-level dynamic routing at N=4096 problematic:
- Each routing weight w_ij ~ 1/K where K = topK sparsity
- Distinguishing "route to neuron j vs neuron k" requires gradients through all K*N active weights simultaneously
- Co-adaptation: routing weights and W_pos jointly move, creating local modes

**Group-level routing** (G=16 groups) reduces decision space from N*K ~ 65536 to G^2 = 256 decisions. Group state S_g = mean(Z[h] for h in group g) provides stable gradient signal. step82 group topology = +2-3pp confirms structural benefit. step83 inter-group routing results mixed (A=-5.35pp, B=-2.14pp vs Ref) — topology itself may matter more than routing between groups.

### Design checklist for new routing mechanisms

- [ ] Signal magnitude conserved per routing step? (sum(w) = 1 or equivalent)
- [ ] Mechanism interact with AH additively (not multiplicatively)?
- [ ] Gradient through K_iter steps bounded (no compound decay)?
- [ ] At target N, routing weights numerically well-separated? (not 1/N)
- [ ] Mechanism avoid re-solving diversity problem AH already solves?

## See Also

- [[antihebbian]] — core diversity mechanism; all gate-death occurs relative to AH stable fixed point
- [[phase_routing]] — step60, comprehensive closure on phase-distance gating; step80 queued for redistribution fix
- [[softmax_routing]] — step73, first clean escape from gate-death via redistribution; Config D = +1.78pp
- [[k_iter]] — iteration depth amplifies gate-death; K_iter=8 baseline, K_iter=12 optimal at N=4096 (step71)