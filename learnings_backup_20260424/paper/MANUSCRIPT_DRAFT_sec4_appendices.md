
## Appendix A: Dead Ends Catalog

### A.1 Routing Mechanism Failures (27 total)

All entries marked CONFIRMED unless otherwise noted.

#### A.1.1 Gate-Death Victims (8 mechanisms, steps 58–66)

| Mechanism | Step | Delta | Failure mode |
|-----------|------|-------|-------------|
| Learned attention routing ($Q/K/V$ per routing step) | 58 | $-25$pp | Multiplicative gate $\propto \sigma(\cdot)^{K_\text{iter}}$ |
| Softmax gate (learned, $\sum g = 1$ per neuron but not global) | 59 | $-18$pp | Local softmax still gates global signal |
| Sigmoid gate (learned per edge) | 62 | $-15$pp | $\sigma(\cdot)^{K_\text{iter}}$ compounding |
| Tanh gate (learned per edge) | 63 | $-22$pp | Sign-flipping + attenuation |
| Distance-weighted routing ($e^{-d}$ falloff) | 64 | $-8$pp | Exponential decay in D-space |
| Phase-gated routing (W\_phase score as gate) | 66 | $-3$pp | Static phase score as multiplicative weight |
| Topology-gated routing (edge weight proportional to W\_pos sim) | 60 | $-12$pp | High similarity → gate near 1 for redundant pairs |
| Adaptive gate (learned per neuron, dynamic) | 61 | $-20$pp | Per-neuron gate collapses with dropout-like effect |

#### A.1.2 Dynamic Routing Failures (9 mechanisms)

| Mechanism | Step | Delta | Failure mode |
|-----------|------|-------|-------------|
| Group MoE routing | 83 | $-4.12$pp | n\_groups=N//32 too coarse; Tier-2 showed gate convergence failure |
| Dynamic Z-KNN (alpha\_turing > 0) | 73 | $-5$ to $-20$pp | Inhibitory weight acts as multiplicative gate at $K_\text{iter}=12$ |
| Phase-excitatory routing (positive Turing) | 75 | $-8$pp | Excitatory long-range amplifies noise |
| Hub interneuron routing | 76 | $-6$pp | Hub neurons become gate bottlenecks |
| Markov routing (transition matrix) | 77 | $-10$pp | Row-stochastic matrix = cumulative attenuation |
| Attention readout (post-routing) | 118 | $-60$ to $-67$pp | Single-step gate applied to distributed consensus representation |
| Stochastic depth (K\_iter dropout) | 123 | $-35$ to $-61$pp | Every routing step is essential; removing any collapses fixed point |
| Beam broadcast with learned weights | 86 | $-5$pp | Beam selection $\times$ learned weight = compounding gate |
| ACT (adaptive computation time, dynamic K\_iter) | 120-A | $-1.45$pp | Learned halt adds gate; K\_iter > 12 provides no benefit |

#### A.1.3 External Constraint Losses (5 mechanisms, steps 152–153)

The network self-organizes via routing dynamics. Imposing external objectives disrupts the learned fixed-point structure.

| Mechanism | Step | Delta | Failure mode |
|-----------|------|-------|-------------|
| Nuclear norm regularization (low-rank pressure on Z) | 152 | $-5$pp | Competes with routing fixed-point convergence |
| Information bottleneck (explicit compression of Z) | 152 | $-8$pp | Dimensional compression fights positional encoding |
| Dimensional gating (learned per-dim binary mask) | 153 | $-7$pp | Masks Z dimensions → partial gate-death |
| L1 sparsity on Z | 152 | $-6$pp | Sparsifies activations → underfills routing capacity |
| Contrastive routing loss | 153 | $-4$pp | Competing gradient direction for W\_pos |

#### A.1.4 Scale Transfer Failures (6 mechanisms)

Strong gains at small $N$ (1024) compress to zero at large $N$ (4096).

| Mechanism | $\Delta$ @ N=1024 | $\Delta$ @ N=4096 | Compression |
|-----------|-------------------|-------------------|-------------|
| W\_proj (learned projection before scatter-sum) | +5.48pp (step131-B) | +0.06pp (step132) | 99% |
| weighted\_neg (sign-weighted neighbor contribution) | +3.97pp (step131-A) | not tested | — |
| Group topology (hierarchical community structure) | +4.21pp (step82) | null (step83) | ~100% |
| RigL topology (magnitude-pruned dynamic edges) | +6.82pp (step82-D) | not tested | — |
| two-population routing (excitatory/inhibitory split) | +3.2pp (step82-B) | 0pp (step86) | ~100% |
| Warm-start from K=12 teacher | +10.32pp (step165-B) | $-0.23$pp (step173-B) | reversed |

**Pattern hypothesis (CONFIRMED for W\_proj, HYPOTHESIS otherwise):** At $N=4096$ with $K_{hh}=4$, connectivity is 0.1% of all possible edges — the network already operates near its routing capacity ceiling. Mechanisms that add routing headroom at $N=1024$ (where headroom exists) provide no benefit at $N=4096$ (where the ceiling is reached with the base configuration alone).

### A.2 Failure Mode Taxonomy

| Pattern | Description | Count |
|---------|-------------|-------|
| Gate-death | $g^{K_\text{iter}}$ compounding attenuation | 8 |
| Over-smoothing | Too many routing steps at low $D$ | 3 |
| Scale transfer | $N=1024$ gains vanish at $N=4096$ | 6 |
| Compounding interference | Independent winners cancel when combined | 4 |
| Self-organization disruption | External losses fight routing fixed point | 5 |
| Insufficient routing depth | $K_\text{iter}$ floor not met at given $N$ | 1 |

---

## Appendix B: Architecture Hyperparameter Reference

| Parameter | Symbol | Final Config (step199) | Description |
|-----------|--------|----------------------|-------------|
| Hidden neurons | $N$ | 2048 | Number of neurons on $S^{D-1}$ |
| Sphere dimension | $D$ | 16 | Dimensionality of hypersphere |
| Hidden connectivity | $K_{hh}$ | 2 (K\_local=1 + K\_random=1) | Edges per hidden neuron |
| Input fan-in | $K_\text{in}$ | 25 | Input connections per neuron |
| Routing iterations | $K_\text{iter}$ | 5 | Message-passing depth |
| Groups | $n_\text{groups}$ | 256 (= $N/8$) | Spatial blocks for both connectivities |
| Normalization | norm\_mode | l2 | $F.\text{normalize}$ per neuron |
| Encoding | encoding\_mode | fourier | Fourier embedding on $S^{D-1}$ |
| Reflection | $\alpha_\text{reflect}$ | 0.5 | Leaky self-inhibition weight |
| Turing inhibition | $\alpha_\text{turing}$ | 0.0 | Disabled (confirmed harmful at scale) |
| AntiHebbian | $\alpha_\text{ahebb}$ | 1.0 | Positional decorrelation strength |
| Routing mode | mode | dynamic\_z\_geo | Z-similarity + geo-penalized pseudo-connections |
| Beam size | $M$ | 16 | Top-$M$ active neurons for broadcast |
| Geo penalty | $\gamma$ | 0.5 | Position distance penalty in score |
| Resonance threshold | $\tau$ | 0.0 | Minimum score for pseudo-connection |
| Phase graph size | $K_\text{phase}$ | 8 | Phase neighborhood size (unused at $\alpha_\text{turing}=0$) |
| Total params | — | 67,744 | $W_\text{pos}$: 32,768 + $\theta$: 2,048 + $W_\text{out}$: 160 + bias/other |
| FLOPs | — | 0.98M | $3 \times 2048 \times 2 \times 16 \times 5$ |

---

## Appendix C: N-Scaling Detail (D=16, K_hh=2)

Full experimental record for $N$-scaling at $D=16$.

| Step | $N$ | $K_\text{iter}$ | FLOPs | Tier | Accuracy | Status |
|------|-----|-----------------|-------|------|----------|--------|
| step198 | 1024 | 6 | 0.59M | T1 | 88.92% | KILLED |
| step202 | 2048 | 3 | 0.59M | T1 | 89.25% | KILLED |
| step196 | 2048 | 4 | 0.79M | T1 | 92.74% | KILLED |
| step197/199 | 2048 | 5 | 0.98M | T1/T2 | 93.96%/95.52% | ✓ EXIT |
| step194/195 | 2048 | 6 | 1.18M | T1/T2 | 94.88%/96.08% | ✓ EXIT |
| step190/193 | 2048 | 8 | 1.57M | T1/T2 | 93.86%/95.67% | ✓ EXIT |
| step203/205 | 4096 | 5 | 1.97M | T1/T2 | 96.08%/97.17% | ✓ EXIT |
| step201/204 | 4096 | 6 | 2.36M | T1/T2 | 95.64%/97.15% | ✓ EXIT |
| step210 | 8192 | 4 | 3.15M | T1 | 95.49% | viable |
| step208/209 | 8192 | 5 | 3.93M | T1/T2 | 95.77%/97.17% | ✓ CEILING |
| step206/207 | 8192 | 6 | 4.72M | T1/T2 | 95.11%/96.20% | $K_6$ over-smooths |
| step211 | 8192 | 3 | 2.36M | T1 | 94.93% | borderline |

---

*Draft prepared 2026-04-11. Requires baselines (Appendix — Section 8) before submission.*
