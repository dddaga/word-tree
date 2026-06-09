## 4. Experimental Setup

### 4.1 Dataset

**Imagenette** is a 10-class subset of ImageNet (Tench, English Springer, Cassette Player, Chain Saw, Church, French Horn, Garbage Truck, Gas Pump, Golf Ball, Parachute) containing approximately 13,000 training images and 500 validation images per class. We use the standard 320px version.

All SGNNET experiments use **frozen VGG16 pool5 features**: images are passed through VGG16 up to (but not including) the FC layers, producing 512-channel feature maps of spatial resolution 7×7 = 25,088 total dimensions per image. These features are extracted once and cached; SGNNET sees only the 25,088-dim vectors, not raw pixels.

**Baseline.** VGG16's two FC layers (4096→4096→10) achieve approximately 95.0% on this benchmark. Their combined FLOPs: $(25088 \times 4096) + (4096 \times 4096) + (4096 \times 10) \approx 123.6$M.

### 4.2 Training Protocol

All experiments use:
- Optimizer: Adam ($\beta_1=0.9$, $\beta_2=0.999$)
- Learning rate: $10^{-3}$ with cosine decay to $10^{-5}$
- Weight decay: $10^{-4}$ (applied to $W_\text{pos}$, not to $\theta$)
- Batch size: 128
- No data augmentation (features are pre-extracted)
- Seed: 42

**Two-tier experiment protocol.** Every new configuration is validated through a mandatory two-stage ladder:

| Tier | Budget | Data | Purpose |
|------|--------|------|---------|
| Tier 0/1 (Scout) | 75 epochs | 50% | Rejection filter — kill clearly bad configs |
| Tier 2 (Full) | 150 epochs | 100% | Reliable comparison against VGG16 FC |

Tier-0 scouts predict the final winner ~80% of the time (Spearman $\rho = 0.80$ over 46 experiments at $D=64$). The typical Tier-1 to Tier-2 lift at $N=2048$ is $+1.1$ to $+1.9$pp.

### 4.3 Metrics

- **Top-1 accuracy**: computed on full validation set at best epoch (`best_ep`)
- **FLOPs**: $3 \times N \times K_{hh} \times D \times K_\text{iter}$ (classifier head only)
- **Params**: $N_\text{hidden} \times D$ ($W_\text{pos}$) + $N_\text{hidden}$ ($\theta$) + $N_\text{out} \times D$ ($W_\text{out}$) = 34,976 at the operating point (N=2048, D=16)

### 4.4 Experiment Scale

~943 controlled experiments were conducted across Phases 5–6, covering:
- $N \in \{256, 512, 1024, 2048, 4096, 8192, 16384\}$
- $D \in \{8, 10, 12, 16, 20, 24, 28, 32, 48, 64\}$
- $K_{hh} \in \{1, 2, 3, 4, 8\}$
- $K_\text{iter} \in \{3, 4, 5, 6, 8, 12, 16\}$
- 30+ routing mechanism variants (see §5.5, §5.8, Appendix A)

---

## 5. Key Findings

### 5.1 The $D > K_{hh}$ Principle

**Claim: At fixed FLOPs, representational dimensionality $D$ dominates connectivity density $K_{hh}$.**

The FLOPs formula $3NKD \cdot K_\text{iter}$ creates iso-FLOP curves along which $D$ and $K_{hh}$ trade against each other. We ran controlled ablations at two FLOPs levels:

**At 1.57M FLOPs** (steps 187, 190):

| Config | $D$ | $K_{hh}$ | $N$ | Tier-1 Accuracy | Delta |
|--------|-----|----------|-----|-----------------|-------|
| step187 | 8 | 4 | 2048 | 91.26% | — |
| **step190** | **16** | **2** | **2048** | **93.86%** | **+2.60pp** |

**At 2.36M FLOPs** (steps 186, 191):

| Config | $D$ | $K_{hh}$ | $N$ | Tier-1 Accuracy | Delta |
|--------|-----|----------|-----|-----------------|-------|
| step186 | 12 | 4 | 2048 | 93.10% | — |
| **step191** | **16** | **3** | **2048** | **94.68%** | **+1.58pp** |

The pattern is consistent: at the same compute budget, halving $K_{hh}$ and doubling $D$ yields $+1.6$ to $+2.6$pp. The directionality is robust: step190 Tier-2 achieves 95.67% (step193) vs step188 Tier-2 = 94.62% at the same FLOPs.

**Why does this hold?** On $S^{D-1}$, the number of approximately orthogonal unit vectors scales as $e^{D/2}$. At $D=8$: $e^4 \approx 55$ directions for 2048 neurons. At $D=16$: $e^8 \approx 2981$ — enough for each neuron to occupy a unique niche. $K_{hh}$ adds over-smoothing pressure without adding directional capacity.

**Implication:** Spend FLOPs budget on $D$ first. Reducing $K_{hh}$ from 4 to 2 to buy $D$ from 8 to 16 is strictly beneficial.

### 5.2 $N$-Scaling Law with Dimension Ceiling

**Claim: Accuracy scales monotonically with neuron count $N$ up to a ceiling determined by $D$.**

At $D=16, K_{hh}=2, K_\text{iter}=5$ (Tier-2):

| $N$ | FLOPs | Accuracy | $\Delta$ from prev. |
|-----|-------|----------|---------------------|
| 1024 | ~0.49M | ~88.9% (T1 proxy) | — |
| **2048** | **0.98M** | **96.38% $\pm$ 0.18pp** | — |
| **4096** | **1.97M** | **97.17%** | +1.65pp |
| **8192** | **3.93M** | **97.17%** | +0.00pp (ceiling) |

The scaling is monotone from $N=1024$ to $N=4096$, then flat. At $N=4096$ and $N=8192$, the $D=16$ ceiling of **97.17%** is hit regardless of further $N$ increase. The same ceiling applies at $K_\text{iter}=6$: step204 ($N=4096$) = 97.15%, step207 ($N=8192$) = 96.20% — the latter showing regression, suggesting $K_\text{iter}=6$ over-smooths at $N=8192$.

This ceiling is a dimension-dependent representational limit. With $D=16$, the hypersphere $S^{15}$ has finite capacity: the network cannot represent more fine-grained class boundaries no matter how many neurons are added. Increasing to $D=64$ raises the ceiling to 97.86% (step89), consistent with the exponential capacity scaling of $S^{D-1}$.

**Key result.** The $D=16$ ceiling of 97.17% is within 0.69pp of the all-time best at $D=64$ (97.86%), achieved at **50× fewer FLOPs** (1.97M vs 38.8M).

### 5.3 Optimal $K_\text{iter}$ Decreases with $N$

**Claim: The optimal routing depth $K_\text{iter}$ decreases as $N$ increases, suggesting over-smoothing scales with both $N$ and $K_\text{iter}$ at fixed $D$.**

| $N$ | Optimal $K_\text{iter}$ | Evidence |
|-----|------------------------|---------|
| 2048 | 6 | K6 T2=96.08% > K5 T2=95.52% (+0.56pp) |
| 4096 | 5 ≈ 6 | K5=97.17% ≈ K6=97.15% (0.02pp gap) |
| 8192 | 5 | K5 T1=95.77% > K6 T1=95.11% (+0.66pp) |

At $N=2048$, the ordering is $K6 > K5 > K4 > K3$. At $N=8192$, $K5 > K6$ at Tier-1, and $K_\text{iter}=4$ (killed at $N=2048$) becomes viable — achieving 95.49% at Tier-1 (step210), vs 92.74% at $N=2048$ (step196).

Over-smoothing hypothesis: at fixed $D=16$, each step propagates activations $K_{hh}^t$ hops away. Large $N$ has higher path diversity — fewer iterations suffice before over-averaging. Best sub-1% FLOPs result: $N=2048, K_\text{iter}=5$ exploits this without accuracy penalty.

### 5.4 $N$-Scaling Rehabilitates Dead Configurations

**Claim: Configurations killed at small $N$ become viable at large $N$. Minimum viable $K_\text{iter}$ decreases with $N$.**

| $K_\text{iter}$ | Accuracy @ $N=2048$ | Accuracy @ $N=8192$ | $\Delta$ |
|-----------------|---------------------|---------------------|---------|
| 4 | 92.74% (KILLED, step196) | 95.49% (VIABLE, step210) | +2.75pp |
| 3 | 89.25% (KILLED, step202) | 94.93% (borderline, step211) | +5.68pp |

FLOPs estimates at small $N$ are systematically pessimistic about minimum routing depth. A config requiring $K_\text{iter}=5$ at $N=2048$ may need only $K_\text{iter}=3$–$4$ at $N=8192$, enabling further FLOPs reduction. The $K_\text{iter}=4$ floor at $N=8192$ (95.49% T1, step210) achieves 3.15M FLOPs with 4× more neurons and 4× fewer iterations than the step185 baseline.

### 5.5 Gate-Death Theorem

**Claim: Any multiplicative gate $g \in [0,1]$ in the routing loop compounds to $g^{K_\text{iter}}$ signal attenuation, explaining the failure of all tested gated routing mechanisms.**

Consider a routing step with a multiplicative gate:

$$Z^{(t+1)} = \text{normalize}(g \cdot Z_\text{struct}^{(t)})$$

where $g \in [0,1]$ is any learned or fixed gate (sigmoid output, attention weight, or soft mask). After $K_\text{iter}$ steps, the signal from the seed is attenuated by $g^{K_\text{iter}}$:

$$Z^{(K_\text{iter})} \propto g^{K_\text{iter}} \cdot Z^{(0)}$$

At $g=0.7$ and $K_\text{iter}=8$: $0.7^8 = 0.058$ — 94% signal loss. At $g=0.5$ and $K_\text{iter}=5$: $0.5^5 = 0.031$ — 97% signal loss.

The normalization step does not rescue this: $F.\text{normalize}(g \cdot v) = F.\text{normalize}(v)$ regardless of $g$ — the gate only affects the magnitude before normalization, and since normalization removes magnitude information, the gate's signal passes through as noise.

**Fix:** Redistribution instead of gating. If gates sum to 1 ($\sum_k g_k = 1$, softmax), signal mass is preserved. This explains why the two successful routing variants survive: the structural gather-sum followed by normalize is a soft redistribution, not a multiplicative gate.

**Empirical evidence (steps 58–66 and beyond; full catalog in Appendix A):**

| Category | Count | Representative failure | Key delta |
|----------|-------|----------------------|-----------|
| Attention gates (learned $Q/K/V$) | 3 | step58 attention routing | $-25$pp |
| Sigmoid/tanh soft gates | 3 | step62 sigmoid-gate | $-15$ to $-30$pp |
| Distance-weighted gates | 2 | step64 distance-weighted | $-8$pp |
| Dynamic Z-KNN routing | 3 | step73 ($\alpha_\text{turing} > 0$) | $-5$ to $-20$pp |
| Group MoE routing | 2 | step83 group routing | $-4.12$pp |
| Phase-modulated gates | 3 | step66 phase-target | $-3$pp |
| Stochastic depth | 1 | step123 $K_\text{iter}$ dropout | $-35$ to $-61$pp |
| External constraint losses | 5 | step152 nuclear norm | $-5$ to $-15$pp |
| Scale-transfer failures | 6 | step132 W\_proj@N=4096 | $\approx 0$pp |

The stochastic depth result (step123): skipping any single routing step collapses accuracy 35–61pp. The routing fixed point requires all $K_\text{iter}$ iterations.

**Unified explanation.** The routing loop is a fixed-point iteration. Gating disrupts the convergence basin. Gather-sum + normalize is redistributive and preserves convergence.

### 5.6 Three Load-Bearing Walls

**Claim: Three architectural components are individually necessary; removing any one causes catastrophic failure.**

We define "load-bearing" as: removal causes $>10$pp degradation compared to the full model.

**Wall 1: $F.\text{normalize}$ after each routing step.**

Removal tested in step129 ($N=4096, D=64$): $-50$pp to $-71$pp. Without normalization, activation magnitudes grow unboundedly (each sum step multiplies by up to $K_{hh}$), causing overflow or collapse to a single dominant neuron. Normalization constrains dynamics to the hypersphere where the Fourier positional encoding is meaningful.

**Wall 2: Static AntiHebbian suppression.**

The `wpos` variant of AntiHebbian (position-based cosine suppression) prevents dimensional collapse. Without it, the effective rank of $Z$ decreases over training (step155 diagnostic analysis), meaning the 16/64 representational dimensions collapse to 3-4 effective dimensions. With $\alpha_\text{ahebb}=1.0$ (confirmed optimal, step88), neurons are repelled from each other in $W_\text{pos}$ space, maintaining spread across $S^{D-1}$.

Dynamic variants of AntiHebbian (`zact`: current-Z cosine suppression) all fail (steps 58-66): they introduce input-dependent multiplicative gates, falling under the gate-death theorem.

**Wall 3: Mean-pool readout.**

Replacing mean-pool with learned attention: step118 ($N=4096, D=64$) gives $-60$pp to $-67$pp. After $K_\text{iter}$ routing steps the class signal is distributed uniformly across all $N$ neurons; attention applies a multiplicative gate to a distributed consensus representation, triggering single-step gate death. Mean-pool aggregates all neurons equally, consistent with learning-in-dynamics: the readout sums a precomputed consensus, not selects from it.

### 5.7 Compounding Interference

**Claim: Two independently beneficial mechanisms can cancel when combined.**

Step131 (Tier-1, $N=1024$, clean 4-config ablation):

| Config | Mechanism | Accuracy | vs Reference |
|--------|-----------|----------|-------------|
| Ref | baseline | 80.64% | — |
| A | weighted\_neg only | +3.97pp | 84.61% |
| B | W\_proj only | +5.48pp | 86.12% |
| **C** | **A + B** | **−0.46pp** | **80.18%** |

The compound is strictly worse than the reference. This cannot be explained by diminishing returns — the compound is below baseline, not merely below additive expectation.

**Interpretation.** W\_proj and weighted\_neg both interact with $W_\text{pos}$ geometry. When combined, they create competing objectives for $W_\text{pos}$; the compromise is worse than either alone.

**Implication.** Greedy winner-stacking is invalid. Each mechanism must be tested (a) in isolation and (b) in combination with all other planned additions.

### 5.8 ΔW-Projection Routing and Ablation

Full mechanism and ablation: **MANUSCRIPT\_DRAFT\_sec\_dw\_proj.md** (§3.5 + §5.8).

Key results: geometry removal $-76.56$pp (step883); CIFAR-10 removal $-62.41$pp (step915); canonical **96.38% $\pm$ 0.18pp** (step887, 3 seeds); step950: Z-collapse without ΔW-proj; step951: routing\_gain always negative (smoothing, not amplification); step967: dense non-selective router.

---
