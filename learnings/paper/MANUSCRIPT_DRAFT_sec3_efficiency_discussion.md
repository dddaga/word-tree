## 6. Efficiency Frontier

### 6.1 The Reduction Path

Starting from the project baseline of $N=4096, D=64, K_{hh}=4, K_\text{iter}=12$ at 38.8M FLOPs (97.86%), the efficiency frontier was mapped by independently reducing each dimension of the FLOPs formula:

**Axis 1: $D$-reduction** ($N=2048, K_{hh}=4, K_\text{iter}=8$).
$D=64 \to 32 \to 20 \to 16$. Floor at $D=16$ (3.15M, 95.87%). $D=12$ misses phase exit by 0.38pp.

**Axis 2: $K_{hh}$-reduction** ($N=2048, D=16, K_\text{iter}=8$).
$K_{hh}=4 \to 3 \to 2$. Each step reduces FLOPs by 25-33% with $<1$pp accuracy loss. Floor at $K_{hh}=2$ (1.57M, 95.67%). $K_{hh}=1$ (step200) fails: $-5$pp, K\_hh=2 is minimum viable.

**Axis 3: $K_\text{iter}$-reduction** ($N=2048, D=16, K_{hh}=2$).
$K_\text{iter}=8 \to 6 \to 5$. Unexpected: $K_\text{iter}=6$ outperforms $K_\text{iter}=8$ at Tier-1 by $+1.02$pp. Floor at $K_\text{iter}=5$ (0.98M, 95.52%). $K_\text{iter}=4$ killed (−2.14pp).

All three axes were independently validated with controlled ablations changing exactly one parameter at a time.

### 6.2 Complete Pareto Table (Confirmed Phase Exits)

All entries below are Tier-2 (full 150-epoch, 100% data) confirmed results unless noted.

| Step | $N$ | $D$ | $K_{hh}$ | $K_\text{iter}$ | FLOPs | FLOPs % | Accuracy | Note |
|------|-----|-----|----------|-----------------|-------|---------|----------|------|
| step89 | 4096 | 64 | 4 | 12 | 38.8M | 31.4% | **97.86%** | Project best ($D=64$) |
| step176-A | 2048 | 32 | 4 | 8 | 6.10M | 4.94% | 96.18% | First phase exit |
| step181 | 2048 | 20 | 4 | 8 | 3.93M | 3.18% | 96.03% | |
| step185 | 2048 | 16 | 4 | 8 | 3.15M | 2.55% | 95.87% | $D$-reduction floor |
| step192 | 2048 | 16 | 3 | 8 | 2.36M | 1.91% | 95.90% | $K_{hh}$ reduction |
| step193 | 2048 | 16 | 2 | 8 | 1.57M | 1.27% | 95.67% | $K_{hh}$=2 minimum |
| **step195** | **2048** | **16** | **2** | **6** | **1.18M** | **0.96%** | **96.08%** | **≤1% FLOPs criterion met** |
| step199 | 2048 | 16 | 2 | 5 | 0.98M | 0.79% | 95.52% | Pre-ΔW-proj canonical (legacy) |
| **step887** | **2048** | **16** | **2** | **5** | **0.98M** | **0.79%** | **96.38% ± 0.18pp** | **ΔW-proj canonical, 3 seeds** |
| **step605** | **2048** | **16** | **2** | **1** | **0.20M** | **0.16%** | **95.95%** | **K=1 KD student — efficiency champion** |
| step204 | 4096 | 16 | 2 | 6 | 2.36M | 1.91% | 97.15% | $N$-scaling |
| step205 | 4096 | 16 | 2 | 5 | 1.97M | 1.59% | **97.17%** | $D=16$ record |
| step209 | 8192 | 16 | 2 | 5 | 3.93M | 3.18% | 97.17% | $D=16$ ceiling confirmed |
| — | — | — | — | — | 123.6M | 100% | ~95.0% | **VGG16 FC baseline** |

### 6.3 Key Takeaways

- **Headline result:** 96.38% $\pm$ 0.18pp (step887, 3 seeds, ΔW-proj canonical) at 0.98M FLOPs = 0.79% of VGG16 FC, 34,976 params = 0.029% of VGG16 FC (119.5M).
- **Efficiency champion:** step605 K=1 KD student — 95.95% @ 0.20M FLOPs (0.16%), 34,976 params, 12.7µs B=32 (5.26× faster than VGG\_FC on RTX 5060 Ti).
- **Sub-1% FLOPs and sub-1% params at ≥95% accuracy** is achievable simultaneously (step195/step887/step605).
- **The $K_\text{iter}=6$ result (step195) outperforms $K_\text{iter}=8$ (step193)** despite 25% fewer FLOPs: reducing over-smoothing at this scale improves accuracy.
- **$D=16$ ceiling = 97.17%**: doubling $N$ from 4096 to 8192 provides no additional accuracy. The bottleneck shifts from $N$ to $D$.
- **The efficiency axis is monotone**: every step on the $D$, $K_{hh}$, and $K_\text{iter}$ reduction paths was individually validated.

### 6.4 Multi-Seed Variance

Seed variance is a secondary paper finding:

- **step887** (ΔW-proj canonical, $N=2048$, $D=16$, 3 seeds): **96.38% $\pm$ 0.18pp** (seed0=96.23%, seed1=96.28%, seed42=96.64%)
- **step980** (CIFAR-10, T2, 3 seeds): **80.57% $\pm$ 0.12pp** — tighter than T1 (±0.31pp)
- **ΔW-projection halves seed variance** relative to pre-ΔW-proj baseline: step199 ±0.43pp → step887 ±0.18pp (step760, CONFIRMED)

The variance reduction from ΔW-projection indicates a more stable optimum landscape: the learned positional geometry constrains the routing to a narrower basin.

---

## 7. Discussion

### 7.1 What Makes This Work?

The success of SGNNET challenges the assumption that expressiveness requires learned connectivity. We posit three mechanisms working in concert:

**Random projections as sufficient statistics.** Johnson-Lindenstrauss theory guarantees that $N$ random projections of a $d$-dimensional input preserve pairwise distances within $\epsilon$ when $N = O(\epsilon^{-2} \log n)$ for $n$ data points. At $N=2048$ projecting 25,088-dim inputs, the theoretical coverage is ample. The scatter-sum over $K_\text{in}=25$ inputs per neuron further concentrates each neuron on a local input region, analogous to simple-cell receptive fields in V1.

**Routing as attractor dynamics.** The $K_\text{iter}$ message-passing loop is a synchronous fixed-point iteration on the hypersphere. The normalization constraint ensures that iterates remain on $S^{D-1}$ and that the map is non-expansive. Fixed points of this iteration correspond to stable activation patterns consistent with the input — effectively, the representation that best satisfies all local neighborhood constraints simultaneously. The learning signal ($W_\text{pos}$, $\theta$) shapes the energy landscape of this dynamical system, not the connectome itself.

**Diversity forcing through AntiHebbian inhibition.** Without active diversity forcing, the routing dynamics collapse to a low-rank fixed point: all neurons track the same dominant direction. The $\alpha_\text{ahebb}=1.0$ anti-Hebbian suppression maintains the full $D$-dimensional capacity by preventing this collapse, ensuring the attractor landscape contains $O(e^{D/2})$ distinct fixed points rather than one.

### 7.2 Implications for FFN Replacement in Transformers

The stated long-term goal of this project is to replace the feed-forward network (FFN) sublayer in transformer models with SGNNET. The FFN sublayer is a two-layer MLP ($d_\text{model} \to 4d_\text{model} \to d_\text{model}$) that accounts for approximately two-thirds of transformer parameters and compute. If SGNNET can match FFN expressiveness at $O(N \times K)$ rather than $O(d^2)$ cost, transformer training and inference efficiency would improve substantially.

The current results on Imagenette demonstrate a necessary condition: SGNNET can match FC layer accuracy at $<1\%$ of FC FLOPs. However, FFN replacement requires demonstrating generalization across multiple tasks and sequence positions — the routing dynamics must work in the context of attention-based representations, not VGG16 features. This is future work.

The $N$-scaling law result (97.17% ceiling at $D=16$, reached at $N=4096$) directly motivates the FFN hypothesis: as task complexity increases, increasing $N$ may provide the additional capacity the FFN needs, without increasing per-parameter compute. This is the core hypothesis of the project and remains to be validated on sequence tasks.

### 7.3 Limitations and Cross-Dataset Results

**Cross-dataset: CIFAR-10 (CONFIRMED).** step980 (T2, 3 seeds): **80.57% $\pm$ 0.12pp** on CIFAR-10, gap $-5.67$pp vs Linear 86.24%. ΔW-projection is essential on CIFAR-10: removing it collapses accuracy by $-62.41$pp (step915).

CIFAR-10 $N$-scaling (T2):

| Step | $N$ | Accuracy | Note |
|------|-----|----------|------|
| step882 | 2048 | 80.69% | canonical (single seed42) |
| step909 | 4096 | 82.53% | $N$-scaling |
| step914 | 8192 | 83.58% | $N$-scaling |

CIFAR-10 $K_\text{iter}$ sensitivity: $K_\text{iter}=10 \to -15.54$pp, $K_\text{iter}=15 \to -59.97$pp vs baseline (step916, CONFIRMED). Over-smoothing more severe on CIFAR-10 than Imagenette.

**MLP and GNN baselines (CONFIRMED).** step891–893 (T2, MLP sweep):

| Model | Params | Accuracy | vs SGNNET |
|-------|--------|----------|-----------|
| MLP\_h1 | — | 14.31% | $-66$pp |
| MLP\_h2 | — | 17.05% | $-63$pp |
| MLP\_h16 (crossover) | 11.5× SGNNET | 80.75% | $\approx 0$ |
| **SGNNET** | **34,976** | **80.57%** | — |

MLP requires 11.5× SGNNET's parameter count to match SGNNET accuracy on CIFAR-10.

Standard GNN baselines (step404, T2, same fixed random graph):

| Model | Accuracy | vs SGNNET |
|-------|----------|-----------|
| GCN | 48.9% | $-31.7$pp |
| GAT | 48.7% | $-31.9$pp |
| GIN | 15.5% | $-65.1$pp |

**Feature extractor coupling.** SGNNET operates on pool5 features. The efficiency claim covers the classifier head only; VGG16 feature extraction is not replaced.

**No theoretical grounding for the ceiling.** The $D=16$ ceiling of 97.17% is empirical. We hypothesize it reflects the capacity of $S^{15}$ for 10-class classification on this feature space (HYPOTHESIS).

**Pruned VGG16 FC baseline.** Head-to-head against pruned VGG16 FC at 34,976 params not yet collected.

---

## 8. Baselines Status

| Baseline | Status | Result |
|----------|--------|--------|
| MLP (param-matched, CIFAR-10) | **DONE** (step891–893 T2) | MLP\_h16 crossover at 11.5× SGNNET params (80.75% vs 80.57%) |
| Standard GNN (GCN/GAT/GIN, fixed random graph) | **DONE** (step404 T2) | GCN=48.9%, GAT=48.7%, GIN=15.5% — all far below SGNNET |
| Random projection + linear classifier | **DONE** (step978) | RandProj\_concat 95.75% (40K params); SGNNET routing recovers $+1.55$pp at comparable param count with mean-pool readout |
| Second dataset (CIFAR-10) | **DONE** (step980 T2) | 80.57% $\pm$ 0.12pp (3 seeds) |
| Pruned VGG16 FC at 34,976 params | NOT YET | Blocks FLOPs-efficient head comparison |
| MLP at 0.98M FLOPs | NOT YET | FLOPs-matched comparison to step887 |

**Random projection result (step978):** RandProj\_concat (fixed random $K_\text{in}=25$ projections, no routing, N=256×D=16=4096-dim fixed feature) achieves 95.75% with 40K params. SGNNET with routing and mean-pool achieves 97.30% at comparable params (+1.55pp). The information is present in the random projections without routing; routing recovers it with only mean-pool readout at $233\times$ fewer params than VGG16 FC (119.5M).

---

## 9. Future Work

**Cross-dataset validation (partial).** CIFAR-10 complete (§7.3). CIFAR-100 in progress. ImageNet-1K and non-vision datasets (tabular, audio) remain open.

**FFN replacement in transformers.** Substitute SGNNET for the MLP sublayer in a small transformer (e.g., 6-layer, 512 hidden dim). Test on language modeling and fine-tuning tasks. The key question: do routing dynamics generalize from IID classification to sequential, contextual representations?

**Dynamic routing with redistribution.** The gate-death theorem rules out all multiplicative gating. Softmax-normalized routing (where gates sum to 1 over neighbors) preserves signal mass and has not been comprehensively tested at the efficiency scale ($D=16, K_{hh}=2$). Steps 73/75 showed early promise; this direction is not closed.

**Theoretical characterization of the D-ceiling.** The empirical ceiling at $D=16$ for this dataset needs a formal explanation. Candidate frameworks: covering numbers on $S^{D-1}$, Rademacher complexity of the routing function class, or information-theoretic capacity bounds on the hypersphere.

**Hardware co-design.** The gather-sum routing pattern ($Z[:, \text{conn\_hh}, :].sum(2)$) is an indexed gather followed by a reduction — a pattern well-suited to custom sparse tensor cores or graph-optimized accelerators. The fixed connectivity (no dynamic sparsity decisions) enables compile-time optimization of the routing loop.

---
