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
| step89 | 4096 | 64 | 4 | 12 | 38.8M | 31.4% | **97.86%** | Project best |
| step176-A | 2048 | 32 | 4 | 8 | 6.10M | 4.94% | 96.18% | First phase exit |
| step181 | 2048 | 20 | 4 | 8 | 3.93M | 3.18% | 96.03% | |
| step185 | 2048 | 16 | 4 | 8 | 3.15M | 2.55% | 95.87% | $D$-reduction floor |
| step192 | 2048 | 16 | 3 | 8 | 2.36M | 1.91% | 95.90% | $K_{hh}$ reduction |
| step193 | 2048 | 16 | 2 | 8 | 1.57M | 1.27% | 95.67% | $K_{hh}$=2 minimum |
| **step195** | **2048** | **16** | **2** | **6** | **1.18M** | **0.96%** | **96.08%** | **≤1% FLOPs criterion met** |
| **step199** | **2048** | **16** | **2** | **5** | **0.98M** | **0.79%** | **95.52%** | **Sub-1% minimum** |
| step204 | 4096 | 16 | 2 | 6 | 2.36M | 1.91% | 97.15% | $N$-scaling |
| step205 | 4096 | 16 | 2 | 5 | 1.97M | 1.59% | **97.17%** | $D=16$ record |
| step209 | 8192 | 16 | 2 | 5 | 3.93M | 3.18% | 97.17% | $D=16$ ceiling confirmed |
| — | — | — | — | — | 123.6M | 100% | ~95.0% | **VGG16 FC baseline** |

### 6.3 Key Takeaways

- **Sub-1% FLOPs and sub-1% params at ≥95% accuracy** is achievable simultaneously (step199).
- **The $K_\text{iter}=6$ result (step195) outperforms $K_\text{iter}=8$ (step193)** despite 25% fewer FLOPs: reducing over-smoothing at this scale improves accuracy.
- **$D=16$ ceiling = 97.17%**: doubling $N$ from 4096 to 8192 provides no additional accuracy. The bottleneck shifts from $N$ to $D$.
- **The efficiency axis is monotone**: every step on the $D$, $K_{hh}$, and $K_\text{iter}$ reduction paths was individually validated. There are no shortcuts — each reduction was tested independently.

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

### 7.3 Limitations

**Single dataset, single backbone.** All results are on Imagenette using frozen VGG16 features. Generalization to other datasets (CIFAR-10, ImageNet) and other feature extractors (ViT, ResNet) is unverified.

**Feature extractor coupling.** SGNNET operates on pool5 features, not raw pixels. Its efficiency claim is for the classifier head only; the VGG16 feature extractor (which SGNNET does not replace) consumes far more compute. End-to-end fine-tuning may change the results.

**No theoretical grounding for the ceiling.** The $D=16$ ceiling of 97.17% is empirical. We hypothesize it reflects the capacity of $S^{15}$ for 10-class classification on this feature space, but a formal characterization is lacking.

**Baselines absent.** The efficiency comparison lacks head-to-head results against MLP at 67K params, GCN/GAT at 67K params, and pruned VGG16 FC. These are required for publication (Section 8).

---

## 8. Baselines Required for Submission

The following baselines are **not yet collected** and block publication.

| Baseline | Purpose | Priority |
|----------|---------|----------|
| MLP (2-layer, 67K params, trained from scratch) | Architecture vs param count — does routing matter, or just capacity? | **Critical** |
| Random projection + linear classifier (67K params) | Isolate routing contribution — is the gain from random projection alone? | **Critical** |
| Pruned VGG16 FC at 67K params | Compare to dense-to-sparse compression at same param count | High |
| Standard GNN (GCN or GAT, 67K params, fixed random graph) | Position SGNNET in GNN literature | High |
| Second dataset (CIFAR-10 or ImageNet-1K via VGG16 features) | Generalization beyond Imagenette | High |
| MLP at 0.98M FLOPs (not param-matched) | FLOPs-matched comparison | Medium |

**Most critical baseline is the random projection + linear baseline.** If a single matrix $W \in \mathbb{R}^{25088 \times 67K}$ (random, fixed, not trained) followed by a linear classifier matches 95.52%, the routing dynamics contribute nothing. This baseline must be run before submitting to any venue.

The second most critical baseline is the 67K-param MLP. If a 2-layer MLP with the same parameter budget matches SGNNET accuracy, the $O(N \times K)$ architecture constraint is not the source of efficiency — the parameter count alone is doing the work.

---

## 9. Future Work

**Cross-dataset validation.** Run SGNNET on CIFAR-10, ImageNet-1K, and at least one non-vision dataset (e.g., tabular via tree-ensemble features). The routing dynamics should be dataset-agnostic.

**FFN replacement in transformers.** Substitute SGNNET for the MLP sublayer in a small transformer (e.g., 6-layer, 512 hidden dim). Test on language modeling and fine-tuning tasks. The key question: do routing dynamics generalize from IID classification to sequential, contextual representations?

**Dynamic routing with redistribution.** The gate-death theorem rules out all multiplicative gating. Softmax-normalized routing (where gates sum to 1 over neighbors) preserves signal mass and has not been comprehensively tested at the efficiency scale ($D=16, K_{hh}=2$). Steps 73/75 showed early promise; this direction is not closed.

**Theoretical characterization of the D-ceiling.** The empirical ceiling at $D=16$ for this dataset needs a formal explanation. Candidate frameworks: covering numbers on $S^{D-1}$, Rademacher complexity of the routing function class, or information-theoretic capacity bounds on the hypersphere.

**Hardware co-design.** The gather-sum routing pattern ($Z[:, \text{conn\_hh}, :].sum(2)$) is an indexed gather followed by a reduction — a pattern well-suited to custom sparse tensor cores or graph-optimized accelerators. The fixed connectivity (no dynamic sparsity decisions) enables compile-time optimization of the routing loop.

---
