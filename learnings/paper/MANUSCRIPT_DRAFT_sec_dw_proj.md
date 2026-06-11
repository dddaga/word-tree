## 3.5 ΔW-Projection Routing

**Claim: ΔW-projection uses learned $W_\text{pos}$ geometry on $S^{D-1}$ to compute message weights, enabling selective signal propagation essential for non-trivial classification accuracy.**

### Direction vectors

For each neuron $i$ and each $K_{hh}$ neighbours $j \in \mathcal{N}(i)$, the **ΔW direction vector** is:

$$\Delta \mathbf{W}_{ij} = \frac{\mathbf{W}_i - \mathbf{W}_j}{\|\mathbf{W}_i - \mathbf{W}_j\|_2} \;\in\; S^{D-1}$$

where $\mathbf{W}_i, \mathbf{W}_j \in \mathbb{R}^D$ are rows of learned position matrix $W_\text{pos}$. This vector points from $j$'s position toward $i$'s position — the direction of increasing "closeness to $i$" in the geometric embedding.

### Projection coefficient and aggregation

At each routing step, the **projection coefficient** for neighbour $j$ is:

$$c_{ij} = \mathbf{Z}_j \cdot \Delta\mathbf{W}_{ij}$$

where $\mathbf{Z}_j \in S^{D-1}$ is $j$'s current activation. Aggregated signal for neuron $i$:

$$\mathbf{Z}_\text{agg}^{(i)} = \sum_{j \in \mathcal{N}(i)} \mathbf{Z}_j \cdot |c_{ij}|$$

$|c_{ij}|$ rather than signed $c_{ij}$ is load-bearing: neighbours whose activation strongly aligns with $\Delta\mathbf{W}_{ij}$ in either direction contribute large positive weight; orthogonal neighbours contribute near-zero. Replacing $|c_{ij}|$ with raw signed $c_{ij}$ costs $-0.59$pp at Tier-1 (step886, CONFIRMED).

### Reflection memory

A scalar-decayed **reflection signal** accumulates per-step change:

$$\mathbf{Z}_\text{ref}^{(t)} = \alpha_r \,\mathbf{Z}_\text{ref}^{(t-1)} + \left(\mathbf{Z}_\text{fwd}^{(t)} - \mathbf{Z}^{(t-1)}\right), \quad \alpha_r = 0.5$$

where $\mathbf{Z}_\text{fwd}^{(t)} = \text{LeakyReLU}_{0.01}(\mathbf{Z}^{(t-1)} - \theta)$ is the thresholded activation and $\theta$ is a learned per-neuron scalar threshold. Reflection carries short-memory residual across iterations. Removing it costs $-0.51$pp at Tier-1 (step886, CONFIRMED). The threshold $\theta$ is neutral: fixing $\theta=0$ yields $+0.15$pp at Tier-1 (step886, CONFIRMED).

### Full routing update

The routing loop for $K_\text{iter}=5$ steps:

$$\mathbf{Z}^{(t)} = F.\text{normalize}\!\left(\text{clamp}\!\left(\mathbf{Z}_\text{agg}^{(t)} + \mathbf{Z}_\text{ref}^{(t)},\; -10,\; 10\right),\; \text{dim}=-1\right)$$

initialised with $\mathbf{Z}_\text{ref}^{(0)} = \mathbf{0}$ and $\mathbf{Z}^{(0)}$ from the seed step. The clamp prevents magnitude explosion; $F.\text{normalize}$ projects back to $S^{D-1}$ at each iteration, enforcing the spherical geometry that $\Delta\mathbf{W}$ direction vectors encode (consistent with Wall 1, §5.6).

### Precomputed geometry

The $\Delta\mathbf{W}$ tensor of shape $[N, K_{hh}, D]$ is computed **once per forward pass**, before the $K_\text{iter}$ loop, from the current $W_\text{pos}$. It is constant across all routing steps and all batch elements. This single-pass cost ($N K_{hh} D$) is one-fifth of a single routing step's cost ($3 N K_{hh} D$ at $K_\text{iter}=5$).

### Structural necessity (step950)

**CONFIRMED (step950):** Replacing ΔW-projection with plain gather-sum while adding explicit learned $W_\text{edge}$ weights causes Z-collapse at initialisation (loss=13.19 at epoch 1 vs loss=2.3 baseline). The ΔW-projection mechanism is structurally load-bearing — the positional geometry must be embedded in the routing direction, not in separate edge parameters that bypass it.

### Routing gain diagnostic (step951)

**CONFIRMED (step951):** Per-class routing gain (ratio of post-routing to pre-routing class discriminability, "Haki diagnostic") is **always negative** across all classes and seeds. Routing does not amplify discriminative signal; it spatially smooths it. Seed Fisher score is the dominant factor in final accuracy — the seed step determines the class-relevant geometry; the routing loop refines it through consensus. This inverts the naive interpretation of "routing as amplification."

### Pathway specialisation diagnostic (step967)

**CONFIRMED (step967):** SGNNET is a **dense non-selective router**. Diagnostic measurements at step967 (3 seeds):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| mean\_act\_frac | 1.0 | All neurons active for all inputs |
| intra\_jaccard | 1.0 | Same neurons active within a class |
| inter\_jaccard | 1.0 | Same neurons active across classes |
| separation | 0.0 | No class-specific pathway separation |

All neurons activate for all inputs regardless of class. There is no pathway specialisation. The correct framing is **collective refinement**: routing iteratively updates all neurons jointly toward a class-discriminative consensus, rather than routing different classes through different subgraphs.

### Proximal gradient interpretation (HYPOTHESIS)

**(HYPOTHESIS)** The $K_\text{iter}$ routing steps can be interpreted as proximal gradient descent on $S^{D-1}$ with $\Delta\mathbf{W}_{ij}$ as the descent direction. Under this interpretation, each routing step takes one proximal step minimising a loss that mixes the seed activation (data term) with neighbourhood consensus (regularisation term), and $F.\text{normalize}$ applies the proximal projection back to the manifold. This framing is not derived from a first-principles loss function; it is a post-hoc geometric interpretation. Direct ablation distinguishing proximal-GD from alternative fixed-point formulations has not been conducted.

---

## 5.8 ΔW-Projection Ablation

**(See §3.5 for mechanism description. This section reports the quantitative ablation.)**

**Claim: ΔW-projection has three load-bearing components and one neutral component. All three are individually necessary; removing any single one degrades performance significantly at Tier-1.**

### Component ablation table

| Component | T0 delta | T1 delta | Verdict |
|-----------|----------|----------|---------|
| Geometry ($W_\text{pos}$ directions vs random) | $-76.56$pp | — (catastrophic) | ESSENTIAL (CONFIRMED) |
| $\|c_{ij}\|$ weighting (abs vs signed) | $-1.83$pp | $-0.59$pp | LOAD-BEARING (CONFIRMED) |
| Reflection memory $\mathbf{Z}_\text{ref}$ (vs $\alpha_r=0$) | $-1.32$pp | $-0.51$pp | LOAD-BEARING (CONFIRMED) |
| Threshold $\theta$ (vs $\theta=0$) | $-0.71$pp | $+0.15$pp | NEUTRAL — T0 artifact (CONFIRMED) |

Source: step883 (T0), step886 (T1), $N=2048$, $D=16$, $K_{hh}=2$, $K_\text{iter}=5$, $\alpha_r=0.5$.

### Geometry ablation

The geometry ablation (D\_rand\_dir, step883) replaces $W_\text{pos}$-derived $\Delta\mathbf{W}_{ij}$ with random unit vectors: $-76.56$pp. This is the strongest evidence that the mechanism exploits the learned positional geometry on $S^{D-1}$, not merely the structural form of projection-weighted aggregation. The mechanism is not a generic attention scheme.

### Cross-dataset transfer (step915)

On CIFAR-10 (step915, T1), removing ΔW-projection (reverting to unweighted gather-sum) collapses accuracy by $-62.41$pp, consistent with the Imagenette ablation. ΔW-projection is essential for non-trivial generalisation across datasets.

### Canonical multi-seed result

**96.38% $\pm$ 0.18pp** (step887, 3 seeds, $N=2048$, $D=16$, 34,976 params). ΔW-projection halves seed variance relative to the unweighted gather-sum baseline (step199, $\pm$0.43pp), indicating a more stable optimum landscape.

---

*Linked from: §3 Architecture (§3.5 anchor) and §5 Key Findings (§5.8 anchor). See also DRAFT\_dw\_proj\_section.md for derivation notes.*
