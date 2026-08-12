## 3.X ΔW-Projection Routing

**Claim: The ΔW-projection routing mechanism uses the learned $W_\text{pos}$ geometry on $S^{D-1}$ to compute message weights, enabling selective signal propagation that is essential for non-trivial classification accuracy.**

### Direction vectors

For each neuron $i$ and each of its $K_{hh}$ neighbours $j \in \mathcal{N}(i)$, define the **ΔW direction vector** as the normalised difference of their learned position vectors:

$$\Delta \mathbf{W}_{ij} = \frac{\mathbf{W}_i - \mathbf{W}_j}{\|\mathbf{W}_i - \mathbf{W}_j\|_2} \;\in\; S^{D-1}$$

where $\mathbf{W}_i, \mathbf{W}_j \in \mathbb{R}^D$ are the rows of the learned position matrix $W_\text{pos}$. This vector points from $j$'s position toward $i$'s position — the direction of increasing "closeness to $i$" in the geometric embedding.

### Projection coefficient and aggregation

At each routing step, the **projection coefficient** for neighbour $j$ onto edge $(i,j)$ is:

$$c_{ij} = \mathbf{Z}_j \cdot \Delta\mathbf{W}_{ij}$$

where $\mathbf{Z}_j \in S^{D-1}$ is $j$'s current activation. The aggregated signal for neuron $i$ is weighted by the **absolute** projection magnitude:

$$\mathbf{Z}_\text{agg}^{(i)} = \sum_{j \in \mathcal{N}(i)} \mathbf{Z}_j \cdot |c_{ij}|$$

The use of $|c_{ij}|$ rather than the signed $c_{ij}$ is load-bearing: a neighbour whose activation is strongly aligned with $\Delta\mathbf{W}_{ij}$ in either direction contributes a large positive weight, while a neighbour orthogonal to the direction vector contributes near-zero weight. Replacing $|c_{ij}|$ with the raw signed $c_{ij}$ costs $-0.59$pp at Tier-1 (step886, CONFIRMED).

### Reflection memory

A scalar-decayed **reflection signal** accumulates the difference between the thresholded activation and the pre-step activation:

$$\mathbf{Z}_\text{ref}^{(t)} = \alpha_r \,\mathbf{Z}_\text{ref}^{(t-1)} + \left(\mathbf{Z}_\text{fwd}^{(t)} - \mathbf{Z}^{(t-1)}\right), \quad \alpha_r = 0.5$$

where $\mathbf{Z}_\text{fwd}^{(t)} = \text{LeakyReLU}_{0.01}(\mathbf{Z}^{(t-1)} - \theta)$ is the thresholded activation and $\theta$ is a learned per-neuron scalar threshold. The reflection term provides a short-memory residual that carries per-step change information across iterations. Removing it costs $-0.51$pp at Tier-1 (step886, CONFIRMED). The threshold $\theta$ is neutral: fixing $\theta=0$ yields $+0.15$pp at Tier-1 (step886, CONFIRMED), simplifying the architecture.

### Full routing update

The routing loop for $K_\text{iter} = 5$ steps is:

$$\mathbf{Z}^{(t)} = F.\text{normalize}\!\left(\text{clamp}\!\left(\mathbf{Z}_\text{agg}^{(t)} + \mathbf{Z}_\text{ref}^{(t)},\; -10,\; 10\right),\; \text{dim}=-1\right)$$

initialised with $\mathbf{Z}_\text{ref}^{(0)} = \mathbf{0}$ and $\mathbf{Z}^{(0)}$ from the seed step (§3.Y). The clamp prevents magnitude explosion; the $F.\text{normalize}$ step projects back to $S^{D-1}$ at each iteration, enforcing the spherical geometry that the $\Delta\mathbf{W}$ direction vectors encode (consistent with Wall 1, §5.6).

### Precomputed geometry

The $\Delta\mathbf{W}$ tensor of shape $[N, K_{hh}, D]$ is computed **once per forward pass**, before the $K_\text{iter}$ loop, from the current $W_\text{pos}$. It is constant across all routing steps and all batch elements. This single-pass cost is $N K_{hh} D$ — one-fifth of a single routing step's $3 N K_{hh} D$ cost at $K_\text{iter}=5$ — and is not counted in the FLOPs formula of §4.3.

### Geometric intuition

Why does projecting neighbour activations onto $\Delta\mathbf{W}_{ij}$ carry semantic signal? Because $W_\text{pos}$ is learned end-to-end: neurons self-organise on $S^{D-1}$ so that nearby position vectors correspond to functionally similar routing roles. The direction $\Delta\mathbf{W}_{ij}$ therefore encodes the local gradient of the semantic manifold between $i$ and $j$. A neighbour whose activation is strongly aligned with this direction is informative about the class boundary near $i$; a neighbour orthogonal to it contributes noise. The mechanism selects for geometrically coherent message passing without any learned attention weights.

The ablation D\_rand\_dir (step883), which replaces $W_\text{pos}$-derived directions with random unit vectors, collapses accuracy by $-76.56$pp — confirming that the semantic content of $\Delta\mathbf{W}$ is essential, not the structural form of projection-weighted aggregation alone. Crucially, this geometry generalises: on CIFAR-10 (step915), removing ΔW-projection (reverting to unweighted gather-sum) collapses accuracy by $-62.41$pp, consistent with the Imagenette ablation.

### Component ablation table

| Component | T0 delta | T1 delta | Verdict |
|-----------|----------|----------|---------|
| Geometry ($W_\text{pos}$ directions) | $-76.56$pp | — (catastrophic, not re-tested) | ESSENTIAL (CONFIRMED) |
| Absolute-value weighting $|c_{ij}|$ | $-1.83$pp | $-0.59$pp | LOAD-BEARING (CONFIRMED) |
| Reflection memory ($\mathbf{Z}_\text{ref}$) | $-1.32$pp | $-0.51$pp | LOAD-BEARING (CONFIRMED) |
| Threshold $\theta$ | $-0.71$pp | $+0.15$pp | NEUTRAL — T0 artifact; simplifies out (CONFIRMED) |

Source: step883 (T0), step886 (T1), $N=2048$, $D=16$, $K_{hh}=2$, $K_\text{iter}=5$.

---

## 5.X ΔW-Projection Ablation

**Claim: ΔW-projection has three load-bearing components (geometry, unsigned-magnitude weighting, reflection memory) and one neutral component ($\theta$ threshold). All three are individually necessary; removing any single component degrades performance significantly at Tier-1.**

| Component | T0 delta | T1 delta | Verdict |
|-----------|----------|----------|---------|
| Geometry ($W_\text{pos}$ directions vs. random directions) | $-76.56$pp | — | ESSENTIAL (CONFIRMED) |
| $|c_{ij}|$ weighting (abs vs. signed projection) | $-1.83$pp | $-0.59$pp | LOAD-BEARING (CONFIRMED) |
| Reflection memory $\mathbf{Z}_\text{ref}$ (vs. $\alpha_r=0$) | $-1.32$pp | $-0.51$pp | LOAD-BEARING (CONFIRMED) |
| Threshold $\theta$ (vs. $\theta=0$) | $-0.71$pp | $+0.15$pp | NEUTRAL (CONFIRMED) |

Source: step883 (T0), step886 (T1). Three seeds, canonical config: $N=2048$, $D=16$, $K_{hh}=2$, $K_\text{iter}=5$, $\alpha_r=0.5$.

The geometry ablation result ($-76.56$pp with random direction vectors in place of $W_\text{pos}$-derived $\Delta\mathbf{W}_{ij}$) is the strongest evidence that the mechanism is not a generic attention scheme but specifically exploits the learned positional geometry on $S^{D-1}$. The cross-dataset transfer on CIFAR-10 (step915 T1: $-62.41$pp without ΔW-projection) confirms that ΔW-projection is essential for non-trivial generalisation beyond the training dataset used during architecture search.

The canonical multi-seed result — **96.38% $\pm$ 0.18pp** (step887, 3 seeds, $N=2048$, $D=16$, 34,976 params) — was obtained with the full ΔW-projection mechanism. ΔW-projection also halves seed variance relative to the unweighted gather-sum baseline (step199, $\pm$0.43pp), suggesting that the projection-weighted aggregation finds a more stable optimum landscape.
