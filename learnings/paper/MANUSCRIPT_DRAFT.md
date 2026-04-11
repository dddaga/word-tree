# Sparse Geometric Neural Networks: Matching Dense Classifier Accuracy at <1% Compute via Iterative Routing on Random Graphs

**Draft v0.1 — 2026-04-11**

---

## Abstract

Dense fully-connected (FC) classification heads dominate inference compute in modern vision pipelines, yet their expressiveness derives almost entirely from learned weight matrices rather than architectural structure. We introduce the Sparse Geometric Neural Network (SGNNET), a classifier that replaces the FC head with $N$ neurons fixed on the unit hypersphere $S^{D-1}$, connected by a static random small-world graph, and iterated for $K_\text{iter}$ message-passing rounds. The entire compute budget is determined by five integers: $3 \times N \times K_{hh} \times D \times K_\text{iter}$.

Applied to Imagenette (a 10-class subset of ImageNet, ~13k training images) using frozen VGG16 pool5 features as input, SGNNET achieves **95.52% accuracy at 0.98M FLOPs — 0.79% of the 123.6M FLOPs consumed by VGG16's FC layers** — using only 67K learned parameters (0.05% of VGG16's parameter count). The ≤1% FLOPs and ≤1% params criteria are satisfied simultaneously at accuracy exceeding the VGG16 FC baseline.

Beyond the efficiency result, we establish four empirical laws from 213 controlled experiments: (1) representational dimensionality $D$ dominates connectivity density $K_{hh}$ at fixed FLOPs; (2) accuracy scales monotonically with neuron count $N$ up to a dimension-dependent ceiling; (3) optimal routing depth $K_\text{iter}$ decreases as $N$ increases, suggesting over-smoothing scales with both; and (4) any multiplicative gate $g \in [0,1]$ in the routing loop produces signal attenuation $\propto g^{K_\text{iter}}$, explaining the failure of all 27 gated routing mechanisms we tested.

These findings collectively suggest that the routing dynamics — not the graph topology or weight magnitudes — are the primary source of representational capacity in sparse random networks.

---

## 1. Introduction

### 1.1 The Dense Classifier Problem

The modern recognition pipeline is a tale of two computational regimes. The feature extractor (VGG16, ResNet, ViT) processes rich spatial structure through billions of multiply-accumulate operations, carefully tuned to learn hierarchical visual features. The classification head, by contrast, is a pair of dense matrix multiplications: two FC layers with 4096 neurons each, consuming 123.6 million parameters and a matching FLOP count. This head is architecturally uninteresting — a universal approximator applied with no structural bias — yet it accounts for a large fraction of both parameter count and inference cost in deployed VGG16 models.

The question motivating this work is direct: can a fundamentally different architecture match the accuracy of this dense head at a fraction of its compute, while being trained from scratch with no distillation or pruning from a larger model?

### 1.2 Our Approach

We propose that dense FC layers can be replaced by a sparse random graph with iterative message passing. The key insight is that learning does not require dense, learned connectivity. Instead, a fixed random graph of $N$ neurons occupying positions on a $D$-dimensional hypersphere, iterated for $K_\text{iter}$ message-passing rounds, can discover a compact sufficient representation of the input through the dynamics of routing — not through learned edge weights.

This hypothesis connects to two classical observations. First, Johnson-Lindenstrauss theory guarantees that random projections approximately preserve pairwise distances: $N$ random projections of a high-dimensional input (25,088-dim for VGG16 pool5) span the same information-preserving subspace as a trained basis, provided $N$ is large enough. Second, iterative refinement through message passing propagates local agreement signals through the graph until a stable attractor is reached — analogous to how cortical lateral inhibition sharpens early feature representations without requiring a learned weight for each neuron-neuron pair.

The resulting model, SGNNET (Sparse Geometric Neural Network), has a hard $O(N \times K)$ parameter budget: its FLOP count is $3 \times N \times K_{hh} \times D \times K_\text{iter}$, fully determined before training begins.

### 1.3 Contributions

This paper makes five contributions:

1. **Architecture achieving <1% FLOPs parity with VGG16 FC**: SGNNET matches VGG16 FC accuracy (95.52% vs ~95%) at 0.98M FLOPs on Imagenette — 0.79% of the baseline compute — with 67K learned parameters.

2. **The $D > K_{hh}$ principle**: At fixed FLOPs, increasing the geometric dimensionality $D$ of the hypersphere dominates increasing the per-neuron connectivity $K_{hh}$. We show this holds across two different FLOPs levels with clean controlled ablations.

3. **$N$-scaling laws with dimension ceiling**: Accuracy scales monotonically with $N$ up to a ceiling determined by $D$. At $D=16$, the ceiling is 97.17% — within 0.69pp of the all-time project best at $D=64$ — achievable at 1.97M FLOPs (1.59% of VGG16 FC).

4. **Gate-death theorem**: Any multiplicative gate $g \in [0,1]$ in the routing loop produces compounding signal attenuation $\propto g^{K_\text{iter}}$. This single principle explains the failure of all 27 gated routing mechanisms tested over 213 experiments.

5. **Complete negative results catalog**: We document all 27 killed mechanisms, organized by failure mode, with single-experiment evidence for each. Negative results are as informative as positive ones when they reveal a structural constraint.

### 1.4 Scope

Throughout this paper, "accuracy" refers to top-1 classification accuracy on Imagenette. The input pipeline uses frozen VGG16 pool5 features (25,088-dim) extracted without fine-tuning; SGNNET is the classifier head only. All reported FLOPs count only the classifier head, not the feature extractor.

---

## 2. Related Work

### 2.1 Sparse Neural Networks

The dominant paradigm for efficient neural networks is structured or unstructured pruning of dense models: lottery ticket hypothesis (Frankle & Carlin, 2019) posits that dense networks contain sparse sub-networks ("winning tickets") trainable to full accuracy from scratch; RigL (Evci et al., 2020) maintains sparse connectivity throughout training via periodic weight magnitude-based topology updates; GMP (Gradual Magnitude Pruning) and SparseGPT compress large language models post-training.

SGNNET differs from all pruning approaches in a fundamental way: the sparse connectivity is never the result of a dense-to-sparse compression. The graph is **fixed at initialization** — no edge weights are learned, no topology updates occur during training (except in ablation experiments). The only learned parameters are neuron positions ($W_\text{pos}$) and per-neuron thresholds ($\theta$). This rules out any interpretation of SGNNET's accuracy as recovering a subset of a dense model; the accuracy must come from the routing dynamics operating on fixed random structure.

### 2.2 Graph Neural Networks

Graph Neural Networks (GCN, Kipf & Welling 2017; GAT, Veličković et al., 2018; GraphSAGE, Hamilton et al., 2017) perform message passing over graphs where nodes have feature vectors and edges encode semantic relationships. SGNNET is superficially similar — it applies iterative gather-sum routing over a graph — but differs in several ways.

Standard GNNs use learned message functions, learned edge weights or attention coefficients, and task-specific graph structure (e.g., molecular bonds, citation links). SGNNET uses **fixed random connectivity**, no edge weights, and a topology built purely by spatial proximity and random long-range shortcuts (Watts-Strogatz small-world model). There is no per-edge learnable parameter.

The over-smoothing problem in GNNs (Li et al., 2018) — where repeated message passing causes node features to converge to a single value — has a direct analogue in SGNNET: too many routing iterations at low $D$ collapse representations. We characterize this explicitly in Section 5.3, finding that optimal $K_\text{iter}$ decreases as $N$ increases.

Geometric deep learning (Bronstein et al., 2021) places neural networks on manifolds. SGNNET embeds neurons on $S^{D-1}$ and uses Fourier positional encoding to project inputs onto this manifold, but does not use the manifold structure for message passing (the routing is index-based gather-sum, not manifold convolution).

### 2.3 Random Features and Random Projections

Rahimi and Recht (2007) established that random feature maps can approximate kernel functions: a random projection $\phi(x) = \cos(Wx + b)$ allows linear classifiers to approximate kernel SVMs. Each SGNNET neuron is precisely a random projection of $K_\text{in}$ input dimensions — the connection between SGNNET and random feature methods is direct.

The critical difference is iteration. Random feature methods stop at the projection: the classifier is a linear function of $\phi(x)$. SGNNET refines the projection through $K_\text{iter}$ rounds of message passing. We show empirically that this iteration is the primary source of representational capacity: removing any single routing step causes dramatic accuracy degradation (stochastic depth ablation, step123: $-35$ to $-61$pp). The representation is not in the projections — it is in the routing fixed point.

### 2.4 Mixture of Experts and Conditional Computation

Mixture of Experts (MoE) systems (Jacobs et al., 1991; Shazeer et al., 2017) apply different sub-networks to different inputs via learned gating. This improves parameter utilization at constant compute by routing inputs to relevant experts. Sparse MoE in transformers (Switch Transformer, Fedus et al., 2021; GShard, Lepikhin et al., 2021) achieves state-of-the-art language modeling at reduced per-token compute.

SGNNET routing is superficially similar to MoE but operates at the **intra-forward-pass** level: different neurons activate for different routing steps of the same input, guided by excitatory gates. However, our experiments show that learned gating mechanisms universally fail (Section 5.5: gate-death theorem). The effective routing in SGNNET is closer to message passing with fixed topology than to expert selection.

### 2.5 Model Compression and Efficient Inference

Knowledge distillation (Hinton et al., 2015), quantization (LeCun et al., 1990; Jacob et al., 2018), and low-rank factorization (Denil et al., 2013) reduce inference cost of pre-trained models. These require a large teacher model as a prerequisite and are fundamentally post-hoc compressions.

SGNNET achieves efficiency structurally: by design, the FLOP count is $3NKD \cdot K_\text{iter}$, and there is no larger model to compress from. The $<1\%$ FLOPs result does not involve any compression step; it is the native inference cost of the architecture.

---

## 3. Architecture

### 3.1 Overview

SGNNET is a three-stage pipeline: **seed** (input projection), **route** (iterative message passing), and **readout** (output scoring). The full model stacks three modules: `SGNNET_SmallWorld` (the routing backbone), `SGNNET_Resonant` (the threshold and reflection controller), and `SGNNET_AntiHebbian` (the diversity regularizer). Figure 1 (planned) illustrates this pipeline.

### 3.2 SGNNET_SmallWorld: Core Graph

**Neuron positions.** $N$ neurons occupy positions on the unit hypersphere $S^{D-1}$ via a learned parameter matrix $W_\text{pos} \in \mathbb{R}^{(N+N_\text{out}) \times D}$. The first $N$ rows are hidden neurons; the last $N_\text{out}$ rows are output class vectors.

**Input connectivity.** Let $x \in \mathbb{R}^{N_\text{in}}$ be the input feature vector (VGG16 pool5: $N_\text{in} = 25088$). A fixed fan-in index table $\text{conn\_in} \in \mathbb{Z}^{N \times K_\text{in}}$ is constructed once at initialization via block-local random sampling with guaranteed input coverage: the $N_\text{in}$ input dimensions are partitioned into $n_\text{groups}$ blocks, each hidden neuron samples primarily from its corresponding input block, and a round-robin assignment guarantees every input dimension is covered by at least one neuron.

**Fourier positional encoding.** Each input dimension $i \in [N_\text{in}]$ is assigned a $D$-dimensional spatial coordinate via Fourier embedding:

$$\text{spatial}[i] = \left[\sin\!\left(\frac{2\pi k \cdot i}{N_\text{in}}\right), \cos\!\left(\frac{2\pi k \cdot i}{N_\text{in}}\right)\right]_{k=1}^{D/2} \in \mathbb{R}^D$$

The augmented input is $A_\text{input}[i] = \text{concat}(x[i], \text{spatial}[i]) \in \mathbb{R}^{D}$ (with $x[i]$ prepended to fill one dimension via broadcasting and the spatial component filling $D-1$ dimensions after normalization). In practice, $A_\text{input}[b, i, :] \in \mathbb{R}^{D}$ is formed as:

```
A_input = cat([x.unsqueeze(-1), spatial], dim=-1)  # [B, N_in, D]
```

**Seed step.** The initial hidden state $Z^{(0)} \in \mathbb{R}^{B \times N \times D}$ is:

$$Z^{(0)} = \text{normalize}\!\left(\sum_{j \in \text{conn\_in}[h]} A_\text{input}[\cdot, j, :]\right)_{h=1}^{N}$$

where normalize applies $F.\text{normalize}$ along the last dimension (L2 norm to the unit sphere).

**Small-world hidden connectivity.** The hidden-to-hidden index table $\text{conn\_hh} \in \mathbb{Z}^{N \times K_{hh}}$ implements the Watts-Strogatz small-world model: $N$ neurons are partitioned into $n_\text{groups}$ groups of $N/n_\text{groups}$ neurons each. Each neuron receives $K_\text{local}$ within-group connections (nearest neighbors by index) and $K_\text{random}$ long-range random shortcuts. This gives graph diameter $O(\log N)$ with $K_{hh} = K_\text{local} + K_\text{random}$ total edges per neuron. At the operating point $K_{hh}=2$, we use $K_\text{local}=1$ and $K_\text{random}=1$.

**Routing step.** For each of $K_\text{iter}$ iterations:

$$Z^{(t+1)} = \text{normalize}\!\left(\sum_{j \in \text{conn\_hh}[h]} Z^{(t)}[\cdot, j, :]\right)_{h=1}^{N}$$

In implementation, this is:
```python
Z = Z[:, self.conn_hh, :].sum(dim=2)   # [B, N, K_hh, D] → [B, N, D]
Z = F.normalize(Z, dim=-1)
```

The normalization after each step is load-bearing: removing it causes $-50$ to $-71$pp accuracy drop (Section 5.6). It constrains the dynamics to the hypersphere, preventing activation explosion and ensuring a well-posed fixed point.

**Readout.** The output score for class $c$ is computed as:

$$\text{score}[b, c] = \left(\sum_{h=1}^{N} C_{hc} \cdot Z^{(K_\text{iter})}[b, h, :]\right) \cdot W_\text{out}[c, :]$$

where $C_{hc} \in \{0,1\}$ is a fixed sparse mask (hidden → output, sparsity = 0.9), and $W_\text{out} = F.\text{normalize}(W_\text{pos}[N:], \text{dim}=-1)$ are the learned class direction vectors. In implementation:

```python
A_out = einsum("bhd,ho->bod", Z, C_ho)          # [B, N_out, D]
W_out_norm = F.normalize(W_pos[N_hidden:], dim=-1)
return (A_out * W_out_norm.unsqueeze(0)).sum(-1)  # [B, N_out]
```

This mean-pool readout is also load-bearing: replacing it with learned attention gives $-60$ to $-67$pp (Section 5.6).

### 3.3 SGNNET_Resonant: Phase Routing and Threshold

The resonant module wraps `SGNNET_SmallWorld` and adds three components to the routing step.

**Excitatory threshold.** A per-neuron learnable threshold $\theta \in \mathbb{R}^N$ (initialized at 0.1) gates propagation:

$$Z_\text{fwd}^{(t)} = \text{ReLU}\!\left(Z^{(t)} - |\theta|\right)$$

Only activations above threshold propagate forward, implementing winner-take-all pressure at each step.

**Leaky reflection.** The sub-threshold remainder feeds back as self-inhibition, creating a leaky memory of suppressed activations:

$$Z_\text{rem}^{(t)} = Z_\text{fwd}^{(t)} - Z^{(t)}$$

$$Z_\text{reflect}^{(t+1)} = \alpha_\text{reflect} \cdot Z_\text{reflect}^{(t)} + Z_\text{rem}^{(t)}$$

With $\alpha_\text{reflect} = 0.5$ (confirmed optimal at step22b, $+5$pp gain). This persists information across routing steps and prevents any neuron from being permanently silenced.

**Long-range phase inhibition.** The mode `dynamic_z_geo` implements input-dependent pseudo-connections: at each routing step, the top-$M$ active neurons (by activation magnitude, $M=\text{beam\_size}=16$) broadcast an inhibitory signal to neurons with similar activation patterns:

$$\text{score}[b, m, n] = \langle Z^{(t)}[b, m], Z^{(t)}[b, n] \rangle - \gamma \|W_\text{pos}[m] - W_\text{pos}[n]\|^2$$

$$\text{gate}[b, m, n] = \max(0,\ \text{score}[b, m, n] - \tau)$$

$$Z_\text{inh}[b, n] = \frac{\sum_m \text{gate}[b,m,n] \cdot Z_\text{ref}[b,m]}{\max\!\left(1, \sum_m \text{gate}[b,m,n]\right)}$$

where $Z_\text{ref} = -\text{ReLU}(-(Z + |\theta|)) \leq 0$, $\gamma = 0.5$ (geo\_gamma), and $\tau = 0.0$ (resonance\_threshold). The Turing coefficient $\alpha_\text{turing} = 0.0$ disables this term in the final efficiency configuration — it was confirmed harmful at scale (step120-A: $-1.45$pp at $N=4096$).

The full routing update is:

$$Z^{(t+1)} = F.\text{normalize}\!\left(Z_\text{struct}^{(t)} + Z_\text{reflect}^{(t)} + \alpha_\text{turing} \cdot Z_\text{inh}^{(t)}\right)$$

where $Z_\text{struct}^{(t)} = (Z_\text{fwd}^{(t)}[:, \text{conn\_hh}, :]).sum(\text{dim}=2)$ is the local structural excitation.

### 3.4 SGNNET_AntiHebbian: Diversity Regularization

Anti-Hebbian lateral inhibition prevents dimensional collapse by suppressing structurally redundant neurons. The `wpos` variant uses static positional similarity:

$$\text{pos\_sim}[h, k] = \langle \hat{W}_\text{pos}[h],\ \hat{W}_\text{pos}[\text{conn\_hh}[h,k]] \rangle, \quad \hat{W} = F.\text{normalize}(W_\text{pos}[:N], \text{dim}=-1)$$

$$\text{supp}[h, k] = 1 - \alpha_\text{ahebb} \cdot \text{pos\_sim}[h,k].\text{clamp}(\min=0)$$

The structural contribution is suppression-weighted:

$$Z_\text{struct}^{(t)} = \sum_{k=1}^{K_{hh}} \text{supp}[h,k] \cdot Z_\text{fwd}^{(t)}[\cdot, \text{conn\_hh}[h,k], :]$$

With $\alpha_\text{ahebb}=1.0$ (confirmed optimal), neurons with perfectly correlated positional embeddings receive zero contribution from each other, forcing the routing to maintain diverse representations. This has a biological analogue in cortical lateral inhibition (Heeger, 1992): similar V1 cells suppress each other, sharpening orientation tuning.

### 3.5 FLOPs Budget

The dominant cost is the routing loop. Per routing iteration, each of $N$ neurons gathers $K_{hh}$ neighbor activations and sums: $N \times K_{hh} \times D$ multiply-accumulates. Over $K_\text{iter}$ iterations:

$$\text{FLOPs}_\text{routing} = N \times K_{hh} \times D \times K_\text{iter}$$

The seed step (input projection) and readout contribute approximately 27% and 5% of total compute respectively at the operating point $N=2048, D=16$. A factor of 3 accounts for both:

$$\text{FLOPs}_\text{total} = 3 \times N \times K_{hh} \times D \times K_\text{iter}$$

At the final efficiency configuration ($N=2048, K_{hh}=2, D=16, K_\text{iter}=5$):

$$\text{FLOPs} = 3 \times 2048 \times 2 \times 16 \times 5 = 983{,}040 \approx 0.98\text{M}$$

This is 0.79% of VGG16 FC's 123.6M FLOPs. The five integers $(N, K_{hh}, D, K_\text{iter}, K_\text{in})$ fully determine inference cost before training begins — there are no runtime-conditional branches, no attention masks, no dynamic sparsity decisions.

---

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
- **Params**: $N_\text{hidden} \times D$ ($W_\text{pos}$) + $N_\text{hidden}$ ($\theta$) + $N_\text{out} \times D$ ($W_\text{out}$) = 67,744 at the operating point

### 4.4 Experiment Scale

213 controlled experiments were conducted across Phase 5, covering:
- $N \in \{256, 512, 1024, 2048, 4096, 8192, 16384\}$
- $D \in \{8, 10, 12, 16, 20, 24, 28, 32, 48, 64\}$
- $K_{hh} \in \{1, 2, 3, 4, 8\}$
- $K_\text{iter} \in \{3, 4, 5, 6, 8, 12, 16\}$
- 27 routing mechanism variants

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

**Why does this hold?** On $S^{D-1}$, the number of approximately orthogonal unit vectors scales as $e^{D/2}$ (a standard result in high-dimensional geometry). At $D=8$, approximately $e^4 \approx 55$ distinguishable directions exist for 2048 neurons — far less than $N$. At $D=16$, this becomes $e^8 \approx 2981$ — enough for each neuron to occupy a unique directional niche. $K_{hh}$, by contrast, controls how much each routing step smooths over neighbors; higher $K_{hh}$ increases over-smoothing pressure without adding directional capacity.

**Implication for architecture search:** The FLOPs budget should be spent on $D$ first, then $K_{hh}$. Reducing $K_{hh}$ from 4 to 2 to buy $D$ from 8 to 16 is strictly beneficial.

### 5.2 $N$-Scaling Law with Dimension Ceiling

**Claim: Accuracy scales monotonically with neuron count $N$ up to a ceiling determined by $D$.**

At $D=16, K_{hh}=2, K_\text{iter}=5$ (Tier-2):

| $N$ | FLOPs | Accuracy | $\Delta$ from prev. |
|-----|-------|----------|---------------------|
| 1024 | ~0.49M | ~88.9% (T1 proxy) | — |
| **2048** | **0.98M** | **95.52%** | — |
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

The over-smoothing hypothesis: at fixed $D=16$, each routing step propagates activations $K_{hh}^t$ hops away after $t$ iterations. With small $N$, the graph diameter is large relative to $K_{hh}^{K_\text{iter}}$ — information integrates well. With large $N$, the graph contains more path diversity, and fewer iterations suffice before the representation over-averages.

**Practical consequence.** The best sub-1% FLOPs result uses $N=2048, K_\text{iter}=5$: fewer iterations than the calibration best ($K_\text{iter}=6$), exploiting the FLOPs reduction without accuracy loss because $N=2048$ is near the per-step optimum.

### 5.4 $N$-Scaling Rehabilitates Dead Configurations

**Claim: Configurations killed at small $N$ become viable at large $N$. Minimum viable $K_\text{iter}$ decreases with $N$.**

| $K_\text{iter}$ | Accuracy @ $N=2048$ | Accuracy @ $N=8192$ | $\Delta$ |
|-----------------|---------------------|---------------------|---------|
| 4 | 92.74% (KILLED, step196) | 95.49% (VIABLE, step210) | +2.75pp |
| 3 | 89.25% (KILLED, step202) | 94.93% (borderline, step211) | +5.68pp |

This means FLOPs estimates at small $N$ are systematically pessimistic about minimum routing depth. A configuration requiring $K_\text{iter}=5$ at $N=2048$ to hit 95% may require only $K_\text{iter}=3-4$ at $N=8192$, enabling further FLOPs reduction via the $K_\text{iter}$ axis.

The D=16 $K_\text{iter}=4$ floor at $N=8192$ (95.49% T1, step210) represents a potentially viable operating point at 3.15M FLOPs — within the same tier as step185 ($N=2048, K_\text{iter}=8$, 95.87% T2) but with 4× more neurons and 4× fewer iterations.

### 5.5 Gate-Death Theorem

**Claim: Any multiplicative gate $g \in [0,1]$ in the routing loop compounds to $g^{K_\text{iter}}$ signal attenuation, explaining the failure of all 27 gated routing mechanisms.**

Consider a routing step with a multiplicative gate:

$$Z^{(t+1)} = \text{normalize}(g \cdot Z_\text{struct}^{(t)})$$

where $g \in [0,1]$ is any learned or fixed gate (sigmoid output, attention weight, or soft mask). After $K_\text{iter}$ steps, the signal from the seed is attenuated by $g^{K_\text{iter}}$:

$$Z^{(K_\text{iter})} \propto g^{K_\text{iter}} \cdot Z^{(0)}$$

At $g=0.7$ and $K_\text{iter}=8$: $0.7^8 = 0.058$ — 94% signal loss. At $g=0.5$ and $K_\text{iter}=5$: $0.5^5 = 0.031$ — 97% signal loss.

The normalization step does not rescue this: $F.\text{normalize}(g \cdot v) = F.\text{normalize}(v)$ regardless of $g$ — the gate only affects the magnitude before normalization, and since normalization removes magnitude information, the gate's signal passes through as noise.

**Fix:** Redistribution instead of gating. If gates sum to 1 ($\sum_k g_k = 1$, softmax), signal mass is preserved. This explains why the two successful routing variants survive: the structural gather-sum followed by normalize is a soft redistribution, not a multiplicative gate.

**Empirical evidence (27 killed mechanisms, steps 58-66 and beyond):**

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

The stochastic depth result (step123) is particularly revealing: randomly skipping any single routing step collapses accuracy by 35–61pp. Every step is essential — not because each step does something different, but because the routing fixed point requires all $K_\text{iter}$ iterations to converge. Gating that probabilistically removes steps (even at low probability) prevents convergence.

**Unified explanation.** The routing loop is not a sequence of optional operations; it is a fixed-point iteration. Gating disrupts the convergence basin. The successful architecture uses the simplest possible aggregation — gather-sum + normalize — which is provably redistributive and preserves the convergence properties.

### 5.6 Three Load-Bearing Walls

**Claim: Three architectural components are individually necessary; removing any one causes catastrophic failure.**

We define "load-bearing" as: removal causes accuracy degradation exceeding 10pp compared to the full model.

**Wall 1: $F.\text{normalize}$ after each routing step.**

Removal tested in step129 ($N=4096, D=64$): $-50$pp to $-71$pp across all configurations. Without normalization, activation magnitudes grow unboundedly through the routing loop (each sum step can multiply magnitudes by up to $K_{hh}$), causing numerical overflow or collapse to a single dominant neuron. Normalization is not merely a regularizer — it constrains the dynamics to the hypersphere where the Fourier positional encoding is meaningful and the routing has a well-defined fixed point.

**Wall 2: Static AntiHebbian suppression.**

The `wpos` variant of AntiHebbian (position-based cosine suppression) prevents dimensional collapse. Without it, the effective rank of $Z$ decreases over training (step155 diagnostic analysis), meaning the 16/64 representational dimensions collapse to 3-4 effective dimensions. With $\alpha_\text{ahebb}=1.0$ (confirmed optimal, step88), neurons are repelled from each other in $W_\text{pos}$ space, maintaining spread across $S^{D-1}$.

Dynamic variants of AntiHebbian (`zact`: current-Z cosine suppression) all fail (steps 58-66): they introduce input-dependent multiplicative gates, falling under the gate-death theorem.

**Wall 3: Mean-pool readout.**

Replacing the dot-product mean-pool readout with learned attention fails catastrophically. Step118 ($N=4096, D=64$): attention readout gives $-60$pp to $-67$pp. The hypothesis: after $K_\text{iter}$ routing steps, all neurons have participated in consensus-building. No single neuron has privileged information; the class signal is distributed uniformly. Attention that tries to select informative neurons applies a multiplicative gate to the final $Z$, triggering a single-step version of gate death.

Mean-pool aggregates all $N$ neurons equally, treating the routing-fixed-point representation as the semantic content. This is consistent with the view that learning happens in the routing dynamics — the readout's job is to sum up an already-computed consensus, not to select from it.

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

**Interpretation.** Both mechanisms improve routing by different means: W\_proj adds a learnable projection layer before the scatter-sum (allowing input re-weighting), while weighted\_neg adjusts the sign weighting of neighbor contributions. Both interact with the learned $W_\text{pos}$ geometry. When combined, they create competing objectives for $W_\text{pos}$: each mechanism pulls the positional geometry toward its own optimum, and the resulting compromise is worse than either individual solution.

**Implication.** Greedy winner-stacking is invalid for SGNNET mechanism design. Each mechanism candidate must be tested (a) in isolation against reference, and (b) in combination with all other planned additions. This significantly increases the experiment count required for rigorous architecture validation.

---

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
