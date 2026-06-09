# Sparse Geometric Neural Networks: Surpassing Dense Classifier Accuracy at 0.16% Compute via Iterative Routing on Random Graphs

**Draft v0.2 — 2026-04-24**

---

## Abstract

Dense fully-connected (FC) classification heads dominate inference compute in modern vision pipelines, yet their expressiveness derives almost entirely from learned weight matrices rather than architectural structure. We introduce the Sparse Geometric Neural Network (SGNNET), a classifier that replaces the FC head with $N$ neurons fixed on the unit hypersphere $S^{D-1}$, connected by a static random small-world graph, and iterated for $K_\text{iter}$ message-passing rounds. The entire compute budget is determined by five integers: $3 \times N \times K_{hh} \times D \times K_\text{iter}$.

Applied to Imagenette (a 10-class subset of ImageNet, $\sim$13k training images) using frozen VGG16 pool5 features as input, SGNNET *surpasses* VGG16 FC accuracy (**96.38% $\pm$ 0.18pp** vs. $\sim$95%, step887 T2 multi-seed) at only **0.20M routing-MACs — 0.16% of the 123.6M FLOPs consumed by VGG16's FC layers** — using only **34,976 learned parameters (0.029% of VGG16's 119.6M parameters)**. In wall-clock time at batch size 32, SGNNET runs at **12.7 µs** versus VGG16 FC's 66.7 µs: **5.26× faster inference** (bench\_step608, RTX 5060 Ti). Knowledge-distillation from a $K_\text{iter}=5$ teacher to a $K_\text{iter}=1$ student (step605) enables this extreme compression without accuracy loss.

The central routing mechanism is **ΔW-projection**: at each message-passing step, neighbor activations are weighted by their absolute projection onto the normalized difference of learned position vectors ($\Delta\mathbf{W}_{ij} = (\mathbf{W}_i - \mathbf{W}_j)/\|\mathbf{W}_i - \mathbf{W}_j\|$). Removing this mechanism collapses accuracy by 62–77pp across both image datasets tested (steps 883, 915). ΔW-projection also halves seed-to-seed variance (±0.43pp → ±0.18pp, step760).

Beyond the efficiency result, we establish five empirical laws from $\sim$943 controlled experiments: (1) representational dimensionality $D$ dominates connectivity density $K_{hh}$ at fixed FLOPs; (2) accuracy scales monotonically with neuron count $N$ up to a dimension-dependent ceiling; (3) optimal routing depth $K_\text{iter}=5$ is universal across datasets, with over-smoothing at $K_\text{iter}>5$ that is dataset-independent; (4) any multiplicative gate $g \in [0,1]$ in the routing loop produces signal attenuation $\propto g^{K_\text{iter}}$, explaining the failure of all 30+ gated routing mechanisms tested; and (5) ΔW-projection — which uses the geometry of learned neuron directions on $S^{D-1}$ — is the load-bearing routing primitive across image classification datasets.

These findings collectively suggest that the routing dynamics — specifically the geometry of $W_\text{pos}$ directions and the ΔW projection — not the graph topology or weight magnitudes, are the primary source of representational capacity in sparse random networks. Cross-dataset results on CIFAR-10 confirm generalization; text and audio modalities remain honest negatives (paper scope: vision classification).

---

## 1. Introduction

### 1.1 The Dense Classifier Problem

The modern recognition pipeline is a tale of two computational regimes. The feature extractor (VGG16, ResNet, ViT) processes rich spatial structure through billions of multiply-accumulate operations, carefully tuned to learn hierarchical visual features. The classification head, by contrast, is a pair of dense matrix multiplications: two FC layers with 4096 neurons each, consuming 119.6 million parameters and 123.6M FLOPs. This head is architecturally uninteresting — a universal approximator applied with no structural bias — yet it accounts for the majority of both parameter count and inference cost in deployed VGG16 models. At batch size 32, VGG16's FC head requires 66.7 µs on modern GPU hardware (RTX 5060 Ti); this is the latency budget we target.

The question motivating this work is direct: can a fundamentally different architecture match — or exceed — the accuracy of this dense head at a small fraction of its compute, while being trained from scratch with no pruning from a larger model?

We answer affirmatively. Our model, SGNNET, achieves **96.38% ± 0.18pp** top-1 accuracy on Imagenette versus the VGG16 FC baseline's $\sim$95%, using **0.16%** of the baseline's FLOPs and **0.029%** of its parameters. This is not parameter-for-parameter matching: it is strict Pareto dominance on four of five efficiency dimensions simultaneously (accuracy, parameters, FLOPs, and wall-time; the exception is peak memory due to intermediate routing tensors).

### 1.2 Our Approach

We propose that dense FC layers can be replaced by a sparse random graph with iterative message passing. The key insight is that learning does not require dense, learned connectivity. Instead, a fixed random graph of $N$ neurons occupying positions on a $D$-dimensional hypersphere, iterated for $K_\text{iter}$ message-passing rounds, can discover a compact sufficient representation of the input through the dynamics of routing — not through learned edge weights.

This hypothesis connects to two classical observations. First, Johnson-Lindenstrauss theory guarantees that random projections approximately preserve pairwise distances: $N$ random projections of a high-dimensional input (25,088-dim for VGG16 pool5) span the same information-preserving subspace as a trained basis, provided $N$ is large enough. Second, iterative refinement through message passing propagates local agreement signals through the graph until a stable attractor is reached — analogous to how cortical lateral inhibition sharpens early feature representations without requiring a learned weight for each neuron-neuron pair.

The resulting model, SGNNET (Sparse Geometric Neural Network), has a hard $O(N \times K)$ parameter budget: its FLOP count is $3 \times N \times K_{hh} \times D \times K_\text{iter}$, fully determined before training begins.

### 1.3 Contributions

This paper makes six contributions:

1. **Architecture strictly Pareto-dominating VGG16 FC on efficiency**: SGNNET surpasses VGG16 FC accuracy (**96.38% ± 0.18pp** vs. $\sim$95%, step887 multi-seed T2) at 0.20M routing-MACs — **0.16% of VGG16 FC compute** — using only **34,976 parameters** (0.029% of VGG16 FC). In wall-time at $B=32$, SGNNET runs at 12.7 µs vs. VGG\_FC at 66.7 µs: **5.26× faster inference** (bench\_step608). Knowledge-distillation from the ΔW-proj teacher ($K_\text{iter}=5 \to K_\text{iter}=1$ student, step605) enables this extreme compression without accuracy loss.

2. **The ΔW-projection routing mechanism**: At each routing step, neighbor activations are weighted by their absolute projection onto $\Delta\mathbf{W}_{ij} = (\mathbf{W}_i - \mathbf{W}_j)/\|\mathbf{W}_i - \mathbf{W}_j\|$ — the learned geometric direction between neuron positions. Removing this mechanism causes 62–77pp accuracy collapse (steps 883, 915). ΔW-projection is load-bearing on both Imagenette and CIFAR-10, confirming cross-dataset generalization. It also halves training seed variance (±0.43pp → ±0.18pp, step760).

3. **The $D > K_{hh}$ principle**: At fixed FLOPs, increasing the geometric dimensionality $D$ of the hypersphere dominates increasing the per-neuron connectivity $K_{hh}$. This holds across two FLOPs levels with clean controlled ablations.

4. **$N$-scaling laws with dimension ceiling**: Accuracy scales monotonically with $N$ up to a ceiling determined by $D$. At $D=16$, the ceiling is 97.30% — achievable at 1.97M FLOPs (1.59% of VGG16 FC).

5. **Gate-death theorem**: Any multiplicative gate $g \in [0,1]$ in the routing loop produces compounding signal attenuation $\propto g^{K_\text{iter}}$. This single principle explains the failure of all 30+ gated routing mechanisms tested over $\sim$943 experiments.

6. **MLP bottleneck and routing capacity advantage**: At the same parameter budget (34,976 params), a 2-layer MLP achieves only 14.3–17.1% on CIFAR-10 due to the N_in=25,088 information bottleneck. SGNNET achieves 80.4% at the same budget via N=2048 parallel nodes. The minimum viable MLP requires 401K parameters (h=16, 11.5× SGNNET) to match SGNNET accuracy; MLPs with h≤12 (8.6×, 301K params) still fail significantly at 67.1% (steps 891–893, T2 confirmed). The routing mechanism — not parameter count — is the source of representational capacity at high-dimensional inputs.

7. **Complete negative results catalog**: We document all 30+ killed mechanisms, organized by failure mode, with single-experiment evidence for each. Negative results include audio modality (ESC-50: −11.5pp vs. linear, step928) and text modality (SST-2/AG News: −1–2pp, steps 410–411), confirming paper scope is vision classification.

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

