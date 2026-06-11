# Sparse Geometric Neural Networks: Surpassing Dense Classifier Accuracy at 0.16% Compute via Iterative Routing on Random Graphs

**Draft v0.3 — 2026-06-11**

---

## Abstract

Dense fully-connected (FC) classification heads dominate inference compute in modern vision pipelines, yet their expressiveness derives almost entirely from learned weight matrices rather than architectural structure. We introduce the Sparse Geometric Neural Network (SGNNET), a classifier that replaces the FC head with $N$ neurons fixed on the unit hypersphere $S^{D-1}$, connected by a static random small-world graph, and iterated for $K_\text{iter}$ message-passing rounds. The entire compute budget is determined by five integers: $3 \times N \times K_{hh} \times D \times K_\text{iter}$.

Applied to Imagenette (a 10-class subset of ImageNet, $\sim$13k training images) using frozen VGG16 pool5 features as input, SGNNET *surpasses* VGG16 FC accuracy (**96.38% $\pm$ 0.18pp** vs. $\sim$95%, step887 T2 multi-seed) at only **0.20M routing-MACs — 0.16% of the 123.6M FLOPs consumed by VGG16's FC layers** — using only **34,976 learned parameters (0.029% of VGG16's 119.6M parameters)**. In wall-clock time at batch size 32, SGNNET runs at **12.7 µs** versus VGG16 FC's 66.7 µs: **5.26× faster inference** (bench\_step608, RTX 5060 Ti). Knowledge-distillation from a $K_\text{iter}=5$ teacher to a $K_\text{iter}=1$ student (step605) enables this extreme compression without accuracy loss.

The central routing mechanism is **ΔW-projection**: at each message-passing step, neighbor activations are weighted by their absolute projection onto the normalized difference of learned position vectors ($\Delta\mathbf{W}_{ij} = (\mathbf{W}_i - \mathbf{W}_j)/\|\mathbf{W}_i - \mathbf{W}_j\|$). Removing this mechanism collapses accuracy by 62–77pp across both image datasets tested (steps 883, 915). ΔW-projection also halves seed-to-seed variance (±0.43pp → ±0.18pp, step760).

Beyond the efficiency result, we establish five empirical laws from $\sim$943 controlled experiments: (1) representational dimensionality $D$ dominates connectivity density $K_{hh}$ at fixed FLOPs; (2) accuracy scales monotonically with neuron count $N$ up to a dimension-dependent ceiling; (3) optimal routing depth $K_\text{iter}=5$ is universal across datasets, with over-smoothing at $K_\text{iter}>5$ that is dataset-independent; (4) any multiplicative gate $g \in [0,1]$ in the routing loop produces signal attenuation $\propto g^{K_\text{iter}}$, explaining the failure of all 30+ gated routing mechanisms tested; and (5) ΔW-projection — which uses the geometry of learned neuron directions on $S^{D-1}$ — is the load-bearing routing primitive across image classification datasets.

These findings collectively suggest that the routing dynamics — specifically the geometry of $W_\text{pos}$ directions and the ΔW projection — not the graph topology or weight magnitudes, are the primary source of representational capacity in sparse random networks. Cross-dataset results confirm CIFAR-10 generalization (§8); audio (ESC-50), time-series (NIFTY50 financial), and text modalities are honest negatives (§8–10). SGNNET does not replicate GPT-2 FFN layer outputs (step989 KILLED: cos_sim=0.19 vs. threshold 0.5); paper scope is VGG16 FC replacement.

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

This paper makes seven contributions:

1. **Architecture strictly Pareto-dominating VGG16 FC on efficiency**: SGNNET surpasses VGG16 FC accuracy (**96.38% ± 0.18pp** vs. $\sim$95%, step887 multi-seed T2) at 0.20M routing-MACs — **0.16% of VGG16 FC compute** — using only **34,976 parameters** (0.029% of VGG16 FC). In wall-time at $B=32$, SGNNET runs at 12.7 µs vs. VGG\_FC at 66.7 µs: **5.26× faster inference** (bench\_step608). Knowledge-distillation from the ΔW-proj teacher ($K_\text{iter}=5 \to K_\text{iter}=1$ student, step605) enables this extreme compression without accuracy loss.

2. **The ΔW-projection routing mechanism**: At each routing step, neighbor activations are weighted by their absolute projection onto $\Delta\mathbf{W}_{ij} = (\mathbf{W}_i - \mathbf{W}_j)/\|\mathbf{W}_i - \mathbf{W}_j\|$ — the learned geometric direction between neuron positions. Removing this mechanism causes 62–77pp accuracy collapse (steps 883, 915). ΔW-projection is load-bearing on both Imagenette and CIFAR-10, confirming cross-dataset generalization. It also halves training seed variance (±0.43pp → ±0.18pp, step760).

3. **The $D > K_{hh}$ principle**: At fixed FLOPs, increasing the geometric dimensionality $D$ of the hypersphere dominates increasing the per-neuron connectivity $K_{hh}$. This holds across two FLOPs levels with clean controlled ablations.

4. **$N$-scaling laws with dimension ceiling**: Accuracy scales monotonically with $N$ up to a ceiling determined by $D$. At $D=16$, the ceiling is 97.30% — achievable at 1.97M FLOPs (1.59% of VGG16 FC).

5. **Gate-death theorem**: Any multiplicative gate $g \in [0,1]$ in the routing loop produces compounding signal attenuation $\propto g^{K_\text{iter}}$. This single principle explains the failure of all 30+ gated routing mechanisms tested over $\sim$943 experiments.

6. **MLP bottleneck and routing capacity advantage**: At the same parameter budget (34,976 params), a 2-layer MLP achieves only 14.3–17.1% on CIFAR-10 due to the N_in=25,088 information bottleneck. SGNNET achieves 80.4% at the same budget via N=2048 parallel nodes. The minimum viable MLP requires 401K parameters (h=16, 11.5× SGNNET) to match SGNNET accuracy; MLPs with h≤12 (8.6×, 301K params) still fail at 67.1% (steps 891–893, T2 confirmed).

7. **Complete negative results catalog**: We document all 30+ killed mechanisms organized by failure mode, with single-experiment evidence. Modality negatives: audio (ESC-50: SGNNET routing hurts vs. no-routing baseline, step964 — routing is vision-specific); text (SST-2/AG News: −1–2pp, steps 410–411); time-series (NIFTY50 financial: all models dir_acc≈50%, efficient market hypothesis, ts_step030). Founding hypothesis (GPT-2 FFN replacement) tested and retired (step989: cos_sim=0.19 vs. threshold 0.5). Paper scope: VGG16 FC replacement, vision classification.

### 1.4 Scope

Throughout this paper, "accuracy" refers to top-1 classification accuracy on Imagenette. The input pipeline uses frozen VGG16 pool5 features (25,088-dim) extracted without fine-tuning; SGNNET is the classifier head only. All reported FLOPs count only the classifier head, not the feature extractor.

We evaluated SGNNET as a replacement for the GPT-2 FFN (Transformer MLP sub-layer) in a separate T0 experiment (step989). The model failed to replicate layer outputs at 4× the FLOPs budget (cosine similarity 0.19 vs. 0.5 threshold), retiring the original founding hypothesis that SGNNET could generalize to Transformer architectures. All Transformer-FFN claims are excluded from this paper.

---

## 2. Related Work

### 2.1 Sparse Neural Networks

The dominant paradigm for efficient neural networks is structured or unstructured pruning of dense models: lottery ticket hypothesis (Frankle & Carlin, 2019) posits that dense networks contain sparse sub-networks ("winning tickets") trainable to full accuracy from scratch; RigL (Evci et al., 2020) maintains sparse connectivity throughout training via periodic weight magnitude-based topology updates; GMP (Gradual Magnitude Pruning) and SparseGPT compress large language models post-training.

SGNNET differs from all pruning approaches in a fundamental way: the sparse connectivity is never the result of a dense-to-sparse compression. The graph is **fixed at initialization** — no edge weights are learned, no topology updates occur during training. The accuracy must come from the routing dynamics operating on fixed random structure.

### 2.2 Graph Neural Networks

Graph Neural Networks (GCN, Kipf & Welling 2017; GAT, Veličković et al., 2018; GraphSAGE, Hamilton et al., 2017) perform message passing over graphs where nodes have feature vectors and edges encode semantic relationships. SGNNET uses **fixed random connectivity**, no edge weights, and topology built purely by spatial proximity and random long-range shortcuts (Watts-Strogatz small-world model). There is no per-edge learnable parameter.

The over-smoothing problem in GNNs (Li et al., 2018) has a direct analogue: too many routing iterations at low $D$ collapse representations. Geometric deep learning (Bronstein et al., 2021) places neural networks on manifolds; SGNNET embeds neurons on $S^{D-1}$ but routing is index-based gather-sum, not manifold convolution.

### 2.3 Random Features and Random Projections

Rahimi and Recht (2007) established that random feature maps can approximate kernel functions. Each SGNNET neuron is precisely a random projection of $K_\text{in}$ input dimensions. The critical difference is iteration: SGNNET refines through $K_\text{iter}$ rounds of message passing. Removing any single routing step causes dramatic accuracy degradation (stochastic depth ablation, step123: $-35$ to $-61$pp). The representation is not in the projections — it is in the routing fixed point.

### 2.4 Mixture of Experts and Conditional Computation

Mixture of Experts (MoE) systems (Jacobs et al., 1991; Shazeer et al., 2017) apply different sub-networks to different inputs via learned gating. SGNNET routing is superficially similar but operates at the intra-forward-pass level. Critically, our experiments show learned gating mechanisms universally fail (Section 5.5: gate-death theorem). The effective routing in SGNNET is closer to message passing with fixed topology than to expert selection.

### 2.5 Model Compression and Efficient Inference

Knowledge distillation (Hinton et al., 2015), quantization, and low-rank factorization reduce inference cost of pre-trained models but require a large teacher model as a prerequisite. SGNNET achieves efficiency structurally: by design, the FLOP count is $3NKD \cdot K_\text{iter}$, and there is no larger model to compress from.
