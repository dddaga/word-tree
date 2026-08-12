# Learning Algorithms for SGNNET — Part 1: Algorithm Survey

**Parent:** [RESEARCH_learning_algorithms.md](RESEARCH_learning_algorithms.md)  
**Date:** 2026-04-11

---

## Architecture Context (What We're Actually Learning)

| Component | What It Is | Learned? |
|-----------|-----------|----------|
| `W_pos` (N × D) | Neuron positions on S^{D-1} | YES — ~67K params for N=2048, D=16 |
| `conn_hh` | Fixed random K-NN graph (K_hh=2) | NO — fixed at init |
| `fc_out` | Linear readout (10 × D) | YES — small |
| Routing kernel | Gather-sum-normalize over conn_hh, K_iter=5 | No params |
| Anti-Hebbian reg | Prevents W_pos collapse | Auxiliary loss term |

**Critical implication:** W_pos lives on S^{D-1}. Any algorithm must either handle Riemannian optimization or tolerate ambient ℝ^D + F.normalize projection.

---

## 1. Backpropagation (Current Baseline)

Reverse-mode autodiff through K_iter=5 routing rounds. Each round: sparse gather-sum + F.normalize. Loss = KL divergence + AH regularizer. Optimizer = AdamW.

**Strengths:** Exact gradient through F.normalize chain; mature hyperparams; handles unrolled routing loop.

**Weaknesses:**
- Gradient attenuates through K_iter × F.normalize (each normalize projects onto tangent plane)
- Global credit assignment: W_pos_i update depends on all N neurons' influence on final loss
- Cannot learn discrete topology (conn_hh stays fixed)
- AH regularizer penalizes global correlation — proxy, not local interaction signal

---

## 2. Predictive Coding Networks (PCN)

**Mechanism:** Hierarchical generative model. Each layer predicts the layer below. Error units compute mismatch. Weight update: local — only pre/post-synaptic activities + local error. Inference (fix neuron activities) and learning (update weights) are interleaved. Inference runs as a fixed-point iteration until convergence, then weights update.

**2024 advance — Error Optimization (EO):** Reparameterizes to optimize over prediction errors instead of states. Eliminates signal decay across layers. Converges orders of magnitude faster than standard PC. Matches backprop performance even on deep models. Source: [PMC11881729](https://pmc.ncbi.nlm.nih.gov/articles/PMC11881729/), [arxiv 2202.09467](https://arxiv.org/abs/2202.09467)

**Match to SGNNET:**
- K_iter routing IS a fixed-point iteration — maps directly to PCN inference phase
- Local update matches K_hh=2 locality
- Defined on "arbitrary graph topologies" (documented)
- Works on S^{D-1}: error signal is angular mismatch

**Obstacles:**
- PCN requires actual convergence in K_iter steps — 5 may be too few. Need convergence diagnostic
- Supervised PC requires output clamping mechanism (non-trivial for dot-product readout)
- New inference learning rate hyperparameter (not well-studied for GNNs)
- Does NOT learn discrete topology

**Cost vs backprop:** +50-100% (inference must converge per sample)  
**Practical feasibility: MEDIUM** — EO variant is the right starting point. Not near-term.

---

## 3. Forward-Forward Algorithm (Hinton 2022)

**Mechanism:** Replaces forward + backward with two forward passes — positive (real data) and negative (corrupted data). Each layer has its own local objective: maximize "goodness" (sum of squared activations) for positive, minimize for negative. No backward pass, no global credit assignment. Source: [arxiv 2212.13345](https://arxiv.org/abs/2212.13345)

**ForwardGNN (ICLR 2024):** Adapts FF for GNNs via two approaches: (1) append class label to node features, create positive/negative label variants; (2) virtual nodes per class — real nodes connect to correct or incorrect virtual class node. Matches or exceeds backprop accuracy on citation graphs. Memory is CONSTANT as depth increases vs backprop's 18× increase. Source: [arxiv 2403.11004](https://arxiv.org/html/2403.11004v1)

**Match to SGNNET:**
- Memory efficiency attractive for scaling K_iter >> 5
- Local objective aligns with K_hh=2 locality
- On S^{D-1}: goodness = dot-product alignment of h_i with W_pos_i (geometric goodness)

**Fundamental problem — shared W_pos:**
- FF assumes each "layer" has independent parameters. SGNNET's W_pos is shared across all K_iter steps.
- Local goodness for step t and step t+1 would use the SAME W_pos → contradictory update signals
- Workaround: step-specific W_pos_t. But this multiplies params by K_iter — defeats parameter efficiency

**Does NOT learn connectivity.**  
**Cost vs backprop:** ~0.8× (2 forward passes vs 1 forward + 1 backward)  
**Practical feasibility: LOW** — shared W_pos is a structural mismatch.

---

## 4. Equilibrium Propagation (EqProp)

**Mechanism:** Energy-based network. Forward pass = relax to energy minimum (free phase). Then nudge outputs softly toward target and re-relax (clamped phase). Weight update = difference in local correlations between phases: Δθ ∝ (∂E/∂θ|clamped − ∂E/∂θ|free). Provably approximates backprop gradient. All computation purely local. Source: [arxiv 1602.05179](https://arxiv.org/abs/1602.05179)

**ICLR 2024 — Jacobian Homeostasis:** Extends EqProp to non-symmetric networks. Adds regularizer to keep Jacobian near-symmetric (generalizes weight symmetry requirement). Scales to ImageNet-32. Source: [arxiv 2309.02214](https://arxiv.org/abs/2309.02214)

**Match to SGNNET:**
- The routing loop IS an energy minimization: gather-sum-normalize minimizes the spread of neuron positions
- Anti-Hebbian reg already captures EqProp's free-phase decorrelation
- Contrastive update has geometric meaning on S^{D-1}: free phase = neurons repel, clamped phase = align toward correct class direction
- Can learn edge WEIGHTS via contrastive correlation (not discrete topology)

**Obstacles:**
- Needs true equilibrium — K_iter=5 may be insufficient. Need K_iter=20-50 to converge
- Clamped phase for dot-product readout needs clear definition
- conn_hh is asymmetric — Jacobian homeostasis required but untested for sparse graph GNNs
- Two equilibrium runs: if K_iter_eq=20, total compute ~4× current

**Cost vs backprop:** 2-4× more expensive  
**Practical feasibility: LOW-MEDIUM** — correct theoretical match, but computationally expensive and requires convergence pre-validation. Phase 7+ research.

---

## 5. Contrastive Hebbian Learning (CHL)

**Mechanism:** Two phases: free phase (network relaxes under input), clamped phase (output units additionally clamped to target class direction). Weight update: Δw_ij = η * (⟨x_i x_j⟩_clamped − ⟨x_i x_j⟩_free). Purely local — each synapse only needs pre/post activities in both phases. Provably equivalent to gradient descent on free-phase energy for symmetric networks. Closely related to EqProp (EqProp generalizes CHL).

**2023/2024 advances:**
- **Dual Propagation (ICML 2023):** Dyadic neurons with free and clamped states in parallel. Up to 100× faster than naive CHL. Source: [ResearchGate](https://www.researchgate.net/publication/368159925_Dual_Propagation_Accelerating_Contrastive_Hebbian_Learning_with_Dyadic_Neurons)
- **Single-Phase CHL (2024):** Eliminates the clamped phase entirely via implicit memory in the update dynamics. ~Same compute as backprop. Source: [arxiv 2402.08573](https://arxiv.org/html/2402.08573)

**Match to SGNNET:**
- K_iter routing IS the free phase. The clamped phase = routing with output neurons softly attracted to target class W_pos
- AH regularizer already does free-phase anti-Hebbian correlation reduction — CHL's free-phase component is already present
- On S^{D-1}: correlation ⟨x_i x_j⟩ = cosine similarity = dot product (matches existing readout)
- Single-phase (2024) has same compute cost as backprop

**Obstacles:**
- Weight symmetry required for convergence proof. Jacobian homeostasis relaxes this but adds complexity
- Still needs equilibrium convergence (though less strict than EqProp in single-phase variant)
- Defining the "clamped state" for the geometric readout requires design work

**Cost vs backprop:** ~1× with single-phase variant  
**Practical feasibility: MEDIUM** — single-phase CHL is the right implementation target. Medium-term experiment (Phase 6).

---

## 6. Oja's Rule (Local Hebbian Learning for S^{D-1})

**Mechanism:** Δw = η * (x * y − y² * w), where y = w·x. Hebbian update with a decay term that naturally normalizes weights to unit norm. Proven to extract principal components. Keeps weight vectors on S^{D-1} by construction. 2024 result: "overcomes challenges of training neural networks under biological constraints" — preserves activation subspaces, mitigates exploding/vanishing signals. Source: [arxiv 2408.08408](https://arxiv.org/html/2408.08408v3)

**Match to SGNNET:**
- W_pos lives on S^{D-1} — Oja's rule was DESIGNED for this manifold
- F.normalize in routing already does the Oja-style projection
- Local: W_pos_i update uses only h_i and its K_hh=2 neighbors
- AH regularizer is redundant under Oja's rule (the y² decay term provides the same anti-collapse effect)
- Unsupervised component naturally decorrelates W_pos (replaces AH as mechanism)

**Critical issue:** Oja's rule is UNSUPERVISED — it learns principal components of the input distribution, not discriminative features for classification. Must be combined with supervised signal. Approaches:
- Supervised Oja: add delta-rule correction from output error
- Hybrid: Oja for W_pos decorrelation + backprop for fc_out only (classification signal still needs path back to W_pos)
- Pretraining with Oja, then fine-tune with backprop

**Does NOT learn connectivity.**  
**Cost vs backprop:** ~0.7× cheaper (no full backward pass for W_pos if Oja is unsupervised)  
**Practical feasibility: HIGH** — 5 lines of code. Immediate Tier-0 candidate.

---

## 7. PEPITA and Direct Feedback Alignment (DFA)

**PEPITA:** Two forward passes. First: standard forward, get output error e. Second: re-run forward with modulated input x' = x + B·e (B = fixed random feedback matrix). Weight update = difference in layer activations between the two runs. No backward pass. Source: [arxiv 2201.11665](https://arxiv.org/pdf/2201.11665)

**DFA-GNN (NeurIPS 2024):** DFA adapted for GNNs. Source: [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/6d0942e288ce41db8d4ebd041e7d1100-Paper-Conference.pdf)

**Performance gap is documented and significant.** Forward-only algorithms "present a performance gap when compared with other biologically inspired learning rules, such as feedback alignment." The current 95.52% from backprop sets a high bar these algorithms are unlikely to meet.

**Structural problems for SGNNET:**
- PEPITA's input perturbation x' = x + B·e assumes a direct path from input to each layer. K_iter routing steps are not "layers above the input"
- DFA's per-layer feedback cannot target shared-W_pos parameters (same issue as FF)
- Fixed random B matrix loses the sparse graph structure of SGNNET

**Does NOT learn connectivity.**  
**Practical feasibility: VERY LOW** — skip in favor of topology learning experiments.

---

## 8. Difference Target Propagation (DTP) / Forward Target Propagation (FTP)

**DTP:** Propagates target activations backward (not gradients). Each layer has an inverse network. Layer-local update: minimize distance between actual and target activation. Gauss-Newton optimization emerges. 30× slower than backprop in classical form. Source: [arxiv 2201.13415](https://arxiv.org/abs/2201.13415)

**FTP (2025):** Estimates layer-wise targets using only feedforward computations. No separate inverse network. Competitive with backprop on MNIST/CIFAR. Source: [arxiv 2506.11030](https://arxiv.org/html/2506.11030)

**Shared W_pos problem:** DTP's layer-local targets assume different parameters per "layer." K_iter steps share W_pos → circular target dependency. FTP partially addresses this but has not been tested on iterative GNN architectures.

**Does NOT learn connectivity.**  
**Practical feasibility: LOW** — shared W_pos is the fundamental obstacle. FTP is a longer-term experiment (Phase 6+) if backprop ever shows clear inadequacy.
