# Learning Algorithms for SGNNET — Part 2: Topology Learning + Recommendations

**Parent:** [RESEARCH_learning_algorithms.md](RESEARCH_learning_algorithms.md)  
**Date:** 2026-04-11

---

## The Core Opportunity: Learning conn_hh

SGNNET's conn_hh (hidden-hidden connectivity, K_hh=2 per neuron) is **fixed random, never trained**.

Step124-B: RigL topology at N=1024 → **+6.82pp** (Tier-0). Largest single mechanism gain in project history.

The topology is likely the biggest remaining accuracy bottleneck. Four approaches for learning it:

---

## Approach 1: RigL (Gradient-Informed Regrowth) — CONFIRMED

**Mechanism:** Every M batches: (1) Prune K_drop edges with lowest magnitude/activation-correlation. (2) Regrow K_drop edges where the GRADIENT of the loss w.r.t. the currently-zero edge weight is largest. Maintains fixed sparsity (K_hh stays constant). Source: [arxiv 1911.11134](https://arxiv.org/pdf/1911.11134)

**Structured RigL (ICLR 2024):** Constant fan-in variant — each neuron always has exactly K_hh incoming edges. This matches SGNNET's structure exactly. Source: [openreview kOBkxFRKTA](https://openreview.net/forum?id=kOBkxFRKTA)

**SGNNET specifics:**
- conn_hh has no scalar weights currently (binary mask). Need to either: (a) add a scalar weight per edge for magnitude-based pruning, or (b) use W_pos GRADIENT MAGNITUDE as the regrowth criterion
- For gradient-based regrowth: compute ∂L/∂(W_pos[i] · W_pos[j]) for all candidate edges (j in K_nn(i, K_candidates=8)), add the K_hh with highest gradient
- Pruning criterion: edges where cos(θ_ij) remains near-zero across training (neurons never co-activate on this edge)

**Why it works:** Gradient tells us which potential connections would most reduce the loss if activated. Starting from random topology and evolving via gradient feedback is exactly supervised graph learning.

**Empirical status in SGNNET:** step124-B = +6.82pp at N=1024, Tier-0. CONFIRMED MECHANISM (needs retest at D=16 efficiency config).

**Implementation sketch:**
```python
def rigl_update(model, loss_fn, loader, K_drop=1, device='mps'):
    # 1. Compute gradient on candidate edges (dense pass on candidates only)
    model.zero_grad()
    x, y = next(iter(loader))
    loss = loss_fn(model(x.to(device)), y.to(device))
    loss.backward()
    
    with torch.no_grad():
        W = F.normalize(model.W_pos, dim=-1)  # [N, D]
        # Candidate edges: K_candidates=8 nearest by W_pos geometry
        candidates = knn(W, k=8)  # [N, 8]
        
        for i in range(N):
            current_neighbors = model.conn_hh[i]  # current K_hh=2 edges
            # Prune: remove edge where activation correlation was lowest
            prune_edge = lowest_coactivation_edge(i, model)
            # Regrow: add candidate with highest gradient magnitude
            grad_scores = [grad_magnitude(i, j, model) for j in candidates[i] if j not in current_neighbors]
            grow_edge = candidates[i][argmax(grad_scores)]
            model.conn_hh[i] = update_edges(current_neighbors, prune_edge, grow_edge)
```

**Cost:** O(N * K_candidates) per topology step. N=2048, K_candidates=8: ~16K ops. Negligible.  
**Frequency:** Every M=5 epochs (or every M=20 batches — experiment to tune).  
**Practical feasibility: HIGH — immediate Tier-0 experiment at D=16.**

---

## Approach 2: Cannistraci-Hebb (CH) Topology — GRADIENT-FREE

**Mechanism:** Local link prediction based on NETWORK STRUCTURE. CH3-L3 predicts new edges by: count common neighbors weighted by their community structure (local triangle density). Higher common-neighbor density → higher edge probability. Prune low-magnitude edges, add CH-predicted edges. Gradient-free — runs entirely outside the training loop. Source: [ICLR 2024 proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/file/c9ef471a579197c4ed99df2aa542ce97-Paper-Conference.pdf)

**Key result:** CHT surpasses fully connected VGG16/GoogLeNet/ResNet50/ResNet152 at **1% sparsity**. SGNNET is 0.05% params — ultra-sparse regime where CHT excels most.

**SGNNET specifics:**
- CH3-L3 on SGNNET's conn_hh: for candidate edge (i,j), count neurons k where k is neighbor of both i and j (common neighbor), weighted by the triangle density around k
- This naturally promotes connections between neurons that "see" the same inputs via different paths — exactly the redundancy-eliminating structure SGNNET needs
- No gradient required, no forward pass needed for topology update

**Why gradient-free matters:** Gumbel-Softmax and RigL require differentiating through edge selection. CH topology update is outside the learning loop entirely — it runs during evaluation epochs, when a few extra seconds of compute are available.

**Implementation:** Requires CH3-L3 link predictor. Formula for edge score: Score(i,j) = Σ_k [1/(deg(k))] * [|N(i) ∩ N(k)| + |N(j) ∩ N(k)|] for k ∈ N(i) ∩ N(j). Where N(x) = neighbors of x in conn_hh. With K_hh=2, this is a very sparse sum — O(K_hh²) per pair. Source: arxiv.org/html/2501.19107v1

**Cost:** O(N * K_candidates * K_hh²) = 2048 * 8 * 4 = 65K operations per topology step. ~0.01 seconds. Free.  
**Practical feasibility: HIGH** — no gradient infrastructure needed. Can run every epoch as a separate topology update phase.

---

## Approach 3: Gumbel-Softmax Differentiable Edge Learning

**Mechanism:** Model each edge as a Bernoulli variable with learned logit. For each neuron i, maintain logits over K_candidates pre-specified neighbors. Use Gumbel-Top-K to differentiably sample K_hh=2 edges. Backward pass uses straight-through estimator (hard discrete edges in forward, soft probabilities in backward).

**2024 advance — Decoupled ST-GS:** Separate temperatures for forward pass (τ_fwd=0.1, near-discrete) and backward pass (τ_bwd=1.0, gradient-smooth). Significantly improves over original ST-GS. Source: [arxiv 2410.13331](https://arxiv.org/abs/2410.13331)

**SGNNET parameterization:**
- `edge_logits = nn.Parameter(torch.zeros(N, K_candidates))` — e.g., N=2048, K_candidates=8
- Cost: 2048 * 8 = 16,384 additional parameters. Negligible vs 67K total
- K_candidates: pre-selected by geometric proximity of W_pos at initialization. Can be fixed or updated periodically

**Forward pass:**
```python
def sample_topology(edge_logits, K_hh=2, tau_fwd=0.1, tau_bwd=1.0, training=True):
    if training:
        # Gumbel-Top-K with straight-through
        gumbel = -torch.log(-torch.log(torch.rand_like(edge_logits) + 1e-10) + 1e-10)
        perturbed = edge_logits + gumbel
        top_k_idx = perturbed.topk(K_hh, dim=-1).indices
        
        # Straight-through: hard in forward, soft in backward
        soft_bwd = F.softmax(edge_logits / tau_bwd, dim=-1)
        hard_fwd = torch.zeros_like(soft_bwd).scatter_(1, top_k_idx, 1.0)
        conn = hard_fwd - soft_bwd.detach() + soft_bwd  # ST trick
    else:
        top_k_idx = edge_logits.topk(K_hh, dim=-1).indices
        conn = ...  # hard selection
    return conn
```

**Training dynamics:** conn_hh changes every forward pass during training (stochastic). This adds regularization — the model must learn W_pos that works for multiple topology samples. May be a feature, not a bug (topology ensemble effect).

**Annealing:** Start with τ=2.0 (near-uniform exploration), anneal to τ=0.1 by epoch 50. K_candidates should include some random non-local candidates (exploration) in addition to geometric near-neighbors.

**Does NOT work well with fixed K_candidates.** K_candidates should be defined dynamically based on current W_pos KNN — update every 10 epochs.

**Cost vs backprop:** ~1.2× (Gumbel sampling + gradient through discrete ops)  
**Practical feasibility: HIGH** — 2024 Decoupled ST-GS is the right implementation. Recommend as Tier-1 experiment after RigL confirms topology learning value.

---

## Approach 4: NRI / Online Relational Inference

**Mechanism:** Variational autoencoder where the latent code IS the graph. Encoder = GNN infers which edges exist. Decoder = GNN predicts dynamics given edge structure. Gumbel-Softmax for discrete edges. Fully differentiable end-to-end. Source: [arxiv 1802.04687](https://arxiv.org/pdf/1802.04687)

**ORI (NeurIPS 2024):** Direct adjacency learning. Adjacency matrix is a trainable parameter updated via AdaRelation (adaptive learning rate sensitive to edge importance). Source: [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/93739)

**Assessment for SGNNET:** ORI's direct adjacency parameterization is equivalent to Approach 3 (Gumbel-Softmax) with an adaptive LR variant. Full NRI with a VAE encoder/decoder is heavy and would significantly increase computational cost. The ORI insight (adaptive LR for edge logits) is a useful technique to layer on top of Approach 3, not a separate experiment.

**Practical feasibility: MEDIUM** — ORI's AdaRelation can be added to Approach 3 as an enhancement.

---

## Approach 5: REINFORCE for Discrete Topology

**Mechanism:** Treat edge selection as a stochastic policy. Estimate gradient via REINFORCE: ∇θ J ≈ (1/B) Σ [∇θ log π(edges|θ) * R] where R = reward (accuracy improvement). High-variance estimator.

**Assessment:** Extremely high variance for N=2048 agents selecting K_hh=2 edges each. Requires order-of-magnitude more forward passes than Gumbel-Softmax for equivalent gradient quality. No precedent at this scale in GNN literature.

**Practical feasibility: NOT RECOMMENDED** — Gumbel-Softmax is strictly superior (lower variance, lower implementation complexity, better gradient signal).

---

## STDP and Biologically-Motivated Topology Learning

**Summary:** STDP is SGNNET's AH regularizer in rate-coded form. The anti-Hebbian regularizer already implements the rate-coded analog of STDP's anti-causal component. F.normalize corresponds to STDP's magnitude normalization. The structural plasticity aspect of STDP (synapse formation/pruning based on co-activation) is subsumed by RigL (gradient-based) and CH topology (structure-based). No additional value from implementing explicit STDP.

---

## Experiment Queue (Topology Learning Priority Order)

### Experiment A: RigL at D=16 Efficiency Config
- **Config:** step199 base (N=2048, D=16, K_hh=2, K_iter=5), add RigL outer loop
- **RigL schedule:** Update every M=5 epochs. Prune 1 edge per neuron with lowest W_pos·W_pos_neighbor cosine. Regrow 1 edge from K_candidates=8 geometric neighbors with highest gradient magnitude
- **Tier-0:** 20ep, 50% data. Control = step199 baseline (95.52%)
- **Expected:** +2-5pp based on step124-B precedent (+6.82pp at N=1024)

### Experiment B: CH Topology at D=16
- **Config:** Same base as A, swap RigL for CH3-L3 link prediction
- **CH schedule:** Update every M=10 epochs. For each neuron, compute CH3-L3 score for 8 candidate edges. Replace lowest-coactivation edge with highest-scoring candidate
- **Tier-0:** 20ep, 50% data

### Experiment C: Gumbel-Softmax K_hh Learning
- **Config:** Add `edge_logits` parameter (N=2048, K_candidates=8). Decoupled ST-GS (τ_fwd=0.1, τ_bwd=1.0). K_candidates = 8 geometric neighbors by W_pos at epoch 0; update K_candidates every 20 epochs
- **Annealing:** τ_bwd: 2.0 → 1.0 → 0.5 over 75 epochs
- **Tier-1:** 75ep (topology needs more epochs to converge)

### Experiment D: Oja's Rule for W_pos
- **Config:** Replace AdamW W_pos update with Oja's rule. Keep AdamW for fc_out. Supervised Oja: Δw_i = η(h_i * y_i − y_i² * w_i) + β * (target_direction_i − w_i) where target_direction comes from output error
- **Tier-0:** 20ep, 50% data. Simple experiment — 5 lines of code

---

## Biologically-Motivated Design Map

| Biological Mechanism | Neural Basis | SGNNET Implementation | Status |
|---------------------|-------------|----------------------|--------|
| Hebbian potentiation | Co-active neurons strengthen | W_pos move together via routing | Implicit in backprop |
| LTD / anti-Hebbian | Co-active neurons weaken lateral inhibition | AH regularizer α=1.0 | CONFIRMED, default |
| Homeostatic plasticity | Maintain activation magnitude | F.normalize after each step | Implemented |
| STDP (rate-coded) | Timing → rate Oja + AH | AH reg is already the rate analog | Redundant with AH |
| Oscillatory binding | Gamma rhythm synchrony | alpha_reflect=0.5 mechanism | Partially explored |
| Structural plasticity | Synapse formation/pruning | conn_hh evolution via RigL/CH | **MISSING — highest priority** |
| Predictive coding | Top-down prediction | PCN-style inference loop | Not yet implemented |
| Contrastive learning | Free vs clamped phase | CHL with single-phase variant | Medium priority |

**The gap is structural plasticity.** Every other bio-plausible mechanism either has a current implementation or a near-term experiment. conn_hh evolution is the missing piece.

---

## Sources

| Source | Key Finding | Confidence |
|--------|-----------|-----------|
| [arxiv 1911.11134](https://arxiv.org/pdf/1911.11134) — RigL | Gradient-informed regrowth, matches dense at 90%+ sparsity | HIGH |
| [ICLR 2024 SRigL](https://openreview.net/forum?id=kOBkxFRKTA) | Constant fan-in variant, 3.4× inference speedup | HIGH |
| [ICLR 2024 CH](https://proceedings.iclr.cc/paper_files/paper/2024/file/c9ef471a579197c4ed99df2aa542ce97-Paper-Conference.pdf) | CHT surpasses dense at 1% sparsity | HIGH |
| [arxiv 2501.19107](https://arxiv.org/html/2501.19107v1) | CH in transformers/LLMs at ultra-sparse | HIGH |
| [arxiv 2410.13331](https://arxiv.org/abs/2410.13331) — Decoupled ST-GS | Decoupled temperatures improve discrete optimization | HIGH |
| [NeurIPS 2024 ORI](https://neurips.cc/virtual/2024/poster/93739) | Adaptive LR for adjacency learning | MEDIUM |
| [arxiv 1802.04687](https://arxiv.org/pdf/1802.04687) — NRI | VAE-based structure learning | MEDIUM |
| [ICML 2023 Dual Prop](https://www.researchgate.net/publication/368159925) | CHL 100× faster via dyadic neurons | MEDIUM |
| [arxiv 2402.08573](https://arxiv.org/html/2402.08573) — Single-phase CHL | Eliminates second phase, ~backprop cost | MEDIUM |
| step124-B in SGNNET | RigL at N=1024: +6.82pp Tier-0 | HIGH (empirical, SGNNET-native) |
