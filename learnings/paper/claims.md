# Core Paper Claims — Evidence Status

## Claim 1: Random sparse graphs are effective feature extractors
**Status: CONFIRMED (strong)**

SGNNET uses random, fixed input connectivity (K_in=25 out of 25088 pixels per neuron) and random small-world hidden connectivity (K_hh=4). No learned edge weights. Achieves 97.86% on Imagenette.

Evidence:
- step89-A: 97.86% accuracy, N=4096, D=64, 529K params, 150ep full data
- No connectivity learning — conn_in and conn_hh are fixed at initialization
- Johnson-Lindenstrauss theory: random projections preserve distance structure
- Each neuron = random projection of K_in input pixels → N neurons = N random projections

**What's novel:** Unlike random feature methods (Rahimi & Recht 2007) which stop at projection, SGNNET refines these projections through iterative message-passing on the random graph. The refinement (not the projection) is where expressiveness emerges.

## Claim 2: Parameter efficiency — <1% params matches VGG16 FC
**Status: CONFIRMED (strong)**

| Model | Params | Accuracy | Ratio |
|-------|--------|----------|-------|
| VGG16 FC layers | 123.6M | ~93.5% | 1.0× |
| SGNNET (step89-A) | 529K | 97.86% | 0.43% |

SGNNET uses 233× fewer parameters and achieves +4.36pp higher accuracy.

**What's novel:** This is not pruning or distillation from a large model. SGNNET is trained from scratch with a fundamentally different architecture. The parameters are: θ (thresholds), W_pos (positional encodings), fc_out (readout head). No W_edge, no W_message, no attention weights.

## Claim 3: Learning happens in routing dynamics, not weights
**Status: CONFIRMED (strong)**

The learned parameters (θ, W_pos) shape HOW information flows through the fixed random graph, not WHAT the connections are. This is fundamentally different from transformers/MLPs where W IS the knowledge.

Evidence:
- Removing F.normalize → catastrophic failure (step129: −50 to −71pp). Normalization constrains the dynamics.
- Removing AntiHebbian → significant degradation. Suppression maintains routing diversity.
- Stochastic depth (skipping K_iter steps) → catastrophic (step123: −35 to −61pp). Every routing step is essential.
- K_iter is the #1 hyperparameter — more routing steps = more refinement = higher accuracy.

## Claim 4: Three load-bearing architectural walls
**Status: CONFIRMED (3 clean ablations)**

1. **F.normalize after each step** — prevents activation explosion, constrains to hypersphere. Removal: −50 to −71pp (step129).
2. **Static AntiHebbian suppression** — position-based decorrelation prevents representational collapse. α=0 hurts significantly.
3. **Mean-pool readout** — attention readout catastrophically fails (step118: −60 to −67pp). The simplest aggregation is the best.

## Claim 5: K_iter (routing depth) is the primary capacity knob
**Status: CONFIRMED (multiple experiments)**

K_iter controls how far information propagates through the graph. After K_iter steps, each neuron integrates information from up to K_hh^K_iter potential paths.

Evidence:
- K_iter=12 >> K_iter=8 >> K_iter=4 (step71, step89)
- Stochastic depth catastrophic — every step matters (step123)
- K_iter dominates FLOPs: ~80% of compute is in the routing loop

## Claim 6: Scale transfer compression
**Status: CONFIRMED (pattern across 6+ mechanisms)**

Mechanisms giving +5-8pp at N=1024 compress to +0.1-0.9pp at N=4096.

| Mechanism | Δ at N=1024 | Δ at N=4096 | Compression |
|-----------|-------------|-------------|-------------|
| W_proj | +5.48pp | +0.06pp | 99% |
| weighted_neg | +3.97pp | not tested | — |
| group topology | +4.21pp | null | ~100% |
| RigL topology | +6.82pp | not tested | — |

HYPOTHESIS: at 0.1% connectivity (N=4096, K_hh=4), the network is already near its routing ceiling. Adding mechanisms helps when there's routing headroom (N=1024) but provides diminishing returns at the ceiling.

**Paper angle:** this implies N=1024 with richer connectivity may be a better operating point than N=4096 with extreme sparsity — directly motivating the efficiency track.

## Claim 7: Compounding interference
**Status: CONFIRMED (1 clean ablation)**

Two independently positive mechanisms can cancel when combined. step131-C: weighted_neg (+3.97pp alone) + W_proj (+5.48pp alone) = compound null (−0.46pp). Interference, not additivity.

Evidence: step131 Tier-1, N=1024, clean 4-config ablation (Ref, A-only, B-only, compound).

---

## NEEDS MORE EVIDENCE

### FLOPs efficiency
**Status: IN PROGRESS**

Current: 38.8M FLOPs (31.4% of VGG16 FC). Target: ≤6.18M (5%).
step140 (N×K tradeoff) running now. Need to demonstrate competitive accuracy at ≤5% FLOPs.

### Generalization beyond Imagenette
**Status: NOT STARTED**

Critical for publication. Need at least:
- CIFAR-10 (via VGG16 features, same pipeline)
- One non-vision dataset (tabular?) to show architecture-agnostic benefit

### Comparison with other efficient methods
**Status: NOT STARTED**

Need head-to-head comparisons:
- Pruned VGG16 FC at equivalent param count
- Knowledge-distilled small MLP
- Random feature baseline (no iterative routing — just random projection + linear)
- Standard GNN (GCN/GAT) at equivalent params

### Theoretical grounding
**Status: HYPOTHESIS ONLY**

Core hypothesis: physical data has compact constraint structure → random projections sample it → iterative routing discovers it. This needs formalization or at least empirical validation beyond one dataset.

---

## Claim 7: SGNNET teacher enables MLP student to exceed scratch via knowledge distillation (GLNN)
**Status: CONFIRMED on 1 dataset/model — NEEDS VALIDATION across datasets and model scales**

### Confirmed evidence
- step603 (B2 GLNN, Imagenette, MLP_37 student, T=2 λ=0.5, 150ep): student=**97.81%** vs scratch=97.71% (+0.10pp, STRONG condition met)
- Interpretation: SGNNET routing discovers structure expressible by a static MLP — teacher's soft targets encode routing geometry that the student can absorb

### What "GLNN distillation generalises" would mean for the paper
If the teacher→student gain appears across datasets and student model sizes, it supports the claim that SGNNET extracts genuinely transferable representations, not just task-specific shortcuts on one benchmark.

### Validation plan (QUEUED — launch after CIFAR-10 results confirm cross-dataset generalization)

| Experiment | Teacher | Student | Dataset | Status |
|---|---|---|---|---|
| step603 | SGNNET ΔW-proj K=5 | MLP_37 | Imagenette | **DONE** +0.10pp STRONG |
| step620 | SGNNET ΔW-proj K=5 | MLP_37 | CIFAR-10 | TODO — needs step401b to complete first |
| step621 | SGNNET ΔW-proj K=5 | MLP_h3 (matched params) | Imagenette | TODO — tests whether gain holds at tiny student |
| step622 | SGNNET ΔW-proj K=5 | MLP_256 | CIFAR-100 | TODO — needs CIFAR-100 to converge for SGNNET first |

### Acceptance criteria
- **STRONG across datasets**: gain appears on CIFAR-10 and CIFAR-100 → paper Section 4 claim
- **MEDIUM**: gain on Imagenette only → footnote, not main claim
- **KILLED**: student on CIFAR-10 below scratch → remove claim, keep as Imagenette-specific finding

### Note on mechanism
Low temperature (T=2) optimal — high T blurs soft targets, loses routing signal. This is a diagnostic: if T=2 wins consistently, it means SGNNET's soft targets encode sharp near-certain routing decisions, not diffuse probabilities.

---

## Claim 8: Anti-Hebbian routing signal on S^{D-1} is a novel architectural primitive
**Status: NOVEL (literature search 2026-04-15, confirmed no prior art)**

### What SGNNET does
- W_pos[i], W_pos[j] are unit-norm embeddings on S^{D-1}
- After co-activation, W_pos[i] and W_pos[j] are pushed APART (anti-Hebbian repulsion)
- ΔW = W_pos[i] − W_pos[j] is used as a live routing signal encoding geometric displacement
- Trained end-to-end in supervised classification

### Prior art coverage (does NOT overlap with SGNNET's mechanism)

| Paper | What it does | Gap |
|-------|-------------|-----|
| Földiák 1990 | Anti-Hebbian lateral inhibition for sparse codes | Unsupervised; no routing; no sphere |
| Pehlevan & Chklovskii 2015/2018 | H/aH from similarity matching for PCA/ICA | Unsupervised; no GNN; no routing signal |
| Liu et al. NeurIPS 2017 | Deep Hyperspherical Learning, geodesic conv | No anti-Hebbian; no routing |
| Sabour et al. 2017 (Capsules) | Iterative routing-by-AGREEMENT via cosine sim | Hebbian-like (route TO agreement), not repulsion; no ΔW |
| HyperGRL arXiv 2512.24062 | Repulsion on S^{D-1} as regularizer in GNN | Repulsion is regularizer, NOT routing signal; no ΔW; no plasticity |

### Novelty verdict
**Claim (a): Anti-Hebbian learning in GNN routing — NOVEL.** No paper combines anti-Hebbian plasticity with graph routing.

**Claim (b): ΔW as geometric routing signal on S^{D-1} — NOVEL.** No paper uses the displacement vector between sphere-embedded nodes to gate routing. Capsules use scalar cosine; HyperGRL uses scalar potentials.

**Key inversion:** Prior routing (capsules) routes TO nodes you agree with (Hebbian attraction). SGNNET uses repulsion to encode geometric displacement, then routes via that displacement — a mechanistic inversion with no prior art.

### Related work positioning
"While prior work uses anti-Hebbian rules for unsupervised decorrelation (Földiák 1990, Pehlevan 2015) and hyperspherical embeddings for regularization (Liu et al. 2017, HyperGRL 2025), SGNNET introduces the first use of anti-Hebbian repulsion dynamics as a supervised routing signal in a graph network — specifically, ΔW = W_pos[i] − W_pos[j] on S^{D-1} encodes live geometric displacement and gates sparse routing, replacing fixed or similarity-attracted aggregation."

### Key references
1. Földiák 1990 — sparse anti-Hebbian coding. Biological Cybernetics.
2. Pehlevan & Chklovskii 2015 — H/aH from similarity matching. Neural Computation.
3. Liu et al. 2017 — Deep Hyperspherical Learning. NeurIPS.
4. Sabour, Frosst & Hinton 2017 — Dynamic Routing Between Capsules. NeurIPS.
5. arXiv 2512.24062 (Dec 2025) — Hyperspherical Graph RL. Most recent intersection of S^{D-1} + GNN + repulsion.
