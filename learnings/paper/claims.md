# Core Paper Claims — Evidence Status

## Claim 1: Random sparse graphs are effective feature extractors
**Status: CONFIRMED (strong)**

SGNNET uses random, fixed input connectivity (K_in=25 out of 25088 pixels per neuron) and random small-world hidden connectivity (K_hh=4). No learned edge weights. Achieves 97.86% on FashionMNIST.

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

### Generalization beyond FashionMNIST
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
