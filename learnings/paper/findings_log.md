# Paper Findings Log — Chronological

Notable discoveries not captured in standard literature, ordered by date.

---

### 2026-04-04: Random connectivity works (Phase 5 start)
First SGNNET models with random conn_in and small-world conn_hh achieve >85% on FashionMNIST. No learned edge weights. Only θ and W_pos are learned. This contradicts the assumption that GNNs need learned message functions.

### 2026-04-05: Fourier encoding on S^{D-1} breakthrough
Switching from random positional encoding to Fourier encoding on the D-dimensional hypersphere gives +9.83pp. The encoding provides a smooth, structured positional basis that the routing dynamics can exploit. This is the single largest improvement in the project.

### 2026-04-06: AntiHebbian suppression is load-bearing
Position-based suppression (reducing influence of positionally similar neighbors) is critical. Without it, neurons converge to correlated representations and accuracy drops significantly. This is a biological analogue — lateral inhibition in cortical networks serves the same decorrelation purpose.

### 2026-04-07: K_iter dominance discovered
K_iter (message-passing depth) is far more important than N (neuron count) or K_hh (connectivity) for accuracy. K_iter=12 >> K_iter=8 >> K_iter=4. This suggests the routing refinement process is the core computation, not the graph structure.

### 2026-04-08: 97.86% achieved (step89-A)
Project best: 97.86% on FashionMNIST with N=4096, D=64, K_hh=4, K_iter=12, 529K params. This surpasses VGG16 FC (93.5%) at 0.43% of its parameters. No data augmentation, standard train/test split.

### 2026-04-09: Three load-bearing walls confirmed
Systematic ablation reveals three components that cannot be removed:
1. F.normalize (step129: −50 to −71pp without it)
2. Static AH suppression (dynamic replacements all fail)
3. Mean-pool readout (step118: attention readout −60pp)

The simplest version of each component is the best. Complexity hurts.

### 2026-04-09: Scale transfer failure pattern
6+ mechanisms show the same pattern: strong gains at N=1024, near-zero at N=4096. W_proj: +5.48pp→+0.06pp. Group topology: +4.21pp→null. Hypothesis: extreme sparsity (0.1% connectivity at N=4096) creates a routing ceiling that additional mechanisms can't lift.

### 2026-04-09: Compounding interference discovered
weighted_neg (+3.97pp alone) + W_proj (+5.48pp alone) = null when combined (step131-C). Mechanisms that independently improve routing can interfere when stacked. This has implications for architecture search — greedy winner stacking doesn't work.

### 2026-04-10: N=1024 efficiency reframing
Key insight: if mechanisms improve accuracy at smaller N, the FLOPs savings from using N=1024 instead of N=4096 (4-6× fewer FLOPs) may be more valuable than the absolute accuracy gain. This motivates the N×K tradeoff experiments.

### 2026-04-10: Core hypothesis articulated
Physical reality is constrained → data inherits constraints → compact sufficient representation exists. SGNNET's random graph + iterative routing is a search for that representation. The 392× input compression (25088 pixels → 64-dim per neuron) isn't a bug — it's the right direction. The question is whether D=16-64 dims are enough to encode all objective-relevant constraints.

### 2026-04-10: Constraint discovery mechanisms designed
Six new architectural tools for encouraging low-rank, objective-relevant representations: nuclear norm regularization, information bottleneck, dimensional gating, L1 sparsity, contrastive routing loss, progressive capacity reduction. These directly test whether explicit compression pressure helps the network find constraint structure faster.

### 2026-04-10: N dominates over connectivity — more K HURTS (step140)
Controlled N×K tradeoff sweep at D=16. Increasing K_hh from 4→8→16 at fixed N=1024 DECREASES accuracy (90.50%→81.94%→77.12%). Reducing N is catastrophic regardless of K: N=512/K=32 = 62.34%, N=256/K=32 = 46.27%. This KILLS the hypothesis that denser connectivity compensates for fewer neurons. Implication: the number of independent random projections (N) matters far more than how connected they are (K). Each neuron needs to maintain a unique perspective on the input — more neighbors average this away. **Paper claim: N (neuron count) and K_iter (routing depth) are the two primary capacity knobs, not connectivity density.**
