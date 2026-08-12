<!-- continued from LEARNINGS_design_2026_04_09_part1.md -->

## Gemma4 / PolarQuant-Inspired Experiment Designs (steps 106-109)

**Date:** 2026-04-09
**Source:** Three ideas from Gemma 4 (PLE per-layer embeddings, MoE 128-expert routing) and PolarQuant (hierarchical polar decomposition), mapped to SGNNET's K_iter message-passing architecture.

### Design Rationale

All three ideas target the same structural observation: SGNNET's K_iter=12 steps are identical. Every step runs the same AH-weighted gather-normalize loop. This is like running the same transformer layer 12 times — functional but leaving temporal specialization on the table.

**Why now:** Prior dynamic routing attempts (9 failures) all introduced multiplicative gates that compound over K_iter. These three ideas avoid that trap:
- step106 (per-step embeddings): purely additive — shifts the activation space, doesn't gate it
- step107 (group-as-expert): sparse activation (on/off), not multiplicative scaling
- step108 (hierarchical polar): topology change only, no learned routing weights at all

### step106: Per-Step Embeddings (PLE → K_iter)

Gemma 4's Per-Layer Embeddings condition each transformer layer differently via a small learned vector. SGNNET analog: condition each K_iter step.

**Core insight:** Adding a D-dimensional bias to Z before routing shifts WHICH neighbors are most similar (and thus how AH redistributes). Different steps see different "views" of the same activation landscape. Early steps might emphasize coarse structure (large Z-bias shifts), late steps fine-grained (small shifts or zero).

**Failure mode analysis:**
- Z-bias too large → dominates Z activations → all neurons look similar → AH collapses. Mitigation: init at zeros, let gradient find the right scale.
- Z-bias learns to undo AH suppression → adversarial. Unlikely: Z-bias is global (same for all neurons), AH is per-pair.
- Edge-scale mode (Config C): scale is a constant per step (not input-dependent), so it's not a gate in the gate-death sense. But if scale < 1, it DOES attenuate. Init at 1.0 and monitor.

**Param budget:** 768 params for mode A (12 × D=64). Current model has ~529K params at N=4096. This is +0.15% — negligible.

### step107: Group-as-Expert MoE (Gemma4 MoE → SGNNET groups)

Gemma 4 uses 128 experts, top-2 active per token. SGNNET has n_groups=8 (step82 winner). Each group is an "expert" — a specialized sub-network.

**Why step83 failed and step107 won't:**

| step83 failure mode | step107 fix |
|---|---|
| Softmax → uniform under K_iter | ReLU: gradient=1 for active, clean 0 for inactive |
| No load balancing → 1-2 groups dominate | L1 regularization on routing weights |
| Batch-level routing (S_g averaged over batch) | Per-token routing (each image gets own group selection) |
| No fallback when routing collapses | Shared expert (group 0 always active) |

**FLOPs implication:** If top-3 of 8 groups active, only 3/8 neurons participate per token = 37.5% of routing FLOPs. At N=4096, this drops from ~39M FLOPs (K_hh=4) to ~15M FLOPs. Combined with K_hh=4 (step86 free lunch), this could approach the 1% FLOPs target at smaller N.

**Risk:** Per-token routing at N=1024 means the router sees 8 group summaries per token. With batch_size=64, that's 64 independent routing decisions per batch — gradients should be stable. But if groups are too small (N=1024/8=128 neurons), the group summary S_g may be noisy.

### step108: Hierarchical Polar Routing (PolarQuant → S^{D-1})

PolarQuant decomposes vectors into hierarchical angles. SGNNET's W_pos lives on S^{D-1} — the unit hypersphere — which has a natural recursive polar decomposition into D-1 angles.

**Key insight:** The polar angles form a TREE. Level 1 (first angle θ_1) splits S^{D-1} into 2 hemispheres. Level 2 splits each into 2 quadrants (4 total). At level L, there are 2^L regions. This is a coarse-to-fine hierarchy that emerges from the geometry, not from a learned parameter.

**Comparison with step82 (n_groups):**

| Property | step82 (random groups) | step108 (polar hierarchy) |
|---|---|---|
| Group assignment | Random, fixed | From W_pos geometry, tracks AH movement |
| Number of groups | Hyperparameter (n_groups=8) | Emerges from hierarchy level (2^L) |
| Cross-group connectivity | K_random parameter | K_cross from parent regions |
| Adaptation | Never changes | Changes as W_pos moves (if rebuilt) |
| New params | 0 | 0 |

**Why this is NOT phase routing (step60):** step60 used phase SHIFTS on activations as a routing signal (multiplicative gate). step108 uses polar ANGLES of W_pos as a topology structure (binary: same region or not). No gates, no signal modification, just neighbor selection.

**Risk:** At D=64 with 3 levels (8 regions), N=1024 gives ~128 neurons per region. K_local=2 within region + K_cross=2 from parent region = K_hh=4 total. This matches current defaults. But the regions may be unbalanced (AH pushes neurons into specific angular positions). Config E (dynamic rebuild) tests whether rebalancing helps.

### step109: Compound (step106 + step107)

Only if both show independent gains. The interaction hypothesis: per-step embeddings could make the MoE router step-aware — different steps activate different expert groups. This is temporal × spatial specialization.

**Why compounding might work (unlike GA rule violations):** Historical compounding failures (step29c, step32, step51, step66) all involved mechanisms that BOTH modified the same signal path (W_pos similarity → routing weights). step106 and step107 operate on orthogonal axes:
- step106: modifies Z (activations) additively
- step107: modifies WHICH neurons participate (topology)
- AH: modifies edge weights (W_pos similarity)

Three orthogonal dimensions. No double-sparsity possible.

---

## FLOPs Path to 1% Target (Design Discussion)

FLOPs goal: ≤1% of VGG16 FC = 1.2M FLOPs.
Current minimum at N=4096: ~30M FLOPs fundamental routing loop.

Two-phase approach:
1. Establish accuracy ceiling at N=4096 (G1, G2 experiments)
2. Then design a separate small-N efficiency experiment:
   - N=512 or N=256, D=32, K_hh=2, K_in reduced
   - Test whether accuracy degrades gracefully at ≤1% FLOPs
   - If yes: this is the "efficiency model" alongside the "accuracy model"

This is NOT a contradiction with the primary goal — both tracks can coexist.
The 1% FLOPs model would be the "inference-optimized" variant.

---

## 50-Experiment Gap Analysis + Transformer-Inspired Experiments (2026-04-09)

### Gap Analysis Summary

After reviewing all 50+ completed experiments, three bottlenecks identified:

1. **Temporal homogeneity of K_iter** — all 12 steps identical. Step106 (Z-bias) partially addresses (+7.42pp). Transformers have unique params per layer.
2. **Input representation** — scatter-sum of K_in=50 features is 392× compression. Never optimized since step55.
3. **Readout head** — mean-pool + linear discards all graph structure. Never ablated.

**Critical gap:** Two largest N=1024 winners (Z-bias +7.42pp, redistribution +3.98pp) NEVER tested at N=4096. This is the #1 priority (step115).

**Scale transfer failures:** Group topology (+3.01pp→0pp), W_phase (+2.88pp→+0.18pp), Turing (+1.68pp→−0.12pp) all lost gains at N=4096. These shared a pattern: mechanisms relying on structural differentiation that gets diluted at N=4096 K_hh=4 (0.1% connectivity). Z-bias and redistribution operate on signal itself, not structure — likely to transfer.

### New Experiments Proposed

**step115 (P0):** Scale Z-bias + redistribution to N=4096. Three configs: A (Z-bias only), B (redistribution only), C (compound). Direct path to 98%+.

**step116 (P1):** RMSNorm / Pre-Norm routing. Tests whether F.normalize() (unit sphere projection) is destroying useful magnitude information. Four configs: RMSNorm after, Pre-route normalize, RMSNorm before, no normalize. Zero params.

**step117 (P1):** Learned input projection. W_proj after scatter-sum: shared [D,D], low-rank, per-neuron bias, GroupNorm. Addresses bottleneck #2.

**step118 (P1):** Attention-pooling readout. Replace mean-pool with learned neuron attention, top-k pool, multi-head readout, or learnable query. Addresses bottleneck #3.

**step119 (P1.5):** Adaptive K_iter per sample. Confidence-based early exit at sample level (NOT per-neuron like killed step34). For efficiency track.

**step120 (P2):** K_iter=16-24 + Z-bias + gradient checkpointing at N=4096. Gated on step115.

**step121 (P2):** Spectral normalization of W_pos. Hard diversity constraint replacing soft safety valve.

**step122 (P2):** DiffPool hierarchical readout. Learned graph coarsening N→64→10.

### Transformer Efficiency Techniques Mapped to SGNNET

| Transformer Technique | SGNNET Analog | Step |
|----------------------|---------------|------|
| Per-layer embeddings (Gemma 4 PLE) | Per-step Z-bias | step106 ✓ (+7.42pp) |
| MoE expert routing | Group-as-expert | step107 ✗ (KILLED) |
| RMSNorm (Llama/Mistral) | Replace F.normalize() | step116 (NEW) |
| LoRA/PEFT low-rank updates | Low-rank input projection | step117 (NEW) |
| Attention pooling / CLS token | Attention readout | step118 (NEW) |
| Early exit / CALM | Adaptive K_iter per sample | step119 (NEW) |
| Gradient checkpointing | Enable K_iter=24+ at N=4096 | step120 (NEW) |
| Spectral regularization | W_pos spectral norm | step121 (NEW) |
| DiffPool graph coarsening | Hierarchical readout | step122 (NEW) |
| Muon/LION/Schedule-free optimizer | Optimizer ablation | step110 (already queued) |

### Additional Transformer-Inspired Experiments (from deep research)

**step123 (P1.5): Stochastic depth training.** During training, randomly skip K_iter step t with probability p=t/24 (later steps skipped more often). Standard regularization technique. At inference, use all 12 steps. Free — zero cost at inference.

**step124 (P1.5): RigL-style topology refinement.** Every 10 epochs, for each neuron: evaluate gradient magnitude for a random sample of ~32 non-neighbor candidates. Swap lowest-gradient existing edge for highest-gradient candidate. Freeze topology for final 30% of training. Sparse-to-sparse training for graph topology.

**step125 (P1.5): AH alpha fine-sweep near 1.0.** step88 showed α=2.0 at 94.17% (better than α=1.5 at 89.48%) — the curve above 1.0 is non-monotone. Values 1.05, 1.1, 1.2, 1.3 never tested. The true optimum may be slightly above 1.0.

**step126 (P2): µP initialization.** Scale init as O(1/√N), readout as O(1/N), LR ∝ 1/√N. Enables hyperparameter transfer from N=256 scouts to N=4096 full runs. Gated on step72 N-scaling curve.

**step127 (P2): Progressive K_iter distillation.** Train K_iter=12 teacher to convergence. Then distill into K_iter=6 student using loss = α·task + (1-α)·MSE(student_out, teacher_out). 50% FLOPs reduction. From diffusion model distillation literature.

**nGPT connection:** nGPT (Loshchilov 2024) constrains all hidden states to the unit hypersphere — exactly what SGNNET does with F.normalize() on S^{D-1}. nGPT shows this constraint helps convergence by conditioning the optimization landscape on the manifold. SGNNET is already doing this. The question (step116) is whether the normalization method matters (F.normalize vs RMSNorm vs pre-norm).

### Priority Reprioritization Rationale

Elevated: step115 (P0, new), step116-118 (P1, new), step110 (P1, was Q4).
Maintained: step102, step103 (P1, unblocked by step105).
Added: step123-127 (P1.5-P2, from transformer research).
Demoted: step114-C (P2, marginal), step92 (P2, group routing has poor track record).
Removed: step109 (step107 KILLED), step84 (step83 hurt), step98 (step90 NULL), G4 (subsumed by step115-B).
