<!-- continued from LEARNINGS_design_2026_04_09_part1.md -->

## Gemma4 / PolarQuant-Inspired Experiment Designs (steps 106-109)

**Date:** 2026-04-09
**Source:** Three ideas from Gemma 4 (PLE per-layer embeddings, MoE 128-expert routing) and PolarQuant (hierarchical polar decomposition), mapped to SGNNET K_iter message-passing architecture.

### Design Rationale

All three ideas target same structural observation: SGNNET K_iter=12 steps identical. Every step runs same AH-weighted gather-normalize loop. Like running same transformer layer 12 times — functional but leaves temporal specialization on table.

**Why now:** Prior dynamic routing attempts (9 failures) all introduced multiplicative gates that compound over K_iter. These three avoid that trap:
- step106 (per-step embeddings): purely additive — shifts activation space, doesn't gate
- step107 (group-as-expert): sparse activation (on/off), not multiplicative scaling
- step108 (hierarchical polar): topology change only, no learned routing weights

### step106: Per-Step Embeddings (PLE → K_iter)

Gemma 4 PLE conditions each transformer layer differently via small learned vector. SGNNET analog: condition each K_iter step.

**Core insight:** Adding D-dimensional bias to Z before routing shifts WHICH neighbors most similar (thus how AH redistributes). Different steps see different "views" of same activation landscape. Early steps emphasize coarse structure (large Z-bias shifts), late steps fine-grained (small shifts or zero).

**Failure mode analysis:**
- Z-bias too large → dominates Z activations → all neurons look similar → AH collapses. Mitigation: init zeros, let gradient find scale.
- Z-bias learns to undo AH suppression → adversarial. Unlikely: Z-bias global (same for all neurons), AH per-pair.
- Edge-scale mode (Config C): scale constant per step (not input-dependent), not gate in gate-death sense. But if scale < 1, DOES attenuate. Init 1.0 and monitor.

**Param budget:** 768 params for mode A (12 × D=64). Current model ~529K params at N=4096. +0.15% — negligible.

### step107: Group-as-Expert MoE (Gemma4 MoE → SGNNET groups)

Gemma 4 uses 128 experts, top-2 active per token. SGNNET has n_groups=8 (step82 winner). Each group = "expert" — specialized sub-network.

**Why step83 failed and step107 won't:**

| step83 failure mode | step107 fix |
|---|---|
| Softmax → uniform under K_iter | ReLU: gradient=1 for active, clean 0 for inactive |
| No load balancing → 1-2 groups dominate | L1 regularization on routing weights |
| Batch-level routing (S_g averaged over batch) | Per-token routing (each image gets own group selection) |
| No fallback when routing collapses | Shared expert (group 0 always active) |

**FLOPs implication:** Top-3 of 8 groups active → only 3/8 neurons participate per token = 37.5% routing FLOPs. At N=4096, drops from ~39M FLOPs (K_hh=4) to ~15M FLOPs. Combined with K_hh=4 (step86 free lunch), could approach 1% FLOPs target at smaller N.

**Risk:** Per-token routing at N=1024 means router sees 8 group summaries per token. With batch_size=64, 64 independent routing decisions per batch — gradients should be stable. But if groups too small (N=1024/8=128 neurons), group summary S_g may be noisy.

### step108: Hierarchical Polar Routing (PolarQuant → S^{D-1})

PolarQuant decomposes vectors into hierarchical angles. SGNNET W_pos lives on S^{D-1} — unit hypersphere — natural recursive polar decomposition into D-1 angles.

**Key insight:** Polar angles form TREE. Level 1 (first angle θ_1) splits S^{D-1} into 2 hemispheres. Level 2 splits each into 2 quadrants (4 total). At level L, 2^L regions. Coarse-to-fine hierarchy from geometry, not learned parameter.

**Comparison with step82 (n_groups):**

| Property | step82 (random groups) | step108 (polar hierarchy) |
|---|---|---|
| Group assignment | Random, fixed | From W_pos geometry, tracks AH movement |
| Number of groups | Hyperparameter (n_groups=8) | Emerges from hierarchy level (2^L) |
| Cross-group connectivity | K_random parameter | K_cross from parent regions |
| Adaptation | Never changes | Changes as W_pos moves (if rebuilt) |
| New params | 0 | 0 |

**Why NOT phase routing (step60):** step60 used phase SHIFTS on activations as routing signal (multiplicative gate). step108 uses polar ANGLES of W_pos as topology structure (binary: same region or not). No gates, no signal modification, just neighbor selection.

**Risk:** At D=64 with 3 levels (8 regions), N=1024 gives ~128 neurons per region. K_local=2 within region + K_cross=2 from parent = K_hh=4 total. Matches current defaults. But regions may be unbalanced (AH pushes neurons into specific angular positions). Config E (dynamic rebuild) tests whether rebalancing helps.

### step109: Compound (step106 + step107)

Only if both show independent gains. Interaction hypothesis: per-step embeddings could make MoE router step-aware — different steps activate different expert groups. Temporal × spatial specialization.

**Why compounding might work (unlike GA rule violations):** Historical compounding failures (step29c, step32, step51, step66) all involved mechanisms that BOTH modified same signal path (W_pos similarity → routing weights). step106 and step107 operate on orthogonal axes:
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
2. Design separate small-N efficiency experiment:
   - N=512 or N=256, D=32, K_hh=2, K_in reduced
   - Test whether accuracy degrades gracefully at ≤1% FLOPs
   - If yes: "efficiency model" alongside "accuracy model"

Not contradiction with primary goal — both tracks coexist.
1% FLOPs model = "inference-optimized" variant.

---

## 50-Experiment Gap Analysis + Transformer-Inspired Experiments (2026-04-09)

### Gap Analysis Summary

After reviewing all 50+ completed experiments, three bottlenecks identified:

1. **Temporal homogeneity of K_iter** — all 12 steps identical. Step106 (Z-bias) partially addresses (+7.42pp). Transformers have unique params per layer.
2. **Input representation** — scatter-sum of K_in=50 features is 392× compression. Never optimized since step55.
3. **Readout head** — mean-pool + linear discards all graph structure. Never ablated.

**Critical gap:** Two largest N=1024 winners (Z-bias +7.42pp, redistribution +3.98pp) NEVER tested at N=4096. #1 priority (step115).

**Scale transfer failures:** Group topology (+3.01pp→0pp), W_phase (+2.88pp→+0.18pp), Turing (+1.68pp→−0.12pp) all lost gains at N=4096. Shared pattern: mechanisms relying on structural differentiation diluted at N=4096 K_hh=4 (0.1% connectivity). Z-bias and redistribution operate on signal itself, not structure — likely transfer.

### New Experiments Proposed

**step115 (P0):** Scale Z-bias + redistribution to N=4096. Three configs: A (Z-bias only), B (redistribution only), C (compound). Direct path to 98%+.

**step116 (P1):** RMSNorm / Pre-Norm routing. Tests whether `F.normalize()` (unit sphere projection) destroys useful magnitude info. Four configs: RMSNorm after, Pre-route normalize, RMSNorm before, no normalize. Zero params.

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

**step123 (P1.5): Stochastic depth training.** During training, randomly skip K_iter step t with probability p=t/24 (later steps skipped more). Standard regularization. At inference, use all 12 steps. Free — zero inference cost.

**step124 (P1.5): RigL-style topology refinement.** Every 10 epochs, for each neuron: evaluate gradient magnitude for random sample of ~32 non-neighbor candidates. Swap lowest-gradient existing edge for highest-gradient candidate. Freeze topology for final 30% training. Sparse-to-sparse training for graph topology.

**step125 (P1.5): AH alpha fine-sweep near 1.0.** step88 showed α=2.0 at 94.17% (better than α=1.5 at 89.48%) — curve above 1.0 non-monotone. Values 1.05, 1.1, 1.2, 1.3 never tested. True optimum may be slightly above 1.0.

**step126 (P2): µP initialization.** Scale init as O(1/√N), readout as O(1/N), LR ∝ 1/√N. Enables hyperparameter transfer from N=256 scouts to N=4096 full runs. Gated on step72 N-scaling curve.

**step127 (P2): Progressive K_iter distillation.** Train K_iter=12 teacher to convergence. Distill into K_iter=6 student using loss = α·task + (1-α)·MSE(student_out, teacher_out). 50% FLOPs reduction. From diffusion model distillation literature.

**nGPT connection:** nGPT (Loshchilov 2024) constrains all hidden states to unit hypersphere — exactly what SGNNET does with `F.normalize()` on S^{D-1}. nGPT shows this constraint helps convergence by conditioning optimization landscape on manifold. SGNNET already doing this. Question (step116): whether normalization method matters (`F.normalize` vs RMSNorm vs pre-norm).

### Priority Reprioritization Rationale

Elevated: step115 (P0, new), step116-118 (P1, new), step110 (P1, was Q4).
Maintained: step102, step103 (P1, unblocked by step105).
Added: step123-127 (P1.5-P2, from transformer research).
Demoted: step114-C (P2, marginal), step92 (P2, group routing poor track record).
Removed: step109 (step107 KILLED), step84 (step83 hurt), step98 (step90 NULL), G4 (subsumed by step115-B).