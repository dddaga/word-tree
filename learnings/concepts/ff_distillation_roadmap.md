# Feed-Forward Layer Distillation — Research Direction

**Date:** 2026-04-21  
**Status:** Discussion — pre-experiment, no results yet  
**Scope:** Beyond SGNNET. General method for compressing large dense FF layers in any network.

---

## Core Goal

Target any deep learning network with large FF layers (transformers, VGG FC, MLP blocks in ResNets, etc.) and compress/distill FF layer using supervised data — original network outputs as soft labels.

SGNNET one architecture tried. This roadmap = broader program.

---

## The Problem

Dense FF layers O(N²) in params and compute. Layer mapping ℝ^2000 → ℝ^2000 = 4M params, 4M MACs. In transformers, FF block typically 4× model dim — dominant cost. Goal: same representation power at O(N·k) cost, k ≪ N.

---

## Lever 1 — Group-Wise Block Decomposition

Break single N×N dense layer into G groups of size (N/G)×(N/G):

```
Dense:  [2000 → 2000]  →  4M params, N² MACs

Group2: [1000 → 1000] × 2 groups  →  2M params, N²/2 MACs
Group4: [ 500 →  500] × 4 groups  →  1M params, N²/4 MACs
GroupG: [ N/G →  N/G] × G groups  →  N²/G params, N²/G MACs
```

Within each group: standard dense matmul. Between groups: optional cross-group interaction (subtraction, addition, learned mixing — expressivity question lives here).

Stacking multiple blocked layers recovers depth. Question: can depth substitute for removed cross-group connectivity?

### Cross-group interaction options
- **None (pure block-diagonal):** each group independent — max sparsity, min expressivity
- **Subtract adjacent groups:** (g_i - g_{i+1}) — difference signal, zero-cost, may capture contrast
- **Learned cross-group projection:** small (G×G) mixing matrix after block matmuls — cheap, recovers some expressivity
- **Alternating group assignment:** shuffle group membership each layer (like ShuffleNet) — breaks local isolation

---

## Lever 2 — Activation Sparsity via LeakyReLU → ReLU Substitution

**Training:** LeakyReLU(x, α≈0.01) — keeps negative activations small but non-zero. Gradients flow through entire network; no dead neurons during training.

**Inference:** swap to ReLU(x) — hard zero for negatives. If network learned to push ~50% activations negative (via LayerNorm + bias), gives guaranteed ~50% activation sparsity at inference.

**Why this works:**  
After LayerNorm, activations zero-mean → ~50% negative by construction. Training with LeakyReLU preserves gradient flow. Inference with ReLU activates sparse CUDA/PyTorch kernel paths (cuSPARSE, `torch.sparse`) that give real speedup when sparsity ≥ 70–80%.

**Training sequence:**
1. Train with LeakyReLU(α=0.01) — full gradient flow
2. Measure actual negative fraction (target ≥ 50%)
3. Swap to ReLU for inference benchmark
4. Compare dense-ReLU vs sparse-ReLU wall-clock

**Random ReLU variant (regularization):**  
During training, randomly sample α ~ Uniform(0, 0.05) per batch. Forces network not rely on specific negative-side values. At inference, α=0 (hard ReLU). Acts as dropout analogue for negative activation regime.

---

## Lever 3 — Genetic Algorithm Over Group Configurations

Grouping structure (G, depth, cross-group connectivity pattern) = discrete combinatorial space. Not differentiable → can't gradient-descent over it.

GA search over configurations:
- **Genome:** (G, depth, cross_group_type, layer_norm_position)
- **Fitness:** val_accuracy / sqrt(FLOPs) — Pareto-efficient configs
- **Population:** 20–50 configs, evolve over 10–20 generations
- **Evaluation:** short T0 scout (20ep, 50% data) per config

Mirrors CNN distiller GA search (step932) but over architectural structure rather than hyperparameters.

---

## Lever 4 — KD Soft Labels as Training Signal

All above trained with KD from original network's FF output:

```
L = KL(log_softmax(student_logits) || teacher_soft_labels)
```

Teacher soft labels encode inter-class similarity (e.g. "tench" and "goldfish" similar). This signal lets compressed student recover most teacher accuracy with far fewer params.

Key insight: teacher does NOT need retraining. Extract soft labels once, store to H5, reuse for all student ablations. Already done for Imagenette (`store_aug.h5`, 18938×10 soft_labels).

---

## Target Networks by Modality

| Modality | Network | FF Layer | N_IN | Current Status |
|---|---|---|---|---|
| Vision (Imagenette) | VGG16 | classifier (25088→4096→4096→1000) | 25088 | ✅ SGNNET 95.95% @ 0.20M FLOPs |
| Vision (CIFAR-10) | VGG16 pool5 | same | 25088 | 80.69% @ N=2048, gap −5.55pp |
| Audio (ESC-50) | VGGish / PANNs | FF block | ~128–2048 | gap confirmed structural (step928–965) |
| Time Series | ? (CNN-LSTM, Transformer) | FF block in encoder | TBD | ts_step010–030 in progress |
| Text | BERT / RoBERTa | FF blocks (768→3072→768) | 768 | gap confirmed negative (step405/410) |

**Paper 1 scope (confirmed):** Imagenette vision only. Audio + TS as honest secondary results or limitations.

**General target criterion:** any network where single FF layer has ≥ 1M params and takes ≥ 10% total inference FLOPs. These layers = max compression leverage.

---

## Lever 5 — Looped Hidden State (Weight-Tied Depth)

**Architecture:**
```
z₀ = W_in · x          # 25088 → 50  (one-time compression)
z_{t+1} = f(z_t + z₀)  # loop K times: 50 → 50, weight-tied, re-inject z₀
ŷ  = W_out · z_K        # 50 → 10    (readout)
```

Total params: W_in (25088×50) + W_loop (50×50) + W_out (50×10) = ~1.26M.
Compare to dense: 25088×10 = 251K (linear) or 25088×4096×... = hundreds of millions (VGG FC).
Loop adds only 2500 params but gives K "thinking steps."

**Key design choice — re-injection:**
Without `+ z₀`: loop is `z_{t+1} = f(z_t)` — iterated application of same map. If spectral radius < 1 (regularization pushes toward), converges to zero independent of input. **Destroys information.**
With `+ z₀`: loop stays anchored to input projection. Each step refines rather than forgets.

**Prior art:**
| Work | Mechanism | Key finding |
|---|---|---|
| Universal Transformer (Dehghani 2018) | Weight-tied transformer layer × T | T=6 sweet spot; matches learned-depth at fewer params |
| DEQ (Bai 2019) | Iterate f(z)=z to fixed point, implicit backprop | Infinite effective depth, O(1) recurrent params |
| Reservoir / Echo State Networks | Fixed random recurrent core, train output only | Surprising accuracy with zero recurrent training cost |
| **SGNNET K_iter** | Same routing function × K iterations | **Our data: K=5 optimal, K>5 → over-smoothing (step916)** |

SGNNET's `K_iter` = exactly this mechanism. step916 result (K>5 collapses) most relevant prior: sweet spot exists, over-iteration causes fixed-point collapse (over-smoothing analogue).

**Hypotheses to test (priority order):**

1. **K sweep with re-injection**: K ∈ {1,2,3,5,10}, `z_{t+1} = f(LayerNorm(z_t + z₀))`. Expect accuracy peak at K=3–5 then drop (same pattern as `K_iter`). This = T0 scout.

2. **Weight-tied vs weight-per-step**: one shared `W_loop` vs K separate matrices. Hypothesis: weight-tied wins on param efficiency; separate wins on accuracy. Crossover = paper finding.

3. **Bottleneck width vs K tradeoff**: width=50 + K=5 better than width=250 + K=1 at similar param count? Hypothesis: loops win when task has iterative structure; width wins for flat lookup.

4. **Re-injection variants**: none / additive (z_t + z₀) / gated (z_t + α·z₀, learned α). Hypothesis: gated marginally better; additive much better than none. Existence of re-injection matters more than form.

5. **Spectral norm on W_loop**: constrain spectral radius ≈ 1 explicitly. Hypothesis: most stable training, recovers fixed-point collapse cases.

**Connection to group decomposition:**
Combine Levers 1+5: compress 25088 → G groups of size H, loop within each group K times, then mix across groups. Parameter-efficient in both axes — group decomposition reduces O(N²) dense cost; looping reduces depth cost.

---

## Canonical Block — Vanilla FF Unit

**Design decision (2026-04-21):** Canonical building block = **vanilla feed-forward unit** — standard dense linear layer + LayerNorm + activation. No graph topology, no learned spatial positions, no ΔW-proj routing. Pure algebra.

```python
class VanillaFFBlock(nn.Module):
    def forward(self, x):
        return activation(LayerNorm(W @ x + b))
```

Groups, sparsity, looping all applied on top of this primitive. Keeps baseline maximally interpretable — every gain attributable to structural modification (grouping / looping / sparsity), not any learned inductive bias.

**Why vanilla FF as canonical:**
- Maximally transparent — no confounds from routing or topology learning
- Directly comparable to dense FF layers being compressed
- Standard primitives → easy benchmark with existing sparse CUDA kernels
- SGNNET comparison then clean: vanilla-grouped vs graph-structured, same task, same KD signal

---

## Relationship to SGNNET

SGNNET = one instantiation of this program:
- Group decomposition via graph topology (`K_in` neighbors = sparse group connections)
- Sparsity via `K_hh` routing (each node receives input from only `K_hh` hidden nodes)
- KD training via VGG FC soft labels (already implemented)
- Looping via `K_iter`=5 (weight-tied routing iterations)

SGNNET's specific contribution: graph structure learned from data (`W_pos` spatial precomputation) gives structural prior that pure block-diagonal decomposition lacks.

**Comparison we want:**

| System | Groups | Sparsity | Looping | Learned structure |
|---|---|---|---|---|
| Vanilla-FF-Grouped | block-diagonal | LeakyReLU→ReLU | optional K loops | ✗ none |
| SGNNET | K_in graph | K_hh routing | K_iter=5 | ✓ W_pos geometry |

Same task (VGG FC distillation, Imagenette), same KD signal, same 5-dim Pareto table. If vanilla-grouped matches SGNNET: learned structure adds no value. If SGNNET wins: graph geometry is load-bearing. Either result = clean paper finding.

---

## Experiment Sequence (Post Paper 1)

1. **Baseline:** VGG FC → Group-decomposed MLP (G=2,4,8,16) with LeakyReLU→ReLU, KD training. Measure accuracy vs FLOPs curve.
2. **Activation sparsity:** Add LeakyReLU→ReLU substitution. Benchmark sparse inference speedup.
3. **Cross-group interaction:** Ablate none / subtract / learned mixing.
4. **GA search:** Evolve over (G, depth, cross_group) space. Find Pareto front.
5. **Compare to SGNNET:** Same task, same KD signal, same evaluation protocol (5-dim Pareto table).
6. **Generalize:** Apply best config to CIFAR-10, audio (if gap closes), TS.

---

---

## Lever 6 — Forward-Forward / Distance-Forward Local Learning

**Reference:** "Advancing the forward-forward algorithm towards high-performance deep local learning" (Xu et al., Neural Networks 2026, https://doi.org/10.1016/j.neunet.2026.108765)

**Core idea:** Replace backprop with layer-local learning rule. Each layer trained to maximize "goodness" score on positive data, minimize on negative data, no gradient flowing between layers. No backward pass across layers → O(1) memory in depth (activations not retained for BP).

**DF (Distance-Forward) variant — key innovations over Hinton's original FF:**
- Reframes FF as **centroid-based metric learning**: each layer's goodness = distance to class centroids rather than sum-of-squares activation
- **N-pair margin loss**: pulls same-class activations together, pushes different-class apart (contrastive objective)
- **Layer-collaboration strategy**: adjacent layers share lightweight "collaboration signal" to reduce information loss from greedy local updates — partially recovers cross-layer credit assignment that pure FF loses
- **SNN extension**: goodness function for temporal spike sequences → event-driven implementation on neuromorphic hardware

**Claimed results (8 datasets):**
- Surpasses existing FF models and other local learning approaches
- < 40% memory cost vs BP training
- Stronger robustness to hardware noise (important for neuromorphic targets)

**Relevance to this program:**
- Memory efficiency aligns with core goal: compress / reduce memory footprint
- Layer-local updates = no full backward graph → fundamentally lower peak activation memory
- N-pair margin loss on centroids conceptually related to KD training: both use soft, relational signals (class similarity) rather than hard one-hot labels
- **Connection to group decomposition (Lever 1):** If each group trained with local FF rule, eliminate cross-group gradient flow entirely → each group = self-contained local learner. Groups become independent inference modules.
- **Connection to looped hidden state (Lever 5):** FF goodness on each loop iteration gives per-step local signal — potential alternative to BPTT through loop

**Unresolved gaps:**
- Accuracy gap vs BP on ImageNet-scale tasks (paper tests 8 datasets but likely not ImageNet-1K)
- Layer-collaboration strategy partially reintroduces cross-layer dependencies — memory savings claim needs careful profiling
- SNN extension performance vs rate-coded equivalent not clearly benchmarked

**Experiment ideas (post-paper-1):**
1. Replace KD-trained FF readout (Lever 4) with DF local learning per group layer
2. Compare: KD-from-teacher (global signal) vs DF-local (no teacher, purely layer-local) on Imagenette VGG distillation task
3. Measure actual peak memory: DF-trained group-MLP vs BP-trained group-MLP at same accuracy
4. Test N-pair margin loss as goodness function on SGNNET's per-iteration hidden state (each `K_iter` step as "layer")

### FF on CNNs (Scodellaro et al., Scientific Reports 2025)

**Reference:** "Training convolutional neural networks with the Forward–Forward Algorithm" (Scodellaro, Kulkarni, Alves, Schröter — Scientific Reports 15, 38461, 2025)

**Problem FF-on-CNNs had to solve:** Standard FF presents positive/negative examples with label embedded in input. For FC layers, straightforward (append one-hot to input). For conv layers, label must be spatially extended — scalar label can't broadcast across all spatial positions of feature map. If label only appears at one corner, conv kernels far from that corner never see it.

**Two labeling strategies:**
- **Fourier-pattern labels**: embed class identity as spatial frequency pattern tiled across input image. Every spatial position carries label info. Frequency encodes class; phase encodes instance.
- **Morphological-transformation labels**: apply class-specific morphological ops (erosion/dilation with class-specific structuring elements) to overlay label info on image texture. More complex, but prevents "shortcut" solutions where network ignores texture and keys only on label pattern location.

**Key findings:**
- Deeper FF-trained CNNs can be optimized successfully — FF previously only demonstrated on shallow FC nets
- Morphology labels outperform Fourier labels on CIFAR-100 (100 classes, more fine-grained) — Fourier patterns create shortcut opportunities on complex datasets
- Class Activation Maps show FF-trained CNNs learn meaningful and complementary features across layers — not layer-collapsed representations
- Scales to CIFAR-100 with carefully designed label sets

**Relevance to this program:**
- **Direct path to CNN distillation**: we distil VGG conv features (`pool5`) into compressed student. If student = CNN trained with FF rather than BP, peak memory drops significantly during student training.
- **Spatial label injection = key primitive**: for Imagenette task (VGG `pool5` features, shape 512×7×7 = 25088-dim), "spatial" structure already collapsed by `pool5`. Input = vector, so FF labeling trivial — no Fourier/morphology trick needed. We have easy case.
- **Harder question**: if we want FF during VGG CNN training itself (not just student training), conv layers need spatial labeling. Scodellaro paper provides that primitive.
- **Shortcut risk real**: their finding that Fourier labels create shortcuts on CIFAR-100 means label design matters. For KD setup, soft-label signal from VGG may act as natural anti-shortcut since it encodes fine-grained inter-class similarity.

**Open gaps:**
- Accuracy gap vs BP on CIFAR-10/100 (not stated explicitly in abstract — need to read results section)
- No ImageNet-scale result
- Memory profiling not provided (unlike Xu et al. DF paper which claims < 40%)
- CNN architecture tested likely shallow (CIFAR-scale); unclear if extends to ResNet/VGG depth

**Experiment ideas linking both FF papers:**
1. **Student FF training**: train compressed group-MLP student (Lever 1) using FF local learning instead of BP KD. Input already flat vector (`pool5`), so labeling trivial. Measure memory savings.
2. **DF + spatial label on conv layers**: apply Xu et al.'s N-pair margin loss with Scodellaro et al.'s morphological spatial labeling to train CNN student end-to-end with FF.
3. **FF goodness as auxiliary loss**: keep BP as main training signal, add per-layer FF goodness as auxiliary regularizer. Does it improve feature quality or just add noise?

---

## Lever 7 — Adiabatic Training + Low-Precision (FP4/FP8) for FC Block Distillation

**Context:** Adiabatic training failed on bare `SGNNET_SmallWorld` (step973) because model has only one param tensor (`W_pos`, 32K params) and K=50 updates/batch gives 0.15% coverage — too sparse to converge in 100ep. But for **group-decomposed FC blocks** (Lever 1), each group has own dense weight matrix. Total params scale as N²/G × depth. At G=8, depth=4, N=2048: ~4 × (256×256) = ~262K params. K=50 then gives 0.019% coverage per batch — still sparse, but hypothesis changes: sparse updates within dense group matrix less destructive than sparse updates on single routing-geometry tensor.

**Mechanism — precision-gated gradient updates (no top-K filter):**
Within each active layer, ALL gradients applied as-is — no explicit top-K masking. Only gradient sparsity comes from **float quantization underflow**: when gradients cast to FP8 or FP4, values below quantization grid's minimum representable magnitude flush to zero. Surviving gradients applied at full optimizer resolution.

Cleaner mechanism than top-K adiabatic:
- No hyperparameter K to tune
- Sparsity determined entirely by gradient magnitude distribution + precision choice
- FP8 kills smallest ~20–40% of gradients; FP4 kills ~60–80%
- "Adiabatic" character comes from **directional layer ordering** (which layer active), not intra-layer gradient masking

| Precision | Gradient kill threshold | Approx. gradient survival rate | Gradient memory |
|---|---|---|---|
| FP32 | none | 100% | 1× |
| FP8 (e4m3) | ~6×10⁻⁸ | ~60–80% | 0.25× |
| FP4 (NF4/e2m1) | ~6×10⁻³ | ~20–40% | 0.125× |

Quantization grid depends on per-tensor scaling factor. Numbers above = rough estimates; actual kill rate depends on gradient distribution of specific layer at training time.

**Directional training — two orderings:**

Group-decomposed block has L layers. Layer 0 closest to VGG `pool5` input; layer L-1 closest to output (class logits). Two strategies:

**Last-first (output → input):**
- Train layer L-1 first (N ep) with quantized gradients, freeze, then train L-2, ..., then layer 0
- Each layer trains against FIXED downstream readout → gradient signal stable and well-defined
- Analogous to greedy layer-wise pretraining (Hinton/Bengio 2006) and step970 `layer_seq` variant
- Risk: early layers trained last see frozen readout fitted without their features — potential info mismatch

**First-first (input → output):**
- Train layer 0 first against full KD soft-label loss, freeze, train layer 1, ..., then L-1
- Each layer compresses representation before next → progressive compression
- Risk: later layers receive frozen compressed features — no gradient flows back into compression
- Advantage: closer to FF local learning paradigm (Lever 6) — each layer trains with only forward signal from frozen predecessor

**Cumulative variant:**
- Unfreeze one more layer per phase (always adding toward other end)
- Recovers more cross-layer signal than pure sequential unfreezing at cost of longer per-phase training

**Comparison matrix:**

| Config | Direction | Gradient precision | Sparsity source |
|---|---|---|---|
| Ref | — (full BP) | FP32 | none |
| LastFirst_FP32 | L-1 → 0 | FP32 | none (baseline direction test) |
| FirstFirst_FP32 | 0 → L-1 | FP32 | none (baseline direction test) |
| LastFirst_FP8 | L-1 → 0 | FP8 | quantization underflow |
| FirstFirst_FP8 | 0 → L-1 | FP8 | quantization underflow |
| LastFirst_FP4 | L-1 → 0 | FP4 | quantization underflow (aggressive) |
| FirstFirst_FP4 | 0 → L-1 | FP4 | quantization underflow (aggressive) |

**Advance criterion:** Any directional config within 2pp of Ref → direction ordering validated. FP8 within 2pp of its FP32 directional counterpart → precision-gated training viable. FP4 within 3pp → aggressive quantization viable.

**What positive result means for paper:**
- Adiabatic + FP8 within 2pp of BP: training memory reduced to ~12% of FP32 baseline (0.25× precision × fewer retained activations from directional unfreezing)
- Adiabatic + FP4 within 3pp: even more aggressive — potentially most memory-efficient path to high-accuracy FC distillation
- Direction comparison: if first-first matches last-first accuracy, validates FF local-learning intuition (Lever 6) from pure optimization angle

**Implementation note — FP4 in PyTorch:**
Native FP4 training not directly supported in PyTorch stable (as of 2026-04). Options:
- `torch.float8_e4m3fn` / `torch.float8_e5m2` — available in PyTorch ≥ 2.1 on CUDA
- FP4 simulation: quantize weights/gradients to 4-bit grid using `torch.quantize_per_tensor` with scale, then cast back to FP32 for optimizer step
- `bitsandbytes` library: has NF4 (normalized float 4) — used in QLoRA, well-tested
- True FP4 hardware: H100 supports FP8, not FP4 natively; FP4 requires simulation or future hardware

---

## Key Open Questions

1. Does group decomposition + depth stacking recover expressivity lost from removing dense connectivity?
2. At what G does accuracy-vs-FLOPs curve cross SGNNET's curve?
3. Does LeakyReLU→ReLU substitution give real sparse-CUDA speedup (requires ≥ 70–80% sparsity)?
4. Is GA search necessary, or does simple G-sweep suffice?
5. For transformer FF blocks (4× model dim, GELU activation) — does this approach transfer?
6. Can DF local learning match KD accuracy on Imagenette VGG distillation task at < 40% memory?
7. Does N-pair margin goodness function work on SGNNET's `K_iter` loop iterations as layer-local signal?