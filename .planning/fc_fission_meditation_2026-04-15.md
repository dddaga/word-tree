# FC Fission — Literature Synthesis + Experiment Designs
# 2026-04-15

Context: MLP_37 (Linear 25088→37→10) hit 97.71% @ 1.86M FLOPs, 929K params.
Question: Can structural compression of that FC layer match MLP_37 at half the params?
If yes: the trick was the bottleneck size, not the routing. If no: architecture matters.

---

## Part 1 — Literature Synthesis

### 1. Block-Diagonal / Grouped Linear ("FC Fission")

The user's proposal is exactly a **block-diagonal linear** (BDL) layer: two parallel blocks of size
N/2 × M/2, followed by an aggregator. Formalized in Nesky et al. (OpenReview 2018) and studied
empirically at 2× and 4× splits. Key finding from that work: (64,2)-BD (64 blocks of size 2)
achieved ~47× parameter compression at only −2% accuracy drop on standard FC benchmarks.

**ResNeXt** (Xie et al., CVPR 2017) is the canonical grouped-convolution paper and provides the
clearest cardinality trade-off curve: at fixed FLOPs, increasing cardinality (= more parallel
groups) consistently beats increasing depth or width. The saturation point: accuracy degrades once
per-group bottleneck width drops below ~4 units. Below that threshold, each group lacks expressive
capacity to capture meaningful correlations.

**MegaBlocks** (Gale et al., MLSys 2023, arXiv:2211.15841) implements MoE as block-sparse matrix
ops, demonstrating 40% training speedup vs Tutel. Key insight: block-diagonal computation is
hardware-efficient once block size ≥ 128 for CUDA — irrelevant at our tiny N/M scale but worth
noting for CUDA path.

**Prior-art prediction:** at 2-chunk, expect <1pp loss. At 4-chunk, 1-2pp. At 8+, ≥3pp unless
the aggregator compensates. Aggregator is the critical component — without it, cross-chunk
interactions are zero.

---

### 2. Monarch Matrices / Butterfly / Structured Sparse

**Monarch** (Dao et al., ICML 2022, arXiv:2204.00595): product of two block-diagonal matrices.
More expressive than naive block-diagonal because off-block interactions happen in the second
factor. Key claim: Monarch achieves 2× training speedup on ViT/GPT-2 at comparable quality;
40% error reduction on PDE/MRI tasks. Monarch has an *analytical* optimal projection from
dense W → Monarch approximation (not heuristic). This is strictly better than naive FC fission:

  - Naive block-diagonal: 0 cross-chunk interaction before aggregator
  - Monarch (= two BDL in sequence): full cross-chunk mixing after first factor

**Monarch Mixer** (Fu et al., NeurIPS 2023, arXiv:2310.12109): replaces attention *and* MLP with
Monarch matrices. Matches BERT-base/large on GLUE with 27% fewer params, 9.1× higher throughput
at seqlen=4K. Relevant because it shows Monarch generalizes beyond language to vision (ViT-style
classification tested).

**For our scale:** N_in=25088, hidden=37, N_out=10. The matrices are so small that Monarch's
block structure might not activate efficiently (designed for d ≥ 512). But the mathematical
property holds: Monarch is a strictly richer class than block-diagonal. If FC fission fails at
4+ chunks, a two-factor Monarch could recover it.

---

### 3. Concat-ReLU (CReLU)

**Shang et al., ICML 2016, arXiv:1603.05201**: key observation is that lower CNN layers learn
filter pairs with *opposite phase* — a natural consequence of ReLU discarding negative
activations. CReLU makes this explicit: emits [ReLU(x), ReLU(-x)], doubling feature width at
zero extra parameter cost.

**When it helps:** consistently in lower convolutional layers of vision CNNs (AlexNet, VGG, ResNet
lower stages). The opposite-phase pairing is an empirical property of convolutional feature maps.

**Known failure modes:**
1. Does NOT help in deeper layers where features are more semantic and phase-paired structure
   breaks down.
2. Does NOT help in FC bottleneck layers: the opposite-phase observation was made for
   *spatial convolutional* filters, not for abstract bottleneck activations.
3. In a 37-unit bottleneck, CReLU emits 74 units → the aggregator Linear(74→10) has more
   params than the bottleneck itself. This partially defeats the param reduction goal.
4. No paper establishes CReLU benefit in MLP layers operating on VGG features (post-pooling,
   after 5 conv stages — these are *not* low-level filters).

**Prior-art prediction:** CReLU is unlikely to help here. The VGG feature vector has gone through
5 max-pool stages. The phase-pairing signal is gone. CReLU adds compute without mechanistic
justification at this layer depth. Low prior for improvement; medium prior for neutral or negative.

---

### 4. Low-Rank Factorization (LoRA-style)

**Hu et al., arXiv:2106.09685 (LoRA)**: N→r→M with r « min(N,M). For LLM fine-tuning:
r=4/8 works for simple tasks, r=16-32 for domain adaptation, r=64+ for complex transfer.
Empirical scaling: Performance(r) ≈ Performance_full − c/log(r), logarithmic diminishing returns.

**NeurIPS 2024 structured FFN paper** (arXiv:2406.16450): systematic comparison at 63% and 32%
parameter retention at scale (up to 1.3B params):

| Method | 63% params | 32% params (perplexity delta vs dense) |
|--------|-----------|----------------------------------------|
| LowRank | +0.40 | +1.09 |
| BlockDense | +0.51 | +1.28 |
| BlockShuffle (= 2-factor BDL) | +0.52 | +1.35 |

LowRank consistently wins at matched parameter budget. BlockShuffle (FC fission analog) is
slightly worse. The gap grows at higher compression (32% params).

**Training instability warning from same paper:** structured layers (especially low-rank) exhibit
loss spikes during training. Their fix: "self-guided training" — dense residual branch for the
first ~50% of training, then gradually zero it. At our tiny scale and 20ep Tier-0, this may not
manifest.

**For our scale** (N=25088→r→10): r=8 gives 25088×8 + 8×10 = 200,784 params (22% of MLP_37's
929K). r=16 gives ~402K params (43%). r=37 = MLP_37 itself. So low-rank at r=8/16 is the most
aggressive compression path.

---

### 5. Depth vs Width Scaling for FC

**Tay et al., arXiv:2109.10686 (Scale Efficiently)**: model *shape* matters for downstream
fine-tuning beyond just total parameter count. Widely-adopted T5-base/large are Pareto-inefficient.
Broader/shallower models often match deeper models at equal compute. Finding: for transformers,
no universal winner — task-dependent.

**"Depth Delusion"** (arXiv:2601.20994, 2025): for transformers, optimal depth scales as
D* ∝ C^0.12 while optimal width scales as W* ∝ C^0.34. Width should grow 2.8× faster than depth
as compute budget grows. Beyond critical depth D_crit ∝ W^0.44, adding layers *hurts*.

**Caveat:** both papers are transformer-specific. For pure MLPs on fixed feature vectors (our
case), the dynamics differ: no residual mixing, no attention, simpler loss surface.

**For pure MLP + fixed VGG features (our task):** MLP_37 is already at depth=1. Going to
depth=2/3 adds expressive power but also regularization difficulty. At 929K params: a 2-layer MLP
would be approximately Linear(25088→36→10) with shared param budget. Whether that beats 1-layer
is *unknown for this exact task* — genuinely an open question. No prior art directly addresses
"depth vs width for MLP on frozen CNN features at 10-class image classification."

---

### 6. Recent 2024-2025 Sparse/Efficient FFN Work

**MonarchMixer** (Fu et al., arXiv:2310.12109): sub-quadratic GEMM-based architecture replacing
attention + MLP. Vision (ViT-style) tested. Architecture-level replacement, not a drop-in layer.

**Hyena** (arXiv:2302.10866): long-convolution attention replacement. No sequence dimension in
our case — irrelevant.

**MoE / FFSplit**: conditional sparse routing over expert groups. Different from FC fission: MoE
activates top-k experts only (learned routing); FC fission activates all groups (static
block-diagonal). MoE has higher expressiveness but degenerates at our scale (37 hidden units).

---

### Summary Prediction for FC Fission

| Approach | Param ratio vs MLP_37 | Prior-art accuracy cost | Notes |
|---|---|---|---|
| 2-chunk FC fission | 0.50× | <1pp likely | ResNeXt/BDL evidence |
| 4-chunk FC fission | 0.25× | 1-3pp likely | Group width = 9 units, risky |
| CReLU MLP_18 | ~0.50× | Neutral or negative | No mechanistic basis here |
| Low-rank r=16 | 0.43× | <1pp at 63% retain | NeurIPS 2024 LowRank result |
| Low-rank r=8 | 0.22× | 1-2pp | 32% retain regime |
| Depth-2 matched-param | 1.0× | Unknown | No prior art for this task |

**Key insight:** at our scale, all structured compression approaches are navigating between
"chunk/group too small to be expressive" (the ResNeXt saturation point) and "aggregator
introduces enough interactions to compensate." The aggregator is the make-or-break component
of FC fission.

---

## Part 2 — Experiment Designs

All experiments: T0 = 20ep, 50% data. Acceptance is relative to MLP_37 baseline (97.71%).
Reference baseline for each comparison: retrain MLP_37 at T0 in the same run for fair comparison.
All use VGG16 features (N_in=25088) → 10 classes.

---

### step606 — FC Fission 2-Chunk (user's exact proposal)

**Architecture:**
```
Input: [B, 25088]
  → chunk A: [B, 12544] → Linear(12544, 19) → ReLU
  → chunk B: [B, 12544] → Linear(12544, 18) → ReLU
  → concat: [B, 37]
  → Linear(37, 10)
```
Total params: 12544×19 + 19 + 12544×18 + 18 + 37×10 + 10 = 238,336 + 225,792 + 380 = **464,518**
(~50% of MLP_37's 929,157 params).

**Variants to sweep in one script:**
- A: No aggregator (direct Linear(37→10) after concat) — baseline FC fission
- B: 1-layer aggregator (concat→Linear(37,24)→ReLU→Linear(24,10))
- C: Learned inter-chunk mixing — concat→Linear(37,37)→ReLU→Linear(37,10)

FLOPs estimate: ~930K MACs (50% of MLP_37's 1.86M).

**Acceptance criteria:**
- STRONG: ≥97.71% at <50% params (beats MLP_37 at half the params)
- MEDIUM: ≥97.0% at <50% params (within 0.7pp at half params)
- VIABLE: ≥96.5% at <50% params (within 1.2pp — still publishable as Pareto point)
- KILL: <96.0% regardless of params

**What it tests:** does the bottleneck dimension (37) drive MLP_37's accuracy, or does the full
N_in→37 interaction matter? If YES to fission: the key variable is bottleneck size, not
connectivity. If NO: cross-feature interactions matter even in a linear layer.

---

### step607 — FC Fission N-Chunk Sweep (2/4/8/16 chunks)

**Architecture:** generalized block-diagonal. For C chunks:
```
Input: [B, 25088]
  → C parallel Linear(25088/C, 37//C + remainder) → ReLU
  → concat: [B, 37]
  → aggregator: Linear(37, 37) → ReLU → Linear(37, 10)
```
Chunk configs: C=2 (12544→19), C=4 (6272→9+1), C=8 (3136→5), C=16 (1568→3)

Params:
- C=2: ~465K (50% of MLP_37)
- C=4: ~232K (25%)
- C=8: ~116K (12%)
- C=16: ~58K (6%)

Note: at C=8, each group output is 4-5 units — near the ResNeXt saturation boundary.
At C=16, groups have 3 units — likely below viability.

**Acceptance criteria:** same scale as step606. Looking for the "cliff" — the chunk count
where accuracy degrades sharply. This gives the scaling law for FC fission on this task.

**Design note:** all C values in a single script, sweep as a config dict. One T0 run covers
the full curve. This is cheap: 5 configs × 20ep × 50% data.

**What it maps:** the block-diagonal accuracy-vs-compression Pareto curve for VGG features.

---

### step608 — CReLU MLP

**Architecture:**
```
Input: [B, 25088]
  → Linear(25088, 18) → CReLU (= cat(ReLU(x), ReLU(-x))) → [B, 36]
  → Linear(36, 10)
```
CReLU implementation: `torch.cat([F.relu(x), F.relu(-x)], dim=-1)`

Params: 25088×18 + 18 + 36×10 + 10 = 451,584 + 18 + 360 + 10 = **451,972** (~49% of MLP_37).

Compare against:
- Matched-param control: MLP_18_relu (Linear(25088, 18)→ReLU→Linear(18,10)) — ~451K params
- MLP_37 reference

FLOPs: 451K MACs (vs 1.86M for MLP_37).

**What it tests:** does CReLU's implicit negative-phase recovery help when applied to VGG
features (post-5-pool)? If the feature space retains phase structure even after deep CNNs,
CReLU should give a free capacity boost. If not, CReLU ≈ MLP_18_relu.

**Acceptance criteria:**
- STRONG: CReLU matches or beats MLP_37 (97.71%) at half params
- USEFUL: CReLU beats MLP_18_relu by ≥0.5pp (confirms CReLU mechanism is active here)
- KILL: CReLU ≤ MLP_18_relu (mechanism is inactive; discard for this feature type)

**Prediction:** KILL is most likely (>60% prior). Worth running because it's cheap (1 config)
and would definitively close the CReLU question for post-CNN features.

---

### step609 — Depth vs Width Sweep (matched param budget ~929K)

**Architecture variants (all ~929K params, same budget as MLP_37):**

Param-matched configs (verified):
| Name | Layer dims | Approx params |
|------|-----------|---------------|
| D1 (ref) | [25088, 37, 10] | ~929K |
| D2 | [25088, 35, 35, 10] | ~880K |
| D3 | [25088, 36, 36, 36, 10] | ~906K |
| D2_narrow_wide | [25088, 18, 74, 10] | ~478K |

D2 uses w=35: 25088×35 + 35×35 + 35×10 = 879,655.
D3 uses w=36: 25088×36 + 36×36×2 + 36×10 = 906,120.
D2_narrow_wide is under-budget (478K) but tests asymmetric bottleneck.
Add BN after each hidden ReLU for D2/D3 only (BN adds ~2× w params, negligible).

**Acceptance criteria:**
- STRONG: D2 or D3 matches D1 (97.71%) — confirms depth works at this scale
- HYPOTHESIS_CONFIRM: D1 wins → bottleneck width is the primary lever, not expressiveness
- USEFUL: D2_narrow_wide beats D1 at ~51% params → "asymmetric bottleneck" finding

**What it tests:** is MLP_37's accuracy driven by the specific dimension "37" or by the
expressive capacity of a matched-param deeper network? Answers the depth-vs-width question
for fixed VGG features.

---

### step610 — Low-Rank MLP

**Architecture:** N→r→M factored as two linear layers without nonlinearity between them
(pure matrix factorization), vs N→r→M with nonlinearity (= standard 2-layer MLP). Test both.

```
Rank variants: r = 8, 16, 24, 32, 37 (r=37 = MLP_37 baseline)

LR_pure (no nonlinearity): Linear(25088, r) → Linear(r, 10)
  Params (r=8):  25088×8 + 8×10 = 200,784 (~22% of MLP_37)
  Params (r=16): 25088×16 + 16×10 = 401,568 (~43%)
  Params (r=32): 25088×32 + 32×10 = 803,136 (~86%)

LR_relu: Linear(25088, r) → ReLU → Linear(r, 10)
  (same param counts, adds nonlinearity)
```

Note: LR_pure(r=37) = MLP_37 (they are identical). So r=37 is the natural reference point
and lets us validate the sweep is internally consistent.

**Acceptance criteria:**
- STRONG: LR_relu(r=16) matches MLP_37 at 43% params → low-rank is a superior compression
- MEDIUM: LR_relu(r=16) within 1pp of MLP_37 → viable Pareto point
- INTERESTING: LR_pure beats LR_relu at same r → nonlinearity hurts (rare but possible for
  nearly-linear feature spaces like frozen VGG features post-linear-classifier training)
- KILL for approach: r=32 (86% params) still significantly worse than MLP_37 → low-rank is
  a poor structural prior for VGG features

**What it tests:** whether VGG16 features are approximately low-rank (concentrated in a small
subspace). If yes, r=8-16 captures most accuracy. If no (features are high-rank), need full r=37.
This has direct implications: if VGG features are low-rank, the entire MLP_37 result collapses to
"you only needed r=16 linear projection" — further pressuring the paper narrative.

---

## Part 3 — Recommendations

### Priority ranking

| Step | EV | Rationale | Cost |
|------|---|---|---|
| **step610** (low-rank) | **Highest** | Most likely to reveal structural property of VGG features; NeurIPS 2024 paper shows low-rank wins at matched params; if r=16 matches MLP_37, collapses paper narrative further AND points to a strong baseline | 5 configs, T0 cheap |
| **step609** (depth/width) | **High** | Answers a clean empirical question; D2 vs D1 at matched params; result is unambiguous either way | 3-4 configs, T0 cheap |
| **step606** (2-chunk fission) | **High** | User's exact proposal; half params; aggregator variants reveal whether cross-chunk interaction matters | 3 variants, T0 cheap |
| **step607** (N-chunk sweep) | **Medium** | Extends step606; only run if step606 shows signal; maps the scaling curve | 5 configs, depends on step606 |
| **step608** (CReLU) | **Low** | Theory says unlikely to help; low expected payoff; BUT cheap and closes the question | 1 config, T0 trivial |

### Fire order

1. **Fire step610 immediately** (lowest hanging fruit, highest information density, directly
   answers "is the feature space low-rank?")
2. **Fire step609 alongside** (independent question, same data, same budget)
3. **Fire step606 alongside** (user's proposal deserves a fair test)
4. **step607 contingent on step606** — if 2-chunk shows ≥VIABLE, sweep chunks. Otherwise skip.
5. **step608 fill a slot** — run when a slot is idle, not a priority.

### Which most likely reframes the paper?

**step610 (low-rank).** If r=8 or r=16 matches MLP_37:
- The VGG feature space has intrinsic rank ≤16 for the 10-class task
- MLP_37 is essentially doing a rank-16 projection accidentally
- SGNNET's graph routing over N=2048 nodes in D=16-dimensional sphere space is doing something
  structurally similar to low-rank projection — but learned geometrically
- Paper reframe: "both MLP_37 and SGNNET exploit the low-rank structure of VGG features;
  SGNNET's advantage is in the geometry of that projection"

**step606 (FC fission).** If 2-chunk matches MLP_37 at half params:
- The original MLP_37 claim ("a 37-unit bottleneck is sufficient") strengthens: even half
  the cross-feature interactions are sufficient
- Paper reframe: "the VGG feature space is highly factorizable; any bottleneck of ~37 effective
  units captures the Imagenette-relevant variance regardless of connectivity pattern"

### If ALL experiments succeed:

The 10-class Imagenette task on VGG features is trivial. Any bottleneck near rank(relevant
variance) works regardless of connectivity. Cross-dataset (CIFAR-100, step601) becomes the
decisive validation — does SGNNET generalize where compressed linears don't?

### Practical launch note

All 5 steps can use the TEMPLATE_experiment.py base. step610 is pure PyTorch, no custom modules.
step606 needs a FissionMLP class (trivial: two linear blocks + cat + aggregator). step608 needs
CReLU (one line: `torch.cat([F.relu(x), F.relu(-x)], dim=-1)`). Smoke-test flag `--help` per
project rules before any launch.

---

## References

| Paper | arXiv |
|-------|-------|
| ResNeXt (grouped convs, cardinality) — Xie et al., CVPR 2017 | — |
| Block-Diagonal Inner Product Layers — Nesky et al. 2018 | openreview HyI5ro0pW |
| Monarch matrices — Dao et al., ICML 2022 | 2204.00595 |
| MonarchMixer — Fu et al., NeurIPS 2023 | 2310.12109 |
| CReLU — Shang et al., ICML 2016 | 1603.05201 |
| LoRA — Hu et al. 2021 | 2106.09685 |
| Scale Efficiently — Tay et al. 2021 | 2109.10686 |
| Depth Delusion 2025 | 2601.20994 |
| Structured FFN NeurIPS 2024 — block-diag vs low-rank | 2406.16450 |
| MegaBlocks — Gale et al., MLSys 2023 | 2211.15841 |
