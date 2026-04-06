# Research: Sparse Attention, Efficient FFN, and SGNNET Implications — Part A: Mechanisms

**Date:** 2026-03-30  
**Source:** Synthesized from literature review

---

## 1. Sparse Attention Mechanisms in Transformers

- **Longformer** (Beltagy et al., 2020): Sliding window of W tokens + global tokens.
  Complexity O(N*W) vs O(N²). Handles sequences up to 4,096 tokens; matches full
  attention on QA benchmarks. Window size W=512 = each token attends to ~1% of a 4K sequence.

- **BigBird** (Zaheer et al., 2020): 3-component sparse pattern: g global tokens +
  w local window tokens + r random tokens. Theoretically sufficient to be a universal
  approximator. BigBird-ETC achieves F1=75.5 on HotpotQA vs Longformer 74.3. Enables
  8x longer sequences on same hardware.

- **Reformer** (Kitaev et al., 2020, ICLR): Locality-sensitive hashing (LSH) replaces
  dot-product attention. O(N log N) complexity. With n_hashes=8 rounds, LSH approximates
  full attention almost exactly. Beneficial only for sequences > ~2K tokens.

- **FlashAttention** (Dao et al., 2022-2024): NOT sparse — computes exact attention
  but restructures memory access. Tiling + recomputation avoids HBM round-trips.
  Up to 7.6x faster than standard attention on GPT-2; up to 20x more memory efficient.
  **Key lesson: IO-awareness often beats algorithmic sparsification for real-world speedups.**

- **2024 trend:** Exphormer (Google) replaces full-graph attention with expander-graph
  edges + virtual global nodes, maintaining good mixing at O(N) cost. Sparse Growing
  Transformer allocates sparse depth at training time via progressive attention looping.

---

## 2. Sparse / Efficient Feed-Forward Layers

- **FFN sparsity in practice:** In 175B OPT (ReLU activations), >95% of FFN neurons are
  zero for any given token. All FFN layers show >90% sparsity — "lazy neuron" phenomenon.

- **ReLU revival (2024, ICLR):** Modern transformers switched from ReLU to GELU/SiLU,
  destroying natural sparsity. "ReLU Strikes Back" shows reintroducing ReLU recovers
  80-95% activation sparsity with comparable perplexity. ReLU² pushes sparsity higher.

- **TurboSparse (June 2024, arxiv 2406.05955):** Applied sparse activation to Mistral/Mixtral
  via dReLU. TurboSparse-Mistral-7B activates only 2.5B params per token (90% FFN sparsity);
  achieves 2-5x decoding speedup; matches/beats dense baseline on benchmarks.

- **Switch Transformer / MoE (Fedus et al., 2021):** Each FFN layer replaced by N_experts
  expert FFNs. Router assigns each token to top-K experts (K=1 for Switch, K=2 for GShard).
  Computation per token: K/N_experts fraction of dense FFN cost. Requires load balancing loss.

- **DeepSeekMoE (Jan 2024, arxiv 2401.06066):** Two innovations:
  (1) Fine-grained expert segmentation — split FFN hidden dim into mN smaller experts.
  (2) Shared expert isolation — K_s experts always activate (universal knowledge); routing
  selects from remaining routed experts only.
  DeepSeekMoE-16B matches LLaMA2-7B at ~40% compute.
  DeepSeek-V3: 256 routed experts, top-8 active, plus 1 shared expert.

- **Mixture of Depths (MoD, Google DeepMind, April 2024):** Top-k router decides which
  tokens participate in each layer's attention+MLP. Routed-out tokens skip entirely.
  MoD transformers: +1.5% better on training objective at identical FLOPs; or ~50% fewer
  FLOPs per forward pass at loss parity. **SGNNET NOTE: step34 showed MoD fails for SGNNET
  routing (all 8 K_iter steps are necessary; no early exit possible).**

- **Unified view:** MoE, sparse FFN, and structured pruning are all input-conditional
  sparsification: `FFN(x) = sum_{k in top-K(router(x))} expert_k(x)`.

---

## 3. Matformer (Matryoshka Transformer)

- **Core idea (arxiv 2310.07707, NeurIPS 2024):** Matryoshka embedding nesting applied
  to transformer FFN blocks. One model trained with nested FFN granularities simultaneously
  — the FFN is structured so the first m neurons of each layer form a valid smaller model.

- **Training procedure:** During each step, randomly sample granularity g (FFN width fraction).
  Optimize loss for that sub-model. All nested sub-models jointly optimized in one run.

- **Result:** After one training run, hundreds of smaller models extractable at different FFN
  widths. Each extracted sub-model outperforms an independently trained model of the same size.

- **SGNNET analog:** Train at N=2048, randomly subsample neurons to N_active ∈ {512, 1024, 2048}
  per training step. After training, evaluate at any N without retraining. Saves ~3x compute
  for the N sweep and provides cleaner comparisons (same seed, same graph structure).

---

## Key Numbers Summary

| Method | Sparsity | Compute vs Dense | Accuracy Impact |
|--------|----------|-----------------|-----------------|
| Longformer | ~1% tokens attend globally | O(N*W) vs O(N²) | Matches full attn on QA |
| BigBird | g+w+r << N per token | O(N) vs O(N²) | +1.2 F1 over Longformer |
| Reformer LSH | ~log(N) buckets | O(N log N) vs O(N²) | Near-exact at n_hashes=8 |
| FlashAttention | 0% (exact) | Same FLOPs, 7-20x faster | No accuracy loss |
| Switch MoE (K=1) | 1/N_experts active | Same FLOPs as 1 expert | Scales model size cheaply |
| DeepSeekMoE | top-8/256 experts | ~40% of comparable dense | Matches LLaMA2-7B at 40% cost |
| MoD | ~50% tokens skip per layer | -50% FLOPs per forward | +1.5% vs isoFLOP vanilla |
| TurboSparse | 3% params active (FFN) | 10x fewer activated params | 2-5x decoding speedup |
| SGNNET signed coupling | all-pairs per step K_iter=8 | ~8N² ops (same as 4x FFN) | +10.93pp over ref at N=512 |

---

## Sources

- [Longformer: arxiv 2004.05150](https://arxiv.org/abs/2004.05150)
- [BigBird: arxiv 2007.14062](https://arxiv.org/abs/2007.14062)
- [Reformer: arxiv 2001.04451](https://arxiv.org/pdf/2001.04451)
- [FlashAttention: arxiv 2205.14135](https://arxiv.org/abs/2205.14135)
- [Mixture of Depths: arxiv 2404.02258](https://arxiv.org/abs/2404.02258)
- [DeepSeekMoE: arxiv 2401.06066](https://arxiv.org/abs/2401.06066)
- [TurboSparse: arxiv 2406.05955](https://arxiv.org/abs/2406.05955)
- [MatFormer: arxiv 2310.07707](https://arxiv.org/html/2310.07707v2)
- [ReLU Strikes Back: ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/6cf669c222ad13f60d503736fb2bd15b-Paper-Conference.pdf)
- [Exphormer: Google Research blog](https://research.google/blog/exphormer-scaling-transformers-for-graph-structured-data/)
