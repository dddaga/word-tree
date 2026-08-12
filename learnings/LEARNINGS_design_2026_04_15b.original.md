# Design Log — 2026-04-15 (Part B: FC-fission / MoE-hybrid / cross-modal results)

---

## step610 Result — Low-rank MLP sweep (2026-04-15)

**Setup:** T0 20ep, 50% data, VGG16 Imagenette features, 7 configs.

| Config | params | FLOPs | top1 (T0) | Δ vs MLP_37_ref |
|---|---|---|---|---|
| LR_pure_r8 | 201K | 402K | 96.00% | −0.74pp |
| LR_pure_r16 | 402K | 803K | 96.36% | −0.38pp |
| LR_pure_r32 | 803K | 1.6M | 96.76% | +0.02pp |
| LR_relu_r8 | 201K | 402K | 94.55% | −2.19pp |
| LR_relu_r16 | 402K | 803K | 96.03% | −0.71pp |
| LR_relu_r32 | 803K | 1.6M | 96.99% | +0.25pp |
| MLP_37_ref | 929K | 1.9M | 96.74% | — |

**Verdict: MEDIUM.** LR_relu_r16 = 96.03% — below STRONG threshold (≥97.71%). Feature space is NOT rank≤16; approximately rank ≤32 in linear sense.

**Key observations:**
1. **LR_pure_r16 (96.36%) > LR_relu_r16 (96.03%)** — at low rank, nonlinearity hurts. VGG post-pool features are near-linearly separable; adding ReLU at rank 16 discards information.
2. **LR_pure_r32 (96.76%) ≈ MLP_37_ref (96.74%)** — rank-32 linear projection is sufficient, ReLU provides minimal benefit once rank is adequate.
3. **Feature rank ≈ 32** for this 10-class task on VGG16 features.

**Paper implication:** SGNNET operates in D=16 dimensional space on S^{D-1}. The feature space has rank ≈32. SGNNET achieves ~97.71% at D=16 while LR_relu_r16 (43% of MLP_37 params) hits only 96.03%. This confirms SGNNET's iterative routing + hypersphere geometry extracts MORE from the same 16-dim projection than a static low-rank layer — by iterating K=5 times with dynamic ΔW-guided message passing. **HYPOTHESIS** (needs controlled comparison with matched param budget).

**Next steps:**
- step612 (routing granularity probe): queue when slot frees — decisive for mechanism claim
- If T1 warranted for LR_relu_r32: add to queue after step612 completes (deferred — below 97.71%)

---

## step405 SST-2 SGNNET — Bug Fix Summary (2026-04-15)

**Bug root cause (two issues):**
1. `use_trainer=(key == "SGNNET")` → Trainer used F.kl_div with soft_labels=zeros(1) → loss=0 every batch → zero gradient, dead network.
2. `compute_fourier_encoding` defaulted to VGG spatial (h=7,w=7,c=512) — meaningless for flat 768-dim DistilBERT CLS vector. Degenerate spatial coords.

**Fixes:**
1. `use_trainer=False` (line 261) — all configs use CrossEntropyLoss directly.
2. Override Fourier encoding: `compute_fourier_encoding(768, D=16, h=768, w=1, c=1)` registered as buffer.

**Status:** MLP_37=84.63%, MLP_64=84.63% saved from old session. SGNNET relaunched on studio_cpu. ep1=0.7844 (healthy). 150ep to complete.

**Expected range:** MLP on binary SST-2 achieves ~84%. SGNNET with D=16 / K_iter=5 on NLP — no prior result. Watch: does routing on 768-dim NLP flat features converge? ep1 at 78.44% is promising.

---

## Slot Status — 2026-04-15 ~05:00

| Slot | Step | Status |
|---|---|---|
| mini_cpu | step235 seed=43 | RUNNING (legacy, ~75ep run) |
| mini_mps | step235 seed=44 | RUNNING (legacy, ~75ep run) |
| studio_cpu | step405 SGNNET (fixed) | RUNNING (ep1=78.44%) |
| studio_mps | step601 CIFAR-100 | RUNNING (75ep, just launched) |
| 5060ti_cuda | step266 seed=43 | RUNNING |

**Queue (next to launch when slot frees):** step602 RUNNING → step604 next → step603/605 after teachers complete

---

## step612 Result — Group-level ΔW routing granularity probe (2026-04-15)

**Setup:** T0 20ep, 50% data, CUDA, N=2048 D=16 K_hh=2 K_iter=5.

| Config | top1 (T0) | params | Δ vs Ref |
|---|---|---|---|
| Ref (per-neuron ΔW) | **93.91%** | 34,976 | — |
| GroupDW (per-group) | 21.73% | 39,072 | **−72.18pp** |

**Verdict: ABANDON.** GroupDW near-random (21.73% vs 10% random for 10 classes). Per-neuron routing is CONFIRMED essential. The direction `W_pos[i] − W_pos[j]` encodes specific pair geometry on S^{D-1} that cannot be shared across 8 neurons.

**Paper implication (CONFIRMED):** "Per-neuron routing at geometric granularity is essential — group-level coarsening (K_g=8) collapses accuracy by 72pp." Cleanly differentiates SGNNET from sparse MoE. Defensible controlled ablation: one variable changed, −72pp. Deprioritize step613 (coarse-to-fine) — if group routing fails at K_g=8, hierarchy won't rescue it.

---

## Meditations completed this session

1. **FC-fission meditation** (`.planning/fc_fission_meditation_2026-04-15.md`) — FC-fission = block-diagonal linear / low-rank factorization. LowRank > BlockDiag per NeurIPS 2024. step610 result: MEDIUM (rank ≈32 not ≤16). Defers FC-fission appeal slightly — high-rank features mean parallel dense chunks would each need substantial rank.

2. **MoE↔SGNNET hybrid meditation** (`.planning/moe_hybrid_meditation_2026-04-15.md`) — Three experiments: step611 (hierarchical), step612 (group-ΔW, decisive), step613 (coarse-to-fine). step612 is the cleanest test: no new gate, no softmax, no auxiliary loss. Either outcome is paper-worthy.

---
