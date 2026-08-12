# SGNNET Research Project — Qwen Briefing

## What this project is

SGNNET is a sparse graph neural network designed to **replace the FC classification head** of VGG16. It does NOT train a new vision model — VGG16's convolutional layers are frozen and pre-extracted into an HDF5 file. SGNNET operates on those 25088-dimensional conv features.

**Goal:** Match or beat VGG16's FC accuracy using <1% params and <5% FLOPs of VGG16's FC layer.

**Status:** All three thresholds met. Paper in preparation.

---

## Architecture

```
Input: pre-extracted VGG16 features (B, 25088)
  → K_in=25 random sparse connections per neuron (fixed, not learned)
  → N=2048 neurons, each D=16 dimensional embedding on S^{D-1}
  → K_iter=5 routing iterations on a small-world graph (K_hh=2 neighbors)
  → Anti-Hebbian repulsion: ΔW = W_pos[i] - W_pos[j] used as routing signal
  → Mean-pool readout → 10-class softmax
```

**Learned parameters:** θ (per-neuron thresholds), W_pos (positional encodings on sphere), fc_out (readout head). No edge weights, no message weights.

**Key claim:** Learning happens in routing dynamics, not in explicit weight matrices.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| N | Number of neurons (default 2048) |
| D | Embedding dimension (default 16) |
| K_in | Input connections per neuron (default 25 at N=2048, 15 at N≥4096) |
| K_hh | Hidden-hidden neighbors in small-world graph (default 2) |
| K_iter | Routing iterations (default 5) |
| α_AH | Anti-Hebbian strength (default 1.0) |
| T | KD temperature (T=1 for near-hard VGG labels) |

---

## FLOPs scope — CRITICAL

**All FLOPs numbers in this project are FC-head-only.** VGG16 conv FLOPs are identical for both systems (frozen, not counted). When claims say "1.85M FLOPs vs VGG16 FC 123.6M FLOPs", this is a head-to-head comparison.

Efficiency champion: **step605 K=1 KD student** — 95.95% @ 0.20M FLOPs, 34,976 params (0.029% of VGG16 FC).

---

## Tier protocol (experiment planning)

| Tier | Budget | Data | Purpose |
|------|--------|------|---------|
| T0 Scout | 20ep | 50% | Rejection filter — advance everything not clearly failing |
| T1 Calibration | 75ep | 50% | Paper-grade comparison |
| T2 Validation | 150ep | 100% | Publication-only |

---

## Dataset and machines

- **Imagenette:** 10-class subset of ImageNet, ~9469 train / 3925 val
- **CIFAR-10:** via VGG16 features (same pipeline)
- **Training slots:** 5060ti_cuda (fastest), mini_mps/cpu, studio_mps/cpu
- **Data:** pre-extracted to `data/store_aug.h5` (Imagenette), `data/cifar10_aug.h5` (CIFAR-10)

---

## Key confirmed findings

- step89-A: 97.86% accuracy, N=4096, D=64, 529K params
- step199: 97.30% @ 0.98M FLOPs (N=2048, D=16)
- step605: 95.95% @ 0.20M FLOPs — efficiency champion, 5.26× faster than VGG_FC
- F.normalize after each step: load-bearing (−50 to −71pp without it)
- AntiHebbian suppression: load-bearing (α=0 hurts significantly)
- K_iter sequential: load-bearing (every step matters, stochastic depth catastrophic)
- ΔW projection (+1.49pp T1, halves seed variance) — default mechanism

---

## Active experiments (check EXPERIMENT_QUEUE.md for current status)

- step977: multi-seed KD vs CE T1 (5 seeds × 2 configs)
- step923: CIFAR-10 K_in=15 T1
- Various signal routing experiments (step873/874/875)

---

## What qwen should NOT do

See `do_not_do.md` for failure mode list.
