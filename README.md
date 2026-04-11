# SGNNET — Sparse Geometric Neural Network

O(N·K) graph neural network achieving VGG16 FC accuracy at <1% of its compute.

---

## Key Result

**95.52% accuracy on FashionMNIST at 0.98M FLOPs — 0.79% of VGG16 FC's 123.6M FLOPs.**

| Metric | VGG16 FC | SGNNET (step199) | Ratio |
|--------|----------|-------------------|-------|
| Accuracy | 95.0% | 95.52% | +0.52pp |
| FLOPs | 123.6M | 0.98M | **0.79%** |
| Params | 123.6M | 67K | **0.05%** |

Both the ≤1% FLOPs and ≤1% params criteria are met simultaneously at ≥95% accuracy.

---

## Architecture

SGNNET stacks three composable modules:

**SGNNET_SmallWorld** — the core sparse graph. N neurons occupy positions on the unit hypersphere S^{D-1}. Connectivity is small-world: K_local nearest-neighbor edges + K_random long-range edges per neuron. Input fan-in K_in projects the 25088-dim FashionMNIST feature map onto N neurons. Fourier encoding maps positions to D-dimensional features.

**SGNNET_Resonant** — iterative message-passing router. Runs K_iter rounds of dynamic Z-geometric routing (mode=`dynamic_z_geo`). Each round propagates activations along K_hh edges per neuron. A reflection term (alpha_reflect) stabilizes oscillations; beam_size=16 limits candidate edge set.

**SGNNET_AntiHebbian (wpos)** — positional diversity regularizer. Penalizes correlated W_pos vectors so neurons spread across the hypersphere rather than collapsing. alpha_ahebb=1.0 is the confirmed optimal strength.

FLOPs budget: `3 × N × K_hh × D × K_iter` — fully determined by five integers.

---

## Efficiency Frontier (D=16, K_hh=2 family)

| Step | N | K_iter | FLOPs | FLOPs % | Accuracy | Note |
|------|---|--------|-------|---------|----------|------|
| step199 | 2048 | 5 | 0.98M | **0.79%** | **95.52%** | Final efficiency config |
| step195 | 2048 | 6 | 1.18M | 0.95% | 96.08% | First ≤1% FLOPs hit |
| step205 | 4096 | 5 | 1.97M | 1.59% | 97.17% | D=16 record |
| step209 | 8192 | 5 | 3.93M | 3.18% | 97.17% | D=16 ceiling confirmed |
| — | — | — | 123.6M | 100% | 95.0% | VGG16 FC baseline |

D=16 ceiling is 97.17% (N=4096/N=8192 both converge here). Project accuracy best remains 97.86% at D=64, step89.

---

## N-Scaling Law (D=16, K_hh=2, K_iter=5, Tier-2)

| N | FLOPs | Accuracy |
|---|-------|----------|
| 1024 | ~0.49M | ~88.9% (T1 proxy) |
| 2048 | 0.98M | 95.52% |
| 4096 | 1.97M | 97.17% |
| 8192 | 3.93M | 97.17% |

N=2048→4096 gains +1.65pp. N=4096→8192 gains 0pp — ceiling at 97.17% for D=16.

---

## Final Efficiency Config (step199)

```
N=2048          # hidden neurons
D=16            # hypersphere dimensionality
K_hh=2          # hidden-to-hidden edges (K_local=1, K_random=1)
K_iter=5        # routing iterations
K_in=25         # input fan-in per neuron
n_groups=256    # max(8, N//8)
norm_mode="l2"
encoding_mode="fourier"

alpha_reflect=0.5
alpha_turing=0.0
alpha_ahebb=1.0
mode="dynamic_z_geo"
resonance_threshold=0.0
beam_size=16
geo_gamma=0.5
K_phase=8

seed=42
epochs=150
batch_size=128
data=100% FashionMNIST
```

FLOPs formula: `3 × 2048 × 2 × 16 × 5 = 983,040 ≈ 0.98M`

---

## Reproducing the Result

**Train from scratch (150 epochs, ~8 min on MPS):**

```bash
cd /Volumes/T9/IndraAstra/dhiraj/neuro_graph
d_env/bin/python3 -u scripts/train_step199_n2048_d16_khh2_kiter5_tier2.py --device mps
# result → results/train_step199_n2048_d16_khh2_kiter5_tier2.json
```

**Evaluate existing result or retrain and report metrics:**

```bash
d_env/bin/python3 scripts/eval_efficiency_config.py
# prints accuracy, params, FLOPs, efficiency ratios
```

To retrain from scratch and evaluate in one step:

```bash
d_env/bin/python3 scripts/eval_efficiency_config.py --train --device mps
```

---

## Project Context

**Primary goal:** Find a deep learning architecture more parameter-efficient than transformers for use as an FFN replacement. SGNNET has a hard O(N×K) parameter budget — FLOPs are determined entirely by N, K_hh, D, K_iter.

**Testbed:** FashionMNIST (via VGG16 feature extractor → 25088-dim input). Accuracy target ≥95% at ≤1% of VGG16 FC FLOPs.

**Status:** Both efficiency criteria met (step199). Next phase: cross-dataset and cross-model generalizability testing.

**Full experiment history:** `learnings/EXPERIMENT_QUEUE.md` — 200+ steps documented with configs, results, and verdicts.
