# Paper 1 — Expanded Scope: Multimodal + Scaling Experiments

**Decided:** 2026-04-21
**Status:** Planning phase — experiments being designed.

Paper 1 is NON-NEGOTIABLE on the following:

---

## Paper 1 Scope (Final)

### What Paper 1 Claims

1. Novel architecture: SGNNET as FC-replacement (sparse O(N·K) graph, Fourier encoding)
2. Efficiency: params ✅ 34,976 (0.029% of VGG_FC), FLOPs ✅ 0.20M (0.16%), wall-time ✅ 5.26×
3. Accuracy: ✅ 95.95% on Imagenette (step605 KD student)
4. Generalizability across modalities: vision, audio, time series
5. Honest negative: text gap documented (SST-2/AG News: SGNNET trails Linear)
6. Scaling law: train unconstrained → calibrate to efficient knee → show compression path

### What Paper 1 Does NOT Cover

- Dynamic routing (Paper 2)
- Hypercomplex representations (Paper 2)
- HNSW inference acceleration (Paper 2)
- Replacing attention layers in transformers (Paper 2)

---

## Modality Coverage Plan

### Vision (STATUS: DONE)

Primary: Imagenette → 95.95% step605, 96.38% step887 ΔW-proj T2
Supporting: CIFAR-10 (steps 882, 909, 914 done), CIFAR-100 (steps in progress)
Gap: Write up K_in crossover rule (K_in=25 at N≤2048, K_in=15 at N≥4096)

### Audio (STATUS: GAP — structural issue)

Current result: ESC-50 −11.5pp structural gap (steps 926/928) — CONFIRMED negative
Root cause hypothesis: feature mismatch (VGG audio features ≠ mel-spectrogram 2D structure)
Plan:
  step960: Audio feature alignment experiment
    - Use pre-computed mel-spectrogram features from audio CNN backbone (e.g. VGGish or PANNs)
    - Match feature dimension to N_IN=25088 or rearchitect for audio N_IN
    - Compare: raw VGG audio features vs properly aligned audio features
    - Goal: close the structural gap or confirm it's architectural (not feature-choice)

If gap closes ≥5pp: include audio as positive result
If gap persists: include as honest modality limitation with analysis

### Time Series (STATUS: IN PROGRESS)

ts/ pipeline defined but all PENDING (ts_step001 through ts_step032)
Financial forecasting: 10y OHLCV Nifty50 + small caps
Baseline: MA windows, CNN+LSTM, CNN+Transformer
SGNNET-TS: ts_step030 (T0) → ts_step031 (alpha scan) → ts_step032 (T1)
Paper claim: SGNNET handles temporal relational structure at low parameter cost

### Text (STATUS: CONFIRMED NEGATIVE — include as Limitations)

SST-2: SGNNET trails Linear baseline
AG News: same
Reason: text lacks continuous geometric embedding structure that SGNNET exploits
This is an honest negative and strengthens the paper's credibility.

---

## Scaling Law / Train-Large-Calibrate-Small

### Motivation

Paper needs to show: this architecture scales, AND can be compressed back to an efficient point.
The "unconstrained training" result establishes the ceiling.
The "calibration" result shows the compression is principled, not just lucky.

### Experiment Design

Phase A — Scaling ceiling (step961):
  N ∈ {2048, 4096, 8192}, D ∈ {16, 32}, K_in ∈ {25, 60}
  T2 (150ep/100%) to establish accuracy ceiling at each scale
  Metrics: accuracy, params, FLOPs, wall-time
  Goal: confirm scaling law holds (accuracy ↑ monotone with N·D)

Phase B — Efficient knee (step962):
  Fix target accuracy = ceiling − 0.5pp (within margin of ceiling)
  Find minimum (N, D, K_in) that meets target
  Show: N=2048, D=16, K_in=60 is within 0.5pp of N=8192, D=32 ceiling
  Metrics: accuracy + efficiency ratio vs ceiling
  Goal: establish the "efficiency is sufficient" claim

Phase C — Calibration transfer (step963):
  Train N=8192, D=32 → extract optimal K_in/K_hh/K_iter settings
  Apply same hyperparameter ratios at N=2048, D=16 (scaled by N)
  Show: hyperparameter scaling law is predictable
  Goal: paper can claim "to deploy at scale X, multiply K_in by X/2048"

### Why This Matters

Without a scaling law section, reviewers will ask "why N=2048? Why D=16?"
This section provides the evidence-based answer: "because N=2048, D=16 is at the Pareto knee."

---

## Modality-Specific Training Observations

### What to Document per Modality

For each modality, report:
1. Feature extraction pipeline (what feeds N_IN)
2. Optimal K_in (coverage → feature discrimination hypothesis)
3. Optimal N (enough capacity for feature space)
4. Convergence behavior (epochs to plateau)
5. Whether ΔW-proj helps (confirmed on vision, unknown on audio/TS)

### Hypothesis: K_in Scales with Feature Dimensionality

Coverage formula: coverage = N * K_in / N_IN
Vision Imagenette: N_IN=25088, K_in=25, N=2048 → coverage=204% → 95.95%
Vision optimal: K_in=60 → coverage=490% → +1.15pp (step951)
Audio: N_IN differs → K_in needs recalibration
Time series: N_IN differs → K_in needs recalibration
Rule: target coverage ≥ 300% as starting point for each modality

---

## Paper 1 Experiment Priority Queue

| Priority | Step | Modality | Status | Blocking what |
|----------|------|----------|--------|---------------|
| P1 | step960 | Audio feature alignment | DESIGN | Audio positive/negative verdict |
| P1 | Write up: 12 audit gaps | Vision | WRITING | Submission |
| P2 | step961 | Vision scaling ceiling | DESIGN | Scaling section |
| P2 | ts_step030-032 | Time series | PENDING | TS modality result |
| P3 | step962 | Vision efficient knee | DESIGN (after 961) | Calibration section |
| P3 | step963 | Cross-modal calibration | DESIGN (after 962) | Calibration transfer |

---

## Updated Paper Structure

1. Abstract — efficiency champion (step605), multimodal scope, honest negatives
2. Introduction — FC replacement problem, SGNNET pitch, paper contributions list
3. Architecture — SmallWorld + Resonant + ΔW-proj (currently ABSENT from draft — HIGH GAP)
4. Vision experiments — Imagenette + CIFAR-10/100 + ablations
5. Audio experiments — ESC-50 (positive or honest negative with analysis)
6. Time series experiments — financial forecasting, SGNNET-TS
7. Scaling section — scaling law + efficient knee + calibration transfer
8. Baselines — FC, MLP, GNN at matched params
9. Limitations — text gap, audio gap (if persists), scaling compute cost
10. Conclusion
