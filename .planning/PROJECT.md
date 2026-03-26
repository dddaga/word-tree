# neuro_graph — SGNNET VGG16 Distillation Experiment

## What This Is

A research project to validate **SGNNET (Sparse Geometric Neural Network)** as a parameter-efficient replacement for the 3 fully-connected (FC) layers of VGG16. The experiment uses knowledge distillation on the Imagenette dataset: the VGG16 CNN backbone extracts 25088-dim feature vectors, and SGNNET learns to replicate the soft probability distribution over 10 classes that the original FC head produces — at ~1% of the parameter count.

## Core Value

Demonstrate that SGNNET can match VGG16 FC accuracy at ≤1% of its parameters, and quantify how PCA-based input compression further reduces compute without sacrificing quality.

## Architecture: SGNNET

SGNNET replaces dense layers with a geometric graph network:

- **N neurons**, each with a D-dimensional weight position W_i (the "crystal lattice") and D-dimensional activation A_i
- **Three neuron types**: input (fixed positions), hidden (learned positions), output (learned positions)
- **Static sparse connectivity C**: [N, N] sparse matrix, pattern fixed at init, values learned
- **Dynamic connectivity**: at each iteration, neurons within proximity radius r* = (box_size/2) / N^(1/D) exchange activations via Gaussian-weighted routing
- **K recursive iterations**: A = normalize(A @ C + dynamic_connectivity(A, W))
- **Self-projection readout**: score_i = dot(A_i, W_i) / ||W_i||
- **Loss**: task (KL divergence for distillation) + safety valve (Coulomb repulsion) + load balance

Key numbers for VGG16 experiment:
- Dense baseline: FC layers have ~123.6M params (25088→4096→4096→1000)
- SGNNET target: ≤1.24M trainable params (1%), 90% sparsity in C

## Requirements

### Validated

- [x] Download Imagenette and extract VGG16 pre-FC features (25088-dim) + soft labels (10-class) into a tensor store — Validated in Phase 1: Data Pipeline
- [x] Evaluate frozen pretrained VGG16 on Imagenette val — record accuracy as the benchmark — Validated in Phase 2: Dense Baseline Benchmark (99.54% top-1, 99.97% mAP)

### Validated

- [x] Implement SGNNET core: W positions, C sparse matrix, dynamic connectivity, K-iterations, self-projection — Validated in Phase 3: SGNNET Core Architecture (649,330 params = 0.53% of VGG16 FC, 51 tests passing)

### Validated

- [x] Train SGNNET wave architecture through three stages (static baseline + two dynamic routing variants) — Validated in Phase 4: Wave Architecture Experiments (Stage A: 10.27% top-1, Stage B: 11.97%, Stage C: 10.52%; binary C mask ceiling identified at ~3.2 KL plateau; wave_comparison.md produced)

### Active
- [ ] Apply PCA to compress 25088-dim input, sweep compression ratios, retrain SGNNET
- [ ] Produce a comparison table: params, FLOPs, accuracy for all three variants (dense / SGNNET full / SGNNET+PCA)

### Out of Scope

- Dynamic topology (Hebbian prune-and-grow from Section 9 of the report) — v2 feature
- Gumbel-softmax differentiable routing — v2 feature
- Full end-to-end VGG16 fine-tuning — distillation only, backbone frozen
- BERT/transformer FFN application — VGG16 FC experiment is v1

## Context

- Dataset: Imagenette (subset of ImageNet, 10 classes, ~9.5k train / ~3.9k val, ~13.4k total)
- VGG16 FC layers: 25088→4096→4096→1000, ~123.6M params total
- SGNNET report: `sparse_geometric_network_report.md` in project root — full architecture spec
- Hardware: Apple Silicon (MPS backend)
- Framework: PyTorch
- Key challenge: N_in=25088 makes the N² connectivity matrix expensive; Phase 3 PCA addresses this
- Tensor store: HDF5 for efficient random access during training

## Constraints

- **Timeline**: 3 days (March 23–25, 2026) — all 4 experimental phases must close
- **Parameters**: SGNNET must use ≤1% of VGG16 FC params (~1.24M active weights)
- **Sparsity**: C matrix must maintain ≥90% sparsity throughout training
- **File size**: Every code file must stay under 250 lines; split into modules as needed
- **Code style**: Top-down — high-level structure first, fill in low-level details progressively
- **Stuck rule**: If blocked on any task for 2+ consecutive attempts, do a web search before attempt 3

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Imagenette over full ImageNet | 13.4k images fits on disk; 10 classes cleanly map to VGG1000 outputs | — Pending |
| Distillation not scratch training | Clean supervised signal in hours on single GPU; direct comparison baseline | — Pending |
| HDF5 tensor store | Efficient random access for 13.4k feature vectors; survives process restarts | — Pending |
| Soft labels = softmax of 10 Imagenette logits from VGG16 | Avoids modifying VGG16 head; directly measures distillation fidelity | — Pending |
| N_in=25088 direct (no adapter) | C_input [25088×256] accepted; 649k total params stays within 1% budget; adapter deferred | Validated in Phase 3 |

---
*Last updated: 2026-03-26 after Phase 4 completion*

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state
