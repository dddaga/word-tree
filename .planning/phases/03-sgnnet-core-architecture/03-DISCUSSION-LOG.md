# Phase 3: SGNNET Core Architecture - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-03-24
**Phase:** 03-sgnnet-core-architecture
**Areas discussed:** N_in strategy, Load balance integration, K-means init data source, C matrix design, Output neuron timing

---

## N_in Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Adapter 25088→48 (Option C) | 1.2M adapter params, N=570, C matrix tiny | |
| Raw N_in=25088, large N | No adapter, N≥25088, C matrix ~252MB | |
| Adapter 25088→512 | 12.8M params — exceeds 1% budget | |
| Spatial encoding D=4 | N_in=25088, A_input=[value, h, w, channel], no adapter | ✓ |

**User's choice:** User rejected all ROADMAP options. Proposed a novel spatial encoding: each input neuron carries a 4D activation [feature_val, h_norm, w_norm, channel_norm], encoding both what the VGG16 feature saw and its position in the 7×7×512 feature map. This sets D=4 globally.

**Notes:** PCA compression (which would reduce N_in) is Phase 5. Phase 3 works with full 25088 input neurons.

---

## Input Neuron W Positions

| Option | Description | Selected |
|--------|-------------|----------|
| Spatial coords [h, w, channel/511, 0] | W encodes spatial location | |
| Match A_input with mean feature | W initialized to mean activation | |
| Input neurons excluded from W geometry | W covers only hidden+output | ✓ |

**User's choice:** Input neurons don't participate in dynamic connectivity. W tensor covers only hidden+output neurons. Input neurons inject signal via C_input (static connections) only.

---

## Input Neuron Seeding vs. Recurrence

| Option | Description | Selected |
|--------|-------------|----------|
| Frozen across K steps | Input activations fixed after loading | |
| Seed-and-step-back | Input neurons seed hidden state in first pass only | ✓ |
| Updated by incoming C | Full recurrence including input neurons | |

**User's choice:** Input neurons seed hidden activations via C_input once before the K-step loop. After seeding, input neurons do not participate in further iterations.

---

## Output Neuron Readout

| Option | Description | Selected |
|--------|-------------|----------|
| Self-projection: dot(A_i, W_i)/||W_i|| | Report design, geometric alignment | ✓ |
| Activation norm: ||A_i|| | Pure sink magnitude | |
| Tiny learned readout: Linear(D→1) | 40 extra params | |

**User's choice:** Self-projection readout (report design). Output neuron positions W_out are learned.

---

## Output Neuron Timing

| Option | Description | Selected |
|--------|-------------|----------|
| Accumulate across all K iterations | Output neurons active throughout | |
| One-shot at final step K | Output neurons only activated at last iteration | ✓ |

**User's choice:** Output neurons are inactive for iterations 1..K-1. Only at step K do they receive signal via C_ho (static) and dynamic routing from hidden to W_out positions. This gives a clean "readout" of the hidden network's final state.

**Notes:** Dead output neurons (score=0) are handled by softmax safely (no NaN). Add ε to logits for extra safety. Phase 4 monitoring flag: if >10% of epochs show all outputs dead → architecture problem.

---

## Compute Budget (N=25610)

| Option | Description | Selected |
|--------|-------------|----------|
| Accept N=25610 | Phase 3 is architecture-only, no training | ✓ |
| Reduce N_hidden | Not meaningful since input neurons dominate | |
| torch.sparse C matrix | Saves memory, slower on MPS | |

**User's choice:** Accept full N for Phase 3 architecture. N_hidden swept in Phase 4 (find minimum N_hidden showing convergence). Dense C matrices.

---

## C Matrix Design

| Option | Description | Selected |
|--------|-------------|----------|
| Single unified C [N, N_hidden+N_out] | Report design | |
| Split C_input + C_internal | Two matrices | |
| Split C_input + C_hh + C_ho | Three matrices, output-timing-aware | ✓ |

**User's choice:** Three separate C matrices:
- C_input: [N_in, N_hidden] — seeding
- C_hh: [N_hidden, N_hidden] — hidden-hidden iterations 1..K-1
- C_ho: [N_hidden, N_out] — hidden-to-output injection at step K only

**User's rationale:** Simpler conceptually. C_input handles the injection, C_hh handles internal dynamics, C_ho is the readout pathway.

---

## Output Neurons as Sources in C_internal

| Option | Description | Selected |
|--------|-------------|----------|
| Pure sinks (only receive) | C_hh rows = hidden only | ✓ |
| Full participants (send and receive) | C_internal includes output as sources | |

**User's choice:** Output neurons are pure sinks — they don't send in C_hh or dynamic connectivity.

---

## Load Balance Integration

| Option | Description | Selected |
|--------|-------------|----------|
| Return (contribution, gate) tuple | Gate exposed to caller | ✓ |
| Module attribute/buffer | Stateful side effect | |
| Recompute from gate matrix | Redundant cdist | |

**User's choice:** dynamic_connectivity returns (activation_contribution, gate). Gate matrix enables selection_count computation in the training loop.

---

## Dead Neuron Prevention

| Option | Description | Selected |
|--------|-------------|----------|
| Soft gate (Gaussian only, no hard gate) | Fully differentiable | |
| Hard gate + load balance loss | Report design | ✓ |
| Hard gate + expanding radius warmup | Scheduled warmup | |

**User's choice:** Hard gate + load balance loss. Simpler, matches report.

---

## K-means Initialization

| Option | Description | Selected |
|--------|-------------|----------|
| Run K-means on seeded hidden activations | Post-seeding representations | |
| PCA project 25088→4, then K-means | Separate PCA step | |
| Skip K-means, use random uniform init | Simple, Phase 3 only needs forward/backward | ✓ |

**User's choice:** Random uniform init for Phase 3. K-means doesn't apply cleanly to this architecture since hidden neurons don't share D-space with the full input representation.

---

## Claude's Discretion

- C_input initialization: exact ≥1-connection-per-input strategy
- register_buffer pattern for C masks
- LayerNorm details (affine=True)
- ε value for logit safety net
- Test data shapes

---

## Deferred Ideas

- Soft gate / probabilistic routing — v2 feature
- Expanding radius warmup — Phase 4 experiment if needed
- K-means init — Phase 4 optional optimization
- Adaptive K — v2 feature
