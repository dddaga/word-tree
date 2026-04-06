# Design Discussions — 2026-04-08

**Parent:** LEARNINGS_design.md (index)

---

## Dynamic Routing: Post-Mortem + Next Generation Design

### Summary of All Attempts

| Generation | Steps | Mechanism | N | Best result |
|------------|-------|-----------|---|-------------|
| Wave-1 | 58, 59, 61, 64, 66 | Multiplicative gates (g∈[0,1]) | 1024 | Killed (−5 to −13pp) |
| Redistribution | 73, 75, 76 | Softmax routing (Σw=1) | 1024 | +1.78pp (step73D), +3.98pp (step75D) |
| Group routing | 82, 83 | Group topology + inter-group softmax | 1024 | step82A=+3.01pp, step83A=−6.01pp |

### Step83 Group Routing Results (Mac Studio)

| Config | Description | top1_best | Δ vs Ref |
|--------|-------------|-----------|----------|
| Ref | Static AH, n_groups=16 | 84.61% | — |
| A | β=0.5, every routing step | 78.60% | −6.01pp |
| B | β=0.5, final step only | 81.96% | −2.65pp |
| C | β=0.1, every step | 84.00% | −0.61pp |

### Failure Mode Analysis: Why Step83 Died Despite Using Softmax

**Hypothesis 1 — Coarse group state (S_g = mean(Z[h])) is too noisy:**
At training start, W_pos is random → all groups produce similar mean vectors → softmax collapses
to uniform. Router gets no gradient signal early in training.

**Hypothesis 2 — Temporal mismatch between AH and router:**
AH: epoch-timescale (cumulative weight updates). Dynamic routing: batch-timescale (per-forward-pass).
These conflict: router adapts to instantaneous batch statistics while AH slowly shapes W_pos manifold.

**Hypothesis 3 — Softmax collapse at group granularity:**
step73/75 used per-neuron redistribution at K_hh=6 level (routing among 6 neighbors).
step83 used per-group routing at G=16 level. Group mean destroys within-group activation diversity.

**Evidence from results:**
- C (β=0.1) = only −0.61pp — degree of interference scales with β (supports co-adaptation)
- A (every step, −6pp) >> B (final step only, −2.65pp) — more opportunities to corrupt AH signal

### Four New Experiment Candidates

**Candidate 1 — Delayed routing activation**
- Freeze router for first N_freeze=30 epochs, let AH establish W_pos structure
- Then unfreeze and train router for remaining 45 epochs
- Config: step83 Config B setup (final-step only) + delayed activation

**Candidate 2 — Stop-gradient on group state Z**
- `S_g = mean(Z[h]).detach()` — treats group states as fixed inputs to router
- Prevents router gradients from backpropagating into W_pos via the AH pathway
- One-line change in SGNNET_GroupRouting.forward()

**Candidate 3 — Entropy regularization on routing weights**
- Add L_entropy = -λ × H(w_group) to loss
- Pushes softmax away from uniform during early training
- λ schedule: anneal from 0.1 at ep1 to 0.0 at ep30

**Candidate 4 — Readout-layer routing only (cleanest test)**
- Dynamic routing ONLY at readout layer (N→N_out), not during K_iter steps
- Cannot interfere with AH since downstream of all K_iter steps

### What Has NOT Been Tried at N=4096

All redistribution routing experiments (step73, 75, 76, 82, 83) were at N=1024.
At N=4096, AH achieves 95.87%. Open questions:
1. Does per-neuron redistribution routing (step75D) add value at N=4096? (gap G4)
2. Does group topology (step82A) add value at N=4096? (gap G2)

---

## Continuous-Position Dynamic Connectivity (Step 87 Design)

### Motivation

Static conn_hh is built once at init from random W_pos. If W_pos learns during training (it does —
AH gradient shapes it), the optimal connectivity changes but conn_hh never follows. Rebuilding
conn_hh from trained W_pos periodically could improve accuracy and enable FLOPs reduction.

### Core Idea

- **No static conn_hh** — connections form from W_pos proximity at each rebuild interval
- **Two timescales**: activation strengths update every batch, positions (conn_hh) update every R batches
- **Softmax routing** (redistribution, not gates) over dynamic neighbours
- **FAISS** for cheap K-NN rebuild: brute force = 1B ops at N=4096; FAISS HNSW ≈ 3M ops (300× cheaper)
- Positions are W_pos (already learnable); no new parameters added

### Mapping to Prior Experiments

| Idea | Prior evidence |
|------|----------------|
| W_pos as positions | Already trains via AH gradient; conn_hh just never follows |
| Periodic conn_hh rebuild | step81 Hebbian rewiring (+1.12pp) — but heuristic, not gradient-driven |
| Dynamic K-NN per step | step50 KILLED: too frequent (every step). Key: rebuild every R batches, not every step |
| Softmax routing | step75D: +3.98pp at N=1024. Never combined with dynamic connectivity |
| Decoupled timescales | Directly addresses step83 failure (temporal mismatch) |

### Critical Design Decisions

1. **Rebuild mechanism**: top-K cosine KNN on W_pos (K=6, same as current K_hh). Fixed K.
2. **Differentiability**: conn_hh rebuild is detached (not differentiable). W_pos learns from AH + routing gradients.
3. **FLOPs reduction path**: Phase 1: train with K=6. Phase 2: prune neighbours below cosine threshold at inference.

### Experiment Configs (Step 87)

N=1024, 50%/75ep, turing=0.0, AH=1.0, reflect=0.5

| Config | Description | Key variable |
|--------|-------------|--------------|
| Ref | Static conn_hh (current baseline) | — |
| A | Rebuild conn_hh from W_pos KNN every R=10 batches | topology plasticity alone |
| B | Rebuild R=10 + softmax routing | topology + redistribution |
| C | Rebuild R=5 + softmax routing | faster adaptation |
| D | Rebuild R=25 + softmax routing | slower adaptation |
| E | FAISS rebuild R=10 + softmax + post-training radius pruning | FLOPs reduction test |

**Key comparisons**:
- A vs Ref: does topology plasticity alone help?
- B vs step81 A (85.55%): gradient-driven vs Hebbian rewiring
- B vs step82 A (85.63%): dynamic topology vs static group topology

### Implementation Notes

- Add `_rebuild_counter` to SGNNET_SmallWorld
- In `forward()`: increment counter, if counter % R == 0 → recompute conn_hh from W_pos cosine KNN (detached)
- FAISS: `index = faiss.IndexFlatIP(D); index.add(W_pos_normed); _, conn_hh = index.search(W_pos_normed, K+1)`
- **Script**: train_step87_proximity_routing.py (EXISTS — see scripts/)
