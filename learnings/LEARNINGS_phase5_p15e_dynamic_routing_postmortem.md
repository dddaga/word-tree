# Phase 5 Part 15e: Dynamic Routing Post-Mortem + Next Directions

**Date:** 2026-04-09
**Parent:** LEARNINGS_phase5_p15_post_wave1.md (TOC)
**Context:** Full analysis of 9+ failed dynamic routing experiments. What failed, why, what next.

---

## Complete Failure Record

| Step | Mechanism | Result | Failure mode |
|------|-----------|--------|--------------|
| step58 | Resonance-gated phase excitatory, per-batch K-NN | ~55% | Gate death |
| step59 | Beam unification (structural + phase channels) | ~55% | Gate death |
| step60 | Phase-distance routing (13 configs, all variants) | 10–19% | Gate death |
| step63 | Activation-gated routing (AGR, soft-attn over candidates) | 55% best | Gate death + co-adapt |
| step65 | Distance-phase exp(−γd) on conn_hh, no AH | 48–49% | Gate death |
| step66 | Phase-target (W_pos=Key/Value, phase=Query, local plasticity) | −15 to −43pp | Co-adaptation adversarial |
| step73 | Softmax routing Config A | collapsed early | Gate death |
| step76 | W_phase sweep (higher turing values) | below Ref | — |
| step83 | Group routing (S_g=mean(Z), softmax over groups) | −2.65 to −6pp | 3 modes (see below) |

**Note:** step73 D, step75 D, step82 A = WINS — redistribution/topology wins, not gates.

---

## Root Cause: Gate-Death Theorem

Core failure: multiplicative gate compounding across K_iter steps:

```
routing contribution at step k: r_h × g_h^k
```

Any gate g < 1 (sigmoid/softmax after normalization), raised to K_iter=8 → g^8 near-zero for g < 0.9. Result:
- "Active" gates (g ≈ 1) carry all signal — degenerates to static routing
- "Selective" gates (g < 0.9) kill signal — gate death

**Why static AH works:** AH modifies W_pos once per epoch (outer loop), not within forward pass. No multiplicative compounding — weights just floats, modified by Hebbian-like update.

**Why redistribution works:** Routing weights sum to 1 (Σw_i = 1) → signal conserved regardless of gate values. step75 Config D gave +3.98pp at N=1024. step73 D gave +1.78pp.

---

## Step83 Specific Failure Modes

Three simultaneous failures in group routing:

1. **Coarse S_g = mean(Z):** Group signal averages ~512 neuron states. Averaging destroys variance → all groups look similar → softmax collapses to uniform.
2. **Temporal mismatch:** AH at epoch timescale; router at batch timescale. AH suppresses weights as router routes through them — adversarial feedback.
3. **Co-adaptation:** W_pos resists router perturbations, router exploits W_pos pattern → degenerate fixed-point, routing effectively static.

**Evidence:**
- C (β=0.1, minimal routing) = only −0.61pp. Interference scales with β.
- A (every step, −6pp) >> B (final step only, −2.65pp): 8× more corruption opportunities.
- step82A (static group topology alone) = +3.01pp. Topology valuable; dynamic routing on top harmful.

---

## What Has NOT Been Tried

1. **Pure proximity architecture (step87 — scripted)**
   - NO conn_hh (eliminates static graph entirely)
   - Connectivity from W_pos proximity at each step
   - Two decoupled state timescales: slow position (W_pos), fast activation (Z)
   - Fundamentally different — no pre-built edges to gate

2. **Redistribution routing at N=4096 (step75 Config D, never scaled)**
   - Σw_i = 1 conserves signal — avoids gate death
   - +3.98pp at N=1024; scaling unknown
   - Gap G4 in priority queue

---

## Next Design Directions

### Proximity-Only (step87, scripted)
Build conn_hh on-the-fly from top-K nearest W_pos neighbors each K_iter step. No pre-computed graph; positions = only inductive bias.
Question: does spatial self-organization emerge from position dynamics alone?

Configs: N=1024, 50%/75ep, turing=0.0, AH=1.0, reflect=0.5
- Ref: static conn_hh (current baseline)
- A: rebuild conn_hh from W_pos KNN every R=10 batches
- B: rebuild R=10 + softmax routing
- C: rebuild R=5 + softmax routing
- D: rebuild R=25 + softmax routing
- E: FAISS rebuild R=10 + softmax + post-training radius pruning

### Redistribution at N=4096 (G4 in queue)
Scale step75 Config D (winner) to N=4096 with K_hh=4, K_iter=12, turing=0.0. If redistribution + AH compound → could be large.

### Four Next-Gen Dynamic Routing Candidates (from step83 post-mortem)

**Candidate 1 — Delayed routing activation**
- Freeze router weights first N_freeze=30 epochs, let AH establish W_pos structure
- Unfreeze for remaining 45 epochs
- Hypothesis: decouples AH and router timescales
- Config: step83 Config B setup (final-step only) + delayed activation; ablate N_freeze={15,30,45}

**Candidate 2 — Stop-gradient on group state Z**
- `S_g = mean(Z[h]).detach()` — treats group states as fixed router inputs
- Prevents router gradients backpropagating into W_pos via AH pathway
- Hypothesis: breaks co-adaptation feedback loop
- Implementation: one-line change in SGNNET_GroupRouting.forward()

**Candidate 3 — Entropy regularization on routing weights**
- Add L_entropy = -λ × H(w_group) to loss, H = routing entropy
- Pushes softmax away from uniform during early training
- Hypothesis: bootstraps router when flat-landscape is problem
- λ schedule: anneal from 0.1 at ep1 to 0.0 at ep30

**Candidate 4 — Readout-layer routing only (cleanest test)**
- Dynamic routing ONLY at readout layer (N→N_out), not during K_iter
- K_iter routing: AH-only (unchanged)
- Cannot interfere with AH — downstream of all K_iter steps
- Expected risk: near-zero; bounded gain ~0.5-1pp

---

## What Has Never Been Tried at N=4096

All redistribution routing experiments (step73, 75, 76, 82, 83) at N=1024. At N=4096, AH already achieves 95.87% (step71 Ref). Two open questions:
1. Does per-neuron redistribution routing (step75D config) add value at N=4096?
2. Does group topology (step82A) add value at N=4096?

Both = DISTILLED_GAPS G2 and G4 (high priority).

---

## The Long Game: What SGNNET Needs to Become

Primary goal: general-purpose FFN replacement for transformers, parameter-efficient, O(N×K).

Key learnings about SGNNET character:
- **Routing/selection machine, not transformation machine.** Each forward pass routes activation through fixed geometric graph.
- **AH = core diversity mechanism.** Every "addition" hurt because AH already solves diversity/sparsity optimally. Further gates re-solve solved problem → redundancy + signal loss.
- **Geometry (W_pos on S^{D-1}) does more than expected.** Input coverage fix (13% missing inputs) cost 6.29pp — 13% of input space responsible for large chunk of representational capacity.
- **N-scaling right axis, but curve needs re-establishing.** N=10000 reversal on buggy code may not persist on patched code. Full input coverage → larger N should benefit more.

What SGNNET must prove for FFN replacement viability:
- Scaling on patched arch up to N=8192+ without reversal
- Performance on harder tasks (CIFAR-10 raw pixels, or language modeling)
- Parameter efficiency at matched accuracy: SGNNET at N=4096 (529K params) vs transformer FFN of similar param count and task accuracy?

Softmax redistribution routing (step73/75 pattern) = key test of input-adaptiveness — what gives transformers expressive power via attention.

---

## Session Summary: 2026-04-08 New Launches

All 4 slots filled in priority order:

| Machine | Slot | Step | Rationale |
|---------|------|------|-----------|
| Mac Mini MPS | 1 | step73 | First conservative routing test |
| Mac Mini CPU | 1 | step76 | alpha_turing sweep + W_phase trained |
| Mac Studio MPS | 1 | step75 | Input-modulated temperature routing |
| Mac Studio CPU | 1 | step81 | Hebbian topology rewiring |