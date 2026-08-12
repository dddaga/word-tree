# Phase 5 Part 15e: Dynamic Routing Post-Mortem + Next Directions

**Date:** 2026-04-09
**Parent:** LEARNINGS_phase5_p15_post_wave1.md (TOC)
**Context:** Full analysis of 9+ failed dynamic routing experiments. What failed, why, and what to try next.

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

**Note:** step73 D, step75 D, step82 A are WINS — redistribution/topology wins, not gates.

---

## Root Cause: Gate-Death Theorem

The core failure is multiplicative gate compounding across K_iter steps:

```
routing contribution at step k: r_h × g_h^k
```

For any gate g < 1 (sigmoid/softmax after normalization), raising to K_iter=8 gives g^8 → near-zero
for g < 0.9. Result:
- Gates that are "active" (g ≈ 1) carry all signal — degenerate to static routing
- Gates that are "selective" (g < 0.9) kill the signal — gate death

**Why static AH works:** AH modifies connection weights W_pos once per epoch (outer loop),
not within the forward pass. There is no multiplicative compounding — the weights are
just floats, modified by a Hebbian-like update rule.

**Why redistribution works:** If routing weights sum to 1 (Σw_i = 1), signal is
conserved regardless of gate values. step75 Config D gave +3.98pp at N=1024.
step73 D gave +1.78pp.

---

## Step83 Specific Failure Modes

Three simultaneous failures in group routing:

1. **Coarse S_g = mean(Z):** Group signal is an average of ~512 neuron states.
   Averaging destroys variance — all groups look similar → softmax collapses to uniform.
2. **Temporal mismatch:** AH operates at epoch timescale; router at batch timescale.
   AH suppresses weights as router tries to route through them — adversarial feedback.
3. **Co-adaptation:** W_pos learns to resist router perturbations, router learns to
   exploit W_pos pattern → degenerate fixed-point where routing is effectively static.

**Evidence:**
- C (β=0.1, minimal routing) = only −0.61pp. Degree of interference scales with β.
- A (every step, −6pp) >> B (final step only, −2.65pp): 8× more opportunities to corrupt AH signal.
- step82A (static group topology alone) = +3.01pp. The topology is valuable; dynamic routing on top is harmful.

---

## What Has NOT Been Tried

1. **Pure proximity architecture (step87 — scripted)**
   - NO conn_hh at all (eliminates the static graph entirely)
   - Connectivity computed purely from W_pos proximity at each step
   - Two decoupled state timescales: slow position (W_pos), fast activation (Z)
   - Fundamentally different from all prior attempts — no pre-built edges to gate

2. **Redistribution routing at N=4096 (step75 Config D, never scaled)**
   - Σw_i = 1 conserves signal — avoids gate death
   - +3.98pp at N=1024; scaling unknown
   - This is gap G4 in the priority queue

---

## Next Design Directions

### Proximity-Only (step87, scripted)
Build conn_hh on-the-fly from top-K nearest W_pos neighbors at each K_iter step.
No pre-computed graph; positions are the only inductive bias.
Question: does spatial self-organization emerge from the position dynamics alone?

Configs: N=1024, 50%/75ep, turing=0.0, AH=1.0, reflect=0.5
- Ref: static conn_hh (current baseline)
- A: rebuild conn_hh from W_pos KNN every R=10 batches
- B: rebuild R=10 + softmax routing
- C: rebuild R=5 + softmax routing
- D: rebuild R=25 + softmax routing
- E: FAISS rebuild R=10 + softmax + post-training radius pruning

### Redistribution at N=4096 (G4 in queue)
Scale step75 Config D (winner) to N=4096 with K_hh=4, K_iter=12, turing=0.0.
If redistribution + AH compound, this could be large.

### Four Next-Gen Dynamic Routing Candidates (from step83 post-mortem)

**Candidate 1 — Delayed routing activation**
- Freeze router weights for first N_freeze=30 epochs, let AH establish W_pos structure
- Then unfreeze for remaining 45 epochs
- Hypothesis: decouples the AH and router timescales
- Config: step83 Config B setup (final-step only) + delayed activation; ablate N_freeze={15,30,45}

**Candidate 2 — Stop-gradient on group state Z**
- `S_g = mean(Z[h]).detach()` — treats group states as fixed inputs to router
- Prevents router gradients from backpropagating into W_pos via the AH pathway
- Hypothesis: breaks co-adaptation feedback loop
- Implementation: one-line change in SGNNET_GroupRouting.forward()

**Candidate 3 — Entropy regularization on routing weights**
- Add L_entropy = -λ × H(w_group) to loss, where H = routing entropy
- Pushes softmax away from uniform during early training
- Hypothesis: bootstraps the router when flat-landscape is the problem
- λ schedule: anneal from 0.1 at ep1 to 0.0 at ep30

**Candidate 4 — Readout-layer routing only (cleanest test)**
- Dynamic routing ONLY at the readout layer (N→N_out), not during K_iter steps
- Routing during K_iter: AH-only (unchanged)
- Cannot interfere with AH since downstream of all K_iter steps
- Expected risk: near-zero; bounded gain ~0.5-1pp

---

## What Has Never Been Tried at N=4096

All redistribution routing experiments (step73, 75, 76, 82, 83) were at N=1024.
At N=4096, AH already achieves 95.87% (step71 Ref). Two open questions:
1. Does per-neuron redistribution routing (step75D config) add value at N=4096?
2. Does group topology (step82A) add value at N=4096?

Both are DISTILLED_GAPS G2 and G4 (high priority).

---

## The Long Game: What SGNNET Needs to Become

Primary goal: a general-purpose FFN replacement for transformers, parameter-efficient, O(N×K).

Key learnings about SGNNET's character:
- **It is a routing/selection machine, not a transformation machine.** Each forward pass routes
  activation through a fixed geometric graph.
- **AH is the core diversity mechanism.** Everything tried as an "addition" has hurt because AH
  already solves the diversity/sparsity problem optimally. Adding further gates re-solves a
  problem that's already solved, creating redundancy and signal loss.
- **The geometry (W_pos on S^{D-1}) is doing more work than we thought.** The input coverage
  fix (13% missing inputs) cost 6.29pp — meaning 13% of input space was responsible for a large
  chunk of representational capacity.
- **N-scaling is the right axis, but the curve needs re-establishing.** The reversal at N=10000
  on buggy code may not persist on patched code. With full input coverage, larger N should benefit more.

What SGNNET needs to prove for FFN replacement viability:
- Scaling behavior on patched arch up to N=8192+ without reversal
- Performance on harder tasks (CIFAR-10 with raw pixels, or a language modeling task)
- Parameter efficiency at matched accuracy: does SGNNET at N=4096 (529K params) match a transformer
  FFN of similar parameter count and task accuracy?

The softmax redistribution routing (step73/75 pattern) is the key test of whether SGNNET can become
input-adaptive — which is what gives transformers their expressive power via attention.

---

## Session Summary: 2026-04-08 New Launches

All 4 slots filled in priority order:

| Machine | Slot | Step | Rationale |
|---------|------|------|-----------|
| Mac Mini MPS | 1 | step73 | First conservative routing test |
| Mac Mini CPU | 1 | step76 | alpha_turing sweep + W_phase trained |
| Mac Studio MPS | 1 | step75 | Input-modulated temperature routing |
| Mac Studio CPU | 1 | step81 | Hebbian topology rewiring |
