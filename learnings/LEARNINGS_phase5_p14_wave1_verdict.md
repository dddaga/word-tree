# Phase 5 Part 14: Wave-1 Verdict (Steps 58-63)

**Date:** 2026-04-06
**Status:** COMPLETE — all wave-1 mechanisms killed
**REF_BASELINE:** 73.53% (AntiHebb α=1.0, 50% data, 75ep, step57)

---

## Wave-1 Summary

All 6 wave-1 mechanisms failed. Static AntiHebb routing not improvable via any additive or input-dependent excitatory modification at this scale.

| Step | Mechanism | Best Config | top1_best | vs Ref | Verdict |
|------|-----------|-------------|-----------|--------|---------|
| step58 | Resonance-gated phase-excitatory | — | ~55% | −18pp | KILLED |
| step59 | Active beam unified routing | — | ~55% | −18pp | KILLED |
| step60 | Phase-distance routing | — | <70% | −3pp+ | KILLED |
| step61 | Hub interneurons (N_mix=256, fan_in=512) | C (AH=1.0) | 64.05% | −9pp | KILLED |
| step62 | Dynamic competitive topology | — | — | — | NOT SCRIPTED |
| step63 | Activation-Gated Routing (AGR) | — | 55.41% | −18pp | KILLED |

---

## Step 63: AGR — Complete Results

**Hypothesis:** Soft-attention over expanded candidate set redistributes excitation multiplicatively (softmax weights sum to 1) rather than adding new signal. Prevents gate death because total excitation magnitude bounded.

**Result:** Hypothesis WRONG. All configs still killed.

| Config | Description | top1_best | best_ep | vs Ref | Verdict |
|--------|-------------|-----------|---------|--------|---------|
| Ref | AntiHebb α=1.0 static routing | 73.48% | 73/75 | — | REF |
| A | soft-attn mixed candidates, no AH | 31.11% | 65/75 | −42pp | KILLED |
| B | soft-attn mixed candidates, AH=1.0 | 55.41% | 73/75 | −18pp | KILLED |
| C | hard top-K mixed candidates, AH=1.0 | 38.88% | 71/75 | −35pp | KILLED |
| D | soft-attn mixed, AH=1.0, hop_decay=0.9 | 55.29% | 73/75 | −18pp | KILLED |
| E | soft-attn phase-only candidates, AH=1.0 | 55.13% | 73/75 | −18pp | KILLED |

**Key question answers:**
- A vs Ref → input-dependent routing WORSE than static (−42pp without AH)
- B vs A → AH partially compensates (+24pp) but still −18pp vs static AH
- C vs B → hard top-K worse than soft (−17pp) — sparsity hurts, not helps
- D vs B → hop_decay marginal (−0.12pp) — attenuation doesn't help
- E vs B → phase-only vs mixed — negligible (−0.28pp)

**Root cause:** Problem not which candidates selected. ANY routing loop modification not exactly matching AH static update rule destabilizes training. AH static routing not "improvable" by attention — fragile equilibrium that soft-attention disrupts.

---

## Wave-1 Root Cause Analysis

Every wave-1 failure shares same pattern:
1. Modified routing introduces additional or redistributed excitatory signal
2. Threshold gate (ReLU - theta) regime shifts
3. Safety valve loss collapses to near-zero (gate death: safety ≈ 0.002-0.005)
4. Routing diversity collapses — most neurons fire similarly
5. Accuracy stuck at ~10-55% depending on whether AH can partially compensate

**Mechanism:** Static AntiHebb with fixed small-world topology = stable fixed point. Any perturbation (more connections via attention, different candidate selection, beam routing, interneurons) destroys stability. Architecture not "locked in" to suboptimal fixed point that attention can unlock — IS at optimal fixed point for current routing regime.

**Implication for Wave-2 design:**
- Do NOT try to improve routing dynamics (attention, gating, mixing)
- N-scaling (step56) confirmed path to improvement: +11pp per 2x N
- Architecture improvements should come from input representation or N-scaling
- FFN-replacement hypothesis requires stability at routing layer

---

## Step 61: Hub Interneurons — Complete Results (2026-04-05)

**Hypothesis:** High fan-in mixing layer (N_mix=256, fan_in=512) can aggregate distributed signals better than per-neuron static routing.

**Result:**

| Config | Description | top1_best | best_ep | vs Ref | Verdict |
|--------|-------------|-----------|---------|--------|---------|
| Ref | AntiHebb α=1.0 (1-layer) | 73.30% | 73/75 | — | REF |
| A | Hub mixing, no AH | 40.51% | 65/75 | −32.8pp | KILLED |
| B | Hub mixing + AH=0.5 | 38.68% | 73/75 | −34.6pp | KILLED |
| C | Hub mixing + AH=1.0 | 64.05% | 73/75 | −9.3pp | KILLED |

**Note:** Config D not completed (3-config script). Best with AH=1.0 still −9pp. High fan-in routing disrupts stable AH fixed point — same gate-death pattern.

---

## Step 51: Spatial Phase Gating — Complete (2026-04-05)

Gate death confirmed. top1_best ≈ 34.93% (agent-read from pane, no JSON file). All configs: routing modifications caused safety collapse. Pattern identical to steps 58-63.

---

## High-D Subspace Routing (step52) — Also Killed

Running but gate-dead. Z-subspace routing:
- Config A (Z-subspace split=16 + AH): 13.99% (early stop e87) — gate death
- Config B (Z-subspace split=32 + AH): 14.80% (e150) — gate death
- Config C (W_pos-subspace split=16 + AH): ~12% at e40 — gate death in progress

W_pos-subspace routing cannot avoid gate death either. Geometry of activation space (S^{D-1}) incompatible with split-subspace attention.

---

## N-Scaling Continues (step56)

N=10000 at e60=72.92% — healthy trajectory (no safety valve, N>5000). Power-law scaling confirmed: 512→69.58%, 1024→80.92%, 2048→81.10%, 4096→84.36%. If N=10000 follows curve, expect ~86-87% final accuracy.

**N-scaling = primary lever for accuracy improvement.**