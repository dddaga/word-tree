# Dynamic Routing: Failure Analysis & Next Experiments

**Status:** Living document. Last updated 2026-04-19. **Dynamic routing direction CLOSED (all mechanisms, 2026-04-19).**
**Cross-links:** [[phase_routing_part2]], [[soft_routing_hnsw]], [[sparse_bfs_routing]], [[softmax_routing]]

## Why Dynamic Routing Keeps Failing (Taxonomy)

### Failure Mode 1: Gate-Death (structural)

**Steps:** 58–66 (wave-1), many others with multiplicative gates.

Any gate g ∈ [0,1] applied to Z propagation decays as g^K over K_iter steps.
At K_iter=5 with g=0.8: 0.8^5 = 0.33×. Network learns g→0 → mechanism becomes decorative.

**Confirmed by:** steps 58–66 crash convergence. SGNNET_Resonant `dynamic_gate` mode.

**Cure:** additive redistribution (softmax weights summing to 1), never multiplicative gates.

---

### Failure Mode 2: Gradient Disconnection from Top-k Selection

**Steps:** step855 Sparse BFS T0 (all configs −2.0 to −2.3pp), step873 BFS M=32 T1 (−24.41pp catastrophic).

`torch.topk` selection is non-differentiable. Gradient only flows through selected values via straight-through estimator. At T1 (75ep), the unstable selection gradients compound: BFS selects different subsets each step, network can't learn which nodes should be consistently active.

**Why T0 looks bad and T1 is catastrophic:** T0 (20ep) hasn't learned enough for the instability to fully manifest. T1 runs long enough for the gradient disconnect to poison the loss landscape.

**Cure:** soft selection (Gumbel-softmax with annealing τ → 0), or NO selection (dense softmax routing with all K_hh neighbors).

---

### Failure Mode 7: Softmax Routing Fixed-Point Collapse at K_hh=2

**Steps:** step894 (B/C/D: −18pp), step895 (A/B/C: −76-78pp, D: −27pp), step896 (A/B/C/D/F: all −19.62pp, identical).

With K_hh=2, softmax(score, dim=-1) produces weights [w, 1-w]. At initialization, AH-logit is near-zero symmetric → w≈0.5 for all neurons → Z_agg ≈ average of both neighbors. This is a locally stable fixed point: with b=0, the gradient of loss w.r.t. b is near-zero because all neurons use the same routing (symmetric), so the bias sees no differential signal to break symmetry.

**Evidence:** A_bias_khh(2p), B_bias_step(5p), C_bias_full(10p), D_temp_only(1p), F_bias_per_node(4096p) ALL converge to identical 74.34%. Parameter count is irrelevant — the fixed point is structural, not capacity-limited.

**Comparison to ΔW-proj:** ΔW-proj modulates per-edge *magnitude* (how much of neighbor j's signal passes), not routing weights. It starts from the geometric prior (relational direction in W_pos space) which provides a non-zero gradient signal from step 1. Softmax routing starts at equal weights and has no gradient to move away from symmetry.

**Cure:** don't replace ΔW-proj with softmax routing. If dynamic routing is desired, add it as an *additive* term to an already-working mechanism, or use a mechanism with non-symmetric initialization (e.g., W_pos distance already breaks symmetry — but step859 showed even that is a T0 artifact).

---

### Failure Mode 8: O(N) Gradient Dominance via Dense Aggregation on Shared Z

**Steps:** step897 v1 (W_pos version: −61pp) and v2 (W_key separate: −54pp). Both collapse at ep1.

Any aggregation over ALL N neurons creates O(N) gradient on the shared recurrent Z:
- Z_dyn[b,i,:] = weighted_mean over j=0..N-1 of Z_fwd[b,j,:] → each Z_fwd[b,i,:] appears in N rows
- ΔW-proj aggregation is O(K_hh=2): only K_hh neighbors contribute to each neuron
- Gradient ratio: N/K_hh = 2048/2 = 1024×
- Over K_iter=5 recurrent steps, the imbalance compounds → Z dynamics fully hijacked by Z_dyn in ep1

Note: FM8 is distinct from FM5 (W_pos parameter hijacking in step897 v1). v2 protected W_pos with a separate W_key but still collapsed because Z is shared — the shared recurrent state is hijacked, not the shared parameter.

**Cure:** limit dynamic gate to the SAME K_hh topology as ΔW-proj. Then both mechanisms have O(K_hh) gradient on Z — balanced. Alternatively, scale Z_dyn by α = K_hh/N ≈ 0.001 (very weak signal) or detach Z_fwd in gate computation.

---

### Failure Mode 9: K_hh Cosine Gate — Co-adaptation & Structural Dependency

**Steps:** step898 (A/B/D: −0.05pp to −0.38pp neutral, C_gate_only: −78.60pp KILL).

K_hh cosine gate: `score_j = Z_fwd[b,i] · W_key[j]`, gate = LeakyReLU(score + b), Z_dyn = Σ gate * Z_nb (no division). Uses K_hh=2 topology (same as ΔW-proj), so FM8 O(N) issue is avoided.

**C_gate_only = −78.60pp** (gate without ΔW-proj): the gate has NO structural prior. Without ΔW-proj providing geometric gradient signal, W_key random init → random routing → catastrophic collapse. The gate is fully dependent on ΔW-proj's gradient scaffold to learn.

**A/B/D neutral (−0.05pp to −0.38pp)** with 32–37K extra params: when added to ΔW-proj, the gate co-adapts to near-zero effect (FM5 — same signal path). The extra mechanism has no benefit because ΔW-proj already handles the routing path completely. Parameter efficiency is ×2 the base model for zero gain.

**Bug found during development:** LeakyReLU(negative, slope=0.01) returns negative values. With K_hh=2 and b≈0, ~25% of neurons have both cosine scores negative → gate.sum < 0 → clamp(1e-6) / negative numerator = 4000× explosion → dead gradient. Fix: use weighted SUM not normalized weighted mean (same pattern as _dw_agg).

**Cure:** dynamic gates additively on top of ΔW-proj do not help because they share the Z aggregation path. The gate's structural prior must come from something orthogonal to ΔW-proj (e.g., θ modulation, Z transformation before aggregation).

---

### Failure Mode 3: Low-D Signal Noise at D=16

**Architecture:** D=16, Z ∈ S^15 (unit sphere after l2-norm).

Dot product between two random unit vectors in R^16: mean=0, std=1/√16=0.25.
With K_hh=2, routing scores for 2 neighbors differ by ~0.25 — extremely low SNR.

**Evidence:** step73 Config A (Z-dot alone, no AH logit) = 55.77% (−28.79pp collapse) on the OLD arch at N=1024/D=64. D=64 has std=0.125 — 2× more stable. At D=16 the noise is 2× worse.

**Implication:** ANY routing mechanism that uses Z-dot as a score, without a strong structural prior (AH), will fail at D=16.

**Cure:** use AH logit as anchor (provides structural stability), or use a richer score (shared W_q projection lifts D=16 to higher scoring dim before comparison).

---

### Failure Mode 4: T0 Artifact (Training Dynamics)

**Steps:** step859/861 soft routing (β-anneal: +0.99pp T0 → 0.0pp T1), step868/874 Z-mem (γ=0.8: +0.33pp T0 → 0.0pp T1), step869/872/875 hub (hub=0.05: +0.33pp T0 → +0.18pp T1 → +0.03pp T2).

Early training (ep 1-20): loss landscape is smooth, many mechanisms provide slight inductive bias boost. By ep 75+ the baseline catches up. The signal is real at ep 20 but the mechanism doesn't survive full training.

**Diagnostic rule:** T0 gains ≤+0.5pp from non-obviously-orthogonal mechanisms are presumed T0 artifacts until T1 confirms. T0 gains ≥+1.0pp from clean ablations are reliable (ρ=0.80 over 46 experiments).

**Cure:** advance T0-artifact candidates only if ≥+0.5pp at T0; below that threshold skip to T1 directly.

---

### Failure Mode 5: Same-Path Co-Adaptation

**Steps:** step859 Config D (soft_routing + ΔW-proj): −60.51pp catastrophic. step876 (hub + Z-mem compound): −0.20pp cancel.

When two mechanisms act on the same signal path (Z propagation), each mechanism captures the variance the other was counting on → catastrophic co-adaptation or cancellation.

**ΔW-proj** modulates Z via a projection gate on the relational axis.
**Soft routing** modulates Z via reweighted aggregation.
Both act on the `Z[:, conn_hh, :]` gather step — SAME path. Result: collapse.

**Confirmed by:** Compounding Rule in CLAUDE.md. Hub and Z-mem also share aggregation path.

**Cure:** mechanisms must be orthogonal (different signal paths). Compose only after isolation ablation.

---

### Failure Mode 6: Dynamic Topology Destruction

**Steps:** step512–514, step521 (deep supervision), step523 (alternating W_pos/edge training), step852 (rebuild cadence ablation — ALL dynamic configs negative, worst −79pp).

`conn_hh` encodes learned connectivity correlated with W_pos geometry. Hot-rebuilding during training destroys this correlation mid-training. W_pos and AH coupling are entangled with the fixed edge structure.

**Confirmed:** step852 — even 10-epoch rebuild cadence is −1.91pp. Static random Watts-Strogatz WINS over data-driven topology. 

**Rule:** topology is frozen post-init. Routing WEIGHTS can be dynamic; routing EDGES cannot.

---

## What HAS Worked

| Step | Mechanism | Result | Status |
|------|-----------|--------|--------|
| step73 D | softmax(Z-dot/τ=0.3 + AH_logit) | +1.78pp (OLD arch N=1024/D=64) | **CONFIRMED old arch only** |
| step75 D | input-modulated temperature | +3.98pp (OLD arch) | **CONFIRMED old arch only** |
| step706 | ΔW-proj (relational direction gate) | +1.49pp T1 | **CONFIRMED current arch** |
| step868 C | Z-mem γ=0.8 | +0.33pp T0 (T0 artifact, fails T1) | CLOSED |
| step869 A | Hub α=0.05 | +0.33pp T0 (artifact, fails T2) | CLOSED |

**step894 closed the critical gap (2026-04-18):** Z-dot+AH softmax was tested on current arch.
Result: ALL configs KILLED (−18pp). **NEW FINDING:** even pure AH-softmax (no Z-dot) loses −18pp.
This means the problem is not just Z-dot noise — **softmax weight selection over K_hh=2 is fundamentally
weaker than ΔW-proj magnitude gating** as a routing mechanism. At K_hh=2, softmax redistributes
between exactly 2 neighbors; the argmax is nearly always the same neighbor. The softmax weights
carry little signal, whereas ΔW-proj modulates the *magnitude* of each neighbor's contribution
based on relational direction — a richer per-edge signal.

step859 (closed) used **W_pos distance** as score — a static structural proxy. NOT Z-dot+AH. Different mechanism.

---

## Completed Experiments Summary (2026-04-19)

| Step | Mechanism | Result | Status |
|------|-----------|--------|--------|
| step894 | Z-dot+AH softmax routing T0 | B/C/D: −18pp | KILLED |
| step895 | Norm-weighted/shared-query/factored-attn/learned-temp T0 | A/B/C: −76-78pp, D: −27pp | ALL KILLED |
| step896 | Biased softmax T0 | A/B/C/D: 74.34% (FM7 fixed-point), E: −2pp, F: −19.62pp | ALL KILLED |
| step897 | Dense cosine gate (v1 W_pos, v2 W_key) T0 | v1: −61pp (FM5+FM8), v2: −54pp (FM8) | KILLED |
| step898 | K_hh cosine gate + ΔW-proj additive T0 | A: −0.38pp, B: −0.31pp, C: −78.60pp, D: −0.05pp | KILLED (C_gate_only proves structural dependency; A/B/D neutral at 2× param cost) |

**Direction CLOSED 2026-04-19:** All dynamic routing mechanisms tested on current arch (N=2048, D=16, K_hh=2) either fail catastrophically or achieve neutral with prohibitive param overhead. The fundamental constraints are: (1) FM7 — softmax K_hh=2 symmetric fixed-point; (2) FM8 — dense aggregation O(N) gradient dominance; (3) FM9 — K_hh gate co-adapts with ΔW-proj on shared path (C_gate_only kill confirms no structural prior). ΔW-proj (+1.49pp) remains the only confirmed routing mechanism for current arch.

---

## Summary Table

| Step | Mechanism | Params | Risk | Priority |
|------|-----------|--------|------|---------|
| step894 | Z-dot+AH softmax routing | 0 | Failure mode 3 (D=16 noise) | HIGH — gap-fills the critical untested case |
| step895 | Norm-weighted / shared W_q / factored attn | 0 / 16 / 128 | T0 artifact | MEDIUM |
| step896 | ΔW-residual as routing weight | 0 | Co-adaptation | MEDIUM |

**Launch order:** step894 first (highest prior probability, smallest mechanism, directly fills the gap from old-arch step73). Then step895 and step896 in parallel once step894 T0 result is known.
