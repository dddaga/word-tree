# Dynamic Routing: Failure Analysis & Next Experiments

**Status:** Living document. Last updated 2026-04-19. **Dynamic routing direction CLOSED (all mechanisms, 2026-04-19). Readout gate direction CLOSED (T1 artifact confirmed, step911 T2).**
**Cross-links:** [[phase_routing_part2]], [[soft_routing_hnsw]], [[sparse_bfs_routing]], [[softmax_routing]]

## Why Dynamic Routing Keeps Failing (Taxonomy)

### Failure Mode 1: Gate-Death (structural)

**Steps:** 58–66 (wave-1), many others with multiplicative gates.

Gate g ∈ [0,1] on Z propagation decays g^K over K_iter steps.
K_iter=5, g=0.8: 0.8^5 = 0.33×. Network learns g→0 → mechanism decorative.

**Confirmed by:** steps 58–66 crash convergence. SGNNET_Resonant `dynamic_gate` mode.

**Cure:** additive redistribution (softmax weights summing to 1), never multiplicative gates.

---

### Failure Mode 2: Gradient Disconnection from Top-k Selection

**Steps:** step855 Sparse BFS T0 (all configs −2.0 to −2.3pp), step873 BFS M=32 T1 (−24.41pp catastrophic).

`torch.topk` non-differentiable. Gradient only flows through selected values via straight-through estimator. At T1 (75ep), unstable selection gradients compound: BFS selects different subsets each step, network can't learn consistent active nodes.

**Why T0 bad, T1 catastrophic:** T0 (20ep) too short for instability to manifest. T1 runs long enough for gradient disconnect to poison loss landscape.

**Cure:** soft selection (Gumbel-softmax with annealing τ → 0), or NO selection (dense softmax routing with all K_hh neighbors).

---

### Failure Mode 7: Softmax Routing Fixed-Point Collapse at K_hh=2

**Steps:** step894 (B/C/D: −18pp), step895 (A/B/C: −76-78pp, D: −27pp), step896 (A/B/C/D/F: all −19.62pp, identical).

K_hh=2 → softmax(score, dim=-1) produces weights [w, 1-w]. At init, AH-logit near-zero symmetric → w≈0.5 for all neurons → Z_agg ≈ average of both neighbors. Locally stable fixed point: with b=0, gradient of loss w.r.t. b near-zero because all neurons use same routing (symmetric), bias sees no differential signal to break symmetry.

**Evidence:** A_bias_khh(2p), B_bias_step(5p), C_bias_full(10p), D_temp_only(1p), F_bias_per_node(4096p) ALL converge to identical 74.34%. Parameter count irrelevant — fixed point structural, not capacity-limited.

**Comparison to ΔW-proj:** ΔW-proj modulates per-edge *magnitude* (how much of neighbor j's signal passes), not routing weights. Starts from geometric prior (relational direction in W_pos space) → non-zero gradient signal from step 1. Softmax routing starts at equal weights, no gradient to escape symmetry.

**Cure:** don't replace ΔW-proj with softmax routing. If dynamic routing desired, add as *additive* term to working mechanism, or use non-symmetric initialization (e.g., W_pos distance breaks symmetry — but step859 showed even that is T0 artifact).

---

### Failure Mode 8: O(N) Gradient Dominance via Dense Aggregation on Shared Z

**Steps:** step897 v1 (W_pos version: −61pp) and v2 (W_key separate: −54pp). Both collapse at ep1.

Aggregation over ALL N neurons creates O(N) gradient on shared recurrent Z:
- Z_dyn[b,i,:] = weighted_mean over j=0..N-1 of Z_fwd[b,j,:] → each Z_fwd[b,i,:] appears in N rows
- ΔW-proj aggregation O(K_hh=2): only K_hh neighbors contribute per neuron
- Gradient ratio: N/K_hh = 2048/2 = 1024×
- Over K_iter=5 recurrent steps, imbalance compounds → Z dynamics fully hijacked by Z_dyn in ep1

Note: FM8 distinct from FM5 (W_pos parameter hijacking in step897 v1). v2 protected W_pos with separate W_key but still collapsed — shared recurrent state hijacked, not shared parameter.

**Cure:** limit dynamic gate to SAME K_hh topology as ΔW-proj. Both mechanisms get O(K_hh) gradient on Z — balanced. Or scale Z_dyn by α = K_hh/N ≈ 0.001 (very weak signal) or detach Z_fwd in gate computation.

---

### Failure Mode 9: K_hh Cosine Gate — Co-adaptation & Structural Dependency

**Steps:** step898 (A/B/D: −0.05pp to −0.38pp neutral, C_gate_only: −78.60pp KILL).

K_hh cosine gate: `score_j = Z_fwd[b,i] · W_key[j]`, gate = LeakyReLU(score + b), Z_dyn = Σ gate * Z_nb (no division). Uses K_hh=2 topology (same as ΔW-proj), FM8 O(N) avoided.

**C_gate_only = −78.60pp** (gate without ΔW-proj): gate has NO structural prior. Without ΔW-proj geometric gradient signal, W_key random init → random routing → catastrophic collapse. Gate fully dependent on ΔW-proj's gradient scaffold.

**A/B/D neutral (−0.05pp to −0.38pp)** with 32–37K extra params: when added to ΔW-proj, gate co-adapts to near-zero effect (FM5 — same signal path). Extra mechanism no benefit — ΔW-proj already handles routing completely. 2× param cost for zero gain.

**Bug found:** LeakyReLU(negative, slope=0.01) returns negative values. K_hh=2, b≈0 → ~25% neurons have both cosine scores negative → gate.sum < 0 → clamp(1e-6) / negative numerator = 4000× explosion → dead gradient. Fix: use weighted SUM not normalized weighted mean (same pattern as _dw_agg).

**Cure:** dynamic gates additive on ΔW-proj don't help — share Z aggregation path. Gate's structural prior must come from something orthogonal to ΔW-proj (e.g., θ modulation, Z transformation before aggregation).

---

### Failure Mode 3: Low-D Signal Noise at D=16

**Architecture:** D=16, Z ∈ S^15 (unit sphere after l2-norm).

Dot product between two random unit vectors in R^16: mean=0, std=1/√16=0.25.
K_hh=2 → routing scores for 2 neighbors differ by ~0.25 — extremely low SNR.

**Evidence:** step73 Config A (Z-dot alone, no AH logit) = 55.77% (−28.79pp collapse) on OLD arch at N=1024/D=64. D=64 has std=0.125 — 2× more stable. D=16 noise 2× worse.

**Implication:** ANY routing using Z-dot as score without strong structural prior (AH) fails at D=16.

**Cure:** use AH logit as anchor (structural stability), or richer score (shared W_q projection lifts D=16 to higher scoring dim before comparison).

---

### Failure Mode 4: T0 Artifact (Training Dynamics)

**Steps:** step859/861 soft routing (β-anneal: +0.99pp T0 → 0.0pp T1), step868/874 Z-mem (γ=0.8: +0.33pp T0 → 0.0pp T1), step869/872/875 hub (hub=0.05: +0.33pp T0 → +0.18pp T1 → +0.03pp T2).

Early training (ep 1-20): loss landscape smooth, many mechanisms give slight inductive bias boost. By ep 75+ baseline catches up. Signal real at ep 20 but mechanism doesn't survive full training.

**Diagnostic rule:** T0 gains ≤+0.5pp from non-obviously-orthogonal mechanisms presumed T0 artifacts until T1 confirms. T0 gains ≥+1.0pp from clean ablations reliable (ρ=0.80 over 46 experiments).

**Cure:** advance T0-artifact candidates only if ≥+0.5pp at T0; below that skip to T1 directly.

---

### Failure Mode 5: Same-Path Co-Adaptation

**Steps:** step859 Config D (soft_routing + ΔW-proj): −60.51pp catastrophic. step876 (hub + Z-mem compound): −0.20pp cancel.

Two mechanisms on same signal path (Z propagation) → each captures variance other counted on → catastrophic co-adaptation or cancellation.

**ΔW-proj** modulates Z via projection gate on relational axis.
**Soft routing** modulates Z via reweighted aggregation.
Both act on `Z[:, conn_hh, :]` gather step — SAME path. Result: collapse.

**Confirmed by:** Compounding Rule in CLAUDE.md. Hub and Z-mem also share aggregation path.

**Cure:** mechanisms must be orthogonal (different signal paths). Compose only after isolation ablation.

---

### Failure Mode 6: Dynamic Topology Destruction

**Steps:** step512–514, step521 (deep supervision), step523 (alternating W_pos/edge training), step852 (rebuild cadence ablation — ALL dynamic configs negative, worst −79pp).

`conn_hh` encodes learned connectivity correlated with W_pos geometry. Hot-rebuilding during training destroys this correlation. W_pos and AH coupling entangled with fixed edge structure.

**Confirmed:** step852 — even 10-epoch rebuild cadence −1.91pp. Static random Watts-Strogatz WINS over data-driven topology.

**Rule:** topology frozen post-init. Routing WEIGHTS can be dynamic; routing EDGES cannot.

---

## What HAS Worked

| Step | Mechanism | Result | Status |
|------|-----------|--------|--------|
| step73 D | softmax(Z-dot/τ=0.3 + AH_logit) | +1.78pp (OLD arch N=1024/D=64) | **CONFIRMED old arch only** |
| step75 D | input-modulated temperature | +3.98pp (OLD arch) | **CONFIRMED old arch only** |
| step706 | ΔW-proj (relational direction gate) | +1.49pp T1 | **CONFIRMED current arch** |
| step868 C | Z-mem γ=0.8 | +0.33pp T0 (T0 artifact, fails T1) | CLOSED |
| step869 A | Hub α=0.05 | +0.33pp T0 (artifact, fails T2) | CLOSED |

**step894 closed critical gap (2026-04-18):** Z-dot+AH softmax tested on current arch.
Result: ALL configs KILLED (−18pp). **NEW FINDING:** even pure AH-softmax (no Z-dot) loses −18pp.
Problem not Z-dot noise — **softmax weight selection over K_hh=2 fundamentally weaker than ΔW-proj magnitude gating** as routing mechanism. K_hh=2 → softmax redistributes between exactly 2 neighbors; argmax nearly always same neighbor. Softmax weights carry little signal; ΔW-proj modulates *magnitude* of each neighbor's contribution based on relational direction — richer per-edge signal.

step859 (closed) used **W_pos distance** as score — static structural proxy. NOT Z-dot+AH. Different mechanism.

---

## Completed Experiments Summary (2026-04-19)

| Step | Mechanism | Result | Status |
|------|-----------|--------|--------|
| step894 | Z-dot+AH softmax routing T0 | B/C/D: −18pp | KILLED |
| step895 | Norm-weighted/shared-query/factored-attn/learned-temp T0 | A/B/C: −76-78pp, D: −27pp | ALL KILLED |
| step896 | Biased softmax T0 | A/B/C/D: 74.34% (FM7 fixed-point), E: −2pp, F: −19.62pp | ALL KILLED |
| step897 | Dense cosine gate (v1 W_pos, v2 W_key) T0 | v1: −61pp (FM5+FM8), v2: −54pp (FM8) | KILLED |
| step898 | K_hh cosine gate + ΔW-proj additive T0 | A: −0.38pp, B: −0.31pp, C: −78.60pp, D: −0.05pp | KILLED (C_gate_only proves structural dependency; A/B/D neutral at 2× param cost) |
| step903 | Epoch topology rebuild T0 | A_std_lr=−0.84pp, B_half_lr=−2.19pp, C_tenth_lr=−8.18pp, D_warmup10=−0.99pp | ALL KILLED (FM6 confirmed) |
| step904 | Node z-score gate (FGSEGNet-style) T0 | best E_wpos_geo=−0.28pp; gates dynamic (H=0.4–0.7) but still hurt (FM10 bottleneck) | ALL KILLED |
| step906 | Top-K activation sparsity T0 | hard top-K −13.7 to −15.9pp; soft τ=1: −1.81pp; gate_H=0.62 (non-discriminative) | ALL KILLED |
| step907 | Readout-gate T0 | Ref=94.04%, **E_ro_geo=+1.12pp ADVANCE**, C_ro_tau_lrn=+0.54pp ADVANCE, A_ro_tau1=+0.36pp neutral, D_ro_topk25=−3.49pp KILL | T1→T2 done, see below |
| step910 | Readout-gate T1 | Ref=95.41%, A=+0.66pp, C=+0.69pp, E=+0.61pp — ALL ADVANCE | T2 done (step911) |
| step911 | Readout-gate T2 | Ref=96.74%, **A=+0.15pp, C=+0.25pp, E=+0.18pp — ALL below ≥+0.5pp threshold** | **T1 ARTIFACT — direction CLOSED** |
| step912 | CIFAR-10 readout gate T0 | Ref=76.03%, A_ro_tau1=+0.40pp(NEUTRAL), E_ro_geo=−0.60pp(**KILL**) | Gate Imagenette-specific |

**Direction CLOSED 2026-04-19:** All dynamic routing mechanisms tested on current arch (N=2048, D=16, K_hh=2) either fail catastrophically or achieve neutral with prohibitive param overhead. Fundamental constraints: (1) FM7 — softmax K_hh=2 symmetric fixed-point; (2) FM8 — dense aggregation O(N) gradient dominance; (3) FM9 — K_hh gate co-adapts with ΔW-proj on shared path (C_gate_only kill confirms no structural prior). ΔW-proj (+1.49pp) remains only confirmed routing mechanism for current arch.

---

### Failure Mode 10: Input-Conditioned Node Gating Before/During K_iter (Information Bottleneck)

**Steps:** step903 (epoch topology), step904 (node z-score gate T0), step906 (top-K activation sparsity T0).

**Motivation:** FGSEGNet v2 uses per-input dynamic channel gating from input features. Hypothesis: gating nodes based on input signal |x_sum| lets SGNNET focus on input-relevant nodes.

**Results across 3 experiments:**
| Mechanism | Best config | Result | Gate entropy |
|-----------|------------|--------|-------------|
| step904 node z-score soft gate | E_wpos_geo (geometric prior) | −0.28pp (best) | 0.4–0.7 (dynamic, not collapsed) |
| step904 node z-score soft gate | D_se_bn | −0.51pp | 0.693 (maximum = uniform) |
| step906 top-K hard gate (75% keep) | A_top75 | −15.06pp | 0.0 (fully binary, collapsed) |
| step906 top-K hard gate (50% keep) | B_top50 | −15.92pp | 0.0 |
| step906 soft sigmoid (τ=1) | E_soft_tau1 | −1.81pp | 0.62 (near-max, non-discriminative) |

**Why NOT gate-death (FM1):** Gate entropy 0.4–0.7 genuinely dynamic (H=0 → static, H=0.693=ln(2) → uniform/non-discriminative). Gates ARE firing differently per input. Yet accuracy still drops.

**Root cause — information bottleneck:** SGNNET K_iter message-passing is collective computation over ALL N nodes. Every node participates in propagation to every other via K_hh graph. Zero-ing even 25% of nodes removes paths from K_hh graph, breaking propagation chains. Unlike CNNs where channels independent, SGNNET nodes connected — removing node removes ALL incident edges.

**Hard gating (top-K):** −13.7 to −15.9pp regardless of K (10%–75% keep). Non-differentiable — gradient disconnect compounds over 20ep.

**Soft gating:** −0.28pp to −1.81pp. Marginal hurt from reduced effective graph connectivity even at moderate sparsity.

**Paper note:** "SGNNET achieves FGSEGNet-style routing efficiency through sparse graph topology (K_hh=2 out of N=2048), not per-input dynamic node gating. Input-conditioned gating during message-passing consistently degrades accuracy due to collective propagation bottleneck."

**Only unexplored safe location:** Readout gate — apply gate AFTER K_iter, only modifying which nodes contribute to class aggregation (step907, currently running).

**Direction CLOSED — step911 T2 DONE (2026-04-19):** T1 gains (+0.61–0.69pp) compressed to +0.15–0.25pp at T2 (150ep/100% data). T1 artifact confirmed. Additionally, E_ro_geo hurts CIFAR-10 (step912: −0.60pp KILL). Paper note: "readout-level gating shows +0.6pp at T1 (50% data) but compresses below 0.25pp at full training; gain does not generalize cross-dataset."

---

## Summary Table

| Step | Mechanism | Params | Risk | Priority |
|------|-----------|--------|------|---------|
| step894 | Z-dot+AH softmax routing | 0 | Failure mode 3 (D=16 noise) | HIGH — gap-fills critical untested case |
| step895 | Norm-weighted / shared W_q / factored attn | 0 / 16 / 128 | T0 artifact | MEDIUM |
| step896 | ΔW-residual as routing weight | 0 | Co-adaptation | MEDIUM |

**Launch order:** step894 first (highest prior probability, smallest mechanism, directly fills gap from old-arch step73). Then step895 and step896 parallel once step894 T0 result known.