# Pending Design Discussions (Not Yet Scripted)

**Purpose:** Architecture/design ideas that were discussed but have no script yet.
**Rule:** Every entry here must eventually produce a script + EXPERIMENT_QUEUE.md entry.

---

## Status Key
- **PENDING** — discussed, no script
- **SCRIPTED** — script exists, in queue or running
- **KILLED** — scripted, ran, failed

---

## SCRIPTED (no longer pending)

### step87 — Pure Proximity Architecture
**Status:** SCRIPTED — train_step87_proximity_routing.py exists

**Hypothesis:** Remove static conn_hh entirely. Connectivity computed from W_pos proximity
(top-K KNN) at each routing step. Two decoupled timescales: slow position (W_pos), fast
activation (Z). No pre-built edges to gate → avoids gate-death by construction.

**Configs:** Ref (static conn_hh) vs A/B/C/D (rebuild R=5/10/25 batches + softmax routing)
+ E (FAISS rebuild + post-training radius pruning for FLOPs reduction).

**Design discussion:** LEARNINGS_design_2026_04_07_08.md (2026-04-08 section)

---

## PENDING (script does not exist)

### G4 — Redistribution Routing at N=4096
**Status:** PENDING — never scripted

**Date discussed:** 2026-04-09 (post-mortem synthesis)

**Hypothesis:** step75 Config D (temperature redistribution routing) gave +3.98pp at N=1024.
Redistribution (Σw=1) avoids gate-death by construction. Never scaled to N=4096 where base is
95.87% (step71 Ref). If the gain scales even partially, this could push past current best.

**Key params to specify before scripting:**
- Base: N=4096, K_hh=4, K_iter=12, turing=0.0, AH=1.0 (current defaults)
- Configs: Ref (static AH) + A/B/C variants of step75-D at N=4096
- Scale: 50%/75ep calibration first, then 100%/150ep if winner

**Design discussion:** LEARNINGS_phase5_p15e_dynamic_routing_postmortem.md

---

### step72 — N-Scaling on Patched Arch
**Status:** PENDING — never scripted (step80 only covered N=512 and N=2048 at 50%/75ep)

**Date discussed:** 2026-04-06 (post-patch priority queue)

**Hypothesis:** The entire step56 N-scaling curve is invalid (buggy architecture). Need to
re-establish the N-scaling law on patched code with current defaults.
At N=2048 the patch gave +11.64pp vs step56. The curve above N=4096 is completely unknown.

**Key params to specify before scripting:**
- N={512, 1024, 2048, 4096, 8192} (add 8192 — not yet tested at any arch)
- turing=0.0, K_iter=12, K_hh=4, AH=1.0, reflect=0.5
- 100%/150ep full run (not 50%/75ep)

**What blocks it:** Nothing — can script immediately.

**Design discussion:** LEARNINGS_phase5_p15_post_wave1.md (priority queue section)

---

### step85 — Stacked SGNNET Parallel Concat-Project (Patched Arch)
**Status:** PENDING — never scripted on patched arch

**Date discussed:** 2026-04-07 (step64 Config F observation)

**Hypothesis:** step64 Config F (2-parallel concat-project fusion) was the ONLY stacking
variant to beat Ref on buggy arch: +1.52pp (74.90% vs 73.38%). best_ep=75/75 — still converging
at run end, suggesting more headroom. On patched arch (Ref ~83.36%), the absolute headroom is
much larger. The series stacking uniformly hurt; parallel concat-project avoids over-smoothing
because the two branches run independently and fuse via concat → linear projection.

**Key params to specify before scripting:**
- N=1024, 50%/75ep, patched arch (input_coverage + alpha_reflect), turing=0.0, AH=1.0
- Configs: Ref (single-layer baseline), A (2-parallel concat-project, same as step64 F),
  B (2-parallel concat-project + K_iter=12 per branch)
- Comparison ref: step82A Ref = 82.62% or step73 Ref = 84.56%

**What blocks it:** Nothing — can script immediately.

**Design discussion:** LEARNINGS_phase5_p15_post_wave1.md (step64 section)

---

### Phase Alignment as Softmax Weight (step60 Redemption)
**Status:** PENDING — never scripted with redistribution fix

**Date discussed:** 2026-04-07 (design synthesis WHY dynamic routing failed)

**Hypothesis:** step60 failed because phase coherence was used as a multiplicative gate.
The redistribution fix: `w_j = softmax(coherence(Z_h, Z_j) / τ, dim=2)`.
Σw=1 over K_hh neighbors → no attenuation → no gate-death. This directly redeems step60
using the redistribution principle that step73/75 validated.

**Distinction from step73:** step73 used dot(Z_h, Z_j) as the score. This uses phase
coherence cos(Z_phase_h, Z_phase_j) as the score — a different information source.

**Key params to specify before scripting:**
- N=1024, 50%/75ep, patched arch
- Configs: Ref (static AH) + A (phase coherence softmax τ=1.0) + B (τ=0.5) + C (τ=2.0)
  + D (hybrid: phase coherence × dot-product score)
- Step number: needs a new step number (step80 was taken by N-scaling)

**What blocks it:** Assign a step number (suggest step89 or next available).

**Design discussion:** LEARNINGS_design_2026_04_07_08.md (step60 revival section)

---

### Delayed Routing Activation (step83 Variant)
**Status:** PENDING — discussed as next-gen group routing candidate

**Date discussed:** 2026-04-08 (step83 post-mortem)

**Hypothesis:** step83 failed partly due to temporal mismatch (AH epoch-timescale vs router
batch-timescale). Fix: freeze router weights for first 30 epochs, let AH establish W_pos
structure, then unfreeze. This decouples the timescales.

**Key params to specify before scripting:**
- N=1024, 50%/75ep, patched arch
- Base: step82 Config A (n_groups=8 group topology)
- Config: step83 Config B setup (final-step routing) + delayed activation
- Ablate N_freeze={15, 30, 45}
- Also test: stop-gradient variant (S_g = mean(Z[h]).detach())

**What blocks it:** Requires deciding whether to combine with stop-gradient or test separately.

**Design discussion:** LEARNINGS_phase5_p15e_dynamic_routing_postmortem.md and
LEARNINGS_design_2026_04_07_08.md (step83 post-mortem section)

---

## SCRIPTED (from research review 2026-04-09)

### step91 — GCNII Initial Residual
**Status:** SCRIPTED — train_step91_gcnii_residual.py exists
**Hypothesis:** h_t = (1-a)*route(h_{t-1}) + a*h_0 prevents over-smoothing at K_iter=12+. Proven at 64 layers in GCNII. Zero new params, orthogonal to AH.
**Design discussion:** RESEARCH_routing_mechanisms.md §7

### step92 — ReLU Group Routing
**Status:** SCRIPTED — train_step92_relu_group_routing.py exists
**Hypothesis:** ReLU(W@S_g) + L1 fixes step83 softmax collapse. ReLU gradient=1 for active routes (no exponential decay under K_iter). Directly addresses gate-death theorem.
**Design discussion:** RESEARCH_routing_mechanisms.md §6

---

## PENDING (from research review 2026-04-09)

### step93 — GRAND-Style Implicit Diffusion
**Status:** PENDING
**Date discussed:** 2026-04-09
**Hypothesis:** Replace explicit h_{t+1}=f(h_t, N) with implicit solve (I-dt*L)h_{t+1}=h_t+dt*h_0. Unconditionally stable at any K_iter depth. Source term h_0 prevents over-smoothing. Different mechanism than AH — about signal propagation stability, not edge selection.
**Key params:** N=1024, 50%/75ep, dt={0.1,0.5,1.0}, fixed-point iters={1,2}, source weight β={0.1,0.2}. Ref=static AH. 4-5 configs.
**What blocks it:** Need to verify implicit solve is compatible with AH weight updates (AH modifies L between epochs, implicit solve uses L within forward pass — should be fine since L is fixed within a forward pass).
**Design discussion:** RESEARCH_routing_mechanisms.md §2

### step94 — Hamiltonian Message Passing
**Status:** PENDING
**Date discussed:** 2026-04-09
**Hypothesis:** Leapfrog integration preserves energy exactly — signal cannot decay over K_iter. Momentum p_i carries information across steps (implicit skip connection). Energy conservation is a hard guarantee against over-smoothing, unlike soft mechanisms.
**Key params:** N=1024, 50%/75ep, dt={0.05,0.1,0.2}, p_i same dim as h_i (D=64). K_iter=12. Potential V from AH-weighted neighbor interactions. 3 configs + Ref.
**What blocks it:** Doubles state size (h+p). Must verify FLOPs stay within budget at N=1024. May need to halve D to compensate (D=32 momentum experiment).
**Design discussion:** RESEARCH_routing_mechanisms.md §4

### step95 — Expander Graph Initialization
**Status:** PENDING
**Date discussed:** 2026-04-09
**Hypothesis:** Replace small-world init with Ramanujan expander graph (deterministic, optimal spectral gap). Better information flow at same K_hh budget. Zero-cost experiment — only changes init, not forward pass.
**Key params:** N=1024, 50%/75ep, K_hh=4. Ref=small-world init. A=Ramanujan expander (d-regular bipartite). B=random d-regular graph. C=expander + n_groups=8 topology.
**What blocks it:** Need to implement Ramanujan expander construction (LPS or Margulis). Simple: use random d-regular graph as proxy (nearly optimal spectral gap for d≥3).
**Design discussion:** RESEARCH_routing_mechanisms.md §1

### step96 — DropMessage During K_iter
**Status:** PENDING
**Date discussed:** 2026-04-09
**Hypothesis:** Randomly drop messages (not edges, not nodes) during K_iter propagation. Proven anti-over-smoothing in AAAI 2023. Trivial to implement — zero new params. At K_iter=12, over-smoothing risk is real; DropMessage is the cheapest countermeasure.
**Key params:** N=1024, 50%/75ep, drop_rate={0.1,0.2,0.3}. Apply during training only (eval = no drop). Ref=no drop. 3 configs.
**What blocks it:** Nothing — can implement in ~5 lines.
**Design discussion:** RESEARCH_routing_mechanisms.md §7

### step97 — Beltrami Flow (Joint Position+Feature Evolution)
**Status:** PENDING
**Date discussed:** 2026-04-09
**Hypothesis:** Jointly evolve features h AND positions W_pos during K_iter. Feature evolution = message passing (current). Position evolution = gradient flow on S^{D-1} manifold toward/away from active neighbors. This IS dynamic connectivity from learned positions — the topology emerges from position dynamics. Naturally compatible with S^{D-1} geometry.
**Key params:** N=1024, 50%/75ep, position_lr_scale={0.01,0.1} (relative to feature update). Position update: W_pos += lr_scale * grad_manifold(V). Project back to S^{D-1} after update. K_iter=12. 3 configs + Ref.
**What blocks it:** Requires manifold gradient computation on S^{D-1} — exponential map/retraction. Medium implementation complexity. Also risk: position updates during K_iter may conflict with AH's epoch-level W_pos updates (same temporal mismatch as step83).
**Design discussion:** RESEARCH_routing_mechanisms.md §2, §5

### step98 — Dynamic Group KNN Topology
**Status:** PENDING
**Date discussed:** 2026-04-09
**Hypothesis:** Binary group-level KNN topology (connected/not) avoids all gate-death issues. Recompute KNN on group centroids every K_iter step (or every other). O(n_groups^2)=O(64) cost — negligible. Changes topology without learned routing weights. Key distinction from step83: no softmax, no learned gates, just binary connectivity.
**Key params:** N=1024, 50%/75ep, n_groups=8, K_group={2,3,4}. Recompute frequency: {every step, every 2 steps, every 4 steps}. Intra-group stays static small-world. 4-5 configs + Ref.
**What blocks it:** Nothing — straightforward implementation.
**Design discussion:** RESEARCH_routing_mechanisms.md §5

---

## PENDING (Gemma4/PolarQuant-inspired designs, 2026-04-09)

### step106 — Per-Step Embeddings for K_iter (PLE-Inspired)
**Status:** PENDING
**Date discussed:** 2026-04-09

**Hypothesis:** K_iter=12 identical steps waste capacity. A small per-step conditioning vector (D-dim, 12 vectors total = 12×D=768 new params) lets different steps specialize: early steps for coarse routing, late steps for fine-grained. Additive injection avoids gate-death — no multiplicative compounding.

**Why it avoids gate-death:** The embedding is ADDED to Z (or to edge weights) before the existing routing step. The routing step itself (AH + gather + normalize) is unchanged. There is no per-step multiplicative factor that compounds. Each step sees a slightly different "view" of the activations, but signal magnitude is preserved.

**Key design choices:**
- **Mode A (Z-bias):** `Z_t = Z_t + emb[t]` — shift activation space per step. AH operates on shifted Z. 12×D params.
- **Mode B (edge-scale):** Edge weights multiplied by `1 + scale[t]` where scale is a scalar per step. 12 scalars = 12 params. Still multiplicative but scale is LEARNED and FIXED (not input-dependent), so it's a constant per step, not a gate.
- **Mode C (AH-modulation):** `alpha_t = alpha_base + delta[t]` — per-step AH strength. 12 scalars. Early steps: weak AH (explore). Late steps: strong AH (exploit).
- **Mode D (Z-bias + initial residual):** Combine with GCNII residual (step91). `Z_t = (1-a)*route(Z_{t-1} + emb[t]) + a*Z_0`. Tests interaction.

**Configs (N=1024, D=64, K_hh=4, K_iter=12, 50%/75ep):**

| Config | Per-step mode | Init | Extra params | Tests |
|--------|--------------|------|-------------|-------|
| Ref | None (identical steps) | — | 0 | Baseline |
| A | Z-bias (learned D-vec per step) | zeros | 768 | Core mechanism |
| B | Z-bias | small random (0.01) | 768 | Does init matter? |
| C | Edge-scale (learned scalar per step) | ones | 12 | Minimal param version |
| D | AH-modulation (per-step alpha offset) | zeros | 12 | Step-specialized diversity |
| E | Z-bias + GCNII residual (a=0.1) | zeros | 768 | Compound with step91 |

**Depends on:** Nothing — fully independent. Can run alongside step91.
**Priority:** P1


*Continued in [PENDING_DISCUSSIONS_part2.md](PENDING_DISCUSSIONS_part2.md) — step107 MoE Routing, step108 Hierarchical Polar, step109 Compound, gap analysis steps 99-101, step163 Progressive KD, Resolved.*
