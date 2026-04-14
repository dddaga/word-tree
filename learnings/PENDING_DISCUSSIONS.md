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

---

### step107 — Group-as-Expert MoE Routing (Gemma4-Inspired)
**Status:** PENDING
**Date discussed:** 2026-04-09

**Hypothesis:** Treat each of n_groups=8 groups as an "expert." A ReLU router selects top-k groups per input token. Only selected groups participate in message passing for that token. Massive FLOPs reduction (only k/8 of neurons active per token) with potential accuracy gain from specialization.

**Why it differs from step83 (KILLED):**
1. **ReLU not softmax** — step83 used softmax which collapsed to uniform under K_iter. ReLU has gradient=1 for active routes (no exponential decay). Directly from ReMoE (ICLR 2025).
2. **L1 load balancing** — step83 had no load balancing. Without it, 1-2 groups dominate. L1 on routing weights encourages sparsity + balance simultaneously.
3. **Shared expert** — one group is always active (like Gemma4's shared expert). Guarantees minimum capacity regardless of routing decisions. Prevents complete signal death for any input.
4. **Token-level routing** — step83 routed at batch level (S_g = mean over all tokens). step107 routes per-token: each input image gets its own group selection.

**Why it avoids gate-death:** ReLU(x) for x>0 has constant gradient 1. No compounding decay. For x<=0, the group is cleanly off (no leaking near-zero signal). The routing decision is binary-ish (on/off) but differentiable.

**Architecture:**
```
S_g = mean(Z[h] for h in group g)     # [B, n_groups, D] — group summary
router_input = flatten(S_g)            # [B, n_groups*D]
route_logits = ReLU(W_route @ S_g.T)   # [B, n_groups] per group
# Group 0 always active (shared expert)
# Top-k selection on remaining groups (k=2 or k=3)
# Active groups run full K_iter message passing
# Inactive groups: Z frozen (no compute)
```

**Configs (N=1024, D=64, K_hh=4, K_iter=12, n_groups=8, 50%/75ep):**

| Config | top_k | Shared expert | L1 lambda | Tests |
|--------|-------|---------------|-----------|-------|
| Ref | all 8 (no routing) | N/A | 0 | step82 baseline |
| A | 3 | Yes (group 0) | 0.01 | Core MoE design |
| B | 2 | Yes (group 0) | 0.01 | Sparser routing |
| C | 3 | No | 0.01 | Is shared expert needed? |
| D | 3 | Yes (group 0) | 0.001 | Weaker L1 |
| E | 3 | Yes (group 0) | 0.01, delayed 30ep | Delayed activation (step83 lesson) |

**Depends on:** step92 (ReLU group routing) results preferred but not required. step105 H2 (does compounding work?) is relevant but this is a fundamentally different approach from compounding — it's sparse activation, not mechanism stacking.
**Priority:** P1 (independent enough to run without step105)

---

### step108 — Hierarchical Polar Routing (PolarQuant-Inspired)
**Status:** PENDING
**Date discussed:** 2026-04-09

**Hypothesis:** Decompose W_pos on S^{D-1} into hierarchical angles via recursive polar transform. Route at coarse levels first (top-level angle = hemisphere), then refine at finer levels. This creates a natural multi-scale topology WITHOUT the n_groups parameter — the hierarchy emerges from the geometry of S^{D-1}.

**Polar decomposition of W_pos (||W_pos||=1):**
```
θ_1 = arccos(W_pos[0])                    # Level 1: hemisphere (2 regions)
θ_2 = arccos(W_pos[1] / sin(θ_1))         # Level 2: quadrant (4 regions)
θ_3 = arccos(W_pos[2] / (sin(θ_1)*sin(θ_2)))  # Level 3: octant (8 regions)
...up to log2(D) levels
```

**Routing mechanism:**
- Level 1: 2 hemispheres. All neurons in same hemisphere are potential neighbors.
- Level 2: 4 quadrants. Refine within hemisphere.
- Level L: 2^L regions. KNN within region at finest level.
- K_hh neighbors selected as: K_local from same finest-level region + K_cross from parent-level region.
- This is a TOPOLOGY change (like step82), not a gate. No learned routing weights.

**Why it's AH-compatible:** The polar angles are derived from W_pos (which AH moves), but the hierarchy is a READ-ONLY structure used for neighbor selection. AH still freely moves W_pos; the hierarchy adapts passively. No antagonism because AH's cosine-similarity suppression and polar-hierarchy neighbor selection operate on different functions of W_pos (cosine vs angle decomposition).

**Risk:** At D=64, log2(64)=6 levels. Level 6 has 64 regions. With N=1024, that's ~16 neurons per region — may be too sparse for KNN. Need K_cross from parent levels to maintain connectivity.

**Configs (N=1024, D=64, K_hh=4, K_iter=12, 50%/75ep):**

| Config | Levels | K_local | K_cross | Tests |
|--------|--------|---------|---------|-------|
| Ref | None (standard KNN) | 4 | 0 | Baseline |
| A | 3 (8 regions) | 2 | 2 | Coarse hierarchy |
| B | 4 (16 regions) | 2 | 2 | Medium hierarchy |
| C | 5 (32 regions) | 3 | 1 | Fine hierarchy |
| D | 3 (8 regions) | 3 | 1 | More local, less cross |
| E | 3 (8 regions), dynamic rebuild every 10 batches | 2 | 2 | Does hierarchy track AH movement? |

**Depends on:** Nothing — topology-only change, independent.
**Priority:** P1

---

### step109 — Compound: Per-Step Embeddings + Group-as-Expert (step106 + step107 winners)
**Status:** PENDING
**Date discussed:** 2026-04-09

**Hypothesis:** Per-step embeddings (step106) and group-as-expert routing (step107) are orthogonal mechanisms:
- step106 specializes WHAT each K_iter step does (temporal specialization)
- step107 specializes WHICH neurons are active per input (spatial specialization)

Combined: different K_iter steps could activate different group subsets, or the per-step embedding could modulate the router input. This is the compound test — only run after step106 and step107 identify winners.

**Why compounding might work here (unlike historical failures):**
- Neither mechanism is a multiplicative gate
- step106 is additive (Z-bias), step107 is sparse activation (on/off)
- No double-sparsity: step106 doesn't remove paths, step107 removes entire groups (coarse, not fine-grained)
- AH operates within active groups only — fewer neurons means AH has less work, not more interference

**Configs (N=1024, D=64, K_hh=4, K_iter=12, n_groups=8, 50%/75ep):**

| Config | step106 winner | step107 winner | Interaction | Tests |
|--------|---------------|---------------|-------------|-------|
| Ref | None | None | — | Baseline |
| A | Best from step106 | Best from step107 | Independent | Simple compound |
| B | Z-bias | top-3 MoE | Per-step router input | Step-varying routing |
| C | AH-modulation | top-3 MoE | Per-step alpha + sparse groups | Double specialization |

**Depends on:** step106 AND step107 results (both must show gain).
**Priority:** P2

---

## PENDING (gap analysis 2026-04-09)

### step99 — W_phase Trained at N=4096
**Status:** PENDING
**Date discussed:** 2026-04-09
**Hypothesis:** step76 showed W_phase trained (turing=0.0) gave +2.88pp at N=1024, but this was never tested at N=4096. At N=4096, turing=0.0 is already default but W_phase is frozen. If the +2.88pp scales even partially, this is free accuracy on top of current 97.38% best.
**Key params:** N=4096, K_hh=4, K_iter=12, 50%/75ep. Ref=frozen W_phase. A=W_phase trained (same lr as other params). B=W_phase trained at 0.1× lr.
**What blocks it:** Nothing — one-line change to include W_phase in optimizer.
**Design discussion:** Gap analysis (step76 never scaled to N=4096)

### step100 — K_in Input Connectivity Sweep
**Status:** PENDING
**Date discussed:** 2026-04-09
**Hypothesis:** K_in (input-to-hidden connectivity) has never been independently swept. Current default K_in=25 was set early and never revisited. For the FLOPs track, K_in reduction is one of the 4 levers identified (N, D, K_hh, K_in). At N=4096 with K_hh=4, the input projection may be over-connected. Reducing K_in could save significant FLOPs with minimal accuracy loss (like K_hh=4 was free lunch).
**Key params:** N=4096, 50%/75ep, K_in={5,10,15,25,50}. All other params at current defaults. 5 configs including Ref (K_in=25).
**What blocks it:** Nothing — parameter already exists in config.
**Design discussion:** Gap analysis (FLOPs track, LEARNINGS_phase5_p15d §Path to 1% FLOPs)

### step101 — Graph Init Comparison (Hidden Assumption)
**Status:** PENDING
**Date discussed:** 2026-04-09
**Hypothesis:** Small-world initialization has been assumed optimal since early phases but never compared against alternatives at current scale (N=4096, patched arch). Random d-regular, lattice, and expander inits may perform differently now that AH reshapes topology during training. If AH converges to similar final topology regardless of init, cheaper inits are equivalent.
**Key params:** N=1024, 50%/75ep, K_hh=4. Ref=small-world (current). A=random d-regular. B=lattice (grid). C=Erdos-Renyi (same avg degree). 4 configs. Can merge with step95 (expander) if both pending when scripted.
**What blocks it:** Nothing.
**Design discussion:** Gap analysis (untested assumption since step9)

---

---

## step163 — Progressive K_iter Distillation (Intermediate State Matching)
**Status:** SCRIPTED — `scripts/train_step163_progressive_kd.py`
**Date discussed:** 2026-04-10

### Motivation

step127 showed that standard output-matching KD HURTS K_iter reduction:
- K=6 scratch:          −4.28pp (best without distillation)
- K=6 distilled α=0.5:  −8.59pp (WORSE than scratch)
- K=6 distilled α=0.7: −12.13pp

**Why step127 failed:** The teacher (K=12) has a routing trajectory — 12 intermediate
Z states that progressively refine the representation. The student (K=6) was only
supervised on the final output. With only final-step supervision, the student has
no signal about HOW to reach that representation — it just sees the destination, not
the path. The routing dynamics are the key mechanism, not just the final activations.

### Hypothesis

Matching intermediate K_iter states forces the student to learn the routing
trajectory, not just the output. If the student can reproduce the teacher's
activation distribution at corresponding intermediate steps, it should acquire
the same routing dynamics with fewer iterations.

### Distillation Mapping (K=12 teacher → K=6 student, 2:1 ratio)

| Student iter | Teacher iter | Interpretation |
|---|---|---|
| k=1 | k=2  | Student's first step should match teacher after 2 steps |
| k=2 | k=4  | Student's second step ≈ teacher's fourth |
| k=3 | k=6  | Halfway through, representations should match |
| k=4 | k=8  | |
| k=5 | k=10 | |
| k=6 | k=12 | Final states must match (same as step127) |

Loss at each step:
```
L_kd_step_k = ||Z_student[k] - Z_teacher[2k]||^2_F  (or cosine similarity loss)
L_total = L_task + α * mean(L_kd_step_k for k in 1..K_student)
```

### Why this might work

1. Each routing step builds on the previous — if step k is well-aligned, step k+1
   starts from a better place. Cumulative alignment > just terminal matching.
2. The intermediate state loss provides dense gradient signal: 6 loss terms instead
   of 1. The student gets corrected at every step of its routing, not just at the end.
3. This is analogous to FitNets (hint-based distillation) but for iterative routing
   rather than layers — proven effective in deep networks.
4. The teacher's intermediate states encode "what should the representation look
   like after this many hops" — exactly the routing curriculum the student needs.

### Configs to test (N=1024, D=16, 75ep 50% data)

| Config | K_iter | Init | Distil | Tests |
|--------|--------|------|--------|-------|
| Ref | 12 | from scratch (SEED=42) | — | teacher baseline |
| A | 6 | same seed scratch (SEED=42) | — | step127 control: shared topology, random weights |
| B | 6 | teacher weights (load_state_dict) | none | warm-start alone, no distil |
| C | 6 | teacher weights | progressive MSE α=0.3 | core hypothesis |
| D | 6 | teacher weights | progressive MSE α=0.5 | α sensitivity |
| E | 8 | teacher weights | progressive MSE α=0.3 | easier alignment (8:12 ratio) |

**Key ablations:**
- A vs B: effect of weight initialization alone (shared topology, with/without warm weights)
- B vs C/D: effect of progressive distillation on top of warm init
- C vs D: α sensitivity

**Teacher init approach:** `student.load_state_dict(teacher.state_dict())`
- Copies ALL parameters AND buffers (conn_hh, conn_in, W_pos, theta, W_in, fc weights)
- K_iter is NOT in state_dict (it's a plain int in the forward loop) — so load works with strict=True
- Topology sharing is automatic — no need to manually copy buffers
- Student starts in teacher's representation space → Z_student[k] and Z_teacher[2k] are immediately comparable

**Config A (scratch control):** Use SEED=42 for student too.
Same seed → same conn_hh/conn_in generation → same topology as teacher.
W_pos, theta, W_in are freshly initialized (not copied). Controls for warm-start effect.

**Teacher must be frozen** during student training. Teacher does K=12 forward pass to collect
Z_teacher[1..12]. Student does K=6 (or K=8) pass, collecting Z_student[1..K_student].
Loss: `L_total = L_task + α * mean(MSE(Z_student[k], Z_teacher[2k].detach()) for k)`

**Implementation note:** Custom wrapper `SGNNET_AH_WithIntermediates` around the AH routing
loop that returns (logits, [Z_1, Z_2, ..., Z_K]) — one list per routing step.

### What blocks it
Nothing — step127 results are in hand, teacher weights can be saved during
step127 training or retrained at the start of step163.

### Step number
**step163** — assign when scripting.

### Expected outcome
If progressive distillation closes the gap (B/C/D > A = −4.28pp), confirms routing
dynamics are the bottleneck in KD, not just capacity. If it fails, K_iter is
intrinsically resistant to compression — K=12 is a hard requirement.

---

## Resolved/Obsolete

### step84 — Phase-Based Inter-Group Routing
**Status:** GATED on step83. Since step83 was KILLED, step84 is deprioritized.
Revisit only if a future group routing experiment shows gain.
**Design discussion:** LEARNINGS_design_2026_04_07_08.md
