<!-- continued from PENDING_DISCUSSIONS_part1.md -->
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
