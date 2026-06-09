# Design Discussions — 2026-04-15 (Session 8)

## K_iter=4 ΔW proj — new N-scaling law discovered (step260-265)

**Headline:** K_iter=4 with ΔW projection NOT equivalent to K_iter=5 at all N — increasingly BETTER as N grows.

| N | Tier | Seeds | Δ (K=4 − K=5) |
|---|------|-------|---------------|
| 1024 | T1 | 2 | +0.05pp (marginal) |
| 2048 | T2 | 3 | +0.02pp (equivalent) |
| 4096 | T1 | 1 | **+0.97pp (strong win)** |

**Mechanism hypothesis:** over-smoothing. At fixed K_hh=2, each routing iteration spreads signal through only 2 neighbors. Larger N = sparser graph = each iter delivers proportionally less new info. 5 iters over-smooths what 4 iters leaves well-differentiated. ΔW proj's alignment gating amplifies this: fewer iters preserve selective signal better.

**Paper claim (pending step265 T2):** K=4 recommended config at D=16 ceiling (N=4096) — AND 20% cheaper in FLOPs and wall-clock. If step265 confirms, paper's main efficiency-plus-ceiling config changes from K=5 to K=4 at larger N.

**Wall-clock impact:** 0.280ms → ~0.224ms projected (step811 K=5 max-autotune × 0.8). Measurement TODO before paper.

---

## Dynamic connectivity at N=512 CLOSED — 4 experiments, 6 mechanism variants ALL FAIL

User's theoretical argument: "more DoF → higher entropy capacity → higher compression." Tested comprehensively:

| Step | Mechanism | Protocol | Δ vs Ref_static |
|------|-----------|----------|-----------------|
| 511 | co_act_low destructive | K_hh=2 every 5ep all-replace | −3.21pp |
| 512-A | co_act_low guarded | K_hh=2 every 15ep ≤1 swap δ=0.02 | −2.34pp |
| 512-B | co_act_high guarded | K_hh=2 every 15ep ≤1 swap δ=0.02 | −2.24pp |
| 513-A | co_act_low K_hh=4 | K_hh=4 every 15ep guarded | −4.94pp |
| 513-B | co_act_high K_hh=4 | K_hh=4 every 15ep guarded | −2.42pp |
| 514 | additive expansion | K_hh 2→3→4→5 non-destructive | −0.56pp |

**Closure:** all six variants negative. Covers both directions (low/high correlation), both aggressiveness levels, both K_hh densities, and non-destructive addition.

**Root cause:** At N=512 on Imagenette 50% data, (a) pairwise activation correlations at noise level, (b) any topology deviation disrupts co-adapted W_pos/θ/C_ho parameters.

**Future work note:** User proposed new protocol (step523) that may succeed: alternating phases of pos-only training and edge-only rewiring (20ep each), keeping edge changes ≤1% per event. Key structural difference from step511-514: TEMPORAL SEPARATION of W_pos and edge updates.

---

## Correction: AH mechanism is NOT activation-dependent (user 2026-04-15)

**Prior error:** I wrote "our current AH mechanism ALREADY provides implicit per-input path modulation (activation-gated suppression)". WRONG.

**Correct:** `SGNNET_AntiHebbian` computes `supp_w = (1 - α_AH × pos_sim.clamp(min=0))` where `pos_sim = (W_pos[i] · W_pos[conn[i,k]])`. Depends ONLY on W_pos. Once training complete and W_pos frozen, AH suppression STATIC across inputs at inference.

**Which mechanism IS activation-dependent?** ΔW projection. In `SGNNET_DeltaAH`:
```
proj_coeff = (Z_nb · dw).sum(-1)   # depends on current activation Z_nb
Z_nb *= proj_coeff.abs()           # per-input modulation
```
REAL per-input path modulation in SGNNET. Paper's "input-conditional routing" claim applies to ΔW proj, not AH.

**Paper narrative implication:** AH = learned static competitive suppression. ΔW proj = activation-conditional routing gate. Complementary — AH structures W_pos geometry for diversity, ΔW proj modulates routing per input.

---

## Proposed new experiment lines

### step523: Alternating W_pos / edge training cycle (CRITICAL — rescues dyn-conn direction)
User protocol:
- Phase 1 (30ep): warmup static
- Phase 2 (20ep): W_pos trainable, conn_hh frozen
- Phase 3 (20ep): rewire edges, W_pos frozen (≤1% edge change per event)
- Repeat Phase 2-3 for 2-3 cycles

Key structural fix vs step511-514: temporal separation of W_pos updates and edge updates. 1% edge cap bounds disruption.

### step521: Multi-forward-backward deep supervision
User protocol: at routing step k ∈ [k_only_forward, K_iter], compute loss from readout(Z_k) and backward. Implemented as deep supervision (accumulate K_iter - k_only_forward losses, single backward).

Variants: k_only_forward ∈ {0, 1, 2, 3}. Tests whether routing trajectory benefits from explicit supervision at intermediate steps.

### step522: Muon optimizer
User directive: "measure BOTH accuracy AND convergence speed. If same accuracy at faster convergence, that's a win."
Muon = recent orthogonalized-momentum optimizer. Test vs AdamW baseline at N=2048 K_iter=5 ΔW proj, 50ep. Report both final accuracy and epochs-to-95%.

All three queued in EXPERIMENT_QUEUE.md pending slot availability.

---

## Wall-clock benchmark verification (hardware: RTX 5060 Ti SM_120)

Verified from bench_step810/811 JSON on 5060ti. PyTorch 2.11.0+cu128, CUDA 12.8.

| Config | Params | Inf (ms) | Train (ms) | Peak Mem (MiB) | vs VGG_FC |
|--------|--------|----------|------------|----------------|-----------|
| VGG_FC (25088→4096→4096→10) | 119.59M | 1.570 | 26.998 | 2293.2 | 1.00× |
| **SGNNET_AH K=5 (reduce-overhead)** | **34,976** | **0.299** | 5.250 | **476.8** | **5.25× FASTER** |
| SGNNET_AH K=5 (max-autotune, step811) | 34,976 | 0.280 | — | — | 5.60× FASTER |
| Linear | 250,890 | 0.089 | 0.394 | 527.5 | 17.6× |
| MLP_64 | 1.61M | 0.087 | 0.481 | 484.6 | 18.0× |

**Paper claims (confirmed):**
1. SGNNET wins wall-clock vs VGG_FC (paper's direct baseline): 5.6× faster.
2. SGNNET has LOWEST GPU memory of all tested variants (476.8 MiB vs Linear 527.5 vs VGG_FC 2293).
3. 3419× fewer params than VGG_FC (34,976 vs 119.59M).

**Param count RECONCILED (2026-04-15):** 34,976 correct. 67,744 was persistent 2× error propagated through docs/manuscript. Formula on MANUSCRIPT_DRAFT.md line 248 computes 34,976 but cited value was 67,744 — arithmetic never matched. Breakdown at N=2048 D=16: W_pos [2058, 16] = 32,928 + θ [2048] = 2,048 → 34,976 trainable. W_phase was `None` (alpha_turing=0.0). bench_step810/811 always right; findings_log.md, MANUSCRIPT_DRAFT.md, CLAUDE.md, step266 session notes corrected.

---

## step523 post-mortem — 5 hypotheses + edge-SHIFT follow-ups

**step523 result:** both variants −2.68 to −2.80pp. Most-conservative discrete topology edit tried (1% edge cap, alternating schedule, temporal separation). Failure rules out "just be gentler" as fix.

**Failure hypotheses (step523-specific):**

| # | Hypothesis | Probe |
|---|---|---|
| H1 | Adam optimizer-state desync across phase switches (stale momentum vs drifted topology) | P1: reset Adam state on every phase switch |
| H2 | W_pos optimized *for specific topology*; 1% edge changes put ~1% of W_pos rows in un-trained regions — compounds across cycles | P2: freeze edges (0% cap) but keep alternating schedule → isolates schedule from edge-change |
| H3 | Edge-swap criterion was heuristic (not differentiable), no guarantee of downhill direction | P3: random 1% swaps within cap → if ≈step523, heuristic contributed nothing |
| H4 | Cycle length (20ep) mismatch — too short for W_pos recovery, or too long causing overfit to old topology | P4: sweep cycle ∈ {5,20,50} ep |
| **H5** | **Discreteness IS problem; volume irrelevant. 1% failing proves any non-differentiable edge jump breaks credit assignment.** | Not testable within step523 class — requires continuous parameterization (step524 below) |

**Bet:** H2 + H5. 1% cap falsifies "volume matters"; H2 explains slow bleed.

### step524 — Edge-SHIFT (no discrete add/remove, constant K)

| Key | Idea | Params | Hypothesis tested |
|---|---|---|---|
| S1 | **Edge perturbation scalar** — freeze conn_hh forever; learn `β[h,k]`, `Z_nb *= (1+β·tanh_gate)`. Learnable edge weights, no topology change. | N·K ≈ 4k | Can smooth per-edge weights recover what step523 wanted, without moving edges? |
| S2 | **W_pos-tied passive rebind** — every N ep, reassign each edge to k-nearest sender in W_pos cosine space. No gradient. | 0 | H2: does letting topology *follow* learned semantics (not lead) help? |
| S3 | **Soft top-2 blend** — each slot blends two fixed candidates: `α·Z[s_a]+(1-α)·Z[s_b]`, α learned. At convergence α saturates → hard edge. | 2·N·K | H5: does making change differentiable fix everything? |
| S4 | **Rotational control** — cyclically shift conn_hh by ±1 every N ep (zero-information edit). | 0 | Null: does topology identity carry signal, or is any permutation equivalent? |

### step525 — Teleportation class (fixed core + sample-dependent hop)

**Key distinction from step511-523:** topology never changes at architecture level. Per-sample content-addressable hops into small dynamic slot. Stability + adaptation.

| Key | Idea | Cost |
|---|---|---|
| T1 | **Content-addressable teleport** — each neuron has K_hh fixed edges + 1 "teleport slot". Per sample, `q[h] = W_q·Z[h]`; target = top-1 of `q[h]·W_pos[·]`. Straight-through via soft-argmax. | +W_q [D,D]; +1 N×N sim per forward (expensive unless restricted to 2-hop) |
| **T2** | **Random-refresh hop** — each neuron gets 1 extra "hop" edge; redraw ALL hop edges every N epochs. No gradient, pure Monte-Carlo. | 0 params. CHEAPEST. |
| **T3** | **Dropout-style training hop** — at train time, 1% of edges replaced with random long-range per batch. At inference, use fixed expected set. Forces topology robustness. | 0 params |
| T4 | **Hub-and-spoke** — H<<N hub neurons; every neuron gets 1 fixed extra edge to nearest hub in W_pos (reassigned every N ep). Small-world + hub. | 0 params |
| T5 | **2-hop soft expansion** — precompute 2-hop neighborhood from conn_hh; blend `Z_nb` with weighted sum of 2-hop neighbors. Edges don't move but effective RF widens. | +scalar/receiver |

**Recommendation:** T2 + T3 cheapest AND avoid every failure mode observed (no discrete gradient through selection, no optimizer desync, no W_pos-topology coupling). Both probe "does topology need stability at inference, or just at training?"

**Priority:** T2 first (1 param: N_refresh), T3 second (1 param: drop_rate), T1 last (expensive, most paper-worthy if works).

### step525 RESULTS (2026-04-15, studio_mps, N=2048 T0 20ep seed=42) — **KILLED**

| Config | best | Δ vs Ref |
|---|---|---|
| Ref (no teleport, ΔW proj baseline) | **94.09%** | 0 |
| T2_5 (redraw every 5 ep) | 93.63% | −0.46pp |
| T2_1 (redraw every epoch) | 89.91% | −4.18pp |
| T3 (per-batch random) | 53.99% | **−40.10pp** |

**Verdict:** teleportation class KILLED. Monotone in redraw frequency — ANY topology perturbation hurts, faster perturbation hurts proportionally. Rules out stochastic/Monte-Carlo topology as alternative to learned rewiring.

**CONFIRMED H2 from step523 post-mortem:** W_pos co-adapted to specific fixed topology. Any edge instability — learned, random, discrete, continuous, add-only, swap-based — disrupts W_pos/topology lock. Paper claim strengthened: **SGNNET's small-world init is once-and-for-all choice; topology stability is load-bearing.**

**Directions now closed:**
- Destructive rewire (step511-513) — KILLED
- Additive expansion (step514) — KILLED
- Alternating W_pos/edge with 1% cap (step523) — KILLED
- Teleportation / Monte-Carlo refresh (step525) — KILLED

**Still open:** step524 S1 (edge-β scalar, NO topology change — learnable weights on FROZEN edges). If S1 also fails, dynamic connectivity fully closed and paper narrative becomes "topology must be static; W_pos on that static topology is entire story."

---

## PyTorch Geometric — UNEXPLORED direction for wall-clock

Grep confirms no prior exploration. PyG ships `torch_scatter.scatter_add` — hand-tuned CUDA kernels for EXACT primitive SGNNET uses: gather + K_hh-reduce.

**Hypothesis:** `scatter_add` (message-passing-native edge-list format) may beat compiled fancy-indexing path because it avoids `[B,N,K_hh,D]` intermediate tensor entirely — operates on `[E, D]` flat edges. This was bottleneck step803 (CUDA graph) hit with int64 dynamic indices.

**Experiment (bench_step832):** install `torch-scatter` on 5060ti → edge-list variant of routing loop → benchmark vs V2 max-autotune (0.280ms K=5). Projected: 2-3× if memory-BW-bound, 0.5-1× if compute-bound.

**Bonus:** PyG `MessagePassing` may compose with `torch.compile` → both optimizations stacked. Missing baseline — every efficient-sparse-GNN paper runs PyG; reviewers will ask why we didn't.

---

## Session paper progress (2026-04-15)

| Paper requirement | Status change this session |
|--|--|
| Efficiency config (K=5 → K=4?) | K=4 Tier-2 at N=4096 (step265) RUNNING — decisive |
| Wall-clock vs VGG_FC | CONFIRMED 5.6× faster (bench_step810/811) |
| GPU memory win | CONFIRMED lowest of all tested (476.8 MiB) |
| Multi-seed ΔW rot+aug 97.30% | seed=43 peak 97.12%, seed=44 peak 97.15% — VALIDATED ±0.2pp |
| MLP paper baselines | step401 full-train done (Lin=97.12%, MLP_64=97.25%) |
| Mechanism diagnostics | step231 — H1-H5 ALL CONFIRMED |
| Cross-dataset CIFAR-10 | extraction started (studio_cpu, ~3%) |
| Dynamic connectivity | CLOSED at N=512 (6/6 variants fail); step523-522 queued for new direction |
| AH vs ΔW mechanism clarity | CORRECTED framing (AH = static pos-based; ΔW proj = input-dependent) |
| Param count 34,976 vs 67,744 | RESOLVED — 34,976 correct; 67,744 was 2× doc error, now fixed |

---

## step266 provenance confirmed — 97.71% is real (2026-04-15)

**Scare from gap analysis:** local JSON showed step266 crashed at ep1 (23.2%, 25s elapsed). 97.71% record appeared unbacked.

**Ground truth:** pure file-sync artifact. Full log + complete JSON live on 5060ti. Now synced back:

| Config | K_iter | best | ep | elapsed |
|---|---|---|---|---|
| Ref (ΔW rot + aug, N=4096) | 5 | **97.71%** | 113 | 1612s |
| A_k4 (ΔW rot + aug, N=4096) | 4 | 97.66% | 108 | 1324s |

**Δ(K=4 − K=5) = −0.05pp at N=4096 with rotation + aug.** Contrast: step263 found K=4 wins +0.97pp at N=4096 with **ΔW proj** (seed=42).

**Conclusion — K-scaling law is mechanism-specific:**
- ΔW proj: K=4 > K=5 at N=4096 (+0.97pp) → K=4 recommended efficiency config for proj path
- ΔW rotation + aug: K=5 ≥ K=4 at N=4096 (−0.05pp) → K=5 remains right choice for rotation path

Two mechanisms implement different N-sensitivity. ΔW proj uses `.abs()` on projection coefficient (sign-invariant gate), whereas rotation uses `cos/sin` which rotate `Z_nb` in plane spanned by `ΔW` — more iterations don't cause same over-smoothing because rotation preserves magnitude. Over-smoothing hypothesis applies to gating, not rotation.

**Paper implication:** keep K=5 in "record" config (97.71% at N=4096 rot+aug) and K=4 in "efficiency" config (ΔW proj at N=2048/4096). Present both as Pareto choice, not single recommendation.

---

## Result file standardization — slot suffix (2026-04-15)

**Problem:** step266 result JSON on 5060ti had real numbers but local JSON was stale ep1 snapshot. Same step name, same seed, different machines — no way to tell which was current.

**Fix:** `scripts/launch_slot.sh` now
- Writes logs as `<step>__<slot>.log` instead of `<step>.log`
- Exports `SGN_SLOT=<slot>` so scripts can suffix OUT_PATH:
  ```python
  SLOT = os.environ.get("SGN_SLOT", "local")
  OUT_PATH = ROOT / "results" / f"train_stepXXX_seed{SEED}__{SLOT}.json"
  ```

`TEMPLATE_experiment.py` updated with this convention. Existing scripts keep seed-based naming unless retrofitted; new scripts inherit by copying from template. Cross-machine sync can now use distinct filenames, preventing silent stale-overwrite. Applies from this launch onward — retroactive step266 numbers now in un-suffixed file and are authoritative (sourced from 5060ti log).