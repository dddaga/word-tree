# Architecture Decisions

## W_pos must have weight_decay=0.0
**Date:** 2026-03-26
**Context:** Switched from Adam to AdamW for all runs.
**What happened:** AdamW with default weight_decay pulled W_pos toward zero, collapsing all neuron positions to the origin. Proximity routing became degenerate (all neurons at the same point → zero distances → undefined phases). Accuracy dropped from 11.97% to 10.11%.
**Fix:** Always set `weight_decay=0.0` on the W_pos parameter group. W_phase (Stage C) can use normal weight decay.
**Where:** `src/training/trainer.py` param_groups, `src/training/experiment_config.py`

---

## Binary C masks: bool dtype, not float32
**Date:** 2026-03-26
**Decision:** Store as `torch.bool` (1 byte vs 4 bytes = 4× memory saving). Cast to `.float()` at the einsum site.
**Why it matters:** At N=10000, C_hh as float32 would be 10000×10000×4 bytes = 400 MB. Bool = 100 MB.
**Watch out for:** `(~mask).float()` works. But `float(~mask)` fails. Always wrap: `(~mask).float()`.

---

## lambda_safety must scale down with N
**Date:** 2026-03-26
**What happened:** At N=1024, safety_valve Coulomb repulsion exploded. At N=2048 it caused a backward shape crash.
**Why:** r* = 0.5/N^(1/D) shrinks as N grows. More neuron pairs violate it. Repulsion term grows proportionally.
**Fix:** Scale by `(256/N)^(1/D)` relative to the tuned N=256 baseline. In `experiment_config.py::scaled_lambda_safety()`.
**Disable entirely:** above N=5000 the cdist matrix is >10 GB — safety loss returns 0.

---

## D=4 is structurally fixed — not a free hyperparameter (PRE-Fourier encoding)
**Date:** 2026-03-26
**Context:** This was TRUE before Fourier encoding was implemented. NOW OUTDATED — D is a free parameter.
**Why D was fixed at D=4:** `A_input = cat([x, spatial_encoding])` produced [B, N_in, 4]: 1 feature + 3 VGG coords.
**Resolution (2026-03-29):** Fourier encoding (`encoding_mode="fourier"`) uses sinusoidal (h,w,c) at multiple
frequencies and works for any D≥4. D=16 Fourier gives 27.54%+ vs D=4 linear's 20.99%.
**Current default:** D=16, encoding_mode="fourier" for all new experiments.

---

## log1p softening of safety_valve_loss
**Date:** 2026-03-26
**Fix:** Return `torch.log1p(raw)` instead of `raw`. log1p(x) ≈ x for small x, but log1p(25) = 3.26 (caps explosive values).
**Where:** `src/sgnnet/losses.py`

---

## Gradient clipping: target optimizer params only
**Date:** 2026-03-29 (bug discovered during regression RCA)
**Bug:** `clip_grad_norm_(model.parameters(), 1.0)` clips ALL params including theta and W_phase
which are NOT in the optimizer. Their gradients accumulate across batches → clip norm is dominated
by accumulated theta/W_phase grads → W_pos gradient clipped to near-zero → model barely learns.
**Fix:** `clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], norm)`
**Current setting:** `grad_clip_norm=float("inf")` (no clipping) — clipping was not the bottleneck
and inf is mathematically equivalent to no clipping but explicit.
**Where:** `src/training/trainer.py`

---

## Small-world topology: K_random ≥ 2 is required
**Date:** 2026-03-26
**What happened:** K_random=0 produces a disconnected graph. At N=256, G=32 groups: only 3.1% of
neurons reachable from node 0. Each group is an isolated island.
**Fix:** Always use K_random ≥ 2. Achieves ~100% global connectivity with O(log N) path length.
**Benchmark:** K_random=1 → 100% connected, avg_dist=4.73 hops. K_random=2 → avg_dist=3.66.

---

## ProximityWave topology rebuild: min-distance guard needed
**Date:** 2026-03-26
**Why:** W_pos positions nearly coincident (d ≈ 0) produce `1/d → ∞` in phase computation.
**Fix:** In `build_knn_conn`, set `dists[dists < r*/4] = inf` before topk.
**Where:** `src/sgnnet/model_proximity_wave.py::build_knn_conn()`

---

## C_input structure matters more than routing
**Date:** 2026-03-26 (confirmed)
**Result:** Block-local C_input (channel-grouped) gives 17.96% vs random 10.17% at N=256/D=4.
Random projection of 25,088 VGG features averages out discriminative structure before routing.
**Current fix:** `_build_fanin_conn` uses block-local bias (group_size_h partitioning with
proportional input regions). Each neuron group preferentially connects to its input region.

---

## ResonantSGNNet Architecture Summary
**Full spec:** `docs/resonant_sgnnet_spec.md`

Three-layer design evolved from diagnosis results:

**Layer 1 — W_pos (position backbone):** structural wiring via k-NN at rebuild, then decoupled.
Carries Z_fwd = relu(Z - θ) along structural edges with geometric phase shift.

**Layer 2 — W_phase (phase receiver):** separate parameter, receiver-not-broadcaster.
Incoming activation collapsed onto W_phase direction (bandpass filter).
Carries Z_ref = -relu(-(Z + θ)) as long-range inhibition (Turing mechanism).
Beam filtering: only top-M active neurons transmit (beam_size hyperparameter, fixed at 32).

**Layer 3 — Readout:** mean aggregation (not sum) — fixes output scale growing with N.

**Two-scale Turing:** structural (local excitatory) + phase (long-range inhibitory) → spontaneous
non-overlapping feature detectors without explicit diversity loss.

**Safety valve constraint:** auxiliary loss must stay ≤ 15% of task loss in steady state.
Enforced by: bounded quadratic repulsion + soft cap relative to task loss.
