# neuro_graph — Learnings & Key Decisions

Human-readable record of decisions, failures, and hard-won insights.
Written so future contributors (and future AI assistants) don't repeat the same mistakes.

GSD planning artifacts (`.planning/`) contain execution history.
This file captures the *why* and *what went wrong* that never makes it into code comments.

---

## Index

- [Architecture decisions](#architecture-decisions)
- [Training stability](#training-stability)
- [Failed experiments](#failed-experiments)
- [MPS / Apple Silicon specifics](#mps--apple-silicon-specifics)
- [Performance benchmarks](#performance-benchmarks)
- [Open hypotheses under investigation](#open-hypotheses-under-investigation)

---

## Architecture Decisions

### W_pos must have weight_decay=0.0
**Date:** 2026-03-26
**Context:** Switched from Adam to AdamW for all runs.
**What happened:** AdamW with default weight_decay pulled W_pos toward zero, collapsing all neuron positions to the origin. Proximity routing became degenerate (all neurons at the same point → zero distances → undefined phases). Accuracy dropped from 11.97% to 10.11%.
**Fix:** Always set `weight_decay=0.0` on the W_pos parameter group. W_phase (Stage C) can use normal weight decay.
**Where:** `src/training/trainer.py` param_groups, `src/training/experiment_config.py`

---

### Binary C masks: bool dtype, not float32
**Date:** 2026-03-26
**Context:** C_input, C_hh, C_ho were originally stored as float32 (0.0/1.0).
**Decision:** Store as `torch.bool` (1 byte vs 4 bytes = 4× memory saving). Cast to `.float()` at the einsum site.
**Why it matters:** At N=10000, C_hh as float32 would be 10000×10000×4 bytes = 400 MB just for one mask. Bool = 100 MB.
**Watch out for:** Expressions like `(~model.C_input_mask).float().mean()` — the `~` on bool returns bool, then `.float()` works. But `float(~mask)` fails. Always wrap: `(~mask).float()`.

---

### lambda_safety must scale down with N
**Date:** 2026-03-26
**Context:** GA tuned lambda_safety=0.691 for N=256 in a D=4 unit box.
**What happened:** At N=1024, the safety_valve Coulomb repulsion exploded. At N=2048 it caused a backward shape crash.
**Why:** r* = 0.5/N^(1/D) shrinks as N grows. r_repel = r*/2 becomes tiny. More neuron pairs violate it. The mean repulsion term grows proportionally.
**Fix:** Scale by `(256/N)^(1/D)` relative to the tuned N=256 baseline. Centralised in `src/training/experiment_config.py::scaled_lambda_safety()`.
**Disable entirely:** above N=5000 the cdist matrix is >10 GB — safety loss returns 0.

---

### D=4 is structurally fixed — not a free hyperparameter
**Date:** 2026-03-26
**Context:** H3 in the ceiling diagnosis initially proposed testing D=8, D=16 for richer output.
**Why D is fixed:** `A_input = cat([x.unsqueeze(-1), spatial_encoding], dim=-1)` always produces a [B, N_in, 4] tensor: 1 feature value + exactly 3 VGG spatial coordinates (channel_norm, h_norm, w_norm). Changing D to 8 would require the spatial encoding to emit 7 coordinates — it only has 3 meaningful ones (C, H, W axes of VGG pool5).
**Implication:** The 4 dimensions are fully semantically occupied at seeding: [intensity, which_channel, spatial_row, spatial_col]. W_out[c] — the output class "trajectory" — is a direction in this [intensity, channel, row, col] space. 40 parameters (D=4 × 10 classes) is the correct capacity given the 4D structure.
**Where to look if readout seems weak:** The problem is in *what activations are being readout*, not the readout mechanism itself. Fix the seeding (H1) and depth (H2) first.

---

### log1p softening of safety_valve_loss
**Date:** 2026-03-26
**Context:** Even with lambda scaling, very large N or unlucky initialisation could push safety loss to 25+, yielding gradient norms of 614,000.
**Fix:** Return `torch.log1p(raw)` instead of `raw`. log1p(x) ≈ x for small x (no distortion at well-separated configs), but log1p(25) = 3.26 (caps explosive values). Smooth and differentiable.
**Where:** `src/sgnnet/losses.py`

---

### Gradient clipping: universal stability
**Date:** 2026-03-26
**Context:** Any loss term explosion (safety valve, degenerate routing edge, reconnection artifact) could corrupt W_pos in a single step and be unrecoverable.
**Fix:** `clip_grad_norm_(model.parameters(), max_norm=1.0)` applied every step in `Trainer.train_epoch()`. Works with GradScaler: unscale first, then clip.
**Where:** `src/training/trainer.py`. Controlled via `grad_clip_norm` parameter (default 1.0).

---

### Small-world topology: K_random ≥ 2 is required
**Date:** 2026-03-26
**Context:** Built SGNNET_SmallWorld with Watts-Strogatz-style connectivity. Tested K_random=0 (pure local groups).
**What happened:** K_random=0 produces a disconnected graph. At N=256, G=32 groups: only 3.1% of neurons reachable from node 0. Each group is an isolated island — information never crosses group boundaries.
**Fix:** Always use K_random ≥ 2 (2 long-range shortcuts per neuron). This achieves ~100% global connectivity with O(log N) average path length.
**Benchmark result:** K_random=1 → 100% connected, avg_dist=4.73 hops. K_random=2 → avg_dist=3.66.

---

### ProximityWave topology rebuild: min-distance guard needed
**Date:** 2026-03-26
**Context:** conn_hh is rebuilt every `reconnect_every` epochs from current W_pos via k-NN. After the first rebuild (epoch 10), ProximityWave N=512 exploded (train_loss → 27,831).
**Why:** W_pos positions that are nearly coincident (d ≈ 0) produce `1/d → ∞` in the phase computation `φ = 2π·d/λ`. k-NN can select near-zero distance neighbours if neurons have collapsed.
**Fix:** In `build_knn_conn`, set `dists[dists < r*/4] = inf` before topk, so degenerate near-zero edges are never selected as neighbours.
**Where:** `src/sgnnet/model_proximity_wave.py::build_knn_conn()`

---

### C_input structure matters more than routing
**Date:** 2026-03-26 (hypothesis, under test)
**Context:** Phase 4 Stage B (phasor routing) only reached 11.97% top-1 at N=256. SmallWorld (no phases) at N=512 reached 16.64%.
**Hypothesis:** The Phase 4 C_input is a random binary mask. Random projection of 25,088 VGG features averages out discriminative structure before routing even begins. Block-local or channel-grouped wiring preserves spatial/semantic structure.
**VGG pool5 structure:** [512 channels × 7×7 spatial] = 25,088. Each channel encodes a specific visual pattern. Random C_input mixes all channels together for every neuron.
**Under test:** `scripts/diagnose_ceiling.py` — ablation across random / block-local / channel-grouped C_input, K_iter depth, and D dimensionality.

---

## Training Stability

### Early stopping monitors train_loss, not val_loss
**Reason:** Validation set is only 3,925 samples across 10 classes. During GA search (15% subset), val set is ~580 samples — too small for stable val_loss signal. Train_loss is more reliable for plateau detection.

### ReduceLROnPlateau: patience=10, factor=0.5
Halves LR when train_loss doesn't improve for 10 epochs. `min_lr=1e-7` prevents full stop.

### FP16 vs FP32: no meaningful difference for this task
**Date:** 2026-03-26
Trained identical configs at 150 epochs: FP32 12.00%, FP16 11.97%. The task is not compute-bound enough to benefit from precision reduction. Use FP16 for speed, not accuracy.

---

## Failed Experiments

### N_hidden=2048 crash: backward shape mismatch
**Date:** 2026-03-26
**Error:** `RuntimeError: shape mismatch: value tensor of shape [4233306] cannot be broadcast to indexing result of shape [5306017]`
**Root cause 1:** `strength.fill_diagonal_(0)` in `wave_routing.py` — in-place mutation on a tensor that's part of the computation graph. PyTorch autograd tracks in-place ops; at large N the resulting gradient shape diverges.
**Root cause 2:** `dists[mask]` boolean indexing in `losses.py` — boolean advanced indexing creates a flat 1D tensor in the graph whose backward shape doesn't match cdist's [N,N] output at large N on MPS.
**Fix 1:** `strength = strength * (1.0 - torch.eye(N_hidden, device=...))` (out-of-place)
**Fix 2:** Element-wise mask multiply instead of boolean indexing.

### N_hidden=1024 loss divergence (16720 → 125501)
**Date:** 2026-03-26
**Context:** Before the fill_diagonal_ fix, N=1024 trained stably for 25 epochs then catastrophically diverged.
**Root cause:** The in-place mutation corrupted gradients silently for ~25 epochs before accumulation caused visible explosion.
**Lesson:** In-place operations on tracked tensors are silent bugs in autograd — they don't error immediately but accumulate graph corruption over time.

### ProximityWave N=512 explosion at epoch 25
**Date:** 2026-03-26
**Error:** train_loss jumped from 2.96 (epoch 2) to 27,831 (epoch 25).
**Root cause:** First topology rebuild at epoch 10 (reconnect_every=10) selected near-zero distance edges, causing `1/d → ∞` in phase computation.
**Fix:** Min-distance guard in `build_knn_conn` + gradient clipping.

### metalcompute pip install failed (Python 3.14)
**Date:** 2026-03-26
**Context:** Wanted to run Metal shaders from Python for sparse kernel benchmarks.
**Error:** Swift bridging module compilation error on Python 3.14.
**Workaround:** Write a standalone Swift script using `MTLDevice.makeLibrary(source:)` directly. No Python bridge needed for benchmarking.
**File:** `src/metal/run_sparse_bench.swift`

---

## MPS / Apple Silicon Specifics

### PYTORCH_ENABLE_MPS_FALLBACK=1 is required
Some ops (e.g. cdist backward, segment_reduce) are not implemented on MPS and silently fall back to CPU if this env var is set. Without it they crash. Always set in experiment scripts.

### CSR sparse tensors: not supported on MPS
`torch.sparse_csr_tensor` operations fail on MPS. Use COO (scatter_add) or fixed fan-in (conn_idx gather) instead.

### Boolean indexing backward: crashes at large N
`tensor[bool_mask]` creates a flat 1D graph node. At large N on MPS, its backward shape diverges from the source tensor's shape. Use element-wise `tensor * float_mask` instead.

### Fixed fan-in gather is not bandwidth-limited by precision
FP16 gives no speedup over FP32 for `Z[:, conn_idx, :].sum(dim=2)`. The operation is random-access memory (cache-miss bound), not sequential bandwidth. Halving data size doesn't help latency.

### MPS sparse algorithm benchmarks (N_in=25088, N_h=256, B=64)
| Method | Time | Notes |
|---|---|---|
| Dense einsum | 3.92 ms | Baseline |
| Fixed fan-in K=100 | 2.33 ms | ~1.7× faster |
| Grouped block-sparse | 0.49 ms | **8× faster** |
| Metal naive_fanin | 0.50 ms | Custom shader |
| COO scatter_add | 124 ms | 37× slower — avoid |
| CSR | N/A | Not supported on MPS |

---

## Performance Benchmarks

### Forward pass scaling (C_hh routing, K=6, B=64, D=4, 3 iterations)
| N | FP32 ms | Notes |
|---|---|---|
| 256 | 0.37 | — |
| 1024 | 1.54 | — |
| 4096 | 5.20 | — |
| 10000 | 12.71 | ~1.3 ms per 1000 neurons |

### Precomputed 2-hop adjacency: 1.65× faster
Replacing 3 sequential K=6 gather steps with 1 step of K=12 (2-hop precomputed neighbours) is 1.65× faster. Only valid for inference — training needs sequential steps for gradient flow through iterations.

---

---

## Translation Invariance: Gap & Fixes

### SGNNET does not have translation invariance — and why
**Date:** 2026-03-26

The seeding step embeds spatial position directly into activations:
```
A_input[b, i] = [x[b,i], channel_norm[i], h_norm[i], w_norm[i]]
```
A golf ball at (row=2, col=3) and the same golf ball at (row=5, col=1) produce
activations in different D-space positions and activate different hidden neurons.
C_input is fixed, so there is no mechanism to recognise them as the same object.

**CNNs get translation invariance from weight sharing** — the same filter applied at
every position. SGNNET has no weight sharing in the connectivity.

### Why dynamic connectivity alone doesn't solve it
Proximity routing connects neurons based on their learned W_pos positions. If W_pos
is in a semantic space, routing *could* aggregate features regardless of position. But
by the time routing fires, spatial coordinates are already embedded in the activations.
Routing works on position-contaminated data.

**The root conflict:** D=4 is doing two jobs at once — encoding feature content AND
spatial origin. These should be separated.

| Dim | Current | For invariance |
|-----|---------|----------------|
| D[0] | feature value | keep |
| D[1] | VGG channel (semantic) | keep |
| D[2] | spatial row | remove or pool |
| D[3] | spatial col | remove or pool |

### Fixes (ordered by invasiveness)

**Fix B — Channel-only C_input fan-in (recommended first step)**
Each hidden neuron samples from *all 49 spatial positions of one VGG channel*,
not a fixed spatial block. A_input drops h/w coordinates. Result: the same feature
activates the same neurons regardless of where it appears in the image.
This is tested in `diagnose_ceiling.py` as `cinput_mode='channel'`.

**Fix C — Routing in semantic space (requires Fix B)**
With channel-based seeding, W_pos organises neurons by feature semantics.
Proximity routing then aggregates evidence from semantically similar neurons,
not spatially nearby ones. Dynamic connectivity becomes genuinely content-driven.

**Fix D — Learnable spatial attention (keeps spatial context when useful)**
Replace hard-coded h/w coordinates with a learnable attention mask over input
positions: `Z[h] = sum_i attn[h,i] * x[i]`. With broad-attention regularisation,
the network approximates translation invariance while retaining spatial reasoning.

**Avoid Fix A** (remove all spatial coords) — completely loses spatial relationships
between features; likely hurts accuracy on classes with strong spatial structure.

---

## Open Hypotheses Under Investigation

### Why is static accuracy capped at ~11-16%?
Three candidates under test in `scripts/diagnose_ceiling.py`:

**H1 — Random C_input destroys discriminative structure**
VGG pool5 = [512ch × 7×7]. Random wiring sums 2500 random input dims per neuron. Channel-grouped wiring (each neuron group → one VGG channel) may preserve semantic features.
*Test:* random vs block-local vs channel-grouped C_input at N=512, 60 epochs.

**H2 — Too few routing iterations**
GA selected K=2 (= 1 hidden iteration). With sparsity=0.90, one C_hh hop reaches ~25 neurons. 3 hops reach the 3-neighbourhood.
*Test:* K_iter = 1, 3, 5 at N=512 with block-local C_input.

**H3 — K_in fan-in size: how many input neurons does each hidden neuron see?**
D=4 is structurally fixed: 1 feature value + 3 VGG spatial coordinates (channel_norm, h_norm, w_norm) from `compute_spatial_encoding`. Increasing D would require redesigning the encoding — extra dimensions are zeros at seeding and contribute nothing.
The real fan-in question: K_in=50 may be too few or too many connections from the 25,088 input pool per hidden neuron.
*Test:* K_in = 10, 50, 200 with block-local C_input, K_iter=3.

### Signal reflection routing (Exp 4)
Strongly negative activations reflected back to source rather than propagating:
```
Z_new[h] = -relu(-Z[h] - θ) + Σ_k weight[h,k] * relu(Z[k])
```
Expected: sparse wavefronts, input-dependent routing channels, self-inhibiting dead zones.
Risk: dying neuron cascade if θ too low. Mitigation: leaky α=0.1 variant.
*Planned:* `scripts/train_exp4_reflection.py` (Plan 05-04).

---

## ResonantSGNNet Architecture (2026-03-26)

Full spec: `docs/resonant_sgnnet_spec.md`

Three-layer design evolved from diagnosis results (H1: block/channel C_input = 17.96% vs random = 10.17%):

**Layer 1 — W_pos (position backbone):** structural wiring via k-NN at rebuild, then decoupled.
Carries Z_fwd = relu(Z - θ) along structural edges with geometric phase shift.

**Layer 2 — W_phase (phase receiver):** separate parameter, receiver-not-broadcaster.
Incoming activation collapsed onto W_phase direction (bandpass filter).
Carries Z_ref = -relu(-(Z + θ)) as long-range inhibition (Turing mechanism).
Beam filtering: only top-M active neurons transmit (beam_size hyperparameter, annealed).

**Layer 3 — Readout:** mean aggregation (not sum) — fixes output scale growing with N.

**Joint repulsion:** W_joint = cat([W_pos, W_phase]) in 2D-dimensional space.
Pauli-like exclusion: neurons must differ in position OR phase (or both).
r_star_2D larger than r_star_D → less aggressive safety valve → N=1024 plateau may resolve.

**Safety valve constraint:** auxiliary loss must stay ≤ 10% of task loss in steady state.
Enforced by: log1p softening + lambda scaling + annealing + hard ceiling if safety > 0.5 * task.

**Simulated annealing:** tau(epoch) couples gate temperature + lambda_safety + beam_size.
Early: hot (explore, spread). Late: cold (commit, crystallize). Mirrors crystal formation from melt.

**Two-scale Turing:** structural (local excitatory) + phase (long-range inhibitory) → spontaneous
non-overlapping feature detectors without explicit diversity loss.
