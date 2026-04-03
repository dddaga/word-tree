# SGNNET Experiment Queue

Persistent record of all planned experiments, their motivation, status, and priority.
Updated after each discussion or result. Drives session continuity across compactions.

**Rule:** All experiments are ablation-structured — one change at a time, then combine
if both individual contributions are positive. Log results in LEARNINGS_phase*.md.

---

## Status Legend
- `DONE` — completed, results in LEARNINGS
- `RUNNING` — currently in progress
- `QUEUED` — ready to implement and run
- `PLANNED` — design agreed, not yet scripted
- `IDEA` — discussed, not yet fully designed

---

## Currently Running

### Step 56: N-scaling sweep (QUEUED for dispatch)
**Script:** `scripts/train_step56_n_scaling.py`
**Log:** `logs/train_step56_n_scaling.log`
**Sweep:** N=[512, 1024, 2048, 4096, 10000] at D=64 K_iter=8 AntiHebb alpha=0.7 wpos
**Status:** Script synced to Mac Studio, awaiting slot (3 experiments running: step29c, step48, step54)
**Key:** Does accuracy plateau at N=1024, or does scaling N help at D=64?

### Threshold / geo sweep (MPS)
**Script:** `scripts/train_thresh_sweep.py`
**Log:** `logs/train_thresh_sweep.log`
**Sweep:** A=thresh0.0+geo, B=thresh0.1+geo, C=thresh0.3+geo, D=thresh0.0+no-geo (iter1 control)
**Key:** Config D (pure dynamic_z) is the regression diagnostic

### 200ep cosine baseline (CPU)
**Script:** `scripts/train_aug_baseline.py --sched cosine --epochs 200 --device cpu`
**Log:** `logs/train_aug_baseline_cosine_200ep.log`
**At:** ~ep130/200

---

## Completed

### LR schedule comparison — DONE (2026-03-28)
**Result:** plateau=19.92%  cosine=19.49%  (Δ=−0.43%, plateau marginally wins)
**Conclusion:** No meaningful difference. Both peaked at e80-90 then declined as LR
collapsed toward zero. Root cause: with 9,469 samples and a slowly-improving loss,
ReduceLROnPlateau fires every ~10 epochs until LR is tiny; cosine reaches eta_min by e150.
**Decision:** Keep plateau as default. Reduce epochs to 120 (one LR drop, then stop).
The bottleneck is data size, not LR schedule — augmented data is the higher-leverage fix.

### Iter2: Dynamic connectivity ablation — DONE (2026-03-28)
**Results (90 epochs CPU):**
```
A. dynamic_z (reference)         23.87%
B. dynamic_z_thresh (thr=0.3)    22.93%  −0.94%
C. dynamic_z_geo (γ=1.0)         22.83%  −1.04%
D. dynamic_z_full (thr+geo)      24.25%  +0.38%  ← best
E. dynamic_z_full N=1024         18.45%  (underconverged at 72ep)
```
**Conclusion:** Individual components both slightly hurt; combination slightly helps.
Interpretation: geo reshapes score distribution so threshold cuts more meaningfully
(distant low-similarity connections pruned, local structured routing emerges).
Margins are within single-seed noise — cannot conclude strongly either way.
N=1024 result invalid (too few epochs). **Use dynamic_z_full as default going forward.**
**Iter1 reference (120ep MPS):** 26.52% — gap explained by 90ep CPU vs 120ep MPS + seed variance.

---

## Queued (implement in order)

### Training infrastructure
**Status:** DONE (scripts written, running)
- `scripts/extract_features_augmented.py` — 2× augmented extraction (original + hflip → 18,938 train; rand_crop excluded — can lose class subject)
- `scripts/test_cosine_lr.py` — cosine vs plateau LR schedule comparison
- `src/training/trainer.py` — `sched_type="cosine"` option added
- `src/training/experiment_config.py` — `trainer_kwargs(n_epochs, sched_type)` updated
**Next:** Run augmented extraction, integrate winners into all future ablations.

---

### Routing dropout ablation
**Motivation:** Standard dropout zeros activations and rescales (inverted dropout) to
preserve expected magnitude. In SGNNET, l2-normalisation after each routing step
handles rescaling automatically — no inverted scaling needed. Dropout of full neuron
Z-vectors (not individual dims) during routing forces the model to route robustly
under partial connectivity, and at test time the full graph acts as an implicit ensemble.

**Key design note:** Drop entire Z[h] vectors (not individual dimensions — zeroing dims
corrupts the geometric direction). Apply only during `_route()`, not seed or readout.
No inverted scaling — l2-normalize handles it. No threshold on rescaling needed.

**Configs (best LR schedule, augmented data, 150 epochs, N=512 D=4 dynamic_z_geo):**
```
dropout_p=0.0   — baseline (no dropout)
dropout_p=0.1   — 10% neuron dropout per routing step
dropout_p=0.2   — 20% neuron dropout per routing step
dropout_p=0.3   — 30% neuron dropout per routing step
```
**What this tells us:** Does routing dropout add regularisation that improves val accuracy?
Does it reduce the train/val gap? Does it hurt by destroying too much signal per step?

---

## Queued (implement in order after iter2 completes)

### Group A: Interneuron fraction sweep
**Motivation:** Currently all N neurons receive direct input AND vote on output.
Introducing interneurons (neurons with no direct input connection) creates a
compression bottleneck — input neurons process local features, interneurons
integrate and mix. The most interesting variant: only interneurons vote on output
(input neurons are excluded from readout), forcing the model to compress through
the interneuron bottleneck before classifying.
Analogous to cortical interneurons that never receive direct sensory input.

**Key variant:** `readout_from=interneurons_only` — input neurons seed routing
but C_ho_mask connects only to interneurons. Interneurons must learn to represent
class-relevant integrated signal.

**Configs (N=512, D=4, best dynamic mode from iter2, 90 epochs CPU):**
```
frac_seeded=1.0  readout=all          — baseline (current)
frac_seeded=0.75 readout=all          — 25% interneurons
frac_seeded=0.50 readout=all          — 50% interneurons
frac_seeded=0.50 readout=interneurons — 50% int., output only from int. neurons
frac_seeded=0.25 readout=interneurons — 75% int., output only from int. neurons
```
**What this tells us:** whether the direct-input-to-readout shortcut is hurting by
bypassing the mixing capacity of routing.

---

### Group B: Connection density sweep
**Motivation:** Current K_hh=6 (K_local=4 + K_random=2). Is the graph too sparse?
Does denser local connectivity help more than more long-range shortcuts?
Also test K_iter=5 (deeper routing at same density) vs K_iter=3.

**Configs (N=512, D=4, best mechanism, 90 epochs CPU):**
```
K_local=4,  K_random=2, K_iter=3  → K_hh=6   (baseline)
K_local=8,  K_random=2, K_iter=3  → K_hh=10  (+67% local)
K_local=4,  K_random=8, K_iter=3  → K_hh=12  (+100% long-range)
K_local=8,  K_random=8, K_iter=3  → K_hh=16  (+167% all)
K_local=4,  K_random=2, K_iter=5  → K_hh=6   (deeper, same density)
K_local=4,  K_random=2, K_iter=10 → K_hh=6   (very deep)
```
**What this tells us:** whether we're bottlenecked by graph sparsity (density helps)
or by something deeper (density doesn't help).
**Compute note:** K_hh=16 is ~2.7× cost of K_hh=6 per routing step.

---

### Group C: Scale N and D
**Motivation:** Ceiling still rising at N=1024. D=4 severely limits the number of
distinguishable neuron directions (S³ vs S⁷ vs S¹⁵). Both axes need exploration.

**How D affects input encoding (must update encoding.py):**
Current D=4: A_input[k] = [feature_value, channel_norm, h_norm, w_norm]
For D>4: use sinusoidal Fourier embedding of spatial coordinates at multiple frequencies
so each (h, w, channel) position maps to a distinct, near-orthogonal seed direction.

```
D=4  (current):  [f, h/6, w/6, c/511]
D=8:  [f, sin(h·π/6), cos(h·π/6), sin(w·π/6), cos(w·π/6),
          sin(c·2π/511), cos(c·2π/511), f_sign·log(1+|f|)]
D=16: full multi-frequency: 2 freqs per spatial axis (h,w,c) + 4 feature encodings
```
After l2-normalisation the seed lands on S^(D-1). Routing remains the same.

**Configs (best mechanism + best interneuron/density from A+B, 90 epochs CPU):**
```
N=512,  D=4   — reference (current best)
N=1024, D=4   — pure N scale (already done; reference point)
N=2048, D=4   — large N
N=4096, D=4   — very large N
N=512,  D=8   — same N, richer directions
N=512,  D=16  — same N, much richer directions
N=1024, D=8   — combined scale
N=2048, D=8   — larger combined
```
**What this tells us:** whether to invest in more neurons or richer representations.
Hypothesis: D=8 at N=1024 will outperform D=4 at N=2048 on accuracy/compute ratio.

---

### Group D: Sequential / Liquid state input
**Motivation:** Currently all 25,088 inputs are seeded simultaneously (stateless
per sample). VGG pool5 is a 7×7×512 spatial feature map — it has inherent spatial
structure. Feed it as T sequential spatial segments, allowing interneurons to
accumulate state across segments.

**The liquid state property:** interneurons are never reset between segments.
After segment t, they carry a summary of all previous segments. Segment t+1 is
injected into input-connected neurons, which then route into the persistent
interneuron state. This is exactly the Liquid AI / LTC model of input-driven
state evolution, but on a spatial domain rather than temporal.

**Key design questions (resolve via ablation):**
1. Segment ordering: raster scan (top-left → bottom-right), random, proximity?
2. Injection method: additive (Z += alpha * new_seed) or replace (Z_input_neurons = new_seed)?
3. K_iter between segments: 1, 3, or 5 routing rounds per segment?
4. Does positional encoding already give this "for free" or does sequential processing
   add something beyond having all positions encoded simultaneously?

**Configs (N=1024, D=8, best from A+B+C, 90 epochs CPU):**
```
baseline         — full simultaneous seeding (reference)
T=4 segments     — 4 quadrants of 7×7, raster order, additive injection
T=7 segments     — 7 row-strips, raster order, additive injection
T=49 segments    — one spatial position at a time (maximum sequential)
T=4 random-order — same segments, random order each sample (tests ordering sensitivity)
```
**When to run:** after Group C confirms best N and D. This changes the forward pass
structure significantly — needs a stable model first.

**Connection to prior art:**
- Vision Transformer: all patches in parallel; ours is sequential with persistent state
- Liquid Time-Constant Networks (LTC): ODE-driven hidden state; ours is discrete routing steps
- State Space Models (Mamba/S4): linear recurrence over sequence; ours is graph recurrence

---

## Additional Planned Experiments

### Group E: Polar/angular input compression (TurboQuant-inspired)
**Status:** IDEA
**Motivation:** VGG pool5 outputs a 25088-dim float32 vector. TurboQuant/PolarQuant
(Google Research, 2026) achieves near-lossless compression at 3-4 bits per value by
converting high-dimensional vectors to polar coordinates (magnitude + angles) and
aggressively quantizing the angular part. Applied to our input, this could:
  1. Reduce storage/preprocessing cost (100KB/sample → 12-16KB)
  2. Force the model to learn from a more structured input representation
  3. Act as a form of geometric regularization on the input space

The key insight: SGNNET already compresses via K_in=50 sparse connections (each
neuron sees 50 of 25088 features). Polar compression would operate at a DIFFERENT
level — on the feature values themselves, not on which features are selected.

**Experiment design (after regression RCA resolves):**
A. Polar-quantized input (3-bit angles): preprocess store.h5 → store_polar3.h5
   - Apply recursive polar decomposition to each 25088-d sample
   - Quantize angles to 3 bits, preserve magnitude in float16
   - Train identical SGNNET on compressed features → compare accuracy
B. Polar-quantized input (4-bit): same as A with 4-bit angles
C. Polar positional encoding: instead of linear [h, w, c] → [sin(θ), cos(θ), r, feature]
   where (r, θ) are polar coordinates of (h/7, w/7) in the 2D spatial plane
   D=4: [r_spatial, sin(θ_spatial), cos(θ_spatial), feature_value]
   This naturally encodes spatial proximity vs. directionality.

**What this tells us:** Is VGG's 25088-d representation more compressible than its
dimensionality suggests? Does polar structure in the input encode useful signal for
geometric routing?
**When to run:** after regression RCA resolves and a stable baseline is established.

---

### Group F: Atomic energy model (alternative safety valve)
**Status:** IDEA
**Motivation:** Current safety valve uses a box [0,1]^D with pair-wise repulsion and
wall repulsion. An alternative inspired by atomic physics:
  - **Positive charge at origin (0^D):** attracts all neurons toward center
  - **Neurons as negative charges:** repel each other (preserves spread)
  - **Near-origin barrier:** if a neuron gets too close to origin (within r_min),
    it repels the center charge too — prevents collapse to a single point

This creates a natural equilibrium: all neurons settle on a spherical shell at
the Bohr radius analog where attraction and repulsion balance. The distribution
is spherically symmetric rather than box-bound, and the "clamp to box" step
is replaced by "clamp to annulus" (r_min ≤ ||W_pos|| ≤ r_max).

**Mathematical form:**
  E_attract = sum_i alpha_a * ||W_pos[i]||²                     (harmonic, soft)
  E_repel   = sum_{i<j} relu(r_star - ||W_pos[i] - W_pos[j]||)² / r_star²  (as before)
  E_barrier = sum_i relu(r_min - ||W_pos[i]||)²                 (near-origin wall)
  Total     = E_attract + lambda_repel * E_repel + lambda_barrier * E_barrier

  With alpha_a chosen so equilibrium radius ≈ 0.5 (center of box).

**Why interesting:** The spherical constraint is arguably more natural for geometric
routing where neuron direction (unit vector) carries meaning. Currently, neurons in
a box may cluster near corners or walls — the atomic model pushes them to a shell
where all directions are equally accessible.

**Implementation:** New `atomic_energy_loss` function in `src/sgnnet/losses.py`.
Replace `W_pos.clamp_(0, box_size)` with `W_pos.div_(W_pos.norm(dim=-1, keepdim=True).clamp(min=r_min))`
to keep neurons on/outside the inner shell.

**Ablation configs (N=512, D=4, dynamic_z_geo, 120ep MPS):**
```
A. Current box safety valve (baseline)
B. Atomic model, alpha_a=0.1 (weak attraction)
C. Atomic model, alpha_a=0.5 (moderate attraction)
D. Atomic model, alpha_a=1.0 (strong attraction — neurons cluster on thin shell)
```
**When to run:** after regression RCA resolves; add atomic_energy_loss to losses.py.

---

### Geometric-biased attention (from iter2 results)
**Motivation:** `dynamic_z_geo` isolates whether adding W_pos to routing score helps.
If iter2 shows geo bias contributes, the next step is to also add geo bias to the
*static structural routing* (conn_hh), not just the dynamic phase routing.
**Depends on:** iter2 results
**Priority:** after Group A

### Learned input connection weights
**Motivation:** Currently input projection is a fixed random gather+sum (no learnable
weights on conn_in). Adding a scalar or D-dim learnable weight per input connection
would add N × K_in = 512 × 50 = 25,600 parameters — still tiny, but gives the model
the ability to select and reweight its local input pool.
**Concern:** may distort the spatial locality structure (neurons learn to ignore their
local chunk and attend globally).
**Priority:** after Group C (scale experiments first to confirm ceiling is capacity-bound)

### Multi-seed (3 seeds per config)
**Motivation:** Single-seed runs at N=512/60 epochs show top1 ranging 13.9–26.5%
across different runs. Need multi-seed to confirm any result with <1% margin.
**When:** run for final winning config before writing up.

---

## Completed Experiments

### Step 1: Safety valve + norm mode sweep (2026-03-27)
**Result:** l2 norm +13% over masked. Bounded quadratic valve fixes N-scaling blowup.
**Details:** `learnings/LEARNINGS_phase5.md` Step 1

### Step 2: W_phase receiver (2026-03-27)
**Result:** Static phase graph -0.2% to -1.1%. Hypothesis revised: static graph = noise.
**Details:** `learnings/LEARNINGS_phase5.md` Step 2

### Step 3: Routing mechanisms (2026-03-27)
**Result:** learnable θ (+6.6%) > turing (+6.5%) > threshold (+4.2%) > reflection (+2.6%)
**Details:** `learnings/LEARNINGS_phase5.md` Step 3

### Step 4: Simulated annealing (2026-03-27)
**Result:** Fixed beam=32 wins. Beam annealing irreversibly reduces capacity.
**Details:** `learnings/LEARNINGS_phase5.md` Step 4

### Iter1: Combined Resonant model (2026-03-27)
**Result:** dynamic_z 26.5% > resonant 25.3% > dynamic_gate 25.1% > baseline 24.4%
N=1024 resonant: 29.0%. dynamic_z still climbing at e120 — not converged.
**Key finding:** Topology change per input matters more than weight modulation on fixed topology.
**Details:** `learnings/LEARNINGS_phase5.md` Combined Run section

---

## Priority Order (current)

```
1. [RUNNING]  Augmented feature extraction (store_aug.h5 — 2× data)
2. [DONE]     Aug baseline (2026-03-29): cosine 18.96% vs plateau 18.68% (Δ=+0.28%). Both collapsed to 9.86% at ep120 — LR hits eta_min. Best around ep60-80. Regression vs iter1 26.52% — likely geo+thresh=0.3 over-pruning on aug data.
3. [DONE]     Routing dropout (2026-03-29): p=0.1 best=18.75% (+0.08% vs baseline). Not meaningful. routing_dropout_p=0.0 going forward. Confirms regression is real.
4. [DONE]     Threshold/geo sweep (2026-03-29): ALL configs identical at 18.96% ep60 — A(thresh=0.0,geo), B(thresh=0.1,geo), C(thresh=0.3,geo), D(thresh=0.0,no-geo). Architecture irrelevant. Bottleneck = aug data distribution.
6. [DONE]     Original-data control (CPU, 2026-03-29): all 4 configs identical 18.80% at ep69. Architecture differences vanish. Root cause = LR decay in refactored trainer, NOT aug data or model config.
7. [DONE]     Constant-LR diagnostic: also gave 18-19%. LR decay NOT the root cause.
8. [RUNNING]  Bugfix verification (CPU): trainer grad-clip bug fixed (was clipping all params incl. non-optimizer theta/W_phase with accumulating grads). Now clips only W_pos. sched=none, 90ep, store.h5. Expected: config D recovers to ~23%+.
5. [DONE]     200ep cosine baseline (2026-03-29): top1_best=20.36% at ep58. < 22% threshold. Regression is in model config, NOT epoch count. 120ep stays as standard.
4. [QUEUED]   Encoding + D sweep (train_encoding_D_sweep.py — fourier vs linear, D=4/8/16)
5. [QUEUED]   Group A: Interneurons (implement next)
6. [QUEUED]   Group B: Connection density
7. [QUEUED]   Group C: N and D scale sweep
8. [PLANNED]  Group D: Sequential / liquid state input
9. [PLANNED]  Multi-seed confirmation of winning config
```

---

## Open Research Questions (to resolve via experiments)

1. Is the ~30% ceiling at N=1024 a **capacity limit** (D=4 too small) or a
   **routing quality limit** (dynamic connectivity not good enough)?
   → Group C D-sweep will answer this

2. Does **interneuron bottleneck** force better representations, or does direct
   input-to-readout shortcut help by providing a "skip connection"?
   → Group A will answer this

3. Does **sequential spatial processing** add anything beyond what simultaneous
   seeding with positional encoding already provides?
   → Group D will answer this

4. What is the **right N for this architecture given D=4**? Is there a point
   where adding neurons stops helping (S³ overcrowding)?
   → Group C N-sweep will reveal the knee in the curve

5. Is **W_pos** actually learning meaningful geometric structure, or are positions
   random noise that happens to support routing? (Visualise W_pos clusters)
   → Can visualise after any training run, no new experiments needed
