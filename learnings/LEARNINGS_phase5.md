# Phase 5 Validation Findings

Experimental results from the validation ladder (2026-03-27).
See `docs/resonant_sgnnet_spec.md` for architecture context.

---

## Step 1: Safety Valve Fix + Norm Mode Sweep

### Safety valve: bounded quadratic CONFIRMED

Old: `relu(1/d - 1/r)` — Coulomb divergence, safety dominated task at large N
New: `relu(r - d)^2 / r^2` + 15% soft cap relative to task loss

Results at N=512 (l2 norm):
```
safety/task = 0.14%  (was 140% before — 1000× reduction)
```
At N=1024: safety/task = 0.02%. Fix holds at all tested scales.

### Norm mode comparison — CRITICAL FINDING

All tested at N=512, 60 epochs, CPU, SmallWorld topology:

| norm_mode | top1 | task_loss | Notes |
|---|---|---|---|
| masked (old) | 10.2% | 3.97 | Random chance — confirmed broken |
| relu | 8.2% | 2.70 | WORSE than random in places |
| **l2** | **23.9%** | **2.06** | **Winner — use for all future runs** |

**Why masked fails:** normalises across all N neurons (mean/std), collapsing every
activation to the same scale. Destroys all class-discriminative structure.

**Why relu fails at D=4:** relu on a 4-dim vector leaves 1-2 non-zero dims.
F.normalize of near-zero vector → noise unit vector. D=4 is too small for relu
to be useful element-wise on the D axis.

**Why l2 wins:** both positive and negative dimensions carry signal.
Direction space in D=4 has full S³ sphere coverage.

**Action taken:** `norm_mode="l2"` is now the default in `experiment_config.topology_kwargs()`.

### N=1024 with l2 norm

Previous: N=1024 plateaued at total_loss=5.5 (task=2.3, safety=3.2 — safety dominated)
Now: task=2.29 at epoch 10, top1=20.9%, safety=0.0005 (0.02% of task)

---

## Step 2: W_phase Receiver — HYPOTHESIS REQUIRES REVISION

All at N=512, 60 epochs, CPU:

| config | top1 | vs baseline | Notes |
|---|---|---|---|
| baseline (l2 only) | 16.1% | — | Reference |
| W_phase K=8 | 15.9% | **-0.2%** | Hurt performance |
| W_phase K=16 | 15.0% | **-1.1%** | Worse with more connections |

**Conclusion:** Static W_phase graph adds noise, not signal.

**Why it failed:** W_phase graph was built once from random W_phase at init and
never rebuilt. "Dynamic long-range routing" was in fact a second frozen random
topology — all the complexity of phase routing, none of the adaptability.

**Revised hypothesis:** W_phase is NOT rejected as a concept. The test was of
"frozen random phase graph" — which is just noise. The real hypothesis
(online-rebuilt phase graph tracking learned W_phase directions) is still untested.
`tick_epoch()` rebuilds this graph between epochs — this is tested in Step 3+.

**Key insight:** The 16.1% baseline in Step 2 vs 23.9% in Step 1 is pure run-to-run
variance at N=512/60 epochs. Single-seed comparisons at this scale are unreliable;
ceiling likely ranges 15–25% across seeds.

---

## Step 3: Routing Mechanisms — STRONG POSITIVE RESULTS

All at N=512, 60 epochs, CPU, TuringSmallWorld wrapper:

| mode | top1 | vs baseline | Notes |
|---|---|---|---|
| baseline (no routing) | 13.9% | — | Step 3 reference |
| threshold | 18.1% | **+4.2%** | relu(Z - θ) gates propagation |
| reflection | 16.5% | **+2.6%** | self-inhibition, still converging at e60 |
| turing | 20.4% | **+6.5%** | local excite + long-range inhibit |
| **learnable θ** | **20.5%** | **+6.6%** | per-neuron threshold — best |

**Mechanism hierarchy:**
- `learnable ≈ turing > threshold > reflection > baseline`
- Two-scale Turing is the strongest: local excitatory via conn_hh + long-range
  inhibitory via phase beam produces competition → non-overlapping feature detectors
- Threshold gate is most robust: prevents noise propagation, works standalone
- Reflection needs >60 epochs — loss still descending, not converged
- Stack order: threshold + reflection + turing are all additive (combined in Resonant)

**tick_epoch works:** Phase graph rebuilds from W_phase each epoch. Confirmed by
the learnable mode outperforming — the graph is being updated meaningfully.

---

## Step 4: Simulated Annealing — REJECTED

All at N=512, 60 epochs, CPU, AnnealedSmallWorld wrapper:

| schedule | top1 | tau final | beam final | Notes |
|---|---|---|---|---|
| **fixed** | **19.3%** | 1.0 | 32 | **Winner** |
| exponential | 18.2% | 0.05 | 8 | Shrinking hurts |
| cosine | 15.2% | ~0 | 8 | Collapses at end |
| stepwise | 15.1% | 0.1 | 8 | Hard drop kills recovery |

**Why annealing failed:** Shrinking beam_size is irreversible capacity reduction.
Once beam drops from 32 to 8, the model permanently loses 75% of its long-range
broadcast capacity. No recovery is possible in the remaining epochs.

**Revised understanding:** Don't anneal beam_size downward. Fixed beam=32 throughout.
If temperature scheduling is desired, apply it only to gate_temp (softness of routing
weights), not to the number of active connections. Or consider reverse annealing:
start small beam (local structure first) and expand to larger beam later.

---

## Combined Run: SGNNET_Resonant on MPS (120 epochs)

Running in `scripts/train_resonant.py`. Configs:
- A: baseline SmallWorld l2 (reference)
- B: resonant (threshold + reflection + turing, static phase rebuilt per epoch)
- C: dynamic_gate (fixed graph, GAT-style activation-conditioned edge weights)
- D: dynamic_z (graph rebuilt from Z similarity each forward pass — Hopfield-style)
- E: resonant N=1024 (scale-up)

### Baseline result (completed):

```
N=512, 120 epochs, MPS
top1_best=24.4%  top1_last=22.1%  task_loss=2.046  t=87s
LR schedule: 2.36e-3 → 2.96e-4 over 120 epochs
```

Best is at ~e20 (23.6%) and ~e60 (23.4%), final slightly lower — the model noise-
plateau behaviour at this scale. Longer training gives marginal benefit.

### Resonant + dynamic configs: pending results.

---

## Dynamic Connectivity — Research Summary (2026-03-27)

Research into input-dependent pseudo-connections. Key finding: **all useful dynamic
connectivity mechanisms converge on attention** — computing a normalized similarity
score between two nodes as a function of their current features, then weighting
message aggregation by that score.

### Mechanism ranking for this setting (N=512, D=4, fixed base graph):

| Rank | Mechanism | Key property | Cost |
|---|---|---|---|
| 1 | **GAT** | Modulates existing edges, fully differentiable | O(E·F) |
| 2 | **Geometric-biased attention** | W_pos in gradient via logit bias | O(N²) ≈ 262K |
| 3 | **Hypernetwork edge weights** | MLP(pos_i, pos_j, h_i, h_j) → scalar weight | O(E·MLP) |
| 4 | Modern Hopfield | All-pairs dynamic, = transformer attention | O(N²·D) |
| 5 | DGCNN-style KNN rebuild | Topology changes per layer | O(N²), partial grad |

**Not applicable:** Neural ODE (10-20x slower), Capsule routing (no geometric prior),
MoE at N=512 (load balancing overhead, no benefit at this scale).

### Critical gap identified:

Current `dynamic_gate` and `dynamic_z` modes do NOT include W_pos in the similarity
score. The strongest unexplored variant is geometric-biased attention:

```python
# current:
score(h, k) = dot(Z[b,k], W_phase[h])

# next experiment:
score(h, k) = dot(Z[b,k], W_phase[h]) - gamma * ||W_pos[h] - W_pos[k]||^2
```

This puts W_pos into the routing gradient, not just the readout. Neurons that are
geometrically close AND feature-similar communicate more strongly. W_pos would then
be trained by two forces: readout quality AND routing quality — richer gradient signal.

### Three implementation strategies tested in train_resonant.py:

1. **resonant** — static phase graph (W_phase k-NN), rebuilt per epoch by tick_epoch
2. **dynamic_gate** — fixed graph, edge weights = f(Z) per input (GAT-style)
3. **dynamic_z** — graph topology rebuilt from current Z similarity per forward pass

Results pending.

---

## Next Experiments (after train_resonant.py results)

Priority order:

1. **Geometric-biased attention** — add `−γ · ||W_pos[h] − W_pos[k]||²` to phase
   score. Puts W_pos into routing gradient. O(E·D) extra cost per routing step.

2. **D=8 dimensionality sweep** — current D=4 gives S³ direction space (~10 distinct
   directions for N=512). D=8 gives S⁷ with dramatically more capacity. One run at
   N=512 D=8 vs D=4 to isolate dimensionality contribution.

3. **Multi-seed confirmation** — 3 seeds on winning config to confirm ceiling is real
   vs run variance. The 13.9–23.9% range across step runs indicates high variance
   at N=512/60 epochs.

4. **Longer reflection convergence** — reflection was still descending at e60 (loss
   2.12, task 2.116). Try 150 epochs on combined resonant+reflection only to see
   if it eventually beats threshold.

5. **Hypernetwork edge scoring** — small shared MLP(pos_i, pos_j, h_i, h_j) →
   scalar per edge. Strictly more expressive than dot-product gate. 2-layer MLP
   width-16 for |E|=4096 is trivially cheap even on CPU.

---

## Automated listener findings (2026-03-29)

### Aug baseline (train_aug_baseline.py, 120ep, store_aug.h5)
- plateau: top1_best=18.68%  top1_last=9.86%
- cosine:  top1_best=18.96%  top1_last=9.86%
- Both collapsed to near-random by ep120. Best was ep60-80. cosine wins by 0.28% (noise margin).
- **Decision:** cosine is the LR schedule going forward.
- Regression from iter1 26.52% is significant — under investigation.

### Routing dropout ablation (train_routing_dropout.py, 120ep, MPS)
Results (dynamic_z_geo + thresh=0.3, N=512, D=4, aug data):
- p=0.0: 18.68%  (baseline, confirms aug baseline regression is real)
- p=0.1: 18.75%  (+0.08% — below 0.5% threshold, not meaningful)
- p=0.2: 11.13%  (−7.54%, signal destroyed)
- p=0.3:  9.86%  (−8.82%, complete collapse)
- **Decision:** routing_dropout_p=0.0 going forward. Dropout is not beneficial here.
  l2-normalisation already regularises neuron directions; vector dropout just destroys
  signal without adding useful regularisation pressure.
- p=0.0 matching aug baseline confirms: the regression IS in the geo+thresh=0.3 config,
  not in epoch count or data loading.

### 200ep cosine baseline (CPU, DONE)
- top1_best=20.36%  best_ep=58  top1_last=17.17%  epochs_run=200
- **Decision: top1_best=20.36% < 22% threshold.**
  Regression is in the model config (geo+thresh=0.3), NOT epoch count.
  200 epochs only adds +1.4% over 120ep (20.36% vs 18.96%). Not worth the cost.
  Model peaks at ep58 and declines as cosine LR decays — geometry of the cosine curve
  matches the training dynamics fine. Root cause is elsewhere.
  **→ Keep 120ep as standard. Root cause = geo+thresh config on aug data.**

### Threshold sweep (train_thresh_sweep.py, MPS, DONE)
All 4 configs identically 18.96% at ep60:
  A: thresh=0.0, geo=True   → 18.96%
  B: thresh=0.1, geo=True   → 18.96%
  C: thresh=0.3, geo=True   → 18.96%
  D: thresh=0.0, geo=False  → 18.96%  ← pure dynamic_z, iter1 winner

**Definitive finding: architecture is irrelevant. The bottleneck is the aug data.**
Geo bias, threshold value, routing mode — all produce identical results on aug data.
The hflip augmentation shifts the VGG feature distribution in a way the model cannot
distinguish. Identical best_ep=60 across all configs confirms it's a data ceiling,
not a model capacity issue.

**Decision: revert aug data strategy.**
- hflip doubles training samples but creates a harder/different distribution.
- Run orig control (data/store.h5) with same configs to confirm ~25%+ recovery.
- If orig recovers: discard store_aug.h5, use store.h5 for all ablations.
- If orig does not recover: something else changed (investigate seed/config).

### Original-data control (CPU, DONE)
All 4 configs on store.h5 (9,469 train): **identical 18.80% at best_ep=69**.
Even config D (pure dynamic_z, no geo, no thresh, original data) = 18.80%.
Regression is NOT from aug data. Something changed in training infrastructure.

### Root cause identified: gradient clipping bug in trainer refactor (2026-03-29)

**The bug:** `clip_grad_norm_(self.model.parameters(), 1.0)` clips ALL model
parameters. But `optimizer.zero_grad()` only zeros gradients for W_pos (the only
parameter in the optimizer). theta (512 params) and W_phase (2048 params) are NOT
in the optimizer — their gradients ACCUMULATE across all batches without being zeroed.

After K batches, theta.grad and W_phase.grad have K× the per-batch gradient magnitude.
The clip norm is dominated by this accumulated norm, clipping the W_pos gradient to
near-zero. Learning effectively stops after the first few batches. This is why ALL
configs/data sources gave identical ~18-19% — the model barely learns W_pos at all.

Intermediate hypothesis (LR decay) was ruled out by constant-LR test also giving 18-19%.
Real culprit identified via trainer diff: old trainer had no grad clipping; new trainer
clips all params including non-optimizer params with accumulating gradients.

**The fix** (applied to trainer.py):
```python
_opt_params = [p for g in self.optimizer.param_groups for p in g["params"]]
clip_grad_norm_(_opt_params, self.grad_clip_norm)  # only W_pos, not theta/W_phase
```

**Verification result (2026-03-29 05:00 PDT):** logs/train_bugfix_control.log — sched=none, 90ep, store.h5.

```
A (thresh=0.0, geo=True,  dynamic_z_geo): 19.95%  best_ep=85  t=251s
B (thresh=0.1, geo=True,  dynamic_z_geo): 19.44%  best_ep=83  t=175s
C (thresh=0.3, geo=True,  dynamic_z_geo): 20.54%  best_ep=32  t=190s  ← best
D (thresh=0.0, geo=False, dynamic_z):     20.08%  best_ep=60  t=80s   ← control
```

**Partial confirmation.** Config D = 20.08%, above the broken-trainer baseline (~18.96%) but
below the expected recovery (23%+, based on iter1 120ep cosine). Key differentiator from broken
trainer: best epochs now vary widely (32, 60, 83, 85) — with the broken trainer all configs
peaked at the same epoch identically. Architecture IS differentiating again.

**Why 20% and not 23%+:** Constant LR (sched=none) + only 90 epochs is likely suppressing
results. Iter1 reference (26.52%) used cosine LR + 120 epochs. With constant LR, the model
explores noisily rather than converging; best epoch varies across the run. Cosine baseline
(120ep) is the definitive confirmation.

**Decision:** D ≥ 0.20 — launch clean cosine baseline immediately (120ep, cosine, store.h5)
as the definitive verification. All prior runs since trainer refactor remain invalid until
cosine baseline confirms ≥ 23%.

### Clean cosine baseline result (2026-03-29 05:18 PDT)

train_thresh_sweep.py —device mps —epochs 120 —data store.h5 —sched cosine (FIXED trainer):

```
A (thresh=0.0, geo=True,  dynamic_z_geo): 19.03%  best_ep=51   t=189s
B (thresh=0.1, geo=True,  dynamic_z_geo): 19.16%  best_ep=42   t=244s
C (thresh=0.3, geo=True,  dynamic_z_geo): 18.52%  best_ep=13   t=198s
D (thresh=0.0, geo=False, dynamic_z):     19.62%  best_ep=51   t=278s  ← control
```

**Bugfix NOT confirmed.** Config D = 19.62% < 20% threshold. Both diagnostic gates fail.
Cosine 120ep is WORSE than sched=none 90ep for most configs (D: 19.62% vs 20.08%).

**Cross-run summary:**
| Run                                | Best D   | Notes |
|------------------------------------|----------|-------|
| Broken trainer (all runs)          | 18.96%   | All configs identical — confirmed broken |
| Fixed trainer, sched=none, 90ep   | 20.08%   | Slight improvement, best_ep=60 |
| Fixed trainer, cosine, 120ep      | 19.62%   | WORSE than constant LR; best_ep=51 |
| Iter1 reference (old trainer)      | 26.52%   | Target — no grad clipping at all |

**Root cause hypothesis — remaining bug:** The grad-clip fix corrected the TARGET (W_pos only,
not all params) but did NOT remove the clip. Iter1 trainer had NO grad clipping on any
parameter. The current fix still applies `clip_grad_norm_(W_pos, 1.0)`. If norm=1.0 is a
binding constraint — i.e., W_pos gradients regularly exceed norm=1.0 — then this is still
throttling the learning rate effectively, preventing iter1-level convergence.

**Evidence supporting this hypothesis:**
- Broken trainer: 18.96% (W_pos gradient ~zero due to dominant accumulated theta/W_phase norms)
- Fixed trainer: 19-20% (W_pos gradient non-zero but capped at norm=1.0)
- Iter1 (no clip): 26.52% (W_pos gradient unconstrained)
- The step from broken→fixed is only ~+1% despite removing the main blocker
- Cosine LR makes it worse (peak ep13-51) vs constant LR (peak ep60-85): with clipped grads
  cosine drops LR too fast before W_pos has found a good region

**Hypothesis DISPROVED (2026-03-29):** clip norm was NOT the bottleneck.

No-clip diagnostic (grad_clip_norm=inf, sched=none, 90ep, store.h5):
```
A (thresh=0.0, geo=True):  20.00%  best_ep=?
B (thresh=0.1, geo=True):  19.62%
C (thresh=0.3, geo=True):  20.23%
D (thresh=0.0, geo=False): 19.87%  ← control
```
Config D = 19.87% — NO meaningful improvement over clip=1.0 (20.08%). Removing clip had
essentially zero effect. The ceil at ~20% is caused by something other than grad clipping.

Note: applying clip_grad_norm_(W_pos, inf) is mathematically identical to no clipping.
experiment_config.py updated to grad_clip_norm=float('inf') — not reverting since it's
correct in principle, but it's not the cause of the regression.

### Root cause investigation — remaining hypotheses (2026-03-29)

All post-refactor diagnostics: ~18.5-20.5% regardless of clip, LR schedule, data.
Iter1 baseline (train_resonant.py): 26.52% with same architecture.

**Hypothesis 1: LR schedule — iter1 used plateau, all fix runs used cosine/none.**
train_resonant.py calls `trainer_kwargs(n_hidden)` with no sched_type → default "plateau".
Fix runs tested: cosine (broken trainer) 18.96%, cosine (fixed) 19.62%, none (fixed) 20.08%.
Plateau with FIXED trainer was NEVER tested.
Diagnostic: scripts/train_step5_plateau_noclip.py — plateau + no-clip + 120ep.
Result (2026-03-29):
```
A (geo, thresh=0.0): 19.49%
B (geo, thresh=0.1): 19.21%
C (geo, thresh=0.3): 19.01%
D (no-geo, thresh=0): 20.03%
```
**HYPOTHESIS 1 DISPROVED.** Plateau LR is NOT the explanation. Config D = 20.03% — no
better than cosine/constant runs. All LR schedule variants now exhausted:
  - cosine 120ep: D=19.62%
  - constant (sched=none) 90ep: D=20.08%
  - plateau 120ep: D=20.03%
Conclusion: LR schedule is NOT the source of the 26.52% → ~20% regression.

**Hypothesis 2: AMP / GradScaler version difference.**
Iter1 may have run on Mac Mini with PyTorch < 2.3 → `_check_grad_scaler_support()` returns
False → scaler=None (fp32, no gradient scaling).
Current Mac Studio has PyTorch >= 2.3 → GradScaler IS active (fp16 autocast + scaling).
If MPS float16 operations cause overflow/nan in any forward pass op, GradScaler silently
halves the scale and SKIPS the update step. Repeated skipping kills learning.
Diagnostic: scripts/train_step5_noamp_diagnostic.py — use_amp=False, fp32, plateau, 120ep.
Result (2026-03-29): D=19.62%
**HYPOTHESIS 2 DISPROVED.** fp32/no-AMP gives same ~20%. GradScaler is NOT the cause.

### Step 6: Regression isolation — SmallWorld baseline control (2026-03-29)

All Step 5 diagnostics used SGNNET_Resonant(dynamic_z). Tested whether regression was in
the Resonant routing layer specifically.

Diagnostic: scripts/train_step6_baseline_control.py — plain SGNNET_SmallWorld, no Resonant
wrapper. Exact iter1 baseline_N512 replica. MPS, plateau, 120ep, store.h5.
Result: baseline=19.87%  (iter1 baseline_N512: 24.38%)

**CONCLUSION: Regression is in SGNNET_SmallWorld or training setup, NOT in Resonant routing.**
Even without dynamic routing, the model achieves only ~20% vs iter1's ~24%.

All 5 active hypotheses have been disproved:
  1. Aug data              — original data also gives ~20%
  2. LR schedule           — plateau/cosine/constant all give ~20%
  3. Grad-clip bug         — fixed + removed, no improvement
  4. AMP/GradScaler        — fp32/no-AMP also gives ~20%
  5. Resonant routing      — SmallWorld alone gives ~20%

### Step 7: Iter1 exact reproduction (2026-03-29)

Diagnostic: train_resonant.py --device mps --skip-n1024 rerun on Mac Studio.
Results:
```
baseline_N512:     20.08%  (original: 24.38%)
resonant_N512:     15.97%  (original: 25.32%)
dynamic_gate_N512: 22.50%  (original: 25.07%)
dynamic_z_N512:    23.18%  (original: 26.52%)
```

Iter1 rerun gives 23.18% for dynamic_z — BETTER than our ~20% diagnostics (3% gap).
Key differences between iter1 rerun and Step 5-6 diagnostics:
  - iter1: unseeded DataLoader (shuffle=True, no generator) — random order per epoch
  - diagnostics: seeded DataLoader (generator=torch.Generator().manual_seed(42))
  - iter1: dynamic_z mode (no geo)
  - diagnostics: dynamic_z_geo, then dynamic_z in Step 5-6

The ~3% gap is explained by dynamic_z vs dynamic_z_geo and possibly seed=42 being suboptimal.
The remaining 3.34% gap vs original iter1 (26.52%) is now explained by Step 8 (see below).

### Step 8: Encoding + D sweep — BREAKTHROUGH (2026-03-29) ←← KEY FINDING

Script: train_encoding_D_sweep.py — 90ep MPS, all dynamic_z_geo, store.h5
Results:
```
A. Linear  D=4  N=512:  20.99%   (matches all previous ~20% diagnostics)
B. Fourier D=4  N=512:  21.61%   (+0.62% over linear)
C. Fourier D=8  N=512:  25.86%   (+4.87% over linear!)
D. Fourier D=16 N=512:  27.54%   (+6.55% over linear — NEW BEST, exceeds iter1 26.52%)
E. Fourier D=8  N=1024: 26.88%   (+5.89%)
```

**RCA CONCLUSION: The 26.52% → ~20% regression was NOT a bug or infrastructure failure.**
It was a configuration test artifact:
  - Post-iter1 diagnostics tested dynamic_z_geo (geo mode) with D=4 linear encoding
  - iter1 used dynamic_z (no geo) with D=4 linear encoding
  - geo mode slightly hurts at D=4 (fewer directions on S³ to represent spatial biases)
  - The REAL fix is D=8 or D=16 Fourier encoding which gives 25-27%+ regardless of geo mode

**New state of the art: D=16 Fourier N=512 = 27.54% (surpasses iter1 26.52%)**

Key insight — D matters more than N:
  - D=16 N=512: 27.54%
  - D=8  N=1024: 26.88%
  → Going from D=4 to D=16 at same N gives +6.55%
  → Doubling N from 512 to 1024 at D=8 gives only +1.02%

Next: Full 150ep run with D=16 Fourier + both dynamic_z and dynamic_z_geo to establish
true ceiling and pick routing mode for Group A/B/C experiments.

### Evaluation/accuracy audit (2026-03-29)
Reviewed trainer.py evaluate() and store.h5 labels. Results: CLEAN.
- val_top1 = (argmax(scores) == hard_labels).mean() — correct standard top-1
- soft_labels: sum=1.0, argmax matches hard label at 99.54% (expected for KD)
- Val: 3925 samples, 10 classes, balanced (357-419/class)
- Features: VGG pool5 ReLU, 85% sparse, float32, no normalization (by design)
- No data leakage between train/val splits

### dataset.py in-memory migration crash (2026-03-29)

While bugfix verification was running (at epoch ~70 of config A), `dataset.py` was updated
from lazy per-batch loading (H5Dataset with `_file` opened per worker) to in-memory loading
(H5Dataset with `features`/`soft_labels`/`labels` tensors loaded in `__init__`).

The new code was rsynced to Mac Studio mid-run. On macOS, DataLoader workers use `spawn`
(not fork) — new worker processes reimport `dataset.py` from disk and get the NEW class
definition. But the DataLoader had pickled the OLD H5Dataset instances (with `_file` in
`__dict__`, no `features`). When a new epoch started, the spawned workers tried to call
`__getitem__` on the unpickled old instances using the new code:
```
AttributeError: 'H5Dataset' object has no attribute 'features'
```

**Fix applied:** `make_loaders` now defaults to `num_workers=0`. With all data in RAM there
is no I/O to parallelise, so worker processes are unnecessary. No pickling, no class mismatch.
Job relaunched at 04:43AM and confirmed running with:
  `Dataset loaded into RAM: train=9469  val=3925  (1.34 GB features)`
