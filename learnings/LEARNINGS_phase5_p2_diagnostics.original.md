# Phase 5: Regression Diagnostics (2026-03-29)

Investigation of the 26.52% → ~20% regression after trainer refactor.
Covers automated listener experiments, infrastructure bugs, and root cause.

---

## Aug Baseline and Initial Regression Symptoms

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

### 200ep cosine baseline (CPU, DONE)
- top1_best=20.36%  best_ep=58  top1_last=17.17%
- **Decision: top1_best=20.36% < 22% threshold.**
  Regression is in model config, NOT epoch count. 120ep stays as standard.

### Threshold sweep (train_thresh_sweep.py, MPS, DONE)
All 4 configs identically 18.96% at ep60:
  A: thresh=0.0, geo=True   → 18.96%
  B: thresh=0.1, geo=True   → 18.96%
  C: thresh=0.3, geo=True   → 18.96%
  D: thresh=0.0, geo=False  → 18.96%  ← pure dynamic_z, iter1 winner

**Definitive finding: architecture is irrelevant at this point.**
Geo bias, threshold value, routing mode — all produce identical results.
Identical best_ep=60 across all configs confirms it's not an architecture issue.

### Original-data control (CPU, DONE)
All 4 configs on store.h5 (9,469 train): **identical 18.80% at best_ep=69**.
Even config D (pure dynamic_z, no geo, no thresh, original data) = 18.80%.
Regression is NOT from aug data. Something changed in training infrastructure.

---

## Root Cause: Gradient Clipping Bug in Trainer Refactor

**The bug:** `clip_grad_norm_(self.model.parameters(), 1.0)` clips ALL model
parameters. But `optimizer.zero_grad()` only zeros gradients for W_pos (the only
parameter in the optimizer). theta (512 params) and W_phase (2048 params) are NOT
in the optimizer — their gradients ACCUMULATE across all batches without being zeroed.

After K batches, theta.grad and W_phase.grad have K× the per-batch gradient magnitude.
The clip norm is dominated by this accumulated norm, clipping the W_pos gradient to
near-zero. Learning effectively stops after the first few batches. This is why ALL
configs/data sources gave identical ~18-19% — the model barely learns W_pos at all.

**The fix** (applied to trainer.py):
```python
_opt_params = [p for g in self.optimizer.param_groups for p in g["params"]]
clip_grad_norm_(_opt_params, self.grad_clip_norm)  # only W_pos, not theta/W_phase
```

### Bugfix verification result (2026-03-29 05:00 PDT)
logs/train_bugfix_control.log — sched=none, 90ep, store.h5:
```
A (thresh=0.0, geo=True,  dynamic_z_geo): 19.95%  best_ep=85
B (thresh=0.1, geo=True,  dynamic_z_geo): 19.44%  best_ep=83
C (thresh=0.3, geo=True,  dynamic_z_geo): 20.54%  best_ep=32  ← best
D (thresh=0.0, geo=False, dynamic_z):     20.08%  best_ep=60  ← control
```

**Partial confirmation.** Config D = 20.08% (above broken 18.96%) but below expected
23%+. Architecture IS differentiating again (best_ep now varies vs. all identical before).

### Clean cosine baseline (2026-03-29 05:18 PDT)
train_thresh_sweep.py --device mps --epochs 120 --data store.h5 --sched cosine:
```
A: 19.03%  best_ep=51    B: 19.16%  best_ep=42
C: 18.52%  best_ep=13    D: 19.62%  best_ep=51  ← control
```

**Bugfix NOT confirmed.** D=19.62% < 20% threshold. Cosine worse than constant LR.

---

## Remaining Hypotheses

**Hypothesis 1: LR schedule.** Plateau with FIXED trainer never tested.
- Result: plateau 120ep D=20.03% — **DISPROVED.** All LR variants exhausted.

**Hypothesis 2: AMP/GradScaler.**
- Result: fp32/no-AMP D=19.62% — **DISPROVED.**

**Hypothesis 3: Grad clip norm=1.0 too restrictive.**
- no-clip (grad_clip_norm=inf): D=19.87% — **DISPROVED.** No improvement.
- `experiment_config.py` updated to `grad_clip_norm=float('inf')` (correct in principle).

**All 5 active hypotheses disproved.** See `LEARNINGS_phase5_p3_breakthrough.md`
for the actual root cause discovered via Step 6-8.

---

## Infrastructure Notes

### dataset.py in-memory migration crash

While bugfix verification was running, `dataset.py` was updated from lazy per-batch
loading to in-memory loading. The new code was rsynced to Mac Studio mid-run.
On macOS, DataLoader workers use `spawn` (not fork) — new worker processes reimport
`dataset.py` and get the NEW class definition. Workers unpickling OLD H5Dataset
instances then call `__getitem__` using new code:
```
AttributeError: 'H5Dataset' object has no attribute 'features'
```

**Fix:** `make_loaders` now defaults to `num_workers=0`. With all data in RAM there
is no I/O to parallelise, so worker processes are unnecessary. No pickling, no mismatch.

### Evaluation / accuracy audit
Reviewed trainer.py evaluate() and store.h5 labels. Results: CLEAN.
- val_top1 = (argmax(scores) == hard_labels).mean() — correct standard top-1
- soft_labels: sum=1.0, argmax matches hard label at 99.54% (expected for KD)
- Val: 3925 samples, 10 classes, balanced (357-419/class)
- No data leakage between train/val splits
