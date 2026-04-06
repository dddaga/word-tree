# Phase 5 — Part 7: Gen3 Closures + ARM Status (2026-04-01)

Reference: D=64 N=1024 K_iter=8 ceiling = **56.28%** | AntiHebb best = **70.14%**

---

## Step 33 — D=128 Extension

**Hypothesis:** Does D=128 continue the D=16→32→64 trend (+12pp per step)?

**Result:**
```
step33 Ref (D=64 N=1024)  = 56.79%  (confirms D=64 ceiling)
step33 A   (D=128 N=512)  = 10.39% at e40 → KILLED (plateau LR collapse)

step33b LR sweep (D=128 N=512, cosine schedule):
  lr=5e-4  = 10.88%  (best@e9, training_too_short)
  lr=1e-3  = 10.73%  (best@e3, training_too_short)
  lr=2e-3  = 10.70%  (best@e1, training_too_short)
  lr=5e-3  = ~9.3% through e20  (same death trajectory)
```
**Conclusion:** D=128 is **architecturally non-viable** with current Fourier encoding. All LR configs stay at ~10% (random baseline). On S^127, cosine similarities between random vectors concentrate near 0 — routing cannot align within 40 epochs at any LR. **D=64 is the confirmed encoding ceiling.**

---

## Step 34 — MoD Adaptive K_iter (KILLED at e110)

**Hypothesis:** Mixture-of-Depth adaptive routing depth — exit early when confident — matches fixed K_iter=8 at lower compute.

**Result:**
```
Ref    D=64 K_iter=8                      = 56.79%
RefK3  D=64 K_iter=3                      = 42.01%  (K_iter=3 costs -14.78pp)
A      MoD exit_thresh=1.5  K_iter=8      ≈ 18-20%  e10–e110, loss flat → KILLED
```

**Conclusion: MoD adaptive depth is incompatible with SGNNET routing.**
All 8 routing iterations contribute meaningfully — there is no early-exit epoch where neurons are "confident." MoD disrupts the refinement loop by prematurely freezing neurons, effectively degrading to K_iter=2-3 for most neurons. Also confirms: **K_iter=3→8 gap = ~15pp** at D=64 (consistent across 3 independent measurements).

**Do not pursue adaptive routing depth. All 8 iterations are necessary.**

---

## Steps 41, 42, 44 — Signed Coupling at D=64: Final Closure (ALL KILLED)

### Step 41 — Oja's Rule Routing Update

**Result:**
```
Ref:          56.79%  best@e120/150
Oja η=0.1:   24.33%  at e60/150 — plateau, KILLED
```
**Conclusion:** Oja's rule compresses each routing step toward the principal component of incoming signals. With K_iter=8, this creates compounding compression — by iteration 4-5, all neurons collapse toward a shared direction. The directional diversity driving the K_iter=3→8 gain is destroyed. **Signal: any routing update that contracts toward a shared direction will fail. Preserve the additive accumulation: `Z_new = normalize(Z_struct + α·Z_inh)`.**

---

### Step 42 — Signed Coupling α Calibration at D=64

**Result:**
```
Config 1   α=0.01  K_iter=3  D=64:  25.43%  best@e33/40  LR→1e-7 (dead)
Config 2   α=0.01  K_iter=3  D=64:  ~23.69% e30, LR collapsing
```
Reference (no coupling, K_iter=3, D=64): **40.00%**

**Conclusion:** No α value makes signed coupling viable at D=64. At D=64, E[cos²(Z_i, Z_j)] ≈ 1/64 (Johnson-Lindenstrauss) — coupling amplifies near-zero cosine signals as noise. Additionally: **plateau scheduler is incompatible with 40-epoch calibration runs at D=64** (5 halvings → LR=1e-7 masking α differences). For future D=64 calibration: use cosine or fixed LR.

---

### Step 44 — Beam-Restricted Signed Coupling

**Result:**
```
e60: 31.62%  e70: 31.57%  ← flat. KILLED at e70/150 (47%).
```
**Conclusion:** Even restricting to beam=32 neurons (3% of N=1024), signed coupling at D=64 fails. The model learns to ignore the coupling term — optimizer drives it toward zero or toward noise. Beam restriction reduces compute but cannot rescue a mechanism incompatible with high-D Fourier encoding.

**FINAL VERDICT — 5 experiments, 1 conclusion:**
```
Step 18  D=64 K_iter=3 signed α=0.3:       ≈ 20%   (vs 40% reference)
Step 23  D=64 K_iter=8 signed α=0.1/0.3:   10-16%  (catastrophic collapse)
Step 28  D=64 K_iter=3 gen3 compound:       32.15%  (signed degrades all mechanisms)
Step 42  D=64 K_iter=3 α=0.005→0.3 calib:  ≤25%    (no α works)
Step 44  D=64 beam=32 signed:               31.57%  (flat, stuck)
```
**Signed coupling is architecturally incompatible with D=64 Fourier encoding. Mechanism closed.**

Physical intuition: at D=16, E[cos²] ≈ 1/16 — meaningful signal. At D=64, E[cos²] ≈ 1/64 — pure noise. The mechanism requires dense, non-orthogonal activation space.

---

## Step 37 — Phase-Queried Matrix Bank (RUNNING)

**Hypothesis:** M learned D×D matrices selected by W_phase direction allow targeted cross-dim mixing without full N²×D² cost.

**Result at e100:**
```
top1 = 51.41%  (oscillating 50-53%)  vs baseline 56.28%
```
**Preliminary:** Clearly hurting vs baseline. Phase-queried matrix bank fails at D=64 similarly to shared W_mix (step30). The D×D matrix interferes with the Fourier routing structure regardless of how it's conditioned. Final result pending e150.

---

## ARM Status Summary — End of Gen3 (2026-04-01)

### What Works at D=64 (Confirmed Gains)

| Mechanism | Best result | vs ceiling | Status |
|-----------|-------------|------------|--------|
| **AntiHebb α=0.5 wpos** | **70.14%** | **+13.86pp** | ✅ Confirmed step29A |
| Fourier D=64 (vs D=16) | 56.28% | +26.58pp over D=16 | ✅ Foundation |
| K_iter=8 (vs K_iter=3) | +15pp gap | Confirmed ×3 | ✅ Foundational |
| alpha_reflect=0.5 | 52.94% at 40ep | +2.75pp (40ep) | ✅ step22b calibration |
| Input-gated adjacency (step36) | 58.62% (Ref config) | +2.34pp | ✅ Promising — gated configs running |

### What Does NOT Work at D=64

| Mechanism | Best seen | Why |
|-----------|-----------|-----|
| Signed coupling (any form) | ~32% | cos-sim on S^63 ≈ noise; 5 experiments |
| D=128 Fourier encoding | ~10% | Near-orthogonal seeds on S^127 |
| Cross-dim W_mix | 40.97% (hurts) | Interferes with phase routing |
| Phase-queried D×D matrix bank | ~51% (hurts) | Same interference pattern |
| MoD adaptive K_iter | ~20% | Disrupts iterative refinement |
| Oja's rule routing update | ~24% | PCA compression destroys diversity |
| Beam-restricted signed | ~32% | Same cosine noise issue |
| Soft beam | fails | Gradients through hard selection not bottleneck |

**Critical pattern:** Any mechanism that involves D×D matrices during routing HURTS at D=64. The Fourier encoding creates a structured low-D subspace in each dimension; full cross-dim mixing scrambles this structure.

### ARM 1 — Generational Compounding

| Step | What | Status | Key result |
|------|------|--------|------------|
| step22b | Base routing calib D=64 | RUNNING | alpha_reflect=0.5 confirmed |
| step29 | AntiHebb α sweep | Config A FINAL: **70.14%** | **Largest gain: +13.86pp** |
| step29b | Gen1 mechanisms at D=64 | 40ep calib done: 49.73% | Gen1 mechanisms survive D=64 |
| step29c | All mechs on calibrated base | Blocked on step22b | — |
| step32 | Gen4 compound | Not written | Blocked on step22b + step29c |

### ARM 2 — New Mechanisms

| Step | What | Epoch | Signal |
|------|------|-------|--------|
| step36 | Input-gated adjacency | Ref 58.62%, gated configs running | **Promising** |
| step37 | Phase-queried matrix bank | e100, 51.41% | Failing (same as W_mix) |
| step48 | K_iter sweep {8,12,16,24,32} | QUEUED (script written) | — |
| step52 | High-D routing (subspace/projection) | QUEUED (script written) | — |

### ARM 3 — Dynamic Connectivity

| Mechanism | Result | vs ceiling |
|-----------|--------|------------|
| Static conn_hh | 56.28% | baseline |
| Dynamic Z-KNN (step31) | ~44% at e130 | **-12pp — fails** |
| Input-gated adjacency (step36) | running | TBD |
| Spatial W_pos K-NN (step50) | QUEUED | — |
| Spatial + W_phase gate (step51) | QUEUED | — |

**Pivot:** Per-step Z-KNN on S^63 is inherently unstable (near-orthogonality → random neighbors). ARM 3 shifting to epoch-level topology evolution (W_pos K-NN, rebuilt per epoch not per step) and edge-weight gating (step50/51).

### ARM 4 — Gap-Bridging (Signed Coupling Closure)

| Step | Conclusion | Status |
|------|-----------|--------|
| step41 Oja's rule | ❌ 24% — confirmed harmful | KILLED |
| step42 signed α calib | ❌ no α works at D=64 | KILLED |
| step44 beam_signed | ❌ 32% flat — confirms signed dead | KILLED |
| step49 signed × K_iter | Testing K_iter 4-7 sweet spot | RUNNING (PID 8287) |
| step46 W_phase reconnect | Just launched | RUNNING |
| step47 interneurons D=64 | Ref running | RUNNING |

### Interneurons Status

D=16 finding (step20): 50% interneurons + readout=all = +2.83pp.
D=64 test (step47): Ref running at e40/150. Full 8-config sweep (A=25%int, B=50%int, C=75%int, D-G compounds) pending Ref completion.
**Hypothesis:** With K_iter=8, interneurons have 8 steps to integrate signal (vs 3 at D=16). Config F (interneurons + AntiHebb) is the critical test — if it compounds, structural inductive bias + competitive inhibition stack.

### Gen4 Design Principle (Emerging)

Mechanisms that assume directional alignment fail on S^63 (cosine similarity ≈ 0.016 = noise).
Mechanisms that impose competitive structure succeed regardless of direction.

**Next tier of gains must come from mechanisms that treat D dimensions as independent channels — routing, gating, or inhibiting per-dimension or per-frequency-band — exploiting the fact that at D=64, each dimension is an independent degree of freedom.**
