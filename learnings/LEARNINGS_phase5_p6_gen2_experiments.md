# Phase 5 — Part 6: Gen2 Experiments (Steps 14–29)

Reference baseline: step9A = **29.22%** (D=16 N=512 K_iter=3, dynamic_z_geo, 150ep MPS)

---

## Step 14 — Top-K Gated Conduction + Excitatory Radiation

**Hypothesis:** Competitive conduction (top-K active neighbors win winner-take-all per neuron) improves routing selectivity. Excitatory radiation adds lateral excitation halo around active neurons.

**Result:**
```
A. k_cond=4  excrad=off         = 24.61%  (-4.61pp)  FAIL
B. k_cond=2  excrad=off         = 23.90%  (-5.32pp)  FAIL
C. k_cond=1  excrad=off         = 24.74%  (-4.48pp)  FAIL  (winner-take-all)
F. k_cond=6  excrad=16 α=0.3   = 29.38%  (+0.16pp)  neutral
G. k_cond=6  excrad=16 α=0.1   = 28.13%  (-1.09pp)
```
**Conclusion:** K-gated conduction always hurts — fewer winners per neuron destroys information flow. Excitatory radiation at best neutral. Neither warranted in Gen3. **Do not include.**

---

## Step 16 — Inhibition Mechanisms (fixed W_pos bug)

**Hypothesis:** Different inhibition types (divisive normalization, refractory, anti-Hebbian) differ in how they suppress competing neurons.

**Result:**
```
A. DivNorm   α=0.5  [mild shunting]         = 31.06%  (+1.84pp)
B. DivNorm   α=1.0  [moderate shunting]     = 30.14%  (+0.92pp)
D. Refract   β=0.9  α_r=1.0  [slow+mild]   = 30.14%  (+0.92pp)
E. Refract   β=0.7  α_r=2.0  [med+strong]  = 33.53%  (+4.31pp)
G. AntiHebb  α=0.3  wpos [spatial surround] = 32.87%  (+3.65pp)
H. AntiHebb  α=0.5  wpos [stronger spatial] = 37.22%  (+8.00pp)  ← WINNER
I. AntiHebb  α=0.3  zact [feature decorr.]  = 31.41%  (+2.19pp)
```
**Conclusion:** Anti-Hebbian inhibition w/ W_pos-weighted spatial surround (H) dominant — spatially nearby neurons suppress each other, forcing spatial diversity. +8pp = second-largest individual gain after signed coupling (+10.93pp). **Must include in Gen4.**

Mechanism: `Z_h -= α × Σ_{j∈wpos_neighbors} cos(Z_h, Z_j) × Z_j`
W_pos-weighted neighbors = neurons geometrically close in learned position space.

---

## Step 17 — Fast W_phase (Intra-Forward Adaptation)

**Hypothesis:** W_phase adapts within single forward pass via Hebbian/attention updates, making phase graph partially input-conditioned.

**Result:** ALL 6 configs beat ref.
```
A. Oja shared     α=0.1    = 30.24%  (+1.02pp)
D. Hopfield       α=0.1    = 30.52%  (+1.30pp)
F. Attention τ=0.25 b=32   = 31.06%  (+1.84pp)  ← WINNER
```
**Conclusion:** Intra-forward phase adaptation consistently helps (+1pp to +1.84pp). Attention-based update treats W_phase as key matrix, retrieves from Z — closest to true input-conditioned dynamic topology. Included in Gen3.

---

## Step 22 — D Dimension Extension

**Hypothesis:** Higher-D Fourier encoding gives neurons richer, more distinguishable seed directions. S^(D-1) capacity grows exponentially with D.

**Result:**
```
Ref  D=16 N=512  K_iter=8  = 36.74%  (150ep baseline — better than step9 due to K_iter=8+150ep)
A.   D=16 N=1024 K_iter=8  = 39.29%  (+2.55pp)
B.   D=32 N=512  K_iter=8  = 49.38%  (+12.64pp)
C.   D=32 N=1024 K_iter=8  = 51.29%  (+14.55pp)
D.   D=64 N=512  K_iter=8  = 49.61%  (+12.87pp)
E.   D=64 N=1024 K_iter=8  = 56.28%  (+19.54pp)  ← DOMINANT WINNER
F.   D=16 N=2048 K_iter=8  = 41.76%  (+5.02pp)
```
**Conclusion:** D dominant axis. Richer directions vastly outperform more neurons:
- D=64 N=1024 (+19.54pp) >> D=16 N=2048 (+5.02pp) at similar param count
- D=64 N=1024 best combo
- On S^63, 1024 neurons comfortably sparse; on S^15 they overcrowded

**New Gen2 base: D=64 N=1024 K_iter=8 (56.28%)**

---

## Step 23 — Signed Coupling × Scale

**Hypothesis:** Combining signed coupling (step18 winner) with N=1024 and K_iter=8 compounds gains.

**Result:**
```
Ref  base  N=512  K_iter=3           = 29.22%
A.   signed α=0.3  N=512  K_iter=3   = 41.10%  (+0.95pp vs step18A=40.15%)
B.   signed α=0.3  N=512  K_iter=8   = 10.68%  ← CATASTROPHIC COLLAPSE
C.   signed α=0.3  N=1024 K_iter=3   = 45.55%  (+5.40pp vs step18A)  ← WINNER
D.   signed α=0.3  N=1024 K_iter=8   = 15.69%  ← CATASTROPHIC COLLAPSE
E.   signed α=0.1  N=1024 K_iter=8   = 10.68%  ← CATASTROPHIC COLLAPSE
F.   signed α=0.5  N=512  K_iter=8   = 12.61%  ← CATASTROPHIC COLLAPSE
```
**CRITICAL CONSTRAINT: Signed coupling + K_iter ≥ 8 = mode collapse, always.**

Mechanism: at K_iter=8, all-pairs feedback = power iteration converging to dominant eigenvector of N×N Gram matrix. K_iter=3 below collapse threshold.

---

## Step 24 — Sparse Signed Coupling

**Hypothesis:** K-NN on W_phase (sparse static graph) approximates full N² coupling at O(N×K) cost.

**Result:**
```
Ref  no coupling                     = 29.22%
A.   K_phase=8   α=0.3  K_iter=3    = 33.27%  (+4.05pp)  recovers 37% of N² gain
B.   K_phase=16  α=0.3  K_iter=3    = 31.16%  (+1.94pp)  18% recovery
C.   K_phase=32  α=0.3  K_iter=3    = 31.72%  (+2.50pp)  23% recovery
D.   K_phase=64  α=0.3  K_iter=3    = 26.19%  (-3.03pp)  HURTS
A.   K_phase=8   α=0.3  K_iter=8    = 31.54%  (+2.32pp)
```
**Conclusion:** Static K-NN on W_phase recovers at most 37% of N² gain. More connections hurt past K=8. W_phase not input-dependent — sparse K-NN on static embeddings cannot replace input-driven Z Z^T coupling.

---

## Step 25 — Beam Routing (Input-Dependent Active Set)

**Hypothesis:** Restricting routing to top-K neurons by Z-norm reduces compute while preserving accuracy.

**Result:**
```
Ref. full N=512                        = 26.85%
A.   route=128  K_iter=3               = 27.31%  (+0.46pp)   15× speedup
B.   route=64   K_iter=3               = 29.73%  (+2.88pp)   53× speedup  ← WINNER
C.   route=32   K_iter=3               = 22.37%  (-4.48pp)   FAIL
D.   route=32   K_iter=8               = 25.50%  (-1.35pp)
F.   dynamic [128→64→32]  K_iter=3     = 29.35%  (+2.50pp)   33× speedup
```
**Conclusion:** route=64 sweet spot — 1/8 neurons active per step, 53× FLOP reduction, +2.88pp vs full-N routing. Route=32 too narrow — cuts semantic signal.

---

## Step 27 — Soft Beam / Threshold-Aligned Beam

**Hypothesis:** Differentiable soft-beam fixes gradient flow through hard beam selection.

**Result:**
```
Ref (full routing)                     = 26.85%
A.  soft τ=1.0  K=32                   = 24.79%  (-2.06pp)  FAIL
B.  soft τ=0.5  K=32                   = 24.41%  (-2.45pp)  FAIL
C.  soft τ=2.0  K=32 (flattest)        = 25.43%  (-1.43pp)  FAIL (best soft)
F.  threshold-aligned K=32             = 26.93%  (+0.08pp)  neutral
G.  Trick1+2 combined                  = 24.15%  (-2.70pp)  FAIL
```
**Conclusion: Soft beam consistently underperforms hard beam.** Not gradient issue — fundamental topology issue (hard gating on FIXED graph prevents neurons from self-organizing into relevant roles). **Threshold-aligned beam clean but no accuracy gain.**

---

## Step 26 — FAISS Conn_Phase Rebuild Frequency

**Hypothesis:** Rebuilding phase-based conn_phase graph more frequently captures faster-changing W_phase dynamics.

**Result:**
```
Ref   rebuild=epoch  K=8    = 29.22%  (baseline)
A     rebuild=10step K=8    = 28.84%  (-0.38pp)  neutral
B     rebuild=5step  K=8    = 29.45%  (+0.23pp)  neutral
C     rebuild=1step  K=8    = 28.71%  (-0.51pp)  mild regression
D     rebuild=epoch  K=16   = 28.61%  (-0.61pp)  mild regression
E     rebuild=5step  K=16   = 28.69%  (-0.53pp)  mild regression
```
**Conclusion:** Rebuild frequency **neutral** at D=16 — all configs within ±0.5pp. Per-epoch rebuilding sufficient. **Do not change rebuild frequency from default.**

---

## Step 29 — K_iter Diagnostic + AntiHebb at D=64

**Hypothesis:** Resolve K_iter vs signed coupling confound from step28 (32.15%).

**Result:**
```
Ref0  D=64 K_iter=3 no-signed  = 40.00%  [diagnostic control]
Ref   D=64 K_iter=8 no-signed  = 56.41%  [replicates step22E ≈56.28% ✓]
A     D=64 K_iter=8 + AntiHebb α=0.5 = 70.14%  (best@e140/150)  ← MASSIVE WIN
B     D=64 K_iter=8 + AntiHebb α=0.3 = ~65%  (still running)
```

**Key findings:**
- K_iter=3→8 at D=64: +16.41pp. Routing depth dominant.
- Signed coupling at D=64 K_iter=3: 32.15% vs 40.00% no-signed → **signed COSTS ~8pp at D=64**
- **AntiHebb α=0.5 at D=64 K_iter=8 = 70.14%** — confirmed FINAL. +13.86pp over D=64 ceiling.

**AntiHebb at D=64 2× stronger than at D=16 (+13.86pp vs +8pp). AntiHebb = Gen4 foundation.**

---

## Step 30 — Cross-Dimensional Mixing (W_mix)

**Hypothesis:** Current routing aggregation dimension-independent. Shared D×D matrix enables full cross-dimensional mixing.

**Result:**
```
Ref   D=16 N=512 K_iter=3     = 29.22%  (baseline)
A     shared W_mix before-norm = 29.12%  (-0.10pp)  neutral
B     shared W_mix after-norm  = 29.35%  (+0.13pp)  neutral
C     per-neuron W_mix [N,D,D] = 28.66%  (-0.56pp)  mild regression
D     shared W_mix once at end = 28.54%  (-0.68pp)  mild regression
E     D=64 N=1024 + W_mix     = 40.97%  (+11.75pp vs D=16, BUT -15.31pp vs step22E=56.28%)
```
**Conclusion:** Cross-dim W_mix **neutral at D=16** (±0.7pp), **hurts at D=64** (-15pp). W_mix interferes with phase-based routing that already provides implicit cross-dim structure through Fourier encoding. **Do not include. See step37 for targeted matrix bank variant.**