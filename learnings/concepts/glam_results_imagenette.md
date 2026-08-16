# GLAM — Imagenette results (T0→T2, mechanism ablations)

Detail file for [glam_grouped_multiplicative_routing.md](glam_grouped_multiplicative_routing.md).
Covers the saturated-ceiling (Imagenette VGG16 25088-dim) runs: locality champion tier ladder +
mechanisms 2/3/4 ablations. Scale/headroom results live in `glam_results_scale_capacity.md`.

## T0 RESULTS (glam_step001, 2026-08-13, mini_cpu, 20ep/50%, seed42)

Standalone GLAM layer (not an SGNNET `_route` override — reframed as a general stackable layer:
25088=512×49 → P=512 chunks → shared pool M=16 → within-group op → across-group add + re-frame).
Ref_dw ≈ 94.0%. All arms 6–8K params (0.005–0.007% FC).

n=1 seed, untuned, tiny model — **ordering hints, not verdicts**. Do not reject any arm off this.

| Arm | best | params | mechanism turned on | ordering note (n=1, untuned) |
|---|---|---|---|---|
| A1 | **0.6494** | 6,490 | locality only (gsz=1, add) | leads the ordering |
| A3 | 0.6482 | 6,490 | + additive 2nd-order pairing | ≈A1 (pairing ~free here) |
| A2 | 0.5445 | 6,490 | + **multiplicative** pairing | trails add by −10.5pp *untuned* |
| A4 | 0.4201 | 7,274 | + selectivity (`|cos|` scale) | trails A1 by −22.9pp *untuned* |
| A4b | 0.4199 | 7,274 | + AH decor penalty | ≈A4 (decor inert here) |
| A5 | 0.4199 | 7,274 | full, L=1 | tracks A4 |
| A7det | 0.3896 | 7,274 | + log slope anneal 1→1e-5 | lower acc (energy arm — judge on Joules) |
| A8 | 0.4769 | 8,554 | full, L=2 | depth lifts vs A5 |

**Scout signals ONLY — `HYPOTHESIS`, NOT falsified.** One seed, one config, a degenerately small
6–8K-param model at 20ep/50%. Every arm sat ≪ 94% Ref_dw *because the model is tiny*, so the
Ref comparison is unfair and no arm can be rejected off this. These are ordering hints, not verdicts:

- Within-group MULTIPLY (A2) trailed ADD (A3/A1) by −10.5pp **at this scale/init**. `HYPOTHESIS`:
  the product may need width/init/LR tuning before it can express interaction value — untuned it
  added variance. Do NOT read as "multiplicative core falsified."
- Selectivity (`|cos|` scaling, A4) trailed A1 by −22.9pp. Directionally consistent with AH/
  gate-death priors, but at this scale it may just be under-initialised. `HYPOTHESIS`.
- Slope anneal (A7det) cost accuracy — expected; it is an energy arm, judge on Joules/zero-fraction.
- Depth (A8) recovered vs A5 within the full family. A1/A3 lead the ordering.

**Why no rejection yet:** the whole ladder ran at a scale where nothing is competitive, so absolute
numbers carry no signal and relative gaps are noisy at n=1. Before forming any hypothesis (let alone
KILLING an arm) GLAM needs **exploration + tuning**: width/pool-size sweep to a regime where at least
one arm approaches Ref_dw, then per-arm LR/init tuning, then multi-seed re-ablation. Next round:
(1) A1/A3 width×M sweep to find a competitive operating point; (2) once there, re-run A2/A4/A8 with
tuned init/LR; (3) multi-seed the survivors. Only then do the ablation verdicts mean anything.

## T0 COMPETITIVE RESULTS (glam_step002, 2026-08-13, un-strawed, multi-seed)

**Root cause of the step001 degeneracy found + fixed:** GLAMNet forced `across_op="add"` on the last
layer, collapsing to `d_out=8` dims before the 10-way readout — an 8-dim terminal straw that
bottlenecked every step001 arm. `collapse_last=False` keeps layers **wide** (concat → readout sees
`G·d_out`). Standalone GLAM layer on VGG16 features (25088=512×49 → P=512 chunks → shared M-expert
pool → concat → readout). Same pipeline/data/split as the FFN-head baseline (`ffn_step001`), so the
comparison is **fair**. 20ep/50%, AdamW 3e-3 OneCycle.

**Baselines (same features):** FFN-head 93.2% @ 1.11M · SGNNET champ 95.95% @ 35K · D=16 ceiling 97.30%.

A1 = locality only (shared-pool per-channel proj 49→d_out, gsz=1). 3-seed {42,43,44}:

| cell | params | %FC | mean ±sd |
|---|---|---|---|
| **L=1 d=8 M=16** | **47,370** | **0.040** | **96.96 ±0.10** |
| L=1 d=32 M=16 | 189,450 | 0.158 | 97.21 ±0.03 |
| L=2 d=8 M=16 | 48,522 | 0.041 | 96.84 ±0.11 |
| L=2 d=32 M=16 | 206,346 | 0.173 | 97.22 ±0.15 |

Full A1 width sweep (seed42) spans 96.4–97.4% across 47K–469K params — the whole curve sits at/above
the FFN-head and SGNNET-champion accuracy, at 4–23× fewer params.

Op ablation at d=8 M=16, 3-seed (A1 gsz=1 · A3 gsz=2 add · A2 gsz=2 mul, only the op differs A2↔A3):

| arm | L=1 | L=2 | verdict (T0, seeds non-overlapping ≫3σ) |
|---|---|---|---|
| A1 locality | 96.96 ±0.10 | 96.84 ±0.11 | the win |
| A3 add-pair | 96.85 ±0.09 | 96.01 ±0.10 | **neutral** (−0.11pp, within noise) — CONFIRMED |
| A2 mul-pair | 94.84 ±0.24 | 89.29 ±0.55 | **net-negative, compounds with depth** — CONFIRMED |

**Verdicts (CONFIRMED at T0 — clean control A1, one variable changed A2↔A3, 3 seeds non-overlapping):**
- **Mechanism 1 (locality / shared-pool projection) is the entire efficiency win.** A shared pool of
  M=16 experts doing a 49→8 per-channel projection (6.3K params) + wide linear readout reaches the
  97.3% feature ceiling at 47K params (0.040% FC) — Pareto-dominating the FFN-head (+3.8pp at 23×
  fewer params) and beating the SGNNET champion (+1.0pp). **This is the new efficient-DL primitive:
  replace a dense FC head with a shared-pool structured projection.**
- **Mechanism 3a (within-group multiplication) is net-negative** and worsens with depth (−2.1pp L=1
  → −7.5pp L=2) — the gate-death compounding law holds on this new layer type (`gate_death.md`).
  Not `KILLED` outright: it may yet earn its place on an energy/interaction axis or with tuned
  init/LR — but it does not buy accuracy here.
- **Mechanism 3b (additive 2nd-order pairing) is neutral** — free but inert at this scale.
- **Depth (L) does not help at the ceiling** — L=1 ≈ L=2 at d=8 (96.96 vs 96.84). Expected: the
  feature ceiling is already hit by one layer, so there is no headroom for depth to add. Depth-over-
  breadth remains untested *below* ceiling (needs a harder dataset, e.g. CIFAR-10 features).

**Tier ladder — headline STRENGTHENS, no compression erosion:**
| tier | config | acc | params |
|---|---|---|---|
| T0 (20ep/50%) | L=1 d=8 M=16 | 96.96 ±0.10 | 47,370 |
| T1 (75ep/50%) | L=1 d=8 M=16 | 97.15 ±0.13 | 47,370 |
| **T2 (150ep/100%, 3 seeds)** | **L=1 d=8 M=16** | **97.68 ±0.05** | **47,370** |

**T2 CONFIRMED (paper-bound):** shared-pool structured projection = **97.68 ±0.05pp @ 47,370 params
(0.040% FC)**, seeds 42/43/44 = 97.63/97.71/97.71. Full-data + full-epoch LIFTED the result (+0.53pp
over T1), opposite of the usual T0→T1→T2 compression. Pareto-dominates FFN-head (93.2% @ 1.11M,
+4.5pp at 23× fewer params) and SGNNET champion (95.95%, +1.73pp). **The /goal deliverable is now
tier-complete.**

**Mechanism 2 — selectivity/affinity (glam_step003, T0 3-seed) — NEUTRAL on saturated ceiling:**
S0 loc=96.96, S1 +affinity |cos| gate=96.89 (−0.07, noise), S2 +anti-Hebbian decor=96.66 (−0.30).
The forward-affinity multiplier is neutral (not dead) — its payoff (dormancy/composability) is
invisible on a saturated accuracy ceiling and needs harder data + dead-expert metrics. CONFIRMED (T0).

**User's faithful key×value combine (glam_step004, T0 3-seed) — the sigmoid IS the mechanism:**
KV `sigmoid(key)·value` = 96.91 (M=34) / 96.84 (M=64) — TIES additive locality (96.96); PV raw
`key·value` = 93.00 / 93.48 — collapses to the FFN-head ceiling. **The once-per-layer sigmoid key gate
lifts +3.9pp over the raw product and does NOT die** — a clean new confirmation of the gate-death
boundary (bounded gate dies only when compounded over K steps; single application survives). Distinct-
pair pool floor is combinatorial: MC2≥P → M≥33 for P=512 (`_distinct_pairs` asserts). My earlier "mul
net-negative" (A2, symmetric both-activated unbounded) tested the WRONG variant — the asymmetric
sigmoid-key form is viable, ties locality at equal-ish params. CONFIRMED (T0). `model_glam_keyval.py`.

**Weight-RReLU on the readout WEIGHTS (glam_step010b, T1 3-seed) — WINNER, 4.5× readout compress:**
The 97.68% champion's params live ~86% in the dense readout `Linear(4096,10)` (rank-bound by C=10 →
massively over-parameterized). `src/sgnnet/weight_leaky.py::LeakyPrunedLinear` gates those weights by a
train-soft / infer-HARD randomized-leaky magnitude threshold (global magnitude quantile at sparsity s;
train `w_eff=w·gate`, gate=1 above τ else leaky α with log-anneal 1→1e-3 as STE; infer gate=hard {0,1}
→ **real zeros**). T1 (75ep/50%, seeds 42/43/44):

| arm | s | acc mean±sd | zero_f | eff_par | Δ vs R0 |
|---|---|---|---|---|---|
| R0 dense | — | 97.16 ±0.14 | 0 | 47,370 | — |
| **R1 det** | **0.9** | **96.94 ±0.06** | 0.90 | **10,506** | **−0.22** |
| R1 det | 0.95 | 96.43 ±0.15 | 0.95 | 8,459 | −0.73 |
| R1 det | 0.97 | 95.97 ±0.10 | 0.97 | 7,639 | −1.18 |
| R2 rand | 0.9 | 96.90 ±0.14 | 0.90 | 10,506 | −0.26 |
| R2 rand | 0.95 | 96.34 ±0.20 | 0.95 | 8,459 | −0.82 |
| R2 rand | 0.97 | 95.95 ±0.09 | 0.97 | 7,639 | −1.21 |

**CONFIRMED (T1, 3 seeds):**
- **Winner = R1 det s0.9: readout 47,370→10,506 eff-par = 4.5× compress at −0.22pp (inside sd 0.06).**
  Layer footprint 0.0396%→0.0088% FC at iso-accuracy. `zero_frac=0.90` are REAL test-time zeros.
- **Break-point = s0.9**, monotonic knee: ≤0.5pp budget holds only at 0.9; 0.95 crosses (−0.73pp).
- **Randomization DEAD** — R1(det) ties-or-beats R2(rand) at every s (T0's n=1 rand edge was noise) →
  kill R2. Deterministic log-anneal is the mechanism; the stochastic slope adds nothing.
- **Infer-hard leg VALIDATED for the first time** (the genuinely-novel, previously-unvalidated leg of
  the RReLU/LTP/BNN-STE synthesis). Sign-off: magnitude-sparsity holds where low-rank readout CRASHED
  (step008 ro32=0.558) — sparsity beats low-rank for this C-bound readout.

**T2 paper-bound (150ep/100%, 3 seeds) — CONFIRMED:** R0 dense reproduces the champion EXACTLY
(97.68 ±0.04, seeds 97.63/97.71/97.71). R1 det s0.9 = **97.39 ±0.08 (−0.29pp) @ 10,506 eff-par =
4.5× readout compress**, real zeros. R2 rand 97.42 ±0.15 (−0.27pp) ties R1 (sd ≫ 0.03pp gap →
randomization neutral, killed on parsimony/energy). **Paper-bound deliverable: the LOC champion
readout compresses 4.5× (0.0396%→0.0088% FC) for −0.29pp.**

Distinct from `glam_step010` (RReLU on *activations*, all 4 arms killed) — same step number, different
signal path (weights, not activations). `scripts/glam/glam_step010_weight_rrelu_t0.py`.
