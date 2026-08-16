# Readout weight-sparsity (glam_step010b) — generality / novelty measurement

**Question (user):** is weight-RReLU-on-readout a *general* step-improvement, or overfit to one
problem type? Measured via two falsification axes — both fire. **Verdict: over-parameterization
harvesting, regime-specific. NOT a general new mechanism.** CONFIRMED (T0 3-seed A-axis; T0
seed-42 C-axis, monotonic).

## Method under test
`src/sgnnet/weight_leaky.py::LeakyPrunedLinear` gates LOC champion readout `Linear(4096,C)`
weights by global magnitude quantile at sparsity `s`; train soft-leaky (log-anneal slope
1→1e-3, STE), infer HARD {0,1} = real zeros. Imagenette C=10 headline: s0.9 = −0.22pp @ 4.5×
readout compress.

## Axis A — novelty vs plain pruning (Imagenette C=10, R1 s0.9, T0 20ep/50%, seeds 42/43/44)
Zero-code baselines via `--a0/--a1` slope endpoints (slope≡gate multiplier).

| arm | flags | acc% | read |
|---|---|---|---|
| one-shot magnitude prune (train dense) | a0=a1=1.0 | 92.98 ±1.23 | naive |
| hard-from-start (mask+STE ep0) | a0=a1=1e-3 | 96.77 ±0.15 | — |
| OURS leaky-anneal | a0=1.0 a1=1e-3 | 96.62 ±0.14 | ties hard-start |

- **PARTIAL PASS.** Ours beats naive one-shot prune **+3.6pp** → method ≠ plain magnitude
  pruning. ✓ CONFIRMED.
- **But the anneal is INERT** — hard-from-start (96.77) ties/beats ours (96.62). The mechanism
  reduces to "STE training under a fixed hard magnitude mask" = established straight-through /
  learned-threshold pruning (LTP). The leaky-anneal schedule adds nothing → **no NEW mechanism**.
  CONFIRMED. Kill the anneal on parsimony; keep hard-mask-from-start.

## Axis B — is the "free compression" a C=10 over-param artifact? (T0 20ep/50%, seed 42)

| dataset | C | R0 dense | s0.9 Δ | s0.95 Δ | s0.97 Δ |
|---|---|---|---|---|---|
| Imagenette | 10 | 97.68 | −0.22 | −0.73 | −1.18 |
| CIFAR10 | 10 | 85.60 | −1.79 | −2.75 | −5.16 |
| CIFAR100 | 100 | 63.97 | **−5.78** | −10.95 | −14.85 |

- **FAILS.** The s0.9 break-point does NOT hold as the readout becomes rank-loaded. −0.22pp
  (Imagenette) → −5.78pp (C=100). CONFIRMED, monotonic in `s` at every C.
- Penalty tracks **how hard the readout works** (∝ dense R0 headroom), not C alone: two C=10
  datasets differ (Imagenette −0.22 vs CIFAR10 −1.79) because Imagenette's 10 easy classes leave
  `Linear(4096,10)` far more over-parameterized than CIFAR10's harder 10.
- R2 (randomized) ties R1 (det) at every (C,s) → **randomization dead confirmed a 3rd time.**

## What this means
The 4.5× "near-free" readout compression on Imagenette was **harvesting the rank-≤C
over-parameterization** of a wide `Linear(4096,10)` readout on an easy task — not a general
property of the mechanism. It is a **legitimate deployment win in that regime** (VGG16 FC-head on
10-class Imagenette IS exactly such a regime — the paper's vehicle), but it is **not a new
discovery that gives a general step-improvement in DL**. As soon as the readout carries real rank
load (more classes / harder task), the compression costs accuracy proportionally.

The general parameter-efficiency win via **locality + information merging** is the **LOC/GLAM
backbone itself** (97.68% @ 47K params, Pareto-dominant) — the readout weight-sparsity is a
regime-specific bonus stacked on top, not the discovery. Generality effort should target the
backbone mechanism, not this add-on. See [[glam_results_imagenette]], [[glam_grouped_multiplicative_routing]].

## Responsibility → params (the /goal): random k-class subsets of CIFAR100
**Q (user):** is param-collapse driven by model *responsibility* (# classes k), not the dataset?
"Collapsing responsibility collapses params — fair?" Test both approaches. Same CIFAR100
features throughout; only k varies (random draws, labels remapped 0..k-1). T0 20ep/50%/seed42.

| k | LOC R0 | R1 s0.9 | Δrdout | plain LIN | param× | Δbb |
|---|---|---|---|---|---|---|
| 10 (draw s0) | 92.90 | 91.50 | −1.40 | 92.60 | 5.3× | +0.30 |
| 10 (draw s1) | 88.10 | 84.60 | −3.50 | 87.80 | 5.3× | +0.30 |
| 50 (draw s0) | 72.88 | 69.22 | −3.66 | 74.06 | 5.9× | −1.18 |
| 100 (full)   | 63.97 | 58.19 | −5.78 | 65.33 | 6.0× | −1.36 |

- **Approach A (readout weight-sparsity) = responsibility mechanism, PARTIAL YES.** Δrdout
  penalty grows with k (−1.40 → −5.78): fewer classes ⇒ `Linear(4096,k)` is rank-≤k
  over-parameterized ⇒ more compressible near-free. Collapsing responsibility DOES collapse the
  compressible-readout budget. **Caveat CONFIRMED:** the two k=10 draws differ (−1.40 vs −3.50),
  so raw k is confounded by per-class difficulty — the readout penalty tracks *how hard the
  readout works* (headroom), of which k is only one driver. Fair to say "fewer classes ⇒ more
  free readout compression"; NOT fair to attribute it to k alone.
- **Approach B (LOC backbone) = k-INDEPENDENT param FACTOR.** param× stays 5.3–6.0× across the
  whole k-axis; the backbone compresses the *feature* dim (25088→4096), not the *class* dim, so
  its param saving does not depend on responsibility. Only the accuracy cost (Δbb) grows with
  difficulty (+0.3 easy → −1.36 at k=100). So the two mechanisms answer the /goal oppositely:
  responsibility collapses **A's** budget but not **B's** factor.
### Nested-ordering trajectory — 5 random orderings, cumulative k (draw-confound removed)
`ktraj.py`, T0 20ep/50%, 5 seeds = 5 random class permutations, prefixes k=2,5,10,20,40,70,100
(each k ⊇ smaller). mean±std over orderings; std tightens as k grows (fewer draws to vary).

| k | LOC R0 | R1 s0.9 | Δrdout | plainLIN | Δbb | param× |
|---|---|---|---|---|---|---|
| 2 | 97.10±2.27 | 96.40 | −0.70 | 97.20 | −0.10 | 3.44 |
| 5 | 93.52 | 92.08 | −1.44 | 93.84 | −0.32 | 4.67 |
| 10 | 88.74 | 87.62 | −1.12 | 88.72 | +0.02 | 5.30 |
| 20 | 83.25 | 81.82 | −1.43 | 83.65 | −0.40 | 5.68 |
| 40 | 75.68 | 72.71 | −2.96 | 75.91 | −0.23 | 5.89 |
| 70 | 69.17 | 64.80 | −4.37 | 69.07 | +0.10 | 5.99 |
| 100 | 63.71±0.33 | 58.28±0.11 | −5.42 | 63.39±0.18 | +0.31 | 6.03 |

**CONFIRMED (5-ordering T0):** the /goal answer splits by approach.
- **A (readout-sparsity) IS responsibility-driven.** Δrdout −0.70 → −5.42pp as k grows (one k=10
  dip, within std). The near-free readout compression is the rank-≤k over-param headroom of
  `Linear(4096,k)`, which shrinks as responsibility k rises. **"Collapsing responsibility collapses
  the compressible-readout budget" = FAIR for A.**
- **B (LOC backbone) is responsibility-INDEPENDENT.** Δbb ≈0 at every k (−0.40…+0.31, all noise):
  backbone matches plain linear at all k, a free param cut. Its param× (3.44→6.03) rises only by
  the **arithmetic** feature-dim ratio 25088/4096≈6.1 — not by responsibility. B collapses
  *feature*-dim params; k does not drive it. **NOT fair to call B's saving responsibility-driven.**

Bottom line: A collapses the **class-dim** (responsibility-coupled, C-fragile — same axis as the
Axis-B C-sweep above); B collapses the **feature-dim** (responsibility-free, ~0 acc cost, the
general win). Data `results/glam/ktraj_t0_seed{0..4}__*.json`; subset draws
`results/glam/{glam_step010,linprobe}_*sub*`. Both stores CIFAR100, features fixed, only k varies.

## Backbone IS general (the resolving control) — LOC vs param-matched plain Linear(25088,C)
Same T0 recipe (20ep/50%/seed42/AdamW+OneCycle). LOC = GLAMNet L1 P512 d_out8 M16 (locality +
shared-pool merge front-end, then Linear(4096,C)). Plain = dense `Linear(25088,C)`.

| dataset | C | LOC acc% | LOC par | LIN acc% | LIN par | Δacc | param× |
|---|---|---|---|---|---|---|---|
| Imagenette | 10 | 96.97 | 47,370 | 97.02 | 250,890 | −0.05 | 5.3× |
| CIFAR10 | 10 | 85.60 | 47,370 | 86.86 | 250,890 | −1.26 | 5.3× |
| CIFAR100 | 100 | 63.97 | 416,100 | 65.33 | 2,508,900 | −1.36 | **6.0×** |

- **The locality+merge backbone gives a STABLE 5–6× param cut across the whole C-axis and
  difficulty range** — including rank-loaded C=100 where the readout weight-sparsity add-on
  collapsed (−5.78pp). **CONFIRMED T1 3-seed on the decisive C=100 case: LOC 63.70 ±0.15 vs plain
  Linear 65.78 ±0.17 = −2.08pp @ 6.0× param cut** (75ep/50%). The param-reduction FACTOR is
  C-stable (~5–6×); the accuracy COST scales with task difficulty: ≈0pp (Imagenette easy, readout
  over-param) → −2.08pp (CIFAR100 hard). A genuine Pareto tradeoff, not a free lunch on hard tasks.
- **Contrast is the whole point:** the add-on's benefit is C-fragile (−0.22→−5.78pp); the
  backbone's tradeoff is C-stable. The GENERAL parameter-efficiency discovery via locality +
  information merging is the **backbone**, not the readout trick. This is what the /goal names.
- Baseline script `scratchpad/linprobe.py`; LOC via `--arm R0`. Results
  `results/glam/linprobe_lin{img,c10,c100}_*.json`, `glam_step010_R0_t0imgloc_*`.

## Scripts / data
`scripts/glam/glam_step010_weight_rrelu_t0.py` (--sweep --store <h5> --a0/--a1). Stores:
data/store.h5 (Imagenette), data/store_cifar10.h5, data/store_cifar100.h5. Result JSONs:
`results/glam/glam_step010_*_t0{oneshot,hardstart,anneal,c10,c100}_*.json`.
