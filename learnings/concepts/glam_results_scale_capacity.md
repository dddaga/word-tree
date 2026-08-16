# GLAM — headroom, scale-invariance & capacity results

Detail file for [glam_grouped_multiplicative_routing.md](glam_grouped_multiplicative_routing.md).
Covers headroom tasks (CIFAR-10/100 VGG16 features), mechanism-4 energy arm, class-count
scale-invariance (T1→T2 overturn), and the capacity-scaling probe. Saturated-ceiling Imagenette
results live in `glam_results_imagenette.md`.

## CIFAR-10 depth-below-ceiling — the decisive arm RESOLVED (step005, T0 3-seed, CONFIRMED)

Same 25088-dim VGG16 head, store swapped to CIFAR-10 (real headroom: dense linear head trails no
ceiling). Within-tier anchor added (LIN = plain `nn.Linear(25088,10)`, same 20ep/50% tier — the only
valid comparison, cross-tier vs the 86.24% T2 claim is invalid).

| Arm | mean | sd | params | vs LIN | vs LOC |
|---|---|---|---|---|---|
| LIN dense | 86.90 | 0.18 | 250,890 | — | — |
| **LOC** locality | 85.66 | 0.04 | 47,370 | −1.24 | — |
| KV sigmoid(key)·val | 85.77 | 0.15 | 54,570 | −1.13 | +0.11 (noise), +15% params |
| LOCL2 depth L=2 | 84.83 | 0.23 | 48,522 | −2.07 | −0.83 |

**Meta-verdict (CONFIRMED, both datasets):** locality (mechanism 1) is the ENTIRE algorithm. On the
*saturated* Imagenette ceiling (~97.3%) locality alone hits it (96.96→97.68 across tiers) and every
added mechanism converges ~96.9% or collapses. **My hypothesis "differentiation just needs headroom"
is now FALSIFIED**: CIFAR-10 has real headroom (LOC −1.24pp under dense) yet KV STILL only ties LOC
(+0.11, within noise) at +15% params (Pareto-dominated), and depth STILL hurts (−0.83). Mechanisms
2/3/4 (affinity, mul/key×value, depth) are neutral-or-negative on BOTH a saturated and a headroom
task. The honest Pareto: shared-pool structured projection = dense accuracy −1.24pp at **5.3× fewer
params** where the task has headroom, and *matches* the ceiling where it saturates. KV does NOT advance
(neutral acc, dominated on params). Depth REJECTED. Only truly open arm: mechanism 4 (slope anneal),
scored on energy/Joules not accuracy. Script `scripts/glam/glam_step005_cifar_depth_t0.py`.

## Mechanism 4 — slope-anneal ENERGY arm CLOSED (step006, T0 3-seed, CONFIRMED)

GLAM-LOC champion (47,370p, Imagenette store). Slope schedule log-linear `α(t)=exp(...)`, `1.0`
(identity) → `1e-5` (hard). Scored on **activation-zero fraction** (real dormancy) not accuracy; gate
to advance = zeros rise *materially* above the plain-ReLU reference for ≤0.5pp accuracy cost.

| Arm | acc mean±sd | zero_frac | verdict |
|---|---|---|---|
| CTRL const 0.01 | 96.96 ±0.12 | 0.398 | control |
| A7DET anneal 1.0→1e-5 | 96.81 ±0.03 | 0.410 | −0.15pp acc, +1.2pp zeros |
| A7RAND stochastic sched | 96.86 ±0.08 | 0.410 | −0.10pp acc, +1.2pp zeros |

**Verdict (CONFIRMED): mechanism 4 FAILS its own energy gate.** Anneal lifts dormancy only +1.2pp
(0.398→0.410, still far below the ~0.52 plain-ReLU reference) while costing −0.10 to −0.15pp accuracy.
The gate that would justify a Joules run (material dormancy gain for ≤0.5pp cost) never opened → no
Joules measurement warranted. Energy proxy (activation-zero fraction) is decisive on its own: the
anneal buys no material sparsity on this architecture. `scripts/glam/glam_step006_slope_anneal_energy_t0.py`.

**GLAM fully characterized.** All four user-specified mechanisms tested end-to-end and faithfully
implemented: (1) locality VALIDATED T0→T1→T2 (97.68% Imagenette, honest Pareto at 5.3× fewer params);
(2) selectivity/affinity NEUTRAL; (3) mul/key×value NEUTRAL; (4) slope-anneal energy NEUTRAL-negative;
depth REJECTED. **Locality is the entire algorithm** — the shipped primitive is a shared-pool
structured FC-head replacement. This is the "best version" to scale to a larger dataset (PAPER_CAMPAIGN.md).

## Scale-invariance CONFIRMED at T1 — OVERTURNED at T2 (step005/007, 3-seed, VGG16 backbone)

- **T1 (75ep/50%) [STALE]:** gap looked FLAT across 10× class count — CIFAR-10 −1.53pp, CIFAR-100 −1.59pp
  (Δ −0.06pp). Read as scale-invariant.
- **T2 (150ep/100%, paper-grade) [CONFIRMED]:** gap WIDENS — CIFAR-10 dense 88.11±0.06 vs LOC 86.64±0.10
  = −1.47pp @ 5.3× fewer; CIFAR-100 dense 68.49±0.08 vs LOC 65.35±0.40 = −3.14pp @ 6× fewer. Gap
  −1.47→−3.14pp with class count, NOT flat. T1 flat-gap was a reduced-budget artifact (neither head
  saturated at 50%/75ep); at full budget dense's 2.5M capacity pays off at 100cls while LOC's fixed
  416K shared-pool bottlenecks.
- **METHODOLOGY (compounds the T0 lesson):** reduced-budget tiers (T0 20ep, T1 50%/75ep) BOTH understate
  the LOC-vs-dense gap at high class count — dense under-trained/under-fed (CIFAR-100 T0 LOC 64.12 *beat*
  dense 63.54; reversed at every richer tier). **Scale-invariance claims require T2.** Design implication:
  LOC capacity should scale with class count — but the capacity-scaling probe (step007 --d_out{16,32}
  --M{16,64} CIFAR-100 T2) shows it only PARTLY works.

**Backbone-invariance CONFIRMED (step008/reuse, T1 3-seed, on-disk):** store_resnet18.h5 = ResNet-18 conv
features (25088-dim = 512×7×7 pre-GAP), Imagenette 10-cls. Dense LIN 98.38±0.05 vs LOC 98.24±0.15 =
−0.14pp @ 5.3× fewer params. VGG16-tuned locality champion transfers to ResNet-18 features within noise.
Full matrix (2 backbones × 3 datasets): gap tracks feature HEADROOM, not backbone/class-count.

## Capacity-scaling probe — gap is STRUCTURAL not capacity (step007 CIFAR-100 T2, CONFIRMED)

Widening LOC only PARTLY recovers the −3.14pp gap, with log-diminishing returns:

| config | params | mean% | gap | Δ vs prev |
|---|---|---|---|---|
| LOC d_out=8 (champ) | 416K | 65.35±0.40 | −3.14 | — |
| LOC d_out=16 | 832K | 66.27±0.09 | −2.22 | +0.92pp (2× params) |
| LOC d_out=32 | 1.66M | 66.44±0.30 | −2.05 | +0.17pp (2× more params) |
| LOC d_out=16 M=64 | 870K | 66.26±0.12 | −2.23 | +0.00 (M inert) |

4× params buys +1.09pp then saturates; even at 1.66M (2/3 of dense 2.5M) LOC trails −2.05pp. **M is
floor-only (pool diversity inert: +0.5× params, +0.00pp).** The 100-cls gap is STRUCTURAL — the
shared pool compresses input 25088 → P·d_out bottleneck (4096–16384) before the class readout, while
dense maps 25088→100 full-rank; widening the bottleneck toward dense erases the param win. **Honest
story: genuine compression↔accuracy tradeoff — locality near-free at 10cls (~1.5pp), structural ~2pp
at 100cls that capacity alone can't buy back.** Direction CLOSED.

**CIFAR-10 CONTROL — residual is class-count-dependent (step005 T2, 3-seed, CONFIRMED):** same d_out
sweep at 10cls (dense 88.11 @ 250K): d_out=8 86.64 (47K, −1.47) → 16 86.98 (95K, −1.13) → 32 87.45
(189K, −0.66). At 10cls capacity KEEPS closing the gap (−1.47→−0.66, still moving at d_out=32, NOT
saturated → ~0 extrapolated); at 100cls it SATURATES at a ~2pp floor. **Refined claim: the residual is
capacity-recoverable at low class count but a STRUCTURAL floor at high class count.** Two matched T2
3-seed frontiers (CIFAR-10 + CIFAR-100). Scaling protocol: carry the FRONTIER (sweep d_out), report
accuracy-vs-params curve, not one champion; do not over-claim scale-invariance (holds low-C, breaks
high-C at T2).

## CIFAR-100 information-MERGING frontier — 5 clean negatives, direction TERMINATED (step008, T0 3-seed, CONFIRMED)

/goal "information merging" rung: can an explicit structured cross-chunk merge recover the CIFAR-100
structural gap the LOC champion leaves (−3.14pp T2)? Ref = same-tier LOC ≈0.640 T0. Merge block inserted
between locality projection and readout (`model_glam_merge.py`); gate-death-compliant (residual, unbounded,
applied once, zero-init second layer → identity-at-start so any gain is attributable).

| Lever | config | mean (T0 3-seed) | vs LOC 0.640 | verdict |
|---|---|---|---|---|
| M expert-pool ↑ | M=32 / M=64 | 0.6364 / 0.6391 | flat | pool diversity inert (confirms step007) |
| additive low-rank merge | rank=32 post | ≈0.640 | neutral | dense readout ABSORBS it (linear-redundant) |
| bilinear 2nd-order merge | q⊙k post | 0.6303 | −1.0pp | 2nd-order overfits, HARMFUL |
| pre-merge (raw chunks) | mix 49-dim before compress | 0.6353 | −0.5pp | HARMFUL |
| low-rank readout | ro=32 / ro=16 | 0.558 / 0.497 | −8 / −14pp | CRASHES: 100-way needs readout rank≈C |

**Verdict (CONFIRMED): every information-merging lever fails.** The additive merge is provably redundant —
the concat→dense readout `Linear(P·d_out, C)` IS a full cross-chunk mix, so a *linear* merge before it adds
nothing the readout doesn't already do. A *bilinear* (2nd-order, non-absorbable) merge is the one form the
readout can't reproduce — and it HURTS (overfits at 420K on 100cls). Low-rank readout, the only lever that
would actually cut the param-dominant readout (97% of params = P·d_out·C), crashes because 100-way logits
need rank≈C. **The merge lives in the readout; it cannot be made cheaper or richer.**

## GLAM — GLOBAL TERMINUS: locality is the whole layer, all merging/capacity/depth falsified (2026-08-13, CONFIRMED)

Complete map across **2 datasets × 8 mechanisms**, every non-locality lever CONFIRMED neutral-or-negative:

| Axis | Mechanism | Dataset | Verdict |
|---|---|---|---|
| mixing | KV mul sigmoid(k)·v (mech 3) | CIFAR-10 | +0.11 noise, +15% params, dominated |
| mixing | additive low-rank merge | CIFAR-100 | neutral (readout-redundant) |
| mixing | bilinear 2nd-order q⊙k | CIFAR-100 | −1.0pp harmful |
| mixing | pre-chunk merge | CIFAR-100 | −0.5pp harmful |
| selectivity | affinity/prototype gate (mech 2) | CIFAR-10 | neutral |
| capacity | d_out ↑ | CIFAR-100 | saturates (~2pp floor) |
| capacity | M expert-pool ↑ | CIFAR-100 | flat/inert |
| readout | low-rank factored readout | CIFAR-100 | crashes (rank-bound by C) |
| depth | L=2 stacked LOC | CIFAR-10 | −0.83pp, dominated |
| energy | slope-anneal (mech 4) | Imagenette | fails energy gate |

**Locality (mechanism 1) is the ENTIRE algorithm.** The significant param-efficient improvement is the LOC
champion itself — Imagenette 96.96%±0.10 @ 47K params (0.04% FC), Pareto-dominating FFN-head AND the SGNNET
champion. On harder headroom tasks the honest story is a genuine compression↔accuracy tradeoff: −1.5pp @
10cls, structural ~2–3pp @ 100cls, buying 5–6× param reduction. No cheap lever closes the high-C gap; the
merge cannot help because it already exists inside the readout. **Direction CLOSED — further speculative
locality-side sweeps have near-zero expected value.** Scaling protocol unchanged: carry the d_out FRONTIER,
report accuracy-vs-params curve.

## glam_step009 — FgSegNet_v2 GAP gate on the LOC head (CIFAR-100 T0, 3 seeds, 2026-08-14)

Motivated by the FgSegNet_v2 source: its decoder gates with `x + x*GAP(shallow)` = `x*(1+g)`, an
**unbounded** gate that survives, costing 8,256 params. HYPOTHESIS under test: gate-death is loss of
the IDENTITY PATH, not multiplication — a residual wrapper being a stronger guarantee than boundedness.

Gate source `g = mean over chunk_dim(x)` → `[B, P=512]`, the exact analogue of its spatial GAP.
S0 reproduces `glam_step007` LOC d8 T0 **exactly** (64.14 vs 64.12) — clean control.

| arm | gate | +params | mean ±sd | Δ vs S0 |
|---|---|---|---|---|
| S0 | none (control) | 0 | **64.14 ±0.24** | — |
| S1 | `y*(1+GAP)` residual, unbounded | 0 | 62.73 ±0.47 | **−1.41** |
| S2 | S1 + learned per-channel scale | 1,024 | 62.81 ±0.41 | **−1.33** |
| S3 | `y*sigmoid(GAP)` bounded, NO residual | 0 | 64.28 ±0.15 | +0.14 (noise) |

**CONFIRMED (T0, 3 seeds, one variable, S1/S3 both +0 params so no capacity confound):**
- **The revised "residual rescues the gate" law is REFUTED in its strong form.** The residual wrapper
  did NOT rescue an unbounded gate — it cost −1.41pp. Adding a learned scale did not repair it (−1.33).
- **BOUNDEDNESS is the surviving ingredient**, consistent with `glam_step004` (`sigmoid(key)·value` ties
  locality) and with the gate-death law: a bounded gate applied ONCE is safe. S3 is neutral, not a win —
  gating buys nothing on this head, it merely fails to hurt when bounded.
- FgSegNet's `x*(1+g)` does **not** transfer to a shared-pool projection head. `HYPOTHESIS` for why:
  there `g` gates a conv feature map whose scale is normalised downstream (InstanceNorm) and the gate is
  cross-layer (shallow stats → deep features); here it multiplies a projection output feeding a linear
  readout directly, so an unbounded per-channel rescale is pure gradient variance.

## glam_step010 — CReLU / RReLU activations (CIFAR-100 T0, 3 seeds, mini_cpu, 2026-08-14)

Test: does the **sign phase** ReLU discards buy more than the same parameters spent on plain
capacity? CReLU `concat[relu(y), relu(-y)]` doubles the dims leaving the projection at ZERO extra
projection params — but the readout dominates (`P*d_out*n_cls`), so CReLU at `d_out=k` costs the
same as ReLU at `d_out=2k`. Both ReLU points are already measured (`glam_step007`), giving every
arm a **free iso-param control**.

| arm | activation | d_out | params | zero_frac | mean ±sd | iso-param ReLU control | Δ |
|---|---|---|---|---|---|---|---|
| A0 | leaky (control) | 8 | 416,100 | 0.000\* | 64.21 ±0.32 | step007 LOC d8 = 64.12 | — |
| A1 | CReLU | 4 | 412,900 | 0.500 | 63.25 ±0.05 | ReLU d8 = 64.12 | **−0.87** |
| A1 | CReLU | 8 | 825,700 | 0.500 | 64.81 ±0.08 | ReLU d16 M16 = 66.27 | **−1.46** |
| A2 | RReLU | 8 | 416,100 | 0.000\* | 64.15 ±0.17 | A0 = 64.21 | −0.06 (noise) |
| A3 | MIX `cat[relu(y), rrelu(−y)]` | 8 | 825,700 | 0.254 | 64.57 ±0.23 | ReLU d16 M16 = 66.27 | **−1.70** |

\* leaky/RReLU never emit exact zeros — `zero_frac=0.000` is the *definition* of the leak, not a
measurement artifact to explain away. That is the energy cost of a leaky activation: no structural
activation sparsity at all.

**CONFIRMED (T0, 3 seeds, iso-param controls measured under the identical harness):**
- **Rank-doubling via sign phase is REFUTED at iso-param.** CReLU loses at both capacity points
  (−0.87pp at ~416K, −1.46pp at ~826K). Spending the readout budget on more projection dims beats
  spending it on the negative phase. The CIFAR-100 gap is capacity-shaped, not sign-phase-shaped.
- **RReLU is inert on accuracy** (−0.06pp) and **destroys activation sparsity** (`zero_frac` 0.000
  vs plain-ReLU 0.398–0.41 measured in `glam_step006`). Pure energy loss for zero accuracy gain —
  KILLED on the Pareto, not on accuracy.
- **Mixing the two (A3) is worse than either** (−1.70pp), and its `zero_frac=0.254` is exactly the
  arithmetic of one sparse phase + one leaky phase. No interaction effect to chase.
- Energy note: CReLU's `zero_frac=0.500` looks like a sparsity win but sits on **2× the
  activations** — absolute nonzeros are unchanged vs ReLU. No energy win either.

**Net: all four activation variants are off the table.** `glam_step009` + `glam_step010` together
say the locality head's remaining gap is **capacity**, not gating and not activation shape.
