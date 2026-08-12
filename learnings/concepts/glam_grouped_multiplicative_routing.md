# GLAM — Grouped Local Anti-hebbian Multiplicative routing

**Status:** proposed 2026-08-13. Parallel research line (separate worker). All causal claims here
are `HYPOTHESIS` until the T0 ablation ladder lands; the constraint table cites prior `CONFIRMED`
evidence. Standing goal: parameter/energy efficiency, judged on the full Pareto table.

## Idea in one line

Replace SGNNET's parameter-free `_route` (`model_smallworld.py:301`) with a small structured
learned operator: a **shared pool of `D×D` experts**, assigned to frozen input groups by
constrained random sampling, combined multiplicatively (key×query), with input-affinity
selectivity and an anti-Hebbian decorrelation penalty. Must pay for itself against the param-free
baseline on accuracy-per-param.

## Four mechanisms (each independently ablated)

1. **Locality / param reduction** — `G` groups (reuse `n_groups`=256), a shared pool of `M`
   experts (`E_m ∈ R^{D×D}`, `M·D²` params, `M≪G·n`). Each group drawn `n=2` experts by seeded
   sampling *without over-reuse*: `p(m) ∝ max(0, 1 − used_m/quota)`, `quota=⌈G·n/M⌉`. Stored as a
   frozen buffer `assign:[G,n]` — a **topology, never rebuilt** (FM6). Params `O(M·D²)`,
   independent of `N`.
2. **Selectivity** — each expert carries prototype `p_m∈R^D` (`M·D` params). Affinity
   `a_{g,m}=cos(z̄_g, p_m)`; contribution scales by `|a|` — unbounded, geometrically primed like
   ΔW-proj, **not** `σ(τa)`. Multiplying the forward path also scales that expert's gradient (one
   mechanism, both effects). Anti-Hebbian term penalises co-firing:
   `L_decor = mean_g offdiag|corr(a_i,a_j)|`.
3. **Mixing** — `q=E_a z`, `k=E_b z`; combine `z' = z + (q ⊙ k)` — residual, unbounded, applied
   **once** after the `K_iter` routing rounds, then `_normalise`. Random second-order pairing:
   experts act on group `g` and `π(g)` via a frozen permutation, so the product mixes patches; a
   second permutation `σ` re-pairs and multiplies once more. Zero-param index buffers.
4. **Leaky-slope decay → hard ReLU** — negative slope `α` schedules log-linearly from `1.0`
   (identity: net starts fully linear) to `1e-5` (numerically hard ReLU) over the whole of
   training as a fraction of total epochs: `α(t)=exp(logα₀+(logα₁−logα₀)·t)`, `t=ep/epochs`.
   Inference always hard ReLU → zeros are real, NVIDIA zero-multiply skipping applies. Scored on
   **energy** (Joules/inference + activation-zero fraction), not accuracy alone.

## Hard constraints from prior evidence (design rules, not cautions)

| Rule | Evidence | Tag |
|---|---|---|
| No bounded `g∈[0,1]` gate per routing step — compounds `g^K_iter`, net nulls it. Structural. | `gate_death.md:5-15`, steps 51–66, 873–916 | CONFIRMED |
| Surviving multiplicative mechanism (ΔW-proj, `Z·proj.abs()`, +1.49pp) is **unbounded, geometrically primed** — nonzero grad from step 1 | `delta_w.md:26-32` | CONFIRMED |
| A branch with no structural prior collapses alone (`C_gate_only` −78.60pp) | step898, `EXPERIMENT_QUEUE_history_part4.md:60` | CONFIRMED |
| Two mechanisms on the same signal path co-adapt (soft_routing+ΔW −60.51pp) | FM5, `dynamic_routing_analysis.md:104-118` | CONFIRMED |
| Topology frozen post-init; weights may be dynamic, edges may not (−79pp) | FM6, `dynamic_routing_analysis.md:126-134` | CONFIRMED |
| AH is position-only/static per input; AH + extra gate "always destructive" (step66 40.33% vs 83.18%) | `antihebbian.md`, `gate_death.md:90-106` | CONFIRMED |
| MoE load-balance loss: +0.21pp, below noise, removed | step79, `INDEX.md:182` | CONFIRMED |
| Random-group membership beat spatial groups +2.14pp | step82, `group_topology.md:41-49` | CONFIRMED |

Two consequences that reshape the naive proposal:
- The key×query product is applied **once, unbounded, residual** (mirrors the ΔW-proj survivor),
  never as a per-step squashed gate. `HYPOTHESIS`: this is what makes multiplication survivable.
- Selectivity shares the AH signal path, so it is ablated with **AH disabled**; a separate arm
  (A4b) tests the "AH + extra gate always destructive" rule on the current arch.
- Load balance uses the **aux-loss-free bias** (per-expert bias on `a`, nudged by usage, no aux
  gradient) *because* the classic balance loss already failed — the gradient-free variant is the
  untried thing. `HYPOTHESIS`.

## Depth over breadth (guiding principle)

Buy capacity with **depth** (`L` stacked GLAM blocks, cost `L·M·D²`), keep breadth (`N`, `M`) at
the minimum that gives the sampler enough distinct pairs (`M≥2n` with slack). The `M∈{16,64}`
sweep exists only to find the pool-size *floor*; freed budget goes into `L`. Depth is exactly
where `g^K` compounding kills naive designs — so the once-applied unbounded residual product is
the enabler, and **depth scaling (A8) is its own measured arm**, not an assumption.

## T0 ablation ladder

T0 = 20ep/50% data, rejection filter (advance every positive/neutral arm). Shared: seed,
N=2048, D=16, K_iter=5, K_in=25, n_groups=256, fourier. **Ref = same-tier T0 Ref_dw ≈ 94.0–94.1%**
(step904/906/907) — never the 95.95% champion or 95.52%; cross-tier comparison is invalid.

| Arm | Content | Isolates / falsifies |
|---|---|---|
| A0 | canonical SmallWorld + ΔW-proj | Ref_dw ~94.0% |
| A1 | grouped shared-expert route, additive `q+k`, no mix/select | H: locality alone ≥ Ref at fewer params |
| A2 | A1 + residual product `z+q⊙k` | H: once-applied unbounded product survives gate-death |
| A3 | A1 + random second-order mixing (additive) | H: cross-patch mixing helps |
| A4 | A1 + selectivity + `L_decor`, **AH disabled** | H: affinity selectivity helps |
| A4b | A4 + AH enabled | tests "AH + gate always destructive" on this arch |
| A5 | A1+A2+A3+A4, L=1 | H: mechanisms compound (expect FM5 exposure) |
| A6 | A0 with N widened to A5 param count | param-matched **width** control |
| A7-det | A5 + log-decayed deterministic leaky slope 1.0→1e-5 | energy arm |
| A7-rand | A7-det but RReLU sampled around scheduled α | schedule vs stochasticity |
| A8 | A5 stacked L∈{2,4}, M at A1 floor | **depth-over-breadth**: depth beats A6 width at equal params? |

Load-bearing: **A6 and A8 are the point** — A6 shows structure beats params; A6-vs-A8 shows depth
beats width at equal params. A2 and A4 both touch `Z`, so A5 is most exposed to FM5 co-adaptation;
read A2/A3/A4 as the real result if A5 fails. Every arm reports the full Pareto row (acc, params,
FLOPs); survivors add wall-time + peak memory via `bench_step608`. An arm losing accuracy while
winning params/FLOPs/energy stays alive. T0→T1→T2 gain compression is severe
(+1.12→+0.18pp, step907→911) — no claim off T0/T1 alone.

## Instrumentation (from run 1)

Affinity/gate entropy per epoch (template `train_step904_node_gating_t0.py`), dead-expert
fraction (`|a|<0.05` for >99% inputs), per-expert grad-norm histogram, branch magnitude ratio
`‖q‖/‖k‖` (product collapse → divergence), activation-zero fraction per layer (mechanism 4).
`TrainingDiagnostics` wired per `CLAUDE_reference.md:48-51`.

## Files

- Model: `src/sgnnet/model_glam.py` (`SGNNET_GLAM(SGNNET_SmallWorld)`, overrides `_route`).
- Script: `scripts/glam/glam_step001_ablation_t0.py` (`--arm A0..A8`, `--M`, `--L`,
  `--slope_epochs`), results → `results/glam/`.
- Queue: `learnings/EXPERIMENT_QUEUE.md` block "QUEUED — GLAM grouped multiplicative routing".
