# Experiment Queue

## Strategic Direction (2026-04-09)
**All experiments from run10 onwards drop the nn.Linear FFN head.** The GNN must classify directly (output_nodes=10, act_strength as logits). `diagnose.ipynb` showed intermediate/input node weights barely changed during training — the FFN was doing most of the classification work. Since the project goal is to prove the *GNN* can replace VGG16's FC layers, keeping an FFN head defeats the purpose. Target: match run6's 85.81% without the FFN.

## Active Investigation
**LN-fix validation series COMPLETE.** run17 (T=4.0) → 86.85% @ep38 (+0.30pp vs run16, HYPOTHESIS — within variance). run18 (T=1.0) → 61.99% @ep40 (≈ run11 at 61.32%, CONFIRMED: new LN does NOT rescue T=1.0 starvation — T remains the dominant lever).

**Next investigation: FLOPs reduction.** Current best (run17) = 0.96B FLOPs/fwd for 86.85%. Target: match/exceed at ≤0.25B FLOPs.

**Cycle 1 — cardinality sweep COMPLETE:** C=200→86.85%, C=100→80.66% (-6.19pp), C=50→71.64% (-9.02pp), C=25→64.54% (-7.10pp), C=4→25.83% (cliff — starvation confirmed). Verdict: cardinality is NOT a free FLOPs lever. ~7-9pp cost per halving; C=4 falls off a cliff. Minimum viable C is between 25 and 100.

**Cycle 2 — beam_width sweep COMPLETE:** beam=0→86.85%, beam=2048→86.50% (-0.35pp), beam=1024→86.68% (-0.17pp), beam=512→85.78% (-1.07pp), beam=256→84.20% (-2.65pp). Verdict: beam is a strong FLOPs lever. **Winner: beam=1024 (N/4) — 2.6x FLOPs at -0.17pp (lossless).** Elbow at N/4–N/8. Cost per halving accelerates below N/8. See `concepts/beam_search.md`.

**Cycle 3 — vector_dim=16 DONE (run27 only):** run27 (N=2048, D=16, C=2 — team member replica) → 29.04% @ep36. CONFIRMED: our softmax routing fails at C=2 (66pp behind team member's 95.52%). run28 (C=25, D=16) deferred — routing mechanism is the priority.

**Cycle 4 — routing mechanism (COMPLETE):** BREAKTHROUGHS: (1) uniform routing beats softmax at C=200 (+2.30pp). (2) Iterations > cardinality as FLOPs lever. (3) D=16 + N=2048: Pareto-dominant. (4) **MILESTONE: run37 (uniform + D=16 + I=5) = 90.45% — broke 90%!** Compound gains: uniform routing + D=16 + full iterations. (5) **Cardinality cliff confirmed between C=10 and C=25.** Pareto frontier at D=16: 0.12B→77.71%, 0.23B→85.91%, 0.45B→89.55%, 0.90B→90.45%. Config sweeps exhausted — reaching 1M FLOPs requires architectural change.

**2026-04-11 architectural refactor (run 17+):** LN moved from pre-update (applied to source mag before aggregation — γ/β had no lasting effect on stored state) to post-update (applied to new_mag after update_activations — γ/β now genuinely learned). Hardcoded in-function mean-subtraction also removed. act_strength scale changed. To reproduce runs 1–16 exactly: `git checkout 7f072c4`. Full details: `concepts/mag_normalization.md`.

---

## Queue (active + pending only)

| Run | Status | Hypothesis | Key delta | Expected result |
|-----|--------|------------|-----------|-----------------|
| run19 | DONE | Cardinality at 200 is overkill; can halve FLOPs with no accuracy loss | C=100 (else = run17) | **80.66% @ep40**, -6.19pp; still improving |
| run20 | DONE | Cardinality can go 4x lower than best config | C=50 (else = run17) | **71.64% @ep40**, -15.21pp vs run17; 4x FLOPs but steep cost |
| run21 | DONE | Aggressive cardinality reduction — find the cliff | C=25 (else = run17) | **64.54% @ep40**, -22.31pp vs run17; ~7-9pp cost per halving confirmed |
| run22 | DONE | Extreme cardinality probe — anchors floor of trend | C=4 (else = run17) | **25.83% @ep40** — starvation confirmed; 2.6x random chance only |
| run23 | DONE | Beam filtering at N/2 preserves accuracy | beam_width=2048 (else = run17) | **86.50% @ep40**, -0.35pp; lossless at N/2 |
| run24 | DONE | Beam at N/4 — moderate filter | beam_width=1024 (else = run17) | **86.68% @ep40**, -0.17pp vs run17; *better* than beam=2048 |
| run25 | DONE | Beam at N/8 — aggressive filter, likely elbow | beam_width=512 (else = run17) | **85.78% @ep40**, -1.07pp vs run17; elbow confirmed between N/4 and N/8 |
| run26 | DONE | Beam at N/16 — cliff probe for winner-take-all | beam_width=256 (else = run17) | **84.20% @ep39**, -2.65pp vs run17; curve steepens below N/8 |
| run27 | DONE | Can our architecture replicate team member's C=2, D=16 result? | N=2048, D=16, C=2, input=1568 (else = run17) | **29.04% @ep36** — CONFIRMED: softmax routing fails at C=2, 66pp behind team member |
| run28 | DEFERRED | D=16 at cliff-level cardinality — does larger dim help? | N=2048, D=16, C=25, input=1568 (else = run17) | Deferred — routing mechanism is the priority now |
| run29 | DONE | Does removing softmax (uniform routing) match run17 at C=200? | **uniform 1/degree routing**, C=200 (else = run17) | **89.15% @ep38** — **+2.30pp vs run17!** Softmax HURTS accuracy. NEW BEST |
| run30 | DONE | Does uniform routing fix starvation at C=2? | uniform routing, **C=2** (else = run17) | **18.39% @ep37** — WORSE than softmax C=2 (29.04%). Sparsity is structural, not routing |
| run31 | DONE | Uniform routing at moderate C — where's the crossover? | uniform routing, **C=50** (else = run17) | **79.54% @ep40** — **+7.90pp vs softmax C=50**. Gain peaks at moderate C |
| run32 | DONE | Uniform routing at C=100 — find the accuracy peak per-FLOP | uniform routing, **C=100** (else = run17) | **85.89% @ep40** — +5.23pp vs softmax, nearly matches run17 at half FLOPs |
| run33 | DONE | Iterations as FLOPs lever — does I=3 maintain accuracy? | uniform routing, **I=3** (else = run29) | **87.16% @ep40** — beats run17 at half FLOPs! Iterations > cardinality as FLOPs lever |
| run34 | DONE | Extreme iteration reduction — does I=2 still work? | uniform routing, **I=2** (else = run29) | **72.05% @ep40** — I=2 below iteration floor; C=50/I=5 (run31) still wins at 0.24B |
| run35 | DONE | Fill iteration curve mid-point | uniform routing, **I=4** (else = run29) | **88.56% @ep40** — smooth curve I=3→5 confirmed, diminishing returns |
| run36 | DONE | Does D=16 push past 89.15% ceiling? | uniform, N=2048, **D=16**, C=200, I=3 | **89.55% @ep40** — NEW BEST at HALF the FLOPs of run29! Pareto-dominant |
| run37 | DONE | Can D=16 + I=5 break 90%? | uniform, N=2048, D=16, C=200, **I=5** | **90.45% @ep40** — BROKE 90%! NEW ABSOLUTE BEST |
| run38 | DONE | Can D=16 + C=100 hold accuracy at half FLOPs? | uniform, D=16, **C=100**, I=3 | **85.91% @ep39** — matched D=8/C=200 at HALF FLOPs! Pareto @ 0.23B |
| run39 | DONE | Can D=16 + C=50 extend Pareto to 0.12B? | uniform, D=16, **C=50**, I=3 | **75.49% @ep40** — below run31 (D=8, C=50, I=5: 79.54%). Low C needs high I |
| run40 | DONE | Does I=5 rescue C=50 at D=16? | uniform, D=16, C=50, **I=5** | **83.06% @ep39** — +7.57pp from I=3→5, but run38 still better at 0.23B |
| run41 | DONE | Can D=16 + C=10 hit viable accuracy at 0.06B FLOPs? | uniform, D=16, **C=10**, I=5 | **27.97% @ep39** — below viability floor, cliff between C=10 and C=50 |
| run42 | DONE | Pinpoint cardinality cliff | uniform, D=16, **C=25**, I=5 | **77.71% @ep40** — viable! Cliff is between C=10 and C=25, not C=25 and C=50. Pareto-dominant at 0.12B tier |

---

## Completed Runs — FFN-free (runs 10–17)

**Note:** All FFN-free runs use output_nodes=10 with act_strength values fed directly into CrossEntropyLoss. No nn.Linear head.

| Run | Date | Status | Val Best | Epochs | N | T | Data% | Key delta from run10 | Verdict |
|-----|------|--------|----------|--------|---|---|-------|----------------------|---------|
| run10 | 2026-04-09 | DONE (stopped ep23) | 40.33% @ep23 | 23/40 | 15454 | 1.0 | 100% | Baseline FFN-free (no FFN, output_nodes=10) | GNN learns without FFN but very slowly; gradient starvation confirmed |
| run11 | 2026-04-09 | DONE | 61.32% @ep37 | 40/40 | 4146 | 1.0 | 100% | total_nodes=4146 (3136+1000+10) | +20.99pp vs run10 (HYPOTHESIS: fewer params, denser connectivity, simpler landscape — confounded) |
| run12 | 2026-04-09 | DONE | 82.19% @ep39 | 40/40 | 15454 | 4.0 | 50% | routing_temperature=4.0 | +41.86pp vs run10; T=4.0 CONFIRMED key lever (single-variable ablation) |
| run13 | 2026-04-10 | DONE | **85.40% @ep40** | 40/40 | 15454 | 7.0 | 50% | routing_temperature=7.0 | +3.21pp vs run12; T=7.0>T=4.0>T=1.0 monotonic; 0.41pp gap from FFN target (85.81%) |
| run14 | 2026-04-10 | DONE | 45.48% @ep40 | 40/40 | 4146 | 1.0 | 50% | topology=layered (delta from run11: layered + 50% data) | -15.84pp vs run11 (HYPOTHESIS: layered worse than flat, but data fraction confounds — not clean ablation) |
| run15 | 2026-04-11 | DONE | 76.97% @ep27 | 40/40 | 15454 | 7→1 | 50% | routing_temperature annealed 7.0→1.0 per-step (delta from run13) | -8.43pp vs run13 (CONFIRMED: annealing HARMFUL — low-T end causes starvation, val collapses to 52.66% by ep40) |
| **run16** | 2026-04-11 | DONE | **86.55% @ep40** | 40/40 | 4146 | 4.0 | 100% | total_nodes=4146 + data_fraction=1.0 (delta from run12) | **EXCEEDED FFN target (+0.74pp); N=4146+T=4.0 compound confirmed; still improving at ep40 — proof-of-concept ACHIEVED** |
| **run17** | 2026-04-11 | DONE | **86.85% @ep38** | 40/40 | 4146 | 4.0 | 100% | **NEW LN** (post-update, learnable γ/β) — all else identical to run16 | +0.30pp vs run16; HYPOTHESIS: new LN helps slightly at best config. Early epochs slower, converges faster in final stage. Train acc lower (81.7% vs 82.3%) → less overfitting |
| run18 | 2026-04-11 | DONE | 61.99% @ep40 | 40/40 | 4146 | 1.0 | 100% | routing_temperature=1.0 (delta from run17) | ≈ run11 (61.32%); CONFIRMED: new LN does NOT rescue T=1.0 gradient starvation. T remains dominant lever under both LN regimes. |

### FFN-free Learnings

- **routing_temperature is the primary lever (CONFIRMED):** T=1.0→40.33%, T=4.0→82.19%, T=7.0→85.40%. Clean single-variable ablations. Gradient starvation at T=1.0 (0.4% nodes carry 50% gradient) is fixed by higher T.
- **GNN nearly matches FFN target at T=7.0:** run13 reached 85.40% without FFN, without full data — only 0.41pp below run6 (85.81% with FFN). Proof-of-concept is nearly complete.
- **Convergence not reached at T=7.0:** run13 best at final epoch (ep40), still improving. More epochs or full data likely closes remaining gap.
- **Fewer nodes helpful (HYPOTHESIS from run11):** 1K intermediates → +20.99pp vs run10. Confounded: fewer params (66K vs 247K), denser connectivity (5% vs 1.3%), simpler landscape. Not a clean ablation.
- **Layered topology worse than flat (HYPOTHESIS from run14):** layered→45.48% vs flat→61.32% (run11), -15.84pp gap. CONFOUNDED: run14 used 50% data vs run11's 100%. Clean ablation (run14 repeated with 100% data) needed to confirm.
- **Temperature annealing 7→1 is HARMFUL (CONFIRMED from run15):** val_best=76.97% @ep27 vs run13 T=7.0 fixed (85.40%), -8.43pp. Single-variable ablation. Low-T end-state causes gradient starvation to return; val collapsed from 76.97% to 52.66% by ep40 as T→1. Do NOT anneal temperature downward.
- **N=4146 + T=4.0 compound exceeds FFN target (HYPOTHESIS from run16):** run16 (N=4146, T=4.0, 100% data) → 86.55% @ep40, +4.36pp vs run12 (N=15454, T=4.0, 50% data, 82.19%), +1.15pp vs run13 (N=15454, T=7.0, 85.40%). Exceeded FFN target (85.81%) by +0.74pp. Still improving at ep40. Two variables changed vs baselines — compound effect HYPOTHESIS. Primary driver likely T=4.0 (CONFIRMED lever) with smaller graph + full data adding further gain. **Proof-of-concept ACHIEVED: GNN without FFN head exceeds FFN-based accuracy.**
- **New post-update LN slightly improves best config (HYPOTHESIS from run17):** run17 (new LN, all else = run16) → 86.85% @ep38, +0.30pp vs run16 (86.55%). Single-variable change (LN placement), but modest gain could be within run-to-run variance. Early epochs slower (ep10: 73.99% vs run16's 76.20%), converges faster in final stage, peaks at ep38 then slightly declines. Train acc lower (81.7% vs 82.3%) → less overfitting, better generalization. Needs γ/β inspection in diagnose.ipynb to confirm learnable params are contributing.
- **T=1.0 starvation persists under new LN (CONFIRMED from run18):** run18 (new LN, T=1.0, else=run17) → 61.99% @ep40, matches run11 (old LN, T=1.0) at 61.32%. Clean single-variable ablation vs run17 confirms T remains the dominant lever for escaping gradient starvation, independent of normalization scheme. The LN placement fix does NOT shift the T scale. Decision gate: no T-sweep follow-up under new LN needed.

---

## Completed Runs — With FFN (runs 1–9)

**Note:** Runs 1–9 all used `nn.Linear(output_nodes=256, 10)` as a classification head after the GNN. These results reflect the combined GNN+FFN system, not the GNN alone.

| Run | Date | Status | Topology | Iters | Key delta | Val best | Epochs | Verdict |
|-----|------|--------|----------|-------|-----------|----------|--------|---------|
| run1 | 2026-03-25 | DONE | flat | 5 | baseline (no layernorm, no dropout, no radiation) | 76.05% | 20/20 | Baseline |
| run2 | 2026-03 | DONE | flat | 5 | total_nodes=30000, cardinality=1000 | <15% | 4 (stopped) | Unstable, inconclusive |
| run3 | 2026-03-25 | DONE | flat | 5 | +layernorm=true | 81.38% | 20/20 | +5.33pp from run1; layernorm CONFIRMED helpful |
| run4 | 2026-03-26 | DONE | flat | 5 | +radiation_targets=32, scattering_prob=0.8, +lr_decay | 16.89% | 3 (stopped) | Collapsed; radiation too aggressive |
| run5 | 2026-03-27 | DONE | flat | 5 | radiation_targets=16, scattering_prob=0.25, 40ep | 71.46% | 40/40 | Radiation still hurts vs run3 (HYPOTHESIS) |
| run6 | 2026-03-28 | DONE | flat | 5 | radiation=0 (off), +dropout=0.2 | **85.81%** | 40/40 | **BEST with FFN — FFN-free target** |
| run7 | 2026-04-02 | DONE | layered | 7 | +layered, temporal_decay=0.8 | 23.08% | 16 (stopped) | Poor; 3 vars changed, confounded |
| run8 | 2026-04-02 | DONE | layered | 15 | +beam_width=500, CPU | 13.32% | 23 (stopped) | Poor; highly confounded |
| run9 | 2026-04-02 | DONE | layered | 7 | +layered only (closest to run6 ablation) | 25.45% | 38/40 | Poor; 60pp gap vs run6 |

### FFN Learnings

- **LayerNorm (CONFIRMED):** +5.33pp in clean run1→run3 ablation. Always on from run3+.
- **Radiation hurts (HYPOTHESIS):** run4/5 worse than run3/run6. No clean single-variable ablation isolating radiation.
- **Dropout (HYPOTHESIS):** added in run6 alongside radiation removal — unconfirmed individually.
- **Layered topology poor with FFN (HYPOTHESIS):** run7/8/9 all <26%. run9 is closest to clean ablation (layered only vs run6) but still confounded by iters change.

---

## Consolidated Results: Accuracy, FLOPs & Parameters

FLOPs per forward pass: `(I-1) * B * N * (8*C*V + 7*C + 24*V)`. Parameters: `2*N*V + [2V if LN] + [n_out*10+10 if FFN]`. See `concepts/flops.md` for derivation.

Variables: `I`=iterations, `B`=batch size, `N`=total_nodes, `C`=cardinality (edges/node), `V`=vector_dim, `n_out`=output_nodes.

| Run | Val Best | Epochs | Params | FLOPs/fwd | FLOPs/step | N | C | I | FFN | T | Data% |
|-----|----------|--------|--------|-----------|------------|-------|-----|---|-----|-----|-------|
| run1 | 76.05% | 20/20 | 249,842 | 3.56B | 10.7B | 15454 | 200 | 5 | Yes | 1.0 | 100% |
| run2 | <15% | 4 (stopped) | 482,570 | 6.91B | 20.7B | 30000 | 1000 | 5 | Yes | 1.0 | 100% |
| run3 | 81.38% | 20/20 | 249,858 | 3.56B | 10.7B | 15454 | 200 | 5 | Yes | 1.0 | 100% |
| run4 | 16.89% | 3 (stopped) | 249,858 | 3.56B | 10.7B | 15454 | 200 | 5 | Yes | 1.0 | 100% |
| run5 | 71.46% | 40/40 | 249,858 | 3.56B | 10.7B | 15454 | 200 | 5 | Yes | 1.0 | 100% |
| **run6** | **85.81%** | 40/40 | 249,858 | 3.56B | 10.7B | 15454 | 200 | 5 | Yes | 1.0 | 100% |
| run7 | 23.08% | 16 (stopped) | 249,858 | 5.34B | 16.0B | 15454 | 200 | 7 | Yes | 1.0 | 100% |
| run8 | 13.32% | 23 (stopped) | 249,858 | 17.8B | 53.4B | 15454 | 200 | 15 | Yes | 1.0 | 100% |
| run9 | 25.45% | 38/40 | 249,858 | 5.34B | 16.0B | 15454 | 200 | 7 | Yes | 1.0 | 100% |
| run10 | 40.33% | 23 (stopped) | 247,280 | 3.56B | 10.7B | 15454 | 200 | 5 | No | 1.0 | 100% |
| run11 | **61.32%** | 40/40 | 66,352 | 0.96B | 2.88B | 4146 | 200 | 5 | No | 1.0 | 100% |
| run12 | **82.19%** | 40/40 | 247,280 | 3.56B | 10.7B | 15454 | 200 | 5 | No | 4.0 | 50% |
| run13 | **85.40%** | 40/40 | 247,280 | 3.56B | 10.7B | 15454 | 200 | 5 | No | 7.0 | 50% |
| run14 | 45.48% @ep40 | 40/40 | 66,352 | 0.96B | 2.88B | 4146 | 200 | 5 | No | 1.0 | 50% |
| run15 | 76.97% @ep27 | 40/40 | 247,280 | 3.56B | 10.7B | 15454 | 200 | 5 | No | 7→1 | 50% |
| **run16** | **86.55%** | 40/40 | 66,352 | 0.96B | 2.88B | 4146 | 200 | 5 | No | 4.0 | 100% |
| **run17** | **86.85%** | 40/40 | 66,352 | 0.96B | 2.88B | 4146 | 200 | 5 | No | 4.0 | 100% |
| run18 | 61.99% | 40/40 | 66,352 | 0.96B | 2.88B | 4146 | 200 | 5 | No | 1.0 | 100% |
| run19 | **DONE** | **80.66% @ep40** | 66,352 | 0.48B | 1.44B | 4146 | 100 | 5 | No | 4.0 | 100% |
| run20 | DONE | **71.64% @ep40** | 66,352 | 0.24B | 0.72B | 4146 | 50 | 5 | No | 4.0 | 100% |
| run21 | DONE | **64.54% @ep40** | 66,352 | 0.12B | 0.36B | 4146 | 25 | 5 | No | 4.0 | 100% |
| run22 | DONE | **25.83% @ep40** | 66,352 | 0.02B | 0.06B | 4146 | 4 | 5 | No | 4.0 | 100% |
| run23 | **DONE** | **86.50% @ep40** | 66,352 | ~0.57B | ~1.71B | 4146 | 200 | 5 | No | 4.0 | 100% |
| run24 | DONE | **86.68% @ep40** | 66,352 | ~0.37B | ~1.11B | 4146 | 200 | 5 | No | 4.0 | 100% |
| run25 | **85.78% @ep40** | 40/40 | 66,352 | ~0.27B | ~0.81B | 4146 | 200 | 5 | No | 4.0 | 100% |
| run26 | **84.20% @ep39** | 40/40 | 66,352 | ~0.22B | ~0.66B | 4146 | 200 | 5 | No | 4.0 | 100% |

Notes:
- FLOPs/fwd = forward pass only. FLOPs/step = fwd + backward (3x with grad checkpointing).
- run2 had C=1000 (5x edges), explaining its high FLOPs despite same architecture.
- run11 has 3.7x fewer FLOPs than run10 (fewer nodes: 4146 vs 15454).
- run12/13 have same per-step FLOPs as run10 but half the steps/epoch (50% data).
