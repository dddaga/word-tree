<!-- continued from EXPERIMENT_QUEUE_history.md -->
## P1 — High value: efficiency gains + input pipeline + confirmed stacking

| Step | Description | Scale | FLOPs | Script |
|------|-------------|-------|-------|--------|
| **step149** | **Input de-squashification** — multi-feature projection + attention scatter | N=1024 | ~3.1M | ✅ |
| **step150** | **Positional max-pool readout** — max-pool + source neuron W_pos identity | N=1024 | ~3.1M | ✅ |
| **step151** | **Input+Output pipeline compound** — step149 winner + step150 winner (gated on both) | N=1024 | ~3.1M | Gated |
| **step144** | **Efficiency stack** — W_proj + RigL + α=1.10 combos at N=1024 | N=1024 | ~6.1M | ✅ |
| **step142** | **Curriculum K_iter** — ramp 4→8→12 during training | N=1024 | ~3-4M | ✅ |
| **step143** | **Heterogeneous neurons** — subpopulations with different θ, edge weights | N=1024 | ~3.1M | ✅ |
| **step127** | **K_iter distillation** — train K=12 teacher, distill to K=6 (50% FLOPs cut) | N=1024 | ~3M | ✅ |
| **step163** | **Progressive K_iter distillation** — teacher init (load_state_dict) + intermediate state matching Z_student[k]↔Z_teacher[t_k]. Configs: Ref(K=12), A(scratch K=6), B(warm-init K=6 no distil), C/D(warm-init+distil α=0.3/0.5), E(K=8 warm-init+distil). Key fixes over step127: shared conn_hh/conn_in via state_dict, warm-start isolates init vs distil effects | N=1024 | ~3M | ✅ |
| step133 | **α calibration Tier-1 at N=4096** | N=4096 | 38.8M | DONE — B=97.27% WINNER (α=1.05), A=97.20% (α=1.10), Ref=96.48% |
| step132 | **Low-rank W_proj Tier-1 at N=4096** | N=4096 | 38.8M | DONE — +0.06pp NULL |

---

## P1.5 — Regularization + remaining transformer techniques

| Step | Description | Scale | Script |
|------|-------------|-------|--------|
| **step158** | **DropMessage regularization** — stochastic edge dropping in K_iter routing, combat over-smoothing. Configs: drop=0.1/0.2/0.3 + late-only variant. 75ep Tier-0 | N=1024 | ✅ |
| **step121** | **Spectral norm / regularization on W_pos** — spectral norm, soft reg, re-orthogonalize | N=1024 | ✅ |
| **step120** | **High K_iter (16-24) + Z-bias + grad checkpoint** at N=4096 | N=4096 | ✅ |
| step72 | **N-scaling patched arch** — full curve N={256-8192} | Multi-N | ✅ |
| step108-C | **Polar routing Tier-1** — region-based lookup, −9% FLOPs | N=1024 | ✅ |

---

## P2 — Lower priority / gated

| Step | Description | Depends on | Script |
|------|-------------|------------|--------|
| step126 | µP initialization for N-scaling | step72 | ✅ |
| step104 | Compound wave+polar | step102+103 (both weak) | ✅ |
| step122 | DiffPool hierarchical readout | step118 showed readout changes catastrophic | ✅ |

---

## P3 — Deprioritized

| Step | Description | Reason |
|------|-------------|--------|
| step92 | ReLU group routing | 0 wins in 3 attempts |
| step85 | Stacked SGNNET parallel | Old design |
| step93/94/97 | GRAND/Hamiltonian/Beltrami | Speculative, no script |

---

## Launch Order (when slots free)

1. ~~step140~~ DONE — N×K tradeoff KILLED
2. **step141** → RUNNING on Mac Studio MPS — Split-D + residual
3. **step155** → RUNNING on Mac Mini CPU — Diagnostics baseline (B,D,E)
4. **step116** → next slot — RMSNorm ablation
5. **step152** → next slot — Constraint discovery (core hypothesis test)
6. **step153** → next slot — Progressive capacity reduction
7. **step149** → next slot — Input de-squashification
8. **step150** → next slot — Positional max-pool readout
9. **step144** → next slot — Efficiency stack (N=1024 winners)
10. **step142** → next slot — Curriculum K_iter
11. **step143** → next slot — Heterogeneous neurons
12. **step127** → next slot — K_iter distillation
13. **step156** → next slot — LayerNorm N=4096 validation (20ep scout, high priority)
14. **step158** → next slot — DropMessage regularization (N=1024 75ep Tier-0)
15. **step121** → next slot — Spectral regularization

---

## Completed Experiments (2026-04-09/10 Session — 24 experiments)

### Winners
| Step | Result | Finding |
|------|--------|---------|
| step116-C ✓ | +2.24pp (N=1024 Tier-1) | LayerNorm with learned affine beats L2 sphere norm. Pending N=4096 validation |
| step128-A ✓ | +8.07pp (N=1024) | weighted_neg β=0.3. Zero params |
| step117-A ✓ | +8.66pp (N=1024) | W_proj [D,D]. 4K params |
| step131-B ✓ | +5.48pp (Tier-1 N=1024) | W_proj confirmed. Compound kills gain |
| step131-A ✓ | +3.97pp (Tier-1 N=1024) | weighted_neg confirmed |
| step124-B ✓ | +6.82pp (D=32 N=1024) | RigL topology, all configs positive |
| step125-B ✓ | +1.53pp (N=4096) | α=1.10 winner. Free param change |
| step132-B ✗ | +0.06pp (Tier-1 N=4096) | Low-rank W_proj NULL at scale (scout +0.94pp → Tier-1 +0.06pp) |

---

## Earlier Completed (Gen4 reference)

| Step | Result | Key finding |
|------|--------|-------------|
| step69-89 | See LEARNINGS files | Established current defaults: AH=1.0, K_hh=4, K_iter=12, turing=0.0, reflect=0.5 |
| step89 ✓ | **97.86%** | **PROJECT BEST** (N=4096, 150ep, full data) |
| step90 ✓ | null at N=4096 | Group topology doesn't scale |
| step105 ✓ | +4.21pp | Phase+AH synergistic confirmed |
| step106 ✓ | +7.42pp | Z-bias N=1024 record |
| step107 ✗ | KILLED | Group MoE routing killed |
| step112 ✗ | KILLED | RNN sequential injection killed |

---

## P-BACKLOG — 12 truly-fresh audit items (added 2026-04-14 session-7)

Per user decision: queue below current high-priority work. Pick up after:
(a) latency-Pareto track completes (step750-759), (b) seed-variance study completes (step760 × 4 configs), (c) ConnGA v2 verdict finalized (step740/741/742).

| Step | Description | Priority tag |
|------|-------------|--------------|
| step117 | N=2048 efficiency-config retest of an N=1024 winner | RUNNING (studio_cpu) |
| step124 | RigL-topology variant at efficiency config | SKIP — step709 confirmed RigL dead |
| step125 | α=1.10 at efficiency config | SKIP — step321 confirms α=1.0 optimal; α=1.25/1.50 both −1.6pp |
| step128 | ConcatReLU mechanism at efficiency config (N=1024 winner retest) | RUNNING (studio_mps) |
| step229 | RigL with fix (previous killed) | SKIP — RigL confirmed dead in step709 |
| step230 ✗ | Gumbel-Softmax differentiable topology — ALL KILLED. Best 48.74% vs ref 91.8% (Δ≈−43pp at 20ep T0). Topology entropy stays near-uniform; ST-GS doesn't learn edge structure. Added to architecture_dead_ends.md. | DONE — KILLED |
| step231 ✓ | Mechanism diagnostics — ALL 5 HYPOTHESES CONFIRMED. H1 class-specificity=0.24, H2 AH diversifies, H3 routing refines 0.09→0.92, H4 input-dependent code (0.30 overlap), H5 W_pos learning essential (+74.8pp). Paper-grade ablation evidence. | DONE — CONFIRMED |
| step401_full ✓ | MLP/Lin paper baselines at 150ep (CUDA) — Lin_direct=97.12% (250K), MLP_2=68.23% (50K), MLP_3=52.41% (75K), MLP_64=97.25% (1.6M). At matched params (75K), SGNNET beats MLP by 44pp. At matched accuracy, MLP needs 24× more params. Paper table ready. | DONE |
| bench_step810 ✓ | Full resource profile across 5 variants (CUDA, compile). SGNNET_AH wins GPU memory (477 MiB, lowest) & inference vs VGG_FC (5.24× faster @ 3419× fewer params). Loses inference vs MLPs (3.4× slower, bandwidth-bound). | DONE |
| bench_step811 ✓ | SGNNET inference opts — V2 max-autotune new best at 0.280ms (8.38× speedup vs eager). fp16/bf16 failed (mixed-dtype in model). CUDA Graph neutral (bandwidth-bound not launch-bound). SGNNET now **5.6× faster than VGG_FC**. | DONE |
| **step260** ✓ | **K_iter × ΔW proj sweep @ T0 — K_iter=4 WINS (94.19% > K_iter=5 93.99%) +0.20pp. NEW Pareto optimum.** K=3 hurts (92.94%), K=6 drops (92.36% over-smoothing). T1 → step261. | **DONE — BREAKTHROUGH** |
| **step261** ✓ | **K_iter=4 ΔW proj Tier-1 (3 seeds × 75ep 50%) — K_iter=5=95.34% σ=0.001, K_iter=4=95.45% σ=0.001. Δ=+0.11pp at 11σ. K_iter=4 WINS statistically (tight σ). 20% fewer iterations at equivalent accuracy.** | DONE |
| step510 ✓ | Warmup checkpoint N=512 ep30 — best 77.30%. Stats saved (mu range [0.28,0.82], sigma [0.002,0.20]). Ready for step511-514. | DONE |
| step262 (3 seeds) ✓ | K_iter=4 ΔW proj T2 (3 seeds): seed42=−0.13pp, seed43=+0.28pp, seed44=−0.08pp. **Mean Δ=+0.02pp. K_iter=4 STATISTICALLY EQUIVALENT to K=5.** | DONE |
| **step263 seed=42** ✓ | **N=4096 K=4 ΔW proj WINS +0.97pp at T1** (95.97% vs 95.01%). Tier-2 validation critical. | DONE |
| **step264 seed=42,43** ✓ | N=1024 T1 K=4 vs K=5: +0.05pp (marginal, 2 seeds agree) | DONE |
| step511 ✗ | Dyn conn v1 (K_hh=2, every 5ep, all-neighbor) — FAILED −3.21pp. Rewire #1 = 90% edges changed (noise-driven). | DONE — KILLED |
| step512 ✗ | Dyn conn v2 (K_hh=2, every 15ep, ≤1 swap, threshold δ=0.02) — both A (co_act_low) and B (co_act_hi) FAILED: −2.34pp / −2.24pp vs Ref_static. **Guarded rewiring still hurts at K_hh=2.** | DONE — KILLED |
| step513 ✗ | Dyn conn at K_hh=4 — A_coact_low −4.94pp, B_coact_hi −2.42pp vs Ref_static. **Dynamic connectivity via correlation-based rewiring KILLED across K_hh={2,4}.** | DONE — KILLED |
| step514 ✗ | Incremental K_hh EXPANSION (2→3→4→5 non-destructive) — FAILED −0.56pp vs Ref_static. **Additive DoF hypothesis DISCONFIRMED at N=512.** 4/4 dyn-conn mechanisms dead at N=512. | DONE — KILLED |
| step730 ✓ | ΔW proj × K_hh=4 N=2048 — Ref=93.15%, Proj=94.98%, Δ=+1.83pp. Proj generalises to K_hh=4. K_hh=2+proj still higher (95.40% mean). | DONE |
| step740 ✗ | ConnGA v2 softmax DONE — 72.18% (Δ=−6.93pp vs ref 79.11%), HURTS | DONE |
| step741 ✗ | ConnGA rank scoring — best_seen=76.05% vs ref 77.81% (Δ=−1.76pp). Gen peaked at Gen2 then degraded. FAILS. | DONE |
| step742 ✗ | ConnGA top_k_avg — best_seen=74.65% vs ref 77.40% (Δ=−2.75pp). ConnGA track CONCLUSIVELY DEAD. | DONE |
| step750 ✓ | K_iter=3 N=4096 K_hh=4 — 95.06% @ep72, T1 done | DONE |
| step751 | K_iter=3 N=8192 K_hh=8 — 94.01% @ep72, latency 1.969ms (28× VGG FC) | DONE |
| step752 | K_iter=2 N=8192 K_hh=8 — 93.30% @ep69, latency 1.676ms. Fails 96% | DONE |
| step754 ✓ | K_iter=2 N=16384 K_hh=16 — 95.26% @ep58, latency=5.272ms (75× VGG FC). Fails 96%. Track CLOSED. | DONE |
| bench_latency_step199 ✓ | N=2048 K_iter=5 AH-only latency — **0.298ms** (4.3× VGG FC) | DONE |
| step760 step729 ✓ | Seed var N=4096 ΔW rot — mean=96.68%, σ=0.18pp, range=0.46pp | DONE |
| bench_latency_step706 ✓ | ΔW proj latency N=2048 — **0.355ms** (5.1× VGG FC, +19% vs AH-only 0.298ms) | DONE |
| step760 step750 ✓ | Seed var N=4096 AH-only K_hh=4 K_iter=3 — mean=94.54%, σ=0.39pp, range=0.89pp | DONE |
| step755 | K_iter=3 N=4096 K_hh=2 — 94.42% @ep71. K_hh=4 beats K_hh=2 by +0.64pp | DONE |
| step760 step199 ✓ | Seed var AH-only (N=2048 D=16 K_hh=2 K_iter=5) — mean=93.82%, σ=0.562pp, range=1.41pp | DONE |
| step760 step706 ✓ | Seed var ΔW proj (same config) — mean=95.402%, σ=0.154pp, range=0.36pp | DONE |
| **step525** | **Teleportation class KILLED (2026-04-15 T0 seed=42 studio_mps):** Ref=94.09%, T2_5 (5ep redraw)=93.63% (−0.46pp), T2_1 (1ep redraw)=89.91% (−4.18pp), T3 (per-batch)=53.99% (−40.10pp). Monotone in redraw frequency — ANY topology perturbation hurts. Confirms W_pos co-adapted to specific fixed topology. Paper claim strengthened: topology stability is load-bearing. | DONE — KILLED |

**Surface-up from D=64 audit:** step65's `exp(−γ·d)` distance weighting was NOT killed; it was silently adopted as the `geo_gamma=0.5` default in beam routing. Paper should note this (not a dead end — absorbed into arch).
