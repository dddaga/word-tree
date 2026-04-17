# SGNNET Experiment Queue (Live)

---

**Historical archive:** [EXPERIMENT_QUEUE_history.md](EXPERIMENT_QUEUE_history.md) — all DONE/KILLED entries, P0–P3 tracks, Completed Experiments.

---

## FINAL EFFICIENCY CONFIG (2026-04-11) — step199

**95.52% @ 0.98M routing MACs — both ≤1% FLOPs AND ≤1% params criteria met simultaneously.**
⚠️ **FLOPs clarification (2026-04-14 audit):** 0.98M = routing-only message-passing MACs (N×K_iter×K_hh×D×2). True per-sample FLOPs ≈ 6.5M (including seed gather K_in=50, AH suppression, normalize, readout). VGG FC = 123M → real ratio ≈ **19× fewer FLOPs** (not 116×). Paper must label these "message-passing MACs". Cross-check with ncu (step800).

| Param | Value |
|-------|-------|
| N | 2048 |
| D | 16 |
| K_hh | 2 (K_local=1, K_random=1) |
| K_iter | 5 |
| K_in | 25 |
| n_groups | 256 (max(8, N//8)) |
| alpha_ahebb | 1.0 |
| alpha_reflect | 0.5 |
| alpha_turing | 0.0 |
| mode | dynamic_z_geo |
| beam_size | 16 |
| geo_gamma | 0.5 |
| K_phase | 8 |
| norm_mode | l2 |
| encoding_mode | fourier |
| seed | 42 |
| epochs | 150 (best_ep=136) |
| data | 100% Imagenette |

**Results:**

| Metric | Value | vs VGG16 FC |
|--------|-------|-------------|
| Accuracy | 95.52% | +0.52pp |
| FLOPs | 0.98M | **0.79%** |
| Params | **34,976** | **0.029%** (CONFIRMED — 67K was STALE pre-spatial-precomputation refactor) |

**Efficiency frontier (D=16, K_hh=2 family):**

| Step | N | K_iter | FLOPs | FLOPs% | Accuracy | Note |
|------|---|--------|-------|--------|----------|------|
| step199 | 2048 | 5 | 0.98M | 0.79% | 95.52% | **Final efficiency config** |
| step195 | 2048 | 6 | 1.18M | 0.95% | 96.08% | First ≤1% FLOPs hit |
| step205 | 4096 | 5 | 1.97M | 1.59% | 97.17% | D=16 record |
| step209 | 8192 | 5 | 3.93M | 3.18% | 97.17% | D=16 ceiling confirmed |
| step89 | 4096 | 12 | 38.8M | 31.4% | 97.86% | Project best (D=64) |

Script: `scripts/train_step199_n2048_d16_khh2_kiter5_tier2.py`
Result: `results/train_step199_n2048_d16_khh2_kiter5_tier2.json`
Eval: `scripts/eval_efficiency_config.py`

---

**GA rule (STRICT):** Every new experiment base = ALL confirmed winners from all prior generations.
**Calibration rule:** When base changes generation/scale, run a 40-50ep param sweep BEFORE full 150ep runs.
**Critical Findings:** See [EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md](EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md)

**Key principle (2026-04-10):** N=1024 winners ARE wins even if they don't scale to N=4096. If mechanisms push N=1024 toward 96%+, that's a massive efficiency gain (4-6× fewer FLOPs than N=4096). Stop dismissing N=1024 results.

---

## Currently Running (5 slots, updated 2026-04-16 session 12)

| Machine:Device | Session | Step | Status | Note |
|---------|---------|------|--------|------|
| mini:mps | — | — | FREE | step297 DONE: K_in=10+aug T2 @ N=16384 — best=97.197% @ ep39, final ep150=96.66%. **+0.31pp N=16384 record** over step291. |
| mini:cpu | — | — | FREE | step411 DEAD: store_agnews_distilbert.h5 missing on mini. Text gap confirmed negative (step405/407), skip unless data recovered. |
| studio:mps | — | — | FREE | step532 cleared. |
| studio:cpu | — | — | FREE | step760 DONE (3 configs): step199 std=0.43pp, step706 ΔW proj std=0.24pp, step750 N=4096 K_hh=4 std=0.37pp. ΔW proj halves variance. |
| 5060ti:cuda | — | — | FREE | step634 DONE: K25=95.49% > K20=95.26% > K15=94.83%. K_in=25 confirmed best at N=2048. K_in=20 does NOT beat K_in=25 (T1 result was noise). |

---

## Priority Queue — Active / QUEUED

### P-PAPER-2026-04-16 — Params efficiency baseline (session 6)

| Step | Description | Status |
|------|-------------|--------|
| **step850** | VGG-FC Pareto curve: MLP_16=97.50%, MLP_128=97.78%, MLP_256=97.91%, MLP_512=97.83%, VGG_37_2L=97.61%, VGG_128_2L=97.73%. **Plain MLPs beat SGNNET at every FLOPs level. Paper pivot: params efficiency is SGNNET's advantage, not FLOPs.** | **DONE** |
| **step851** | MLP param-threshold crossover: h=4 (100K)=95.00% (trails SGNNET 95.52%), **h=6 (150K)=96.31% (CROSSOVER)**. Min MLP to beat SGNNET = 150K = **4.3× SGNNET's 34,976** (updated from 2.22× — old ratio used stale 67K param count). Paper claim: "SGNNET achieves VGG-level accuracy at 4.3× fewer params than minimum competitive MLP." | **DONE** |
| **step615** | Fair MLP at matched params: best-practices MLP (LeakyReLU, He init, label smooth 0.1, dropout 0.3, cosine LR) at h=3 (75K) and h=2 (50K). Tests if best-optimized MLP at 67K-equivalent params can close the gap vs SGNNET. D_skip_h3 missing from MODEL_MAP (not implemented). | **DONE (A/B/C/E)** |
| **bench_step832** | PyG scatter vs fancy-index compiled: V_ref_c=0.151ms (6.55× over eager). scatter_add compiled=0.217ms — fancy-index wins. torch.compile max-autotune is the kernel speedup path. | **DONE** |
| **step852** | **conn_hh rebuild cadence ablation** (8 configs, T0 @ N=2048). Ref=91.92%, A_wpos_static=88.18% (-3.74pp), B_wpos_batch=65.12%, E_wpos_ep10=90.01% (-1.91pp) best dynamic. **ALL dynamic rebuilds trail static. Static random Watts-Strogatz WINS. Data-driven topology kills learning mid-run. Also: W_pos-based static KNN worse than random (-3.74pp).** | **DONE — KILLED** |
| **step853** | **C_ho sparsity ablation** (8 configs T0 mini_cpu + T1 5060ti). T0: Ref=91.69%, D_very(0.98)=91.85% (+0.16pp), dense=84.87% (-6.82pp). T1: Ref=93.91%, **D_very=94.27% (+0.36pp CONFIRMED)**. Sweet spot at sparsity~0.98 (208/class). K_ho=10/class collapses regardless of selection (F_tiny=-17pp, G_geometric=-23pp). Dense readout catastrophic (-6.82pp). New default: sparsity=0.98. | **DONE — D_very T1 WIN** |
| **step856** | **C_ho sparsity=0.98 multi-seed T1** (5 seeds, 75ep, 50% data). Confirms step853 D_very +0.36pp is not noise. Same protocol as step760 seed variance. | QUEUED |
| **step855** | **Sparse BFS routing T0** (5 configs, 20ep, 50% data). Tests beam-gated broadcaster selection: only top-M active nodes broadcast per K_iter step. Configs: Ref/A_fixed_M16/B_cascade/C_quiet_zero/D_readout_active. HYPOTHESIS: 128× routing FLOP reduction at M=16 with <0.5pp accuracy loss. Script: `scripts/train_step855_sparse_bfs.py`. | QUEUED (mini_cpu) |
| **step859** | **Soft distance-weighted routing T0** (5 configs, 20ep, 50% data). Tests softmax over static K_hh neighbor positions (W_pos distance). β annealing 0.5→3.0. Configs: Ref/A_soft_β1/B_soft_anneal/C_soft_ah/D_soft_dwproj — tests both AH and ΔW-proj on soft routing. HYPOTHESIS: W_pos gradient through distance term improves topology. Script: `scripts/train_step859_soft_routing.py`. | QUEUED (mini_cpu) |

### P-PAPER-2026-04-15 — Scripts added from V3 gap-analysis + design log (2026-04-15)

All scripts smoke-tested with `--help`. Launch via `scripts/queue_submit.sh` (once the queue controller is running) or `scripts/launch_slot.sh`.

| Step | Description | Scale | Script | Status |
|------|-------------|-------|--------|--------|
| **step267** | ΔW rot + aug + K=4 @ N=4096 (V3 Gap 2.1) — **SUPERSEDED.** step266 Ref (K=5)=97.71%, A_k4 (K=4)=97.66% ALREADY COMPLETE. Local JSON was stale sync artifact; authoritative result synced from 5060ti. No new script needed. | N=4096 | (n/a) | **DONE — confirmed from synced step266 log** |
| **step268** | ΔW proj + aug + K=4 combo @ N=2048 Tier-1 (V3 Gap 2.2) — stack K=4 equivalence + aug gain. 4-config ablation: Ref K5 no-aug, A K4 no-aug, B K4 aug (COMBO), C K5 aug. | N=2048 | `scripts/train_step268_dwproj_aug_k4.py` | **DONE** |
| **step403b** | Matched-FLOPs MLP_37 baseline — MLP_37=97.71% @ep36 at 1.86M FLOPs. Paper baseline confirmed. | N_in=25088→h=37→10 | `scripts/train_step403b_matched_flops_mlp.py` | **DONE (studio_mps)** |
| **step404** | GCN / GAT / GIN baselines. GCN=48.9%, GAT=48.7%, GIN=15.5% vs SGNNET=95.52%. SGNNET crushes all GNN baselines by ~47pp. | N=2048 | `scripts/train_step404_gnn_baselines.py` | **DONE** |
| **step405** | SST-2 cross-modal SGNNET (V3 Gap 2.8 paper-blocker) — DistilBERT CLS [768-d] → Linear / MLP_64 / SGNNET comparison. 2-phase: `--phase extract` (one-time) then `--phase train`. Requires `pip install transformers datasets h5py`. | N=2048 D=16 | `scripts/train_step405_sgnnet_sst2.py` | **DONE — Linear=84.63%, MLP_64=84.52%, SGNNET=83.60% (−1.03pp). Text gap confirmed.** |
| **step524-S1** | Edge-β scalar on frozen topology. Ref=93.94%, S1_init0=93.91% (−0.03pp), S1_init1=93.07% (−0.87pp). | N=2048 | `scripts/train_step524_s1_edge_beta.py` | **DONE — KILLED. Edge-β adds no value. Dynamic direction CLOSED.** |
| **step526/527** | INT8 QAT sweep. Part A weight-only: −0.20pp (lossless). W+Z: −1.27pp. Part B QAT accum=1: −0.97pp viable; cliff at accum=4 (−16.31pp). fp32 ref=87.90%. | N=2048 | `scripts/train_step526_int8_qat.py` | **DONE (mini_mps)**. Result: `results/train_step527_int8_qat_k4_seed42__mini_mps.json` |
| **bench_step830** | K=4 direct wall-clock measurement (V3 Blocker-7) — currently paper says "20% reduction projected"; this measures it. 6 variants: K5/K4 × eager/reduce-overhead/max-autotune fp32. Prints pass/fail vs the ≤0.85× criterion. | N=2048 bs=32 | `scripts/bench_step830_k4_wallclock.py` | QUEUED (5060ti only) |

### P-SPEED — Seed Gather Optimization (2026-04-15)

**Spatial precomputation (mathematical identity) — SHIPPED to model_smallworld.py.**
FLOPs: seed 1.64M → 0.05M (16× less). Memory: 14× less. Speed: 5–10× (device-dependent).
Correctness verified (max_diff=1.19e-07 CPU/MPS/CUDA). T0 training: 91.77% — STABLE.

| Step | Description | Script | Status |
|------|-------------|--------|--------|
| **bench_step831** | CUDA seed speedup validation — DONE: 5.26× @B=128, 1.98× @B=32. Correctness ✓ | `scripts/bench_step831_seed_opt_cuda.py` | **DONE** |
| **step630** | K=1 clean benchmark. Ref_k5=95.46% (1.97ms B1), Scratch_k1=91.69% (0.82ms), Distill_k1=92.13% (+0.44pp over scratch). Speedup vs K=5: 2.3× B1, 3.0× B32. Verdict: MEDIUM — distillation helps but −3.3pp cost too large for efficiency claim. | `scripts/train_step630_k1_clean_benchmark.py` | **DONE** |
| **step631** | K_in sweep T1. Ref=94.01%, K15=93.66% (−0.36pp ADVANCES), K10=92.84% FAILS, K5=91.26% FAILS. K_in=15 minimum viable. 26.7× compound seed reduction. | `scripts/train_step631_kin_sweep.py` | **DONE** |
| **step632** | K_in=15 T2. Ref_k25=95.46%, A_k15=95.13% (Δ=−0.33pp). **CONFIRMED for paper.** 26.7× seed reduction, <0.5pp cost. | `scripts/train_step632_kin15_t2.py` | **DONE** |
| **step633** | K_in plot sweep K_in=1..25. K_in=20=94.50% is the true knee (beats K_in=15=93.89% and K_in=25=94.01%). Publishable curve. Both 5060ti+mini_cpu results consistent. | `scripts/train_step633_kin_plot_sweep.py` | **DONE** |
| **step621** | GLNN distillation Imagenette. h=256: −0.18pp. h=2: collapsed (−60pp). KILLED at both student sizes. | `scripts/train_step621_glnn_imagenette_mlp.py` | **DONE — KILLED** |

### P-CUDA — Deferred

| Step | Description | Script | Status |
|------|-------------|--------|--------|
| **step530** | **Triton fused gather+sum+L2norm kernel** — 1.17-1.18× over torch.compile max-autotune (K_iter=5, N=2048, D=16, B=32/128, 5060ti). Kernel integrated into `model_smallworld._route()` with CUDA+l2+power-of-2-D guard. Fallback=eager on CPU/MPS. | **DONE — integrated** |
| **step531** | T0 smoke-test: Triton _route() vs eager Ref. Verifies accuracy preserved after integration. 20ep 50% data. **91.85% PASS (>87% floor). 3 bugs found+fixed in SGNNET_AntiHebbian_CUDA: (1) cudagraph_mark_step_begin unconditioned on _compiled, (2) supp_w cached with live grad_fn, (3) supp_w detached killed W_pos hidden-row gradients. Guard-rail test added: tests/test_cuda_model_training.py.** | **DONE** |

---

## Aug N-scaling + K_in compound (2026-04-16 session)

| Step | Description | Status |
|------|-------------|--------|
| **step269** | K=5+aug T2. Ref=95.29%, C_aug_k5=95.46% (+0.18pp). Aug confirmed T2. | **DONE** |
| **step270** | K_iter=3/2 + aug sweep T1. K_iter=3: −5.10pp no-aug, −2.78pp aug. K_iter=2: −2.78pp. KILLED. | **DONE** |
| **step271** | K_in=15+aug compound T1. Ref=93.85%, C_k15_aug=94.55% (+0.69pp). Compound ADVANCES. | **DONE** |
| **step272** | N=4096+aug T1. Ref=95.85%, A_n4096_aug=96.71% (+0.87pp). ADVANCES to T2. | **DONE** |
| **step273** | N=4096+aug T2. Ref=97.12%, A_n4096_aug=97.68% (+0.56pp). NEW D=16 ACCURACY RECORD. | **DONE** |
| **step274** | K_in=15+aug compound T2. Ref=95.29%, C_k15_aug=95.46% (+0.18pp). CONFIRMED. 26.7× seed reduction publishable. | **DONE** |
| **step275** | N=8192+aug T1. Ref=94.80%, A_n8192_aug=96.46% (+1.66pp). ADVANCES to T2. | **DONE** |
| **step276** | N=8192+aug T2. Ref=96.94%, A_n8192_aug=97.38% (+0.43pp). CONFIRMED. Aug scale-invariant. | **DONE** |
| **step277** | K_in=15+aug @ N=4096 T1. Ref=96.23%, C_k15_aug=96.56% (+0.33pp). ADVANCES to T2 (step279). | **DONE** |
| **step278** | N=1024+aug T1. Ref=88.15%, A_n1024_aug=89.45% (+1.30pp). Completes bottom of scaling curve. | **DONE** |
| **step279** | K_in=15+aug @ N=4096 T2. Ref=97.12%, C_k15_aug=97.30% (+0.18pp). CONFIRMED. | **DONE** |
| **step280** | N=1024+aug T2. Ref=90.57%, A_n1024_aug=91.11% (+0.54pp). Aug N-scaling curve COMPLETE. | **DONE** |
| **step281** | K_in=15+aug compound @ N=8192 T1. Ref=94.90%, C_k15_aug=96.05% (+1.15pp). ADVANCES to T2. | **DONE** |
| **step282** | K_in=15+aug compound @ N=8192 T2. Ref=96.94%, C_k15_aug=97.30% (+0.36pp). CONFIRMED. | **DONE** |
| **step283** | K_in=15+aug compound @ N=1024 T1. Ref=88.13%, C_k15_aug=89.78% (+1.66pp). ADVANCES to T2. | **DONE** |
| **step284** | K_in=15+aug compound @ N=1024 T2. Ref=90.52%, C_k15_aug=91.31% (+0.79pp). CONFIRMED. | **DONE** |
| **step285** | K=4+aug @ N=2048 T2. Ref=95.46%, B_k4_aug=94.17% (−1.30pp). K=4 routing KILLED. K_hh=2 is minimum viable. | **DONE — KILLED** |
| **step286** | N=16384+aug T1. Ref=93.43%, A_n16384_aug=95.54% (+2.11pp). ADVANCES. Non-monotonic T1 gain. | **DONE** |
| **step287** | N=16384+aug T2. Ref=95.87%, A_n16384_aug=96.87% (**+0.99pp** — strongest aug T2 delta at any N). | **DONE** |
| **step288** | K_in=15+aug@N=16384 T1. C_k15_naug=94.70% (+1.27pp), D_k15_aug=95.92% (+2.50pp). **ANOMALY: K_in=15 > K_in=25 at N=16384.** | **DONE** |
| **step289** | K_in=20,10 T0 @ N=16384. K_in=20=91.01% (-2.42pp DEAD), K_in=10=92.43% (-0.99pp ADVANCES). | **DONE** |
| **step290** | K_in=10,20 T1 @ N=16384. Validates K_in anomaly curve. | **DONE** |
| **step291** | K_in=15 compound T2 @ N=16384. C_k15_naug=**96.13%** @ep65, D_k15_aug=**96.89%** @ep74. K_in=15+aug matches K_in=25+aug (96.87%) at 40% fewer seed connections. **CONFIRMED.** | **DONE** |
| **step293** | K_in=15 no-aug T1 @ N=4096,8192. N=4096=96.18% (+0.33pp), N=8192=95.18% (+0.38pp). Crossover between N=2048-4096. | **DONE** |
| **step294** | K_in=15 no-aug T2 @ N=4096 = 97.07% (Ref T2=97.12%, Δ=-0.05pp — tied). T1 delta compressed to parity. | **DONE** |
| **step296** | K_in=10+aug T1 @ N=16384. A_k10_aug=**96.56%** @ep57 (+3.13pp vs Ref, +1.60pp vs K_in=10 alone). ADVANCES to T2 (step297). | **DONE** |
| **step297** | K_in=10+aug T2 @ N=16384. Best=**97.197% @ ep39** (+0.31pp over step291 K_in=15+aug=96.89%), final ep150=96.66%, 11669s, 278,688 params. New N=16384 record; still below step279 N=4096 (97.30%) — N=16384 capacity-limited at D=16. Result: `results/train_step297_kin10_aug_n16384_t2_seed42__mini_mps.json`. | **DONE — 97.20% N=16384 record** |
| **step606** | K=1 + K_in=15 compound at N=2048 T1. Ref_k25=95.69%, A_k15_KD=94.96% (-0.73pp), B_k15_scratch=95.29% (-0.40pp). **COMPOUND FAILS at N=2048** — K_in=15 hurts at K=1 when no routing to compensate. Need to retest at N>=4096. | **DONE — compound KILLED at N=2048** |
| **step607** | K=1 pure-KD T2 @ N=2048. A_pure_kd=95.92% (-0.76pp vs teacher), B_balanced=95.95% (-0.74pp). T2 OVERFITS T1 (step605=96.33% at 75ep was better). Early-stop @ep75 for best. | **DONE** |
| **bench_step608** | K=1 wall-time bench. SGNNET K=1 @ B=32 = **12.7us** (5.3× faster than VGG_FC 66.7us, 3418× fewer params). K=1 vs K=5 = 2-2.5× wall-time reduction. Memory anomaly: K=5 B=128 regresses (45.4us > B=32's 31.2us) — CUDA optim candidate. | **DONE** |

---

## P0 GAP-CLOSE (2026-04-16, user directive): Close SGNNET-vs-FC gap at low-dim features

**Mindset:** instead of accepting "SGNNET stops working below N_in=1000", actively close the gap so SGNNET wins on Pareto efficiency at low-dim too.

**Hypothesis:** default SGNNET (N=2048, K_in=25, K_iter=5) is tuned for VGG16's N_in=25088. At N_in=768 (DistilBERT):
- N=2048 may be overparameterized
- K_in=25 covers 3.3% of features (vs 0.1% at N_in=25088) — too dense
- K_iter=5 may be excessive for low-dim features

| Step | Description | Status |
|------|-------------|--------|
| **step410** | SST-2 config sweep. All 5 configs trail Linear=84.63%. Best: Ref_orig=83.72% (−0.91pp). Compression worsens monotonically — D_combined worst at −2.06pp. **Gap NOT closed. Architecture investigation needed.** | **DONE** |
| **step411** | AG News config sweep — same 5 configs vs Linear=91.18%, MLP_64=92.53%. Launched studio_mps 2026-04-16. h5 on mac-studio. | **RUNNING (studio_mps)** |
| **step412** | If step410 winner ≥ Linear: validate scaling on SST-5 (5-class text) and RTE (low-data). | QUEUED (depends on 410) |

**Exit criterion:** SGNNET config X on text tasks Pareto-dominates (accuracy, params, FLOPs, wall-time) vs Linear/MLP_64. If yes → paper claim expands from "VGG-FC-replacement" to "general high-efficiency FC replacement." If no → scope refinement stands with honest low-dim failure mode documented.
| **step295** | K_in=5 T1 @ N=16384. A_k5=93.78% (+0.36pp vs Ref, −1.18pp vs K_in=10). K_in=5 continues to improve but sublinear — K_in=10 still stronger. | **DONE** |
| **step633** | K_in=1..25 sweep. K_in=15 is knee (−0.08pp vs K_in=25). K_in=20/25 identical. Publishable curve. | **DONE** |
| **step634** | K_in=20 T2 validation @ N=2048. Ref_k25=95.36%, A_k20=95.29% (Δ=−0.07pp), B_k15=95.06% (Δ=−0.30pp). K_in=20 ≈ K_in=25. K_in=15 confirmed <0.5pp cost. T1 crossover was noise. | **DONE — K_in=15 paper default stands.** |
| **step760** | Seed variance (5 seeds T1): step199 ±0.43pp (mini+studio CPU agree); step706 ΔW proj ±0.20–0.24pp (3-device confirmed: CUDA/mini_cpu/studio_cpu); step729 N=4096 ΔW rot ±0.20pp (CUDA); step750 N=4096 K_hh=4 ±0.39pp (studio_mps). ΔW halves variance. step729 studio_cpu CPU cross-device running; step750 mini_cpu CPU cross-device running. | **DONE (4 configs). CPU cross-device ongoing.** |

## Paper-critical experiments (2026-04-15 session)

| Step | Description | Status |
|------|-------------|--------|
| **step601** | CIFAR-100 complexity test — SGNNET_DeltaProj=37.81% vs MLP_37=58.66% (−20.85pp). FAIL: rescue hypothesis NOT validated. SGNNET underperforms MLP at 100 classes. | **DONE (studio_mps)** |
| **step610** | Low-rank MLP sweep — 7 configs: LR_pure/relu × r=8,16,32 + MLP_37_ref. T0 20ep 50% data. MEDIUM: LR_relu_r16=96.03% (not STRONG). Feature rank NOT ≤16; ≈rank 32 linearly. LR_pure_r16 (96.36%) > LR_relu_r16 → nonlinearity adds noise at low rank. | **DONE (studio_mps)** |
| **step612** | Group-level ΔW routing granularity probe. Result: Ref=93.91%, GroupDW=21.73%, Δ=−72pp. **ABANDON — per-neuron routing confirmed essential.** Paper: neuron-level specialization cannot be coarsened to group level. | **DONE (5060ti_cuda)** |
| **step602** | B2 GLNN teacher — SGNNET ΔW-rotation 75ep + cache soft logits at T=4. | **RUNNING (5060ti_cuda)** |
| **step603** | B2 GLNN student — T2_lam05 STRONG=97.81% (+0.10pp over scratch). T=2 λ=0.5 optimal. | **DONE (5060ti_cuda)** |
| **step604** | B1 consistency-DEQ teacher — best=96.69% @ep75. Cache 1.7GB saved. | **DONE (5060ti_cuda)** |
| **step605** | B1 consistency-DEQ student — K=1 student, 6 configs 75ep. ep30=95.26%, learning. | **RUNNING (5060ti_cuda)** |
| **step405** | SST-2 cross-modal — Linear=84.63%, MLP_64=84.52% (MPS). SGNNET=83.60% (CPU, −1pp). SGNNET competitive. NOTE: MPS run fails (49%, numerical issue) — use CPU result. | **DONE** |
| **step406** | ESC-50 audio cross-modal. Linear=64.5%, MLP_64=64.5%, SGNNET=50.5% (-14pp). KILLED. Audio fails. | **DONE — KILLED** |
| **step407** | AG News 4-class text cross-modal (DistilBERT features). Linear=91.18%, MLP_64=92.53%, SGNNET=90.91% (−0.27pp vs Linear, −1.62pp vs MLP_64). Text gap confirmed on multi-class. | **DONE** |

## GLNN distillation validation — cross-dataset/scale (Claim 7)

| Step | Description | Depends on | Status |
|------|-------------|-----------|--------|
| **step620** | B2 GLNN distillation on CIFAR-10 — same protocol as step603 (T=2 λ=0.5 optimal config only, 100ep). Win: student > scratch MLP_37 on CIFAR-10. Accept if +δ; kill if −. | step401b complete + CIFAR-10 teacher trained | TODO (script needed) |
| **step621** | B2 GLNN on Imagenette with MLP_h3 student (67K matched params). Does gain hold at tiny student? | step603 teacher logits (already cached) | TODO (script needed) |
| **step622** | B2 GLNN on CIFAR-100 with MLP_256 student. SGNNET barely converges at 100 classes — may produce no useful soft targets. | CIFAR-100 SGNNET training result (step614) | TODO — only run if step614 SGNNET reaches >50% |

**Priority:** step621 first (reuses existing teacher cache, cheap). step620 second (needs CIFAR-10 teacher). step622 last (conditional on CIFAR-100 SGNNET viability).

---

## User-proposed 2026-04-15 (queued, pending slot)

| Step | Description | Status |
|------|-------------|--------|
| **step523** | **Alternating W_pos / edge training cycle.** Ref=77.68%, A_low=74.85% (−2.83pp), B_high=75.03% (−2.65pp). Both fail. Edge rewiring destroys weights; recovery incomplete. **KILLED — dynamic connectivity direction CONFIRMED dead (7/7 negative including step511-514).** | **DONE** |
| **step521** | Deep supervision on K_iter routing. Ref=94.85%, A_ds_3to5=88.36% (-6.5pp), B_ds_2to5=88.87% (-6.0pp). **KILLED — routing disrupted by intermediate supervision.** C/D still running but expected dead. | **DONE — KILLED** |
| **step522** | Muon optimizer vs AdamW at N=2048 K=5 ΔW proj. Ref_adamw=95.64% (ep_to_95=38), A_muon=95.39% (ep_to_95=42). Muon −0.25pp vs AdamW at T1. No convergence benefit. | **DONE — KILLED. AdamW remains default.** |
| **direct K=4 wall-clock bench** | Measure SGNNET K=4 inference latency directly (currently projected 0.224ms based on 20% reduction). Add to bench_step811 variant list. | TODO |
| **step524** | **Edge-SHIFT probes** (post-step523 follow-up). 6 configs at N=1024 T0: Ref / P1 step523+Adam-reset / P2 alt-schedule+0%cap / S1 edge-β scalar / S2 W_pos-passive-rebind / S4 cyclic-shift-null-control. Tests H1-H2-H5 of step523 failure + 2 continuous-parameterization alternatives + 1 null control. ~4h one slot. Design in LEARNINGS_design_2026_04_15.md. | TODO (script) |
| **step526** | **INT8 QAT + inference impact + grad-accum sweep.** Part A: fp32 train → quant eval for 3 modes (saturate/modular/crt) × {W only, W+Z}. Tests hypothesis: L2-norm at D=16 bounds components to ±0.25 × scale=100 → int8 range ±25, so wrap never fires. Part B: QAT from scratch (fake-quant+STE forward, fp32 master), grad_accum ∈ {1,4,16,64}. Telemetry: wrap_rate per forward. Script ready: `train_step526_int8_qat.py`. | TODO (launch) |
| **Param count reconciliation** | Bench reports SGNNET=34,976; training reports 67,744. Diff ≈ 32K. Find missing component (likely K_in=25 seed projection). Resolve before paper. | TODO |
| **bench_step832** | PyG scatter vs fancy-index. V_ref_c (compiled) = **0.151ms** (6.55× over eager). Scatter_add compiled = 0.217ms — fancy-index wins. CUDA Graph approach (0.896ms) slower. Best: `torch.compile(max-autotune)` on fancy-index. | **DONE** |
| **bench_step830** | K=4 vs K=5 wall-clock. K=4 compiled=0.140ms vs K=5=0.154ms → **1.10× compiled, 1.18× eager**. Paper claim updated: "10–18% latency reduction" (was "20% projected"). | **DONE** |

## Parked — Resume after arch experiments complete

### bench_step840: Concurrent users / hardware democratization claim
**Parked (2026-04-15). Resume after arch experiments.**

Original simple framing (max batch size before OOM → max concurrent users) was superseded by a more powerful architectural vision:

**Vision (user directive):** Break SGNNET into layers where each K_iter routing step is its own microservice. Workflow manager dispatches concurrent requests across steps; all transfers stay within GPU memory. This is pipeline-parallel inference — each routing "layer" processes a different request at every clock cycle, multiplying effective throughput by K_iter.

**Why this matters for the paper:**
- VGG16 full model = ~528MB. Head sizes are negligible (SGNNET 0.27MB vs FC 494MB). Head memory is not the bottleneck.
- The real advantage: SGNNET's K_iter routing steps are structurally identical and independently schedulable — ideal for pipeline parallelism. Standard FC has no such decomposition.
- Claim: "SGNNET's homogeneous routing steps enable pipeline-parallel inference on commodity 16GB hardware, multiplying concurrent user capacity by K_iter without additional memory overhead."

**What needs to be built for validation:**
1. Pipeline-parallel inference harness: K_iter=5 steps as 5 stages, each stage a separate forward kernel call
2. Workflow manager: routes batch[i] to stage[i % K_iter], maintains ring buffer of in-flight requests
3. Benchmark: max sustained throughput (requests/sec) vs VGG16-FC and MLP_37 at 16GB budget
4. Script: `bench_step840_pipeline_concurrent.py` — TODO (write after arch experiments done)
