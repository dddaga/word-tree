# SGNNET Experiment Queue — Historical Archive Part 3

*Archived 2026-05-13. Sections: P-PAPER-2026-04-16/15, P-SPEED, Aug N-scaling, P0-GAP-CLOSE, GLNN, user-proposed, Meditation P0. Continued: [part4](EXPERIMENT_QUEUE_history_part4.md)*

---

## Priority Queue — Active / QUEUED

### P-PAPER-2026-04-16 — Params efficiency baseline (session 6)

| Step | Description | Status |
|------|-------------|--------|
| **step850** | VGG-FC Pareto curve: MLP_16=97.50%, MLP_128=97.78%, MLP_256=97.91%, MLP_512=97.83%, VGG_37_2L=97.61%, VGG_128_2L=97.73%. **Plain MLPs beat SGNNET at every FLOPs level. Paper pivot: params efficiency SGNNET advantage, not FLOPs.** | **DONE** |
| **step851** | MLP param-threshold crossover: h=4 (100K)=95.00% (trails SGNNET 95.52%), **h=6 (150K)=96.31% (CROSSOVER)**. Min MLP to beat SGNNET = 150K = **4.3× SGNNET 34,976** (updated from 2.22× — old ratio used stale 67K param count). Paper claim: "SGNNET achieves VGG-level accuracy at 4.3× fewer params than minimum competitive MLP." | **DONE** |
| **step615** | Fair MLP at matched params: best-practices MLP (LeakyReLU, He init, label smooth 0.1, dropout 0.3, cosine LR) at h=3 (75K) and h=2 (50K). Tests if best-optimized MLP at 67K-equivalent params can close gap vs SGNNET. D_skip_h3 missing from MODEL_MAP (not implemented). | **DONE (A/B/C/E)** |
| **bench_step832** | PyG scatter vs fancy-index compiled: V_ref_c=0.151ms (6.55× over eager). scatter_add compiled=0.217ms — fancy-index wins. torch.compile max-autotune = kernel speedup path. | **DONE** |
| **step852** | **conn_hh rebuild cadence ablation** (8 configs, T0 @ N=2048). Ref=91.92%, A_wpos_static=88.18% (-3.74pp), B_wpos_batch=65.12%, E_wpos_ep10=90.01% (-1.91pp) best dynamic. **ALL dynamic rebuilds trail static. Static random Watts-Strogatz WINS. Data-driven topology kills learning mid-run. Also: W_pos-based static KNN worse than random (-3.74pp).** | **DONE — KILLED** |
| **step853** | **C_ho sparsity ablation** (8 configs T0 mini_cpu + T1 5060ti). T0: Ref=91.69%, D_very(0.98)=91.85% (+0.16pp), dense=84.87% (-6.82pp). T1: Ref=93.91%, **D_very=94.27% (+0.36pp CONFIRMED)**. Sweet spot at sparsity~0.98 (208/class). K_ho=10/class collapses regardless of selection (F_tiny=-17pp, G_geometric=-23pp). Dense readout catastrophic (-6.82pp). New default: sparsity=0.98. | **DONE — D_very T1 WIN** |
| **step856** | **C_ho sparsity=0.98 multi-seed T1** (5 seeds, 75ep, 50% data). Ref mean=93.92%, D_very mean=93.97% (+0.05pp). Paired per-seed deltas: +0.28, +0.40, -0.38, +0.07, -0.11pp. **D_very NEUTRAL — single-seed T1 +0.36pp was lucky seed42. Sparsity default stays 0.90.** | **DONE — D_very NEUTRAL (not confirmed at multi-seed)** |
| **step855** | **Sparse BFS routing T0** (5 configs, 20ep, 50% data). ALL KILLED: A=-2.10pp, B=-2.23pp, C=-2.20pp, D=-2.05pp. Direction CLOSED. | **DONE — KILLED** |
| **step859** | **Soft distance-weighted routing T0** (5 configs, 20ep, 50% data). Tests softmax over static K_hh neighbor positions (W_pos distance). β annealing 0.5→3.0. Configs: Ref/A_soft_β1/B_soft_anneal/C_soft_ah/D_soft_dwproj — tests both AH and ΔW-proj on soft routing. HYPOTHESIS: W_pos gradient through distance term improves topology. Script: `scripts/train_step859_soft_routing.py`. | **DONE — KILLED (B_soft_anneal +0.99pp was T0 artifact; step861 T1=0.0pp confirmed; soft routing CLOSED)** |

### P-PAPER-2026-04-15 — Scripts from V3 gap-analysis + design log (2026-04-15)

All scripts smoke-tested with `--help`. Launch via `scripts/queue_submit.sh` or `scripts/launch_slot.sh`.

| Step | Description | Scale | Script | Status |
|------|-------------|-------|--------|--------|
| **step267** | ΔW rot + aug + K=4 @ N=4096 (V3 Gap 2.1) — **SUPERSEDED.** step266 Ref (K=5)=97.71%, A_k4 (K=4)=97.66% ALREADY COMPLETE. Local JSON was stale sync artifact; authoritative result synced from 5060ti. No new script needed. | N=4096 | (n/a) | **DONE — confirmed from synced step266 log** |
| **step268** | ΔW proj + aug + K=4 combo @ N=2048 Tier-1 (V3 Gap 2.2) — stack K=4 equivalence + aug gain. 4-config ablation: Ref K5 no-aug, A K4 no-aug, B K4 aug (COMBO), C K5 aug. | N=2048 | `scripts/train_step268_dwproj_aug_k4.py` | **DONE** |
| **step403b** | Matched-FLOPs MLP_37 baseline — MLP_37=97.71% @ep36 at 1.86M FLOPs. Paper baseline confirmed. | N_in=25088→h=37→10 | `scripts/train_step403b_matched_flops_mlp.py` | **DONE (studio_mps)** |
| **step404** | GCN / GAT / GIN baselines. GCN=48.9%, GAT=48.7%, GIN=15.5% vs SGNNET=95.52%. SGNNET crushes all GNN baselines by ~47pp. | N=2048 | `scripts/train_step404_gnn_baselines.py` | **DONE** |
| **step405** | SST-2 cross-modal SGNNET (V3 Gap 2.8 paper-blocker) — DistilBERT CLS [768-d] → Linear / MLP_64 / SGNNET comparison. 2-phase: `--phase extract` then `--phase train`. Requires `pip install transformers datasets h5py`. | N=2048 D=16 | `scripts/train_step405_sgnnet_sst2.py` | **DONE — Linear=84.63%, MLP_64=84.52%, SGNNET=83.60% (−1.03pp). Text gap confirmed.** |
| **step524-S1** | Edge-β scalar on frozen topology. Ref=93.94%, S1_init0=93.91% (−0.03pp), S1_init1=93.07% (−0.87pp). | N=2048 | `scripts/train_step524_s1_edge_beta.py` | **DONE — KILLED. Edge-β adds no value. Dynamic direction CLOSED.** |
| **step526/527** | INT8 QAT sweep. Part A weight-only: −0.20pp (lossless). W+Z: −1.27pp. Part B QAT accum=1: −0.97pp viable; cliff at accum=4 (−16.31pp). fp32 ref=87.90%. | N=2048 | `scripts/train_step526_int8_qat.py` | **DONE (mini_mps)**. Result: `results/train_step527_int8_qat_k4_seed42__mini_mps.json` |
| **bench_step830** | K=4 vs K=5 wall-clock (5060ti_cuda, B=32). K5_ma=0.154ms, K4_ma=0.139ms. **K4/K5=0.901 → 9.9% faster, NOT 20%.** Paper claim "20% wall-clock reduction" REVISED to ~10%. | N=2048 bs=32 | `scripts/bench_step830_k4_wallclock.py` | **DONE — paper claim revised** |

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
| **step633** | K_in plot sweep K_in=1..25. K_in=20=94.50% true knee (beats K_in=15=93.89% and K_in=25=94.01%). Publishable curve. Both 5060ti+mini_cpu results consistent. | `scripts/train_step633_kin_plot_sweep.py` | **DONE** |
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
| **step285** | K=4+aug @ N=2048 T2. Ref=95.46%, B_k4_aug=94.17% (−1.30pp). K=4 routing KILLED. K_hh=2 minimum viable. | **DONE — KILLED** |
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
| **step606** | K=1 + K_in=15 compound at N=2048 T1. Ref_k25=95.69%, A_k15_KD=94.96% (-0.73pp), B_k15_scratch=95.29% (-0.40pp). **COMPOUND FAILS at N=2048** — K_in=15 hurts at K=1 when no routing to compensate. Need retest at N>=4096. | **DONE — compound KILLED at N=2048** |
| **step607** | K=1 pure-KD T2 @ N=2048. A_pure_kd=95.92% (-0.76pp vs teacher), B_balanced=95.95% (-0.74pp). T2 OVERFITS T1 (step605=96.33% at 75ep was better). Early-stop @ep75 for best. | **DONE** |
| **bench_step608** | K=1 wall-time bench. SGNNET K=1 @ B=32 = **12.7us** (5.3× faster than VGG_FC 66.7us, 3418× fewer params). K=1 vs K=5 = 2-2.5× wall-time reduction. Memory anomaly: K=5 B=128 regresses (45.4us > B=32 31.2us) — CUDA optim candidate. | **DONE** |

---

## P0 GAP-CLOSE (2026-04-16, user directive): Close SGNNET-vs-FC gap at low-dim features

**Mindset:** instead of accepting "SGNNET stops working below N_in=1000", actively close gap so SGNNET wins on Pareto efficiency at low-dim too.

**Hypothesis:** default SGNNET (N=2048, K_in=25, K_iter=5) tuned for VGG16 N_in=25088. At N_in=768 (DistilBERT):
- N=2048 may be overparameterized
- K_in=25 covers 3.3% of features (vs 0.1% at N_in=25088) — too dense
- K_iter=5 may be excessive for low-dim features

| Step | Description | Status |
|------|-------------|--------|
| **step410** | SST-2 config sweep. All 5 configs trail Linear=84.63%. Best: Ref_orig=83.72% (−0.91pp). Compression worsens monotonically — D_combined worst at −2.06pp. **Gap NOT closed. Architecture investigation needed.** | **DONE** |
| **step411** | AG News config sweep. Linear=91.18%, MLP_64=92.53%. All SGNNET configs trail Linear: Ref_orig=90.34%(-0.84pp), B_low_kin=90.34%(-0.84pp), C_low_kiter=90.34%(-0.84pp), A_small=89.16%(-2.02pp), D_combined=88.67%(-2.51pp). Text gap CONFIRMED — SGNNET loses monotonically on text. Paper scope = vision only. | **DONE — text gap CONFIRMED** |
| **step892** | CIFAR-10 MLP h-sweep crossover (T1: 75ep, 50%, seed=42). h=4(2.9×)=40.5%, h=6(4.3×)=45.1%, h=8(5.7×)=45.5%, **h=16(11.5×)=81.3% ← CROSSOVER**, h=32(23.0×)=85.0%, h=64(45.9×)=84.2%. Cliff at h=8→h=16 (35pp jump). T2 confirmed by step893. | **DONE** |
| **step893** | CIFAR-10 MLP crossover T2 (150ep, 100% data, seed=42). h=8=45.8%(-34.6pp), h=12=67.1%(-13.3pp), **h=16=80.75%(+0.33pp) ← T2 CROSSOVER**. Paper claim CONFIRMED: SGNNET needs 11.5× fewer params than min-viable MLP on CIFAR-10. | **DONE** |
| **step412** | If step410 winner ≥ Linear: validate scaling on SST-5 (5-class text) and RTE (low-data). | QUEUED (depends on 410) |

**Exit criterion:** SGNNET config X on text tasks Pareto-dominates (accuracy, params, FLOPs, wall-time) vs Linear/MLP_64. If yes → paper claim expands from "VGG-FC-replacement" to "general high-efficiency FC replacement." If no → scope refinement stands with honest low-dim failure mode documented.
| **step295** | K_in=5 T1 @ N=16384. A_k5=93.78% (+0.36pp vs Ref, −1.18pp vs K_in=10). K_in=5 continues to improve but sublinear — K_in=10 still stronger. | **DONE** |
| **step633** | K_in=1..25 sweep. K_in=15 knee (−0.08pp vs K_in=25). K_in=20/25 identical. Publishable curve. | **DONE** |
| **step634** | K_in=20 T2 validation @ N=2048. Ref_k25=95.36%, A_k20=95.29% (Δ=−0.07pp), B_k15=95.06% (Δ=−0.30pp). K_in=20 ≈ K_in=25. K_in=15 confirmed <0.5pp cost. T1 crossover was noise. | **DONE — K_in=15 paper default stands.** |
| **step760** | Seed variance (5 seeds T1): step199 ±0.43pp (mini+studio CPU agree); step706 ΔW proj ±0.20–0.24pp (3-device confirmed: CUDA/mini_cpu/studio_cpu); step729 N=4096 ΔW rot ±0.20pp (CUDA); step750 N=4096 K_hh=4 ±0.39pp (studio_mps). ΔW halves variance. step729 studio_cpu CPU cross-device running; step750 mini_cpu CPU cross-device running. | **DONE (4 configs). CPU cross-device ongoing.** |

## Paper-critical experiments (2026-04-15 session)

| Step | Description | Status |
|------|-------------|--------|
| **step601** | CIFAR-100 complexity test — SGNNET_DeltaProj=37.81% vs MLP_37=58.66% (−20.85pp). FAIL: rescue hypothesis NOT validated. SGNNET underperforms MLP at 100 classes. | **DONE (studio_mps)** |
| **step610** | Low-rank MLP sweep — 7 configs: LR_pure/relu × r=8,16,32 + MLP_37_ref. T0 20ep 50% data. MEDIUM: LR_relu_r16=96.03% (not STRONG). Feature rank NOT ≤16; ≈rank 32 linearly. LR_pure_r16 (96.36%) > LR_relu_r16 → nonlinearity adds noise at low rank. | **DONE (studio_mps)** |
| **step612** | Group-level ΔW routing granularity probe. Ref=93.91%, GroupDW=21.73%, Δ=−72pp. **ABANDON — per-neuron routing confirmed essential.** Paper: neuron-level specialization cannot coarsen to group level. | **DONE (5060ti_cuda)** |
| **step602** | B2 GLNN teacher — SGNNET ΔW-rotation 75ep + cache soft logits at T=4. | **DONE** |
| **step603** | B2 GLNN student — T2_lam05 STRONG=97.81% (+0.10pp over scratch). T=2 λ=0.5 optimal. | **DONE (5060ti_cuda)** |
| **step604** | B1 consistency-DEQ teacher — best=96.69% @ep75. Cache 1.7GB saved. | **DONE (5060ti_cuda)** |
| **step605** | B1 consistency-DEQ student — K=1 student, 6 configs 75ep. ep30=95.26%, learning. | **DONE — 95.95% @ 0.20M FLOPs, efficiency champion** |
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
| **step523** | **Alternating W_pos / edge training cycle.** Ref=77.68%, A_low=74.85% (−2.83pp), B_high=75.03% (−2.65pp). Both fail. Edge rewiring destroys weights; recovery incomplete. **KILLED — dynamic connectivity direction CONFIRMED dead (7/7 negative incl step511-514).** | **DONE** |
| **step521** | Deep supervision on K_iter routing. Ref=94.85%, A_ds_3to5=88.36% (-6.5pp), B_ds_2to5=88.87% (-6.0pp). **KILLED — routing disrupted by intermediate supervision.** C/D still running but expected dead. | **DONE — KILLED** |
| **step522** | Muon optimizer vs AdamW at N=2048 K=5 ΔW proj. Ref_adamw=95.64% (ep_to_95=38), A_muon=95.39% (ep_to_95=42). Muon −0.25pp vs AdamW at T1. No convergence benefit. | **DONE — KILLED. AdamW remains default.** |
| **direct K=4 wall-clock bench** | Measure SGNNET K=4 inference latency directly (currently projected 0.224ms based on 20% reduction). Add to bench_step811 variant list. | TODO |
| **step524** | **Edge-SHIFT probes** (post-step523 follow-up). 6 configs at N=1024 T0: Ref / P1 step523+Adam-reset / P2 alt-schedule+0%cap / S1 edge-β scalar / S2 W_pos-passive-rebind / S4 cyclic-shift-null-control. Tests H1-H2-H5 of step523 failure + 2 continuous-parameterization alternatives + 1 null control. ~4h one slot. Design in LEARNINGS_design_2026_04_15.md. | TODO (script) |
| **step526** | **INT8 QAT + inference impact + grad-accum sweep.** Part A: fp32 train → quant eval for 3 modes (saturate/modular/crt) × {W only, W+Z}. Tests hypothesis: L2-norm at D=16 bounds components ±0.25 × scale=100 → int8 range ±25, wrap never fires. Part B: QAT from scratch (fake-quant+STE forward, fp32 master), grad_accum ∈ {1,4,16,64}. Telemetry: wrap_rate per forward. Script ready: `train_step526_int8_qat.py`. | TODO (launch) |
| **Param count reconciliation** | Bench reports SGNNET=34,976; training reports 67,744. Diff ≈ 32K. Find missing component (likely K_in=25 seed projection). Resolve before paper. | TODO |
| **bench_step832** | PyG scatter vs fancy-index. V_ref_c (compiled) = **0.151ms** (6.55× over eager). Scatter_add compiled = 0.217ms — fancy-index wins. CUDA Graph approach (0.896ms) slower. Best: `torch.compile(max-autotune)` on fancy-index. | **DONE** |
| **bench_step830** | K=4 vs K=5 wall-clock. K=4 compiled=0.140ms vs K=5=0.154ms → **1.10× compiled, 1.18× eager**. Paper claim updated: "10–18% latency reduction" (was "20% projected"). | **DONE** |

## Meditation P0 — 2026-04-17 (step860–863)

From meditation 001 (step267→step859). Scripts written and smoke-tested.

| Step | Description | Script | Slot | Status |
|------|-------------|--------|------|--------|
| **step860** | K=1 KD student @ N=4096 T0. Ref_k5=89.81%, A_k1_scratch=34.37% (−55.44pp!), B_k1_kd=21.10% (−68.71pp!). Routing degenerates completely at large N+K=1. **KILLED.** | `scripts/train_step860_k1_n4096_t0.py` | 5060ti_cuda | **DONE — KILLED** |
| **step861** | Soft routing T1 — NEGATIVE. B_soft_anneal=0.9381=Ref (0.0pp). T0 +1.22pp early-epoch artifact. D_soft_dwproj=0.3659 (catastrophic). Soft routing KILLED. | `scripts/train_step861_soft_routing_t1.py` | 5060ti_cuda | **DONE — KILLED** |
| **step862** | CIFAR-10 cross-dataset T0. Paper requirement (≥2 datasets). VGG pool5 512-dim features. Configs: Linear, MLP_37, MLP_256, Ref_SGNNET. N=512, D=8. | `scripts/train_step862_cifar10_crossdataset.py` | 5060ti_cpu | QUEUED (fringe slot) |
| **step863** | D probe T0: Ref_D16=91.75%, A_D8=89.89% (−1.86pp marginal), B_D8_K10=75.29% (KILLED), C_D12=91.46% (−0.28pp ADVANCES). D=12 advances to T1 (step871). D floor = D=12. | `scripts/train_step863_d8_efficiency.py` | 5060ti_cuda | **DONE** |
| **step864** | D floor+beam ablation T0: D=6 KILLED (−6.93pp), D=4 KILLED (−12.76pp), D=4+K=10 catastrophic. Rbeam_M8=Ref (insensitive), Rbeam_M32=+0.05pp (insensitive). BFS_M32=+0.15pp ADVANCES. BFS_M16/M64 neutral. D=12 confirmed floor. | `scripts/train_step864_d_floor_beam_m.py` | 5060ti_cpu | **DONE** |

**Parking lot (wait for above results):**
- step865: K_hh=1 probe — low priority (5060ti_cpu candidate)
- step867: K=1 + soft routing — DEAD (step861 killed soft routing)

| **step866** | HNSW eval-mode T0. 4 configs: Ref_dw(AH)=91.72%, A_soft_static=91.92%(+0.20pp), B_beam_topk=91.87%(+0.15pp), C_beam_wider=91.90%(+0.18pp), D_beam_train=91.80%(+0.08pp). All advance vs AH but ALL below ΔW baseline (~93.96%). Beam/soft routing not competitive with ΔW-proj. | `scripts/train_step866_hnsw_eval_mode.py` | mini_mps | **DONE — KILLED vs ΔW** |
| **step868** | Z-memory retention T0. Ref_dw=93.96%, A_g03=94.04%(+0.08pp neutral), B_g05=94.04%(+0.08pp neutral), **C_g08=94.29% (+0.33pp ADVANCES)**, D_g09=93.63%(-0.33pp KILLED). gamma=0.8 sweet spot. | `scripts/train_step868_zmem_retention_t0.py` | mini_cpu | **DONE — C_g08 ADVANCES** |
| **step869** | Hub aggregation T0: Ref=93.91%, A_hub005=94.24% (+0.33pp ADVANCES), B_hub03=93.27% (-0.64pp), C_hub10=88.00% (catastrophic), D_hub_beam=92.61% (-1.30pp). alpha=0.05 only sweet spot — larger values homogenize representations. | `scripts/train_step869_hub_aggregation_t0.py` | 5060ti_cuda | **DONE** |
| **step872** | Hub aggregation T1 (75ep, 50% data). Ref_dw=95.36%, **A_hub005=95.54% (+0.18pp VIABLE)**. Light global context consistently helps but below +0.2pp STRONG threshold. Advances to T2. | `scripts/train_step872_hub_t1.py` | 5060ti_cuda | **DONE — VIABLE** |
| **step870** | D_very+ΔW compound T2 (150ep, 100% data). Ref_dw=96.92%, C_compound=96.43% (**-0.48pp KILL**). T1 synergy (+0.18pp) NOT hold at T2. ΔW alone base. D_very+ΔW compound direction closed. | `scripts/train_step870_dvery_dw_compound_t2.py` | studio_mps | **DONE — KILLED** |
| **step871** | D=12+ΔW T1. Ref_dw=95.34%, A_d12=93.91%(-1.43pp), **B_d12_dw=93.63%(-1.71pp KILLED)**, C_d12_comp=93.86%(-1.48pp). ΔW-proj WORSENS D=12 (projection direction insufficient info). D=16 hard floor for ΔW-proj family. | `scripts/train_step871_d12_dw_t1.py` | studio_cpu | **DONE — KILLED** |
| **step873** | BFS M=32 T1. Ref=95.36%, A_bfs_m32=70.96% (**-24.41pp CATASTROPHIC KILL**). Completely collapses at T1 — dynamic top-M selection creates unstable routing gradients. BFS direction CLOSED. | `scripts/train_step873_bfs_m32_t1.py` | 5060ti_cuda | **DONE — KILLED** |
| **step874** | Z-memory T1. Ref=95.29%, A_g08=95.29% (**0.00pp NEUTRAL**). T0 +0.33pp early-epoch artifact. Z-mem NOT advance to T2. Direction CLOSED. | `scripts/train_step874_zmem_t1.py` | mini_mps | **DONE — NEUTRAL** |
| **step875** | Hub T2. Ref=96.59%, A_hub005=96.61% (**+0.03pp NEUTRAL**, non-canonical params 67744). T1 +0.18pp not hold at full training. Hub direction CLOSED. NOTE: MPS path double-counts W_pos (67744 vs 34976); inflated baseline explains ref>95.52%. | `scripts/train_step875_hub_t2.py` | studio_mps | **DONE — NEUTRAL** |
| **step865** | K_hh=1 efficiency probe T0. Ref=93.73%, A_khh1=93.25% (**-0.48pp VIABLE**). 50% routing MACs reduction advances to T1 (step878). | `scripts/train_step865_khh1_t0.py` | mini_cpu | **DONE — VIABLE** |
| **step876** | Hub+Z-mem compound T0. Ref=93.96%, A_hub005=94.11%(+0.15pp), B_zmem_g08=94.42%(+0.46pp), C_compound=93.76%(**-0.20pp CANCEL**). Hub+Z-mem interact negatively — DO NOT compound. Run each independently. | `scripts/train_step876_hub_zmem_compound_t0.py` | studio_cpu | **DONE — CANCEL** |
| **step877** | BFS+Hub compound T0 (20ep, 50% data). 4 configs: Ref, bfs32, hub005, compound. Tests same-path cancellation. | `scripts/train_step877_bfs_hub_compound_t0.py` | 5060ti_cpu | **DONE — KILLED (BFS diverged ep1)** |
| **step878** | K_hh=1 T1 (75ep, 50% data). Ref=95.36%, A_khh1=95.36% (−0.41pp STRONG). Advancing to T2 (step885). | `scripts/train_step878_khh1_t1.py` | 5060ti_cuda | **DONE** |
| **step889** | K_hh=1+K_in=15 compound T2 (150ep, 100% data). Ref=96.64%, C_compound=95.21% (-1.43pp @ 0.57× FLOPs). **CONFIRMED — PAPER CLAIM VALID.** | `scripts/train_step889_compound_t2.py` | 5060ti_cuda | **DONE** |
| **step862** | CIFAR-10 cross-dataset T0 (20ep, 50% data). Paper requirement ≥2 datasets. VGG16 pool5 features N_in=25088. Configs: Linear, MLP_37, MLP_256, Ref_SGNNET (N=1024, D=8). | `scripts/train_step862_cifar10_crossdataset.py` | mini_cpu | **DONE — superseded by step882 T2 (Linear=86.24%, SGNNET=80.69%, -5.55pp MARGINAL)** |
| **step879** | Z-mem gamma fine-scan T0. γ=0.80 confirmed peak (+0.54pp T0). Curve: g070=+0.25, g075=+0.31, **g080=+0.54**, g085=+0.33. Peak at 0.80 but T1 0.00pp — Z-mem direction CLOSED. | `scripts/train_step879_zmem_gamma_scan_t0.py` | mini_cpu | **DONE — g080 T0 peak but T1 NEUTRAL → Z-mem CLOSED** |
| **step880** | K_hh=3 probe T0. Ref=93.91%, A_khh3=93.78% (**-0.13pp WORSE**). More local edges hurt. K_hh=2 confirmed Pareto optimal. K_hh curve: K_hh=1(-0.48pp) < K_hh=2(ref) > K_hh=3(-0.13pp). | `scripts/train_step880_khh3_t0.py` | studio_cpu | **DONE — K_hh=2 confirmed optimal** |
| **step881** | ΔW-proj T2 multi-seed seeds=[0,1,42] for paper error bars. Non-canonical (studio, 67744 params). Use for variance estimation; paper numbers need CUDA re-run. | `scripts/train_step881_multiseed_t2.py` | studio_mps | **DONE — non-canonical Mean=96.44% ±0.26pp; step887 canonical (34,976 params) replaces for paper** |
| **step882** | CIFAR-10 cross-dataset T2 (150ep, 100% data). Linear=86.24%, SGNNET=80.69% (Δ=−5.55pp). 7.4× fewer params at -5.55pp cost. MARGINAL — paper-presentable as honest cross-dataset. | `scripts/train_step882_cifar10_cross_t2.py` | mini_mps | **DONE — MARGINAL** |
| **step883** | ΔW-proj component ablation T0 (20ep, 50% data). 5 configs: Ref_dw/A_sign(clamp0)/B_no_ref(α_r=0)/C_no_theta(θ=0)/D_rand_dir. Paper ablation table: which components essential? | `scripts/train_step883_dwproj_ablation_t0.py` | mini_cpu | **DONE — D_rand_dir=-76.56pp (geometry ESSENTIAL); T1 confirmed in step886** |
| **step884** | K_hh=1 + K_in=15 compound efficiency probe T0 (20ep, 50% data). Configs: Ref_dw/A_khh1/B_kin15/C_compound. Tests ~45% total FLOPs reduction. SUCCESS: C_compound within -1.5pp → ultra-efficient config for paper. | `scripts/train_step884_khh1_kin15_compound_t0.py` | studio_cpu | **DONE — non-canonical; step884_canonical (5060ti_cpu) C_compound=-1.89pp T0 artifact; step889 T2 CONFIRMED -1.43pp** |

