# SGNNET Experiment Queue (Live)

---

**Historical archive:** [EXPERIMENT_QUEUE_history.md](EXPERIMENT_QUEUE_history.md) — all DONE/KILLED entries, P0–P3 tracks, Completed Experiments.

---

## FINAL EFFICIENCY CONFIG (2026-04-11) — step199

**95.52% @ 0.98M routing MACs — both ≤1% FLOPs AND ≤1% params met simultaneously.**
⚠️ **FLOPs clarification (2026-04-14 audit):** 0.98M = routing-only message-passing MACs (N×K_iter×K_hh×D×2). True per-sample FLOPs ≈ 6.5M (incl seed gather K_in=50, AH suppression, normalize, readout). VGG FC = 123M → real ratio ≈ **19× fewer FLOPs** (not 116×). Paper must label "message-passing MACs". Cross-check with ncu (step800).

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
**Calibration rule:** When base changes generation/scale, run 40-50ep param sweep BEFORE full 150ep runs.
**Critical Findings:** See [EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md](EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md)

**Key principle (2026-04-10):** N=1024 winners ARE wins even if no scale to N=4096. If mechanisms push N=1024 toward 96%+, massive efficiency gain (4-6× fewer FLOPs than N=4096). Stop dismissing N=1024 results.

---

## Currently Running (updated 2026-06-22)

**Slot policy:** 5060ti first, then Mini. Studio excluded.

| Machine:Device | Status | Note |
|---------|--------|------|
| mini:mps | FREE | — |
| mini:cpu | FREE | — |
| 5060ti:cuda | FREE | — |

## QUEUED — meditation 005 (2026-07-19)

**Theme: measure the GOAL, not the proxy.** 36+ experiments optimized surrogates (MACs, FLOPs, params). The paper title claims *memory footprint* (bytes) + *energy* (Joules) — neither was ever measured. These P0s convert the headline from proxy to measured and fill named paper holes. All 4 scripts `--help` smoke-tested ✅.

| Step | Script | Slot | Tier | Status | Motivation |
|---|---|---|---|---|---|
| step997 | `scripts/bench_step997_energy_joules.py` | 5060ti_cuda | bench | DONE | **champion 34.81 mJ/inf vs VGG_FC 131.58 mJ/inf → 3.78× fewer Joules** (P=88.7W vs 62.4W but 5.4× faster wall). First measured energy result. Eager PyTorch — compiled would widen. |
| cnn_step037 | `scripts/cnn_distiller/bench_cnn_step037_walltime.py` | mini_mps | bench | DONE | **GA5 13.86ms vs Dense 20.90ms; MAC 1.42× → wall 1.51× → MACs PREDICT latency at GA5 point.** POC-C's 2.7× was at extreme 8.2× depthwise (memory-bound); GA5's modest cut tracks. Pareto survives on wall-time. |
| step998 | `scripts/train_step998_int8_champion.py` | mini_cpu | bench | DONE | **136.6KB fp32 → 34.2KB int8, 4.0× compression, wrap_rate=0, MSE 7.8e-6.** step526 recipe lossless on champion. Accuracy delta still needs champion .pt (3-seed retrain-with-save). |
| step999 | `scripts/train_step999_pruned_vgg_fc_baseline.py` | mini_cpu | T-base | DONE | **pruned VGG_FC @ 34,980 params = 10.17% = CHANCE.** Iso-param baseline collapses (99.97% sparsity). Champion beats it by ~86pp. Answers "isn't it just a small FC?" — strongest new paper asset. |
| ffn_step002 | `scripts/ffn_baseline/ffn_step002_squish_mix_t0.py` | mini_mps | T0 | DONE | **Squish-then-MIX (correct intended arch: per-channel squish→concat→nonlinear cross-channel FFN→logits). T0: b1=92.7%, b5=93.2% — SAME ~93% plateau as no-mix step001.** FALSIFIES the "missing mixing" explanation for the 2.9pp gap: mix and no-mix both cap at 93% at 1–5% budget. SGNNET 95.95% @0.03% is NOT recoverable by any FFN head. Hard ReLU only (no RReLU). T1 re-run pending for paper-exact protocol. |

**cnn_step036 DONE-KILLED (mini_mps, 2026-06-22):** GA5+side=True T0. **66.47% @ep20 — KILLED (prediction-based).** T0 score ≈ k=5 T0 (66.39%). Empirical calibration: k=5 T0~66.4% → T1=73.22% → −1.51pp NO-GAIN. side=True predicted T1 ~73.3% → also NO-GAIN. No T1 warranted. 540K params, 142.3M MACs. **Conclusion: GA5 side=False IS optimal.** Reference configs (Ref/D_small_s/F_wide) use side=True with k=7 + different channel ratios — side branch benefit is architecture-specific, not portable to GA5's (48,96,384) k=3 layout. **CNN GA architecture search COMPLETE.** Result: `results/cnn_step036_ga5side_t0__mini_mps.json`.

**cnn_step035 DONE-NEGATIVE (mini_mps, 2026-06-22):** GA5 k=5 T1. **73.22% @ep70, Δ=−1.51pp vs k=3 T1 (74.73%) — NO-GAIN. k=5 ACTIVELY WORSE than k=3.** Not just neutral — larger dw kernel hurts. k=5 adds 8K params and 4.8M MACs while LOSING 1.51pp. Conclusion: k=3 is optimal for GA5 (48,96,384). Larger receptive field provides no benefit at this scale. k=7 now LOW priority (k=5 worse → k=7 likely worse). **dw_kernel search CLOSED. GA5 k=3 = optimal.** Result: `results/cnn_step035_ga5k5_t1__mini_mps.json`.

**cnn_step034 DONE-ADVANCE (mini_mps, 2026-06-22):** GA5 k=5 T0. **66.39% @ep20 — ADVANCED (T0 rejection filter).** Script flagged NO-GAIN but threshold was wrong (T0 vs T1 baseline). Real gap vs estimated k=3 T0 (~68-70%) is only 2-4pp — not clearly failing. ConvNeXt default k=7; Ref/D_small_s k=7 both STRONG. Advance to T1. 531K params, 146.1M MACs. Result: `results/cnn_step034_ga5k5_t0__mini_mps.json`.

**cnn_step033 DONE-KILLED (mini_mps, 2026-06-22):** GA5+Mixup(α=0.4) T0. **63.90% @ep20, Δ=−10.83pp — KILLED. α=0.4 catastrophically over-regularizes.** GA5+Mixup α=0.2 at ep20 ≈ 72% (step030 seeds); α=0.4 = 63.9%, gap ~8pp. Gap too large to recover in 75ep. Conclusion: stronger Mixup hurts GA5 — optimal α=0.2 confirmed. Next: k=5 kernel size ablation (step034). Result: `results/cnn_step033_ga5_mixup04_t0__mini_mps.json`.

**cnn_step032 DONE-KILLED (mini_mps, 2026-06-22):** GA6 C=(64,128,384)+Mixup T0. **68.64% @ep20 — KILLED (Pareto invalid).** T0 score comparable to GA1-GA4 (65-68%) so NOT a training failure. KILLED on Pareto grounds: 197.7M MACs > Ref (183.2M) and 576K params > GA5 (523K). GA6 Pareto-dominated by GA5 on efficiency with no accuracy upside. Paper table complete; GA6 adds nothing. Result: `results/cnn_step032_ga6_t0__mini_mps.json`.

**cnn_step031 DONE-NEGATIVE (mini_mps, 2026-06-22):** GA5+CutMix(α=0.2) T1. **75.18% @ep75, Δ=+0.45pp vs GA5+Mixup T1, NO-GAIN** (threshold 75.23%, miss by 0.05pp). CutMix effect amplified at GA5 vs GA2 scale (+0.45pp vs +0.20pp) but still below tier threshold. Verdict: CutMix NOT superior to Mixup at T1 significance. Next: architecture expansion GA6. Result: `results/cnn_step031_ga5_cutmix_t1__mini_mps.json`.

**cnn_step030 DONE-STRONG (mini_mps, 2026-06-22):** GA5 C=(48,96,384)+Mixup(α=0.2) T2 4-seed. **77.79% ±0.26% STRONG** — beats Ref T2 (77.61%) by +0.18pp, beats GA2 T2 (77.32%) by +0.47pp. Per-seed: s1=77.48%@ep65, s2=77.58%@ep50, s3=78.06%@ep90, s42=78.04%@ep50. 523K params, 141.3M MACs. Multiple-dip instability confirmed all 4 seeds (ep55-85 range) — best checkpoint secured before dips. C1/C2 width CONFIRMED as lever: single arch change from GA2 broke 77.32% ceiling. Next: step031 GA5+CutMix T1 (augmentation variant scout). Result: `results/cnn_step030_ga5_mixup_t2__summary__mini_mps.json`.

**cnn_step029 DONE-ADVANCE (mini_mps, 2026-06-22):** GA5 C=(48,96,384)+Mixup(α=0.2) T1. **74.73% @ep75, Δ=+1.12pp vs GA2, ADVANCE.** 523K params, 141.3M MACs. C1/C2 width is architecture bottleneck — CONFIRMED. All 6 regularization methods on GA2 capped at 73.94% (ceiling ±0.60pp); single arch change (C1: 32→48, C2: 64→96) breaks ceiling by +1.12pp. Next: T2 multi-seed (step030). Result: `results/cnn_step029_ga5_t1__mini_mps.json`.

**cnn_step028 DONE-NEGATIVE (mini_mps, 2026-06-22):** GA2+MixCutMix(α=0.2) T1. **73.94% @ep75, Δ=+0.33pp, NO-GAIN** (threshold 74.11%). Best regularization result to date (+0.33pp), but still below advance threshold. Regularization direction CONFIRMED CLOSED after 6 experiments (steps 021–028): all methods within ±0.60pp of baseline 73.61%. Architecture bottleneck CONFIRMED — cannot break past ~74% T1 with current GA2 arch via regularization alone. Result: `results/cnn_step028_mixcutmix_t1__mini_mps.json`.

**cnn_step027 DONE (mini_mps, 2026-06-22):** GA2+CutMix(α=0.2) T1. **73.81% @ep65, Δ=+0.20pp, NO-GAIN** (threshold 74.11%). CutMix vs Mixup baseline: +0.20pp — nearly identical. Architecture is the bottleneck, not regularization type. All methods (Mixup/CutMix/LS) within ±0.63pp. Result: `results/cnn_step027_cutmix_t1__mini_mps.json`.

**cnn_step026 DONE-NEGATIVE (mini_mps, 2026-06-22):** GA2+Mixup loss weight ablation T1. A(feat=0.30,dkd=0.50,ce=0.20): 68.84% Δ=−4.77pp. B(feat=0.20,dkd=0.60,ce=0.20): 64.71% Δ=−8.90pp. Both NO-GAIN. Monotonic degradation confirmed. **feat_cos at 0.50 is load-bearing — cannot be traded for DKD weight.** Loss-weight direction CLOSED. Result: `results/cnn_step026_loss_weight_t1__mini_mps.json`.

**cnn_step025 DONE-NEGATIVE (mini_mps, 2026-06-22):** GA2+Mixup(α=0.2)+LabelSmoothing T1. ε=0.05: 73.04% @ep75 Δ=−0.57pp NO-GAIN. ε=0.1: 70.78% @ep65 Δ=−2.83pp NO-GAIN. Both worse than baseline step021 (73.61%). LS direction CLOSED — over-regularization atop Mixup. DKD already provides soft targets; LS double-softens CE → conflicts. Result: `results/cnn_step025_ga2_mixup_ls_t1__mini_mps.json`.

**step986 T2 DONE (5060ti_cuda, 2026-06-21):** CIFAR-10 N=16384 T2. **SCALING EXTENDS. Mean=84.66% ±0.06pp** (seed42=84.60%, seed43=84.75%, seed44=84.63%, 3 seeds 150ep 100%data). Full CIFAR-10 scaling curve T2: N=2048=80.57%, N=4096=82.53%, N=8192=83.55%, N=16384=**84.66%**. Gap to linear ceiling (86.24%): **−1.58pp**. Monotonic scaling confirmed. N=16384 is NOT a ceiling. Results: `results/train_step986_cifar10_n16384_t2_seed42__5060ti_cuda.json` (84.60%), seed43 (84.75%), seed44 (84.63%).

**cnn_step022 DONE (mini_mps, 2026-06-22):** GA2+Mixup T2 multi-seed. **77.32% ±0.56% EFF-PARETO.** seed=1: 76.59% @ep50, seed=2: 77.61% @ep65, seed=3: 77.04% @ep60, seed=42: 78.06% @ep50. vs step020 (no Mixup): **+0.52pp mean, −0.36pp std** (76.80%±0.92% → 77.32%±0.56%). Mixup CONFIRMED beneficial: suppresses ep25 spike universally (peak shifted ep25-35 → ep50-65), raises mean, halves variance. Still EFF-PARETO (0.29pp below STRONG=77.61%). seed=42 best seed: 78.06% (+0.81pp vs no-Mixup 77.25%). Result: `results/cnn_step022_ga2_mixup_multiseed_t2__mini_mps.json`.

**cnn_step004 T2 DONE (mini_mps + mini_cpu, 2026-06-21):** CNN distiller Pareto table, 4-seed T2. Anchors:
| Config | T2 mean | MACs | Params | δ vs Ref |
|---|---|---|---|---|
| F_wide (128/256/256 k=7) | 79.03% ±0.41pp | 559.8M | 825K | +1.42pp |
| Ref (64/128/256 k=7) | 77.61% ±0.45pp | 183.2M | 419K | — |
| D_small_s (32/64/128 k=7) | 75.82% ±1.02pp | 57.5M | 150K | −1.79pp |
D_small_s achieves 75.82% at 3.2× fewer MACs and 2.8× fewer params than Ref — EFF-PARETO anchor confirmed. F_wide = accuracy ceiling. CNN distiller Pareto table complete (3 anchors). Paper: report means ±std for all 3.

**cnn_step018 DONE (mini_mps, 2026-06-21):** EfficientVGG GA T1. All 4 EFF-PARETO. Results:
| Config | T1 | MACs | Params | T0→T1 |
|---|---|---|---|---|
| GA1: C=(32,64,384) k=3 exp=1 crelu | 72.61% @ep75 | 85.6M | 471K | +5.43pp |
| GA2: C=(32,64,384) k=3 exp=2 crelu | 72.51% @ep70 | 98.5M | 481K | +4.18pp |
| GA3: C=(32,96,384) k=3 exp=2 crelu | 72.97% @ep75 | 118.0M | 516K | +5.17pp |
| GA4: C=(32,64,256) k=3 exp=1 (no crelu) | 72.76% @ep65 | 62.9M | 304K | +7.10pp |
Finding: ep35 dip CONFIRMED systematic across all configs (LR≈1.70e-4). GA3 late convergence — C2=96 slows early but wins final. GA4 hypothesis CONFIRMED: no-crelu closes gap (T0 gap 1.52pp → T1 gap 0.26pp vs GA1). All advance to T2. Log: `logs/train_cnn_step018_ga_evgg_t1__mini_mps.log`.

**cnn_step019 DONE (mini_mps, 2026-06-21):** EfficientVGG GA T2. 150ep 100%data. Results:
| Config | T2 | MACs | Params | T1→T2 | Verdict |
|---|---|---|---|---|---|
| GA1: C=(32,64,384) k=3 exp=1 crelu | 75.72% @ep35 | 85.6M | 471K | +3.11pp | **WEAK** |
| GA2: C=(32,64,384) k=3 exp=2 crelu | 77.25% @ep25 | 98.5M | 481K | +4.74pp | **EFF-PARETO** |
| GA3: C=(32,96,384) k=3 exp=2 crelu | 76.89% @ep25 | 118.0M | 516K | +3.92pp | **EFF-PARETO** |
| GA4: C=(32,64,256) k=3 exp=1 (no crelu) | 76.10% @ep30 | 62.9M | 304K | +3.34pp | **EFF-PARETO** |
GA4 most efficient (62.9M MACs, 304K params). GA2 highest accuracy (0.36pp below STRONG). GA1 WEAK — exp=1 + C3=384 can't reach EFF-PARETO. exp=2 (inverted bottleneck) key differentiator. Result: `results/cnn_step019_ga_t2_seed42__mini_mps.json`. Log: `logs/train_cnn_step019_ga_evgg_t2__mini_mps.log`.

**cnn_step015 DONE-C2 (2026-06-21):** MultiScaleCNN NO-CReLU T1, best=37.58%@ep37. C2 CONFIRMED — arch fundamentally limited regardless of CReLU. Removing CReLU gives only +0.58pp vs Ref; 35pp below EfficientVGG T1. Log: `logs/train_cnn_step015_nocrelu_t1__mini_mps.log`.

**cnn_step010 DONE-KILLED (2026-06-21):** GA_2 T1+SGDR — SGDR HARMFUL. best=41.32%@ep19 (cycle1=T0 reproduced), then LR restart DISRUPTS basin: cycle2 starts at 41.32%, never improves (ep25=27%, ep30=33%, ep35=33%...). Cap at 41.32%=T0. **SGDR cycle restarts destroy the learned basin. Phase 2 (no restart) correct approach; confirmed by step012.** Log: `logs/train_cnn_step010_sgdr_t1__mini_mps.log`.

**cnn_step013 DONE-CAPS (2026-06-21):** MultiScaleCNN Ref T1+twophase — best=37.17%@ep57. Phase 1=36.15%, Phase 2 delta=+1.02pp. vs GA_2 T1: −4.51pp. Confirms Ref caps early even with two-phase schedule. MultiScaleCNN family ceiling 37-42% regardless of LR strategy. Verdict: CAPS EARLY. Result: `results/cnn_step013_ref_twophase_t1_seed42__mini_mps.json`.

**cnn_step012 DONE-CAPS (2026-06-21):** GA_2 T1+twophase — best=41.68%@ep21 (+0.36pp vs T0 41.32%). Phase 1 reproduced T0 exactly (41.32% @ep19). Phase 2 adds only 0.36pp then plateaus. A_prime CONFIRMED: two-phase prevents SGDR disruption but GA_2 still caps at 41.68% — MultiScaleCNN arch limit confirmed. Log: `logs/train_cnn_step012_twophase_t1__mini_mps.log`.

**cnn_step011 DONE-KILLED (2026-06-21):** GA_2 T1+nofeat (wf=0, no feat cosine loss) — best=36.61%@ep8, then degraded. Worse than GA_2 T0 41.32%. Eliminates hypothesis B2 (feat cosine incompatibility = cause). **Feat cosine was helping; removing it makes degradation worse.** Root cause A (AdamW trap) or C2 (arch mismatch) — not feat cosine. Log: `logs/train_cnn_step011_nofeat_t1__mini_mps.log`.

**cnn_step009 DONE-FAILED (2026-06-21):** GA_2 T1+warmup — BELOW_THRESHOLD. best=33.07%@ep24, FROZEN ep24-75. −44.28pp vs Ref T2=77.35%. Warmup made it WORSE than step007 (−2.37pp). Both step007 and step009 confirm same failure mode: model finds basin early (ep8-24) then actively degrades. Log: `logs/train_cnn_step009_warmup_t1__mini_mps.log`.

**cnn_step007 DONE-FAILED (2026-06-21):** GA_2 T1 — PATHOLOGICAL TRAINING. Best=35.44%@ep8, then degraded for 67 epochs (val ~24-30%). Final: best=35.44%, −41.91pp vs Ref T2=77.35%. Root cause: same initial LR (3e-4) with T_max=75 keeps LR high for too long; CReLU in stages 1+2 (C=32/64) unstable at high sustained LR. Ref T1 (step004) worked at same LR — architecture-specific instability. Log: `logs/train_cnn_step007_ga2_t1__mini_mps.log`.

**cnn_step008 DONE (mini_cpu, 2026-06-21):** 2×2 ablation — dilation {1,4} × CReLU placement {early=(T,T,F), late=(F,T,T)}. C=(32,64,384) exp=1 crelu varies. 20ep 50%data. Results:
| Config | MACs | Best | Δ_ref | Time | Verdict |
|---|---|---|---|---|---|
| dil1_crelu_early | 105.9M | 40.56% | +3.74pp | 6685s | ADVANCE |
| dil4_crelu_early | 105.9M | 39.82% | +3.00pp | 9490s | ADVANCE (42% slower, −0.74pp) |
| dil1_crelu_late  | 93.1M  | 37.68% | +0.86pp | 6916s | ADVANCE |
| dil4_crelu_late  | 93.1M  | 37.83% | +1.01pp | 8967s | ADVANCE |
Dilation effect (early crelu): dil4−dil1=−0.74pp. Dilation effect (late crelu): +0.15pp (negligible). CReLU early vs late (dil=1): +2.88pp. CReLU early vs late (dil=4): +1.99pp. **VERDICT: dil=4 KILLED (CONFIRMED)** — 42% slower AND −0.74pp with early CReLU (best placement), +0.15pp with late CReLU (negligible). **CReLU early > late: CONFIRMED** (+2–3pp, both dilation settings). Result: `results/cnn_step008_ablation_seed42__mini_cpu.json`.

**cnn_step006 DONE (2026-06-21):** GA arch search complete. Winners:
- GA_2: C=(32,64,384) dil_rates=(1,) crelu=(T,T,F) exp=1, 105.9M MACs, T0=41.32% (+4.51pp)
- GA_1: C=(32,64,384) dil_rates=(1,) crelu=(F,T,T) exp=1, 93.1M MACs, T0=37.99% (+1.17pp)
- GA_4: br=2, 226.2M MACs, T0=30.73% (−6.09pp) → KILLED
Key finding: GA never tested true dilation (n_br=1 → dil_rates=(1,) always). GA_2 vs GA_1 gap is CReLU placement only.

**cnnc_step002 verdict (2026-06-17):** B_isomac=75.11% (−0.46pp), C_isomac=75.26% (−0.31pp). Both NEUTRAL — below ±0.5pp threshold AND no efficiency story (2–3× more params at same MACs). **cnnc line PARKED.** Ref_deep=74.47% (−1.10pp) KILLED.

**Completed this session (session 40):**
- **step986 T2** (5060ti_cuda): **DONE. SCALING EXTENDS.** N=16384 CIFAR-10 T2 = **84.60%** @ep149 (150ep, 100% data, seed=42). Result: `results/train_step986_cifar10_n16384_t2_seed42__5060ti_cuda.json`.
- **step994** (5060ti_cuda): **DONE.** N=16384 multi-seed. seed=43=84.75%, seed=44=84.63%. **Mean=84.66% ±0.06pp** (3 seeds: 42=84.60, 43=84.75, 44=84.63). Scaling curve: 2K=80.57±0.12 → 4K=82.53 → 8K=83.55 → 16K=**84.66±0.06**. sec6_scaling.md updated.
- **cnn_step004 T2 (ALL CONFIGS)** (mini_mps + mini_cpu): **DONE.** Full Pareto table seed=42: Ref=**77.35%** (419K params, 183.2M MACs), F_wide=**79.39%** (+2.04pp, 825K params, 559.8M MACs), D_small_s=**74.52%** (−2.83pp, 150K params, 57.5M MACs — 3.2× fewer MACs, 2.8× fewer params vs Ref). Results: `results/cnn_step004_t2_seed42__mini_mps.json` (Ref+D_small_s), `results/cnn_step004_t2_seed42__mini_cpu.json` (F_wide). F_wide best@ep26, T1→T2=+4.97pp.
- **cnn_step005** (mini_mps): **DONE.** CNN distiller 4-seed variance (seed=42 from step004, seeds 1–3 from step005). **F_wide = clear winner: 79.03% ±0.42pp (stable). D_small_s high variance: 75.82% ±1.02pp (seed=42 low outlier 74.52%). Ref stable: 77.36% ±0.26pp.**

  | config | seed=42 | seed=1 | seed=2 | seed=3 | mean | std |
  |---|---|---|---|---|---|---|
  | Ref | 77.35% | 77.10% | 77.73% | 77.27% | 77.36% | ±0.26pp |
  | D_small_s | 74.52% | 76.08% | 75.64% | 77.02% | 75.82% | ±1.02pp |
  | F_wide | 79.39% | 79.03% | 79.31% | 78.39% | 79.03% | ±0.42pp |

  seed=3 timings: Ref=5152s, D_small_s=3717s (3.2× fewer MACs), F_wide=8175s. D_small_s seed=42 likely low outlier — std=1.02pp driven by it. Paper: report means±std for all 3 configs.
- **step995** (mini_mps): **DONE (NEGATIVE).** ResNet-18 T0: best=**30.70%** @ep19. VGG16-SPECIFIC — SGNNET routing requires VGG16 feature geometry. backbone-agnostic claim FAILED. Result: `results/train_step995_resnet18_t0_seed42__mini_mps.json`.

## QUEUED — meditation 004 (2026-06-17)

| Step | Script | Slot | Tier | Status | Motivation |
|---|---|---|---|---|---|
| step994 | `scripts/train_step994_cifar10_n16384_multiseed.py` | 5060ti_cuda | T2 | DONE | N=16384 seeds 43+44 done. Mean=84.66%±0.06pp (3 seeds). |
| cnn_step005 | `scripts/cnn_distiller/train_cnn_step005_multiseed.py` | mini_mps | T2 | DONE | CNN distiller multiseed variance (4 seeds × 3 configs). F_wide=79.03%±0.42pp winner. |
| step995 | `scripts/train_step995_resnet18_backbone_t0.py` | mini_mps | T0 | DONE (NEGATIVE) | ResNet-18 T0: best=30.70% @ep19. VGG16-SPECIFIC — routing requires VGG feature geometry. |

## QUEUED — session 42 (2026-06-21) — new CNN-GA line

| Step | Script | Slot | Tier | Status | Motivation |
|---|---|---|---|---|---|
| cnn_step006 | `scripts/cnn_distiller/train_cnn_step006_ga_t0.py` | mini_mps | T0 | DONE | GA arch search complete. GA_2 winner: T0=41.32% (+4.51pp vs Ref), 105.9M MACs. GA_4 KILLED (−6.09pp). Key: dil never tested (n_br=1 bug). |
| cnn_step007 | `scripts/cnn_distiller/train_cnn_step007_ga2_t1.py` | mini_mps | T1 | DONE (FAILED) | GA_2 T1 PATHOLOGICAL: best=35.44%@ep8, degraded 67ep. LR instability with CReLU at C=32. |
| cnn_step008 | `scripts/cnn_distiller/train_cnn_step008_dil_crelu_ablation_t0.py` | mini_cpu | T0 | DONE | dil=4 KILLED (42% slower, −0.74pp vs dil=1 early). CReLU early > late: +2.88pp (CONFIRMED). |
| cnn_step009 | `scripts/cnn_distiller/train_cnn_step009_warmup_t1.py` | mini_mps | T1 | DONE (FAILED) | best=33.07%@ep24, −2.37pp vs step007 (warmup made it worse). Same optimizer trap. |
| cnn_step010 | `scripts/cnn_distiller/train_cnn_step010_sgdr_t1.py` | mini_mps | T1 | DONE-KILLED | SGDR harmful: cap at 41.32%=T0, restarts destroy basin. |
| cnn_step011 | `scripts/cnn_distiller/train_cnn_step011_nofeat_t1.py` | mini_mps | T1 | DONE-KILLED | No-feat worse (36.61%@ep8→degraded). Feat cosine was helping. B2 ELIMINATED. |
| cnn_step012 | `scripts/cnn_distiller/train_cnn_step012_twophase_t1.py` | mini_mps | T1 | DONE | 41.68%@ep21 (+0.36pp). Two-phase CONFIRMED. A_prime CONFIRMED. Arch cap confirmed. |
| cnn_step013 | `scripts/cnn_distiller/train_cnn_step013_ref_twophase_t1.py` | mini_mps | T1 | DONE | Ref 37.17%@ep57. Phase 2 delta=+1.02pp. MultiScaleCNN arch limited. |
| cnn_step014 | `scripts/cnn_distiller/train_cnn_step014_ga3_twophase_t1.py` | mini_cpu | T1 | DONE (CAPS) | GA_3 38.50%@ep30 (−3.18pp vs GA_2). Wider C2=192 WORSE. Arch cap confirmed not width-limited. |
| cnn_step015 | `scripts/cnn_distiller/train_cnn_step015_nocrelu_t1.py` | mini_mps | T1 | DONE (C2) | best=37.58%. CReLU NOT bottleneck. MultiScaleCNN arch cap CONFIRMED. |
| cnn_step016 | `scripts/cnn_distiller/train_cnn_step016_ga_evgg.py` | mini_mps | T0 | DONE | EfficientVGG GA search. GA converged: C1=32,C2=64,k=3,side=0 in all top-5. Top: GA1(85.6M fit=0.4761), GA2(98.5M), GA3(118M), GA4(62.9M). |
| cnn_step017 | `scripts/cnn_distiller/train_cnn_step017_ga_evgg_t0.py` | mini_mps | T0 | DONE | All 4 EFF-PARETO: GA1=67.18%, GA2=68.33%, GA3=67.80%, GA4=65.66%. GA4 (no crelu) 12pp behind at ep5, closes to 1.52pp by ep20. |
| cnn_step018 | `scripts/cnn_distiller/train_cnn_step018_ga_evgg_t1.py` | mini_mps | T1 | DONE | All 4 EFF-PARETO: GA1=72.61%, GA2=72.51%, GA3=72.97%, GA4=72.76%. ep35 dip systematic. |
| cnn_step019 | `scripts/cnn_distiller/train_cnn_step019_ga_evgg_t2.py` | mini_mps | T2 | DONE | GA1=75.72% WEAK; GA2=77.25% EFF-PARETO; GA3=76.89% EFF-PARETO; GA4=76.10% EFF-PARETO (62.9M, most efficient). |
| cnn_step020 | `scripts/cnn_distiller/train_cnn_step020_ga2_multiseed_t2.py` | mini_mps | T2 | DONE | GA2 multi-seed T2 (4 seeds). mean=76.80% ±0.92% EFF-PARETO. seed42=77.25% seed1=75.87%@ep35 seed2=78.09%@ep25 seed3=76.00%@ep30. ep25 peak pattern confirmed (3/4 seeds). Mixup needed for STRONG. |
| cnn_step021 | `scripts/cnn_distiller/train_cnn_step021_ga2_mixup_t1.py` | mini_mps | T1 | DONE | GA2+Mixup(α=0.2) T1: best=73.61% @ep65, Δ=+1.10pp vs baseline 72.51%. ADVANCE. Peak shifted ep25→ep65 (Mixup eliminates early-peak overfit, CONFIRMED). elapsed=843s. |
| cnn_step022 | `scripts/cnn_distiller/train_cnn_step022_ga2_mixup_multiseed_t2.py` | mini_mps | T2 | DONE | GA2+Mixup(α=0.2) T2 multi-seed. mean=77.32% ±0.56% EFF-PARETO (gap −0.29pp to STRONG). |
| cnn_step023 | `scripts/cnn_distiller/train_cnn_step023_ga2_mixup_alpha_sweep_t1.py` | mini_mps | T1 | DONE | α=0.3 ADVANCE (74.24%, +0.63pp); α=0.4 NO-GAIN (73.58%); α=0.5 NO-GAIN (73.45%). Non-monotonic peak — α=0.3 Goldilocks. |
| cnn_step024 | `scripts/cnn_distiller/train_cnn_step024_ga2_mixup_a03_multiseed_t2.py` | mini_mps | T2 | DONE | GA2+Mixup(α=0.3) T2 multi-seed. **77.17% ±0.82% EFF-PARETO.** seed1=76.28%@ep65, seed2=78.50%@ep65, seed3=77.04%@ep60, seed42=76.87%@ep115. α=0.3 WORSE than α=0.2 (77.17% vs 77.32%). Gap to STRONG: 0.44pp. α sweep closed — Mixup α not the lever. Result: `results/cnn_step024_ga2_mixup_multiseed_t2__mini_mps.json`. |
| cnn_step025 | `scripts/cnn_distiller/train_cnn_step025_ga2_mixup_ls_t1.py` | mini_mps | T1 | DONE-NEGATIVE | GA2+Mixup(α=0.2)+LS. ε=0.05: 73.04% NO-GAIN (−0.57pp). ε=0.1: 70.78% NO-GAIN (−2.83pp). LS direction CLOSED. |
| cnn_step026 | `scripts/cnn_distiller/train_cnn_step026_loss_weight_t1.py` | mini_mps | T1 | DONE-NEGATIVE | Loss weight ablation CLOSED. A(feat=0.30): 68.84% Δ=−4.77pp. B(feat=0.20): 64.71% Δ=−8.90pp. Monotonic: less feat_cos = worse. feat_cos@0.50 is load-bearing and irreplaceable. |
| cnn_step027 | `scripts/cnn_distiller/train_cnn_step027_cutmix_t1.py` | mini_mps | T1 | DONE-NEGATIVE | GA2+CutMix(α=0.2) T1. 73.81% @ep65, Δ=+0.20pp, NO-GAIN. |
| cnn_step028 | `scripts/cnn_distiller/train_cnn_step028_mixcutmix_t1.py` | mini_mps | T1 | DONE-NEGATIVE | GA2+MixCutMix(α=0.2) T1. 73.94% @ep75, Δ=+0.33pp, NO-GAIN. Best regularization result — still below threshold. Regularization direction CLOSED. |
| cnn_step029 | `scripts/cnn_distiller/train_cnn_step029_ga5_t1.py` | mini_mps | T1 | DONE-ADVANCE | GA5 C=(48,96,384)+Mixup(α=0.2) T1. **74.73% @ep75, Δ=+1.12pp vs GA2. ADVANCE.** C1/C2 width is architecture bottleneck — CONFIRMED. Breaks regularization ceiling (+1.12pp vs best GA2 reg). 523K params, 141.3M MACs. |
| cnn_step030 | `scripts/cnn_distiller/train_cnn_step030_ga5_mixup_t2.py` | mini_mps | T2 | DONE-STRONG | GA5 C=(48,96,384)+Mixup(α=0.2) T2 4-seed. **77.79% ±0.26% STRONG** — beats Ref (77.61%) +0.18pp, GA2 (77.32%) +0.47pp. 523K params, 141.3M MACs. CNN GA search COMPLETE. |

**Non-experiment P0 tasks (equal priority):**
- [x] Update `learnings/paper/MANUSCRIPT_DRAFT_sec6_scaling.md` §9.3: DONE — line 52 has 84.66%±0.06pp, monotonic confirmed, gap −1.58pp to linear
- [ ] Decide CNN distiller scope: Paper 1 appendix vs Paper 2
- [ ] Start LaTeX conversion: sec1_abstract first

**Completed this session (session 38):**
- **step993 T1** (mini_mps): DONE. **Additive dynamic KILLED at T1 — T0 signal noise.** Ref=73.50% (75ep), A=−1.47pp, B=−1.73pp, C=−1.47pp. Sign reversal from T0 (+0.67→+1.51pp). T0 at N=512 20ep had insufficient signal; 75ep reveals the additive term hurts. **Vision debt: additive dynamic connectivity (brief §3.5) KILLED-CONFIRMED.**
- **step990 T0 v2** (mini_mps): DONE. ALL ADVANCE T0 (noise). Ref=61.27% (N=512). A=+0.67pp, B=+1.51pp, C=+1.15pp. T1 step993 KILLED — direction CLOSED.
- **step991 T0 v2** (mini_cpu): DONE. **Hebbian prune-grow KILLED-CONFIRMED.** Ref=85.40%. A_hebbian_random=47.21% (−38.19pp), B_hebbian_wpos=71.11% (−14.29pp), C_hebbian_fast=35.59% (−49.81pp). All massively below Ref. Hebbian prune-grow epoch-boundary rewiring catastrophically disrupts learned ΔW-proj routing. **Vision debt: Hebbian prune-grow (brief §9) RETIRED.**
- **step992 T0** (5060ti_cuda): DONE. **K-means init KILLED-CONFIRMED.** Ref=85.58%. A_kmeans=83.75% (−1.83pp), B_kmeans_classaware=82.47% (−3.11pp). Random init superior to both K-means variants. **Vision debt: K-means init (brief §6.1) RETIRED.**
- **step990 T0 v1** (mini_mps): INVALID. Ref=11.6% (bare SmallWorld, AH chain broken in v1 script). v2 relaunched (see RUNNING above).
- **step991 T0 v1** (mini_cpu): INVALID. Ref=14% (bare SmallWorld, AH chain broken in v1 script). v2 superseded above.
- **step986 T1** (5060ti_cuda or mini): DONE. N=16384 CIFAR-10 T1 — 82.85% @ep69 (75ep, 50%). Scaling curve continues. **Consider T2 (150ep) for paper scaling section.** N-scaling: 80.57→82.53→83.55% (N=2048→4096→8192 T2) + 82.85% T1 at N=16384 (T2 pending).
- **step982 T2** (5060ti_cuda): **DONE-NEGATIVE.** Ref=80.58%, A_aug=55.96%, Δ=−24.62pp [KILL]. Best ep88 then degraded (ep120: 52.86%, ep150: 53.09%). 100k augmented VGG16 features catastrophically worse than 50k standard. HYPOTHESIS: augmented features higher within-class variance; 150ep insufficient for 2× dataset. Paper CIFAR-10 claim unaffected (step980: 80.57% ±0.12pp, standard training). Augmentation at training time DOES NOT help SGNNET on CIFAR-10 at T2.

**Completed this session (session 36):**
- **step985 T0** (5060ti_cuda): DONE (v1 KILLED — BUG). **v1 PhaseGate was symmetric: gate=[s_i,−s_i]+softmax → always 50/50 regardless of s. Zero routing signal.** Ref=91.57%. A/B/C all 11-12% (random). Root cause: softmax of antisymmetric inputs always produces uniform weights. **v2 relaunched (2026-05-13) with correct asymmetric [relu(s), alpha*relu(-s)] + sum-divide normalization.** See v2 RUNNING entry below.
- **step982 T1** (mini_cpu): DONE. Ref=78.60%, A_aug=79.63%, Δ=+1.03pp → **ADVANCES to T2**.
- **step984 T0** (mini_mps): DONE. N=16384 best=80.23% @ep17. Non-monotonic T0 underfit. Script: `scripts/train_step984_cifar10_n16384_t0.py`. Result: `results/train_step984_cifar10_n16384_t0_seed42__mini_mps.json`.

**Completed this session (session 35):**
- step981: N-scaling figure generated (`figures/scaling_law.pdf`) — no training
- step983/ts_step040: TS eval diagnostic — ROOT CAUSE: MSE→mean predictor→50% dir_acc. Fix: directional loss.
- step956: refractory T1 DONE — Ref=95.57%, B_β05_αr1=+0.03pp NEUTRAL, C_β09_αr1=−0.36pp KILL. Direction CLOSED.

**Completed prior sessions (session 34):**
- **step958 T0** (5060ti_cuda): DONE. **LayerNorm routing — L2-normalize load-bearing, alternatives kill.** Ref(L2)=93.94%. A_layernorm=91.36% (−2.57pp KILL, PR↑3.55). B_rmsnorm=90.80% (−3.13pp KILL, PR↑2.71). C_rmsnorm_learned=90.80% (−3.13pp KILL). D_scaledl2=93.94% (±0.00pp NEU). **Finding: LayerNorm raises PR 1.83→3.55 (activates more dims) but hurts −2.57pp. L2-norm angular structure geometrically essential for ΔW-proj routing — not limitation, load-bearing. PR metric no predict accuracy gains.** All norm variants KILLED. L2-normalize confirmed canonical.
- **step957 T0** (mini_cpu): DONE. **Matryoshka D-nesting — MRL raises PR but no help accuracy.** Ref=94.04% (PR=1.83). A_uniform(d=[2,4,8], α=1.0)=93.96% (−0.08pp NEU, PR→2.37). B_decay2(α=0.5)=94.01% (−0.03pp NEU, PR→2.32). C_d48only=93.89% (−0.15pp NEU, PR→2.18). **Finding: All MRL configs raise PR (more dims used) but accuracy flat or slightly down. Low PR (≈2.3) property of representation, not bug — forcing more dims with aux losses no help. D=16 may genuinely encode in ~2 effective dims.** All NEUTRAL. No T1.
- **step966 T0** (5060ti_cuda): KILLED. **Backward reward scoring — training/eval path mismatch, all configs ~13% (random chance).** Custom `forward_with_Z` loop trains different pathway than `model(x)` eval. Root cause: model trained but weights learned through custom BFS routing path no align with standard Resonant eval. Also moot: step967 confirmed zero pathway specialization (no class-selective routing exists to reward). **Direction CLOSED.**

**Completed this session (session 34):**
- **step980** (5060ti_cuda): DONE. **CIFAR-10 multi-seed T2 — authoritative paper variance.** 3 seeds × canonical DeltaW (N=2048, K_iter=5), 150ep T2, 100% data. SGNNET T2: **80.57% ± 0.12pp** (s0=80.43%, s1=80.56%, s42=80.73%). Gap vs Linear 86.24%: **−5.67pp**. Variance ±0.12pp (tighter than T1 ±0.31pp, consistent with step887 ±0.18pp pattern). Seed42 T2=80.73% vs step882 seed42=80.69% — **bit-consistent**. **Paper claim: "SGNNET achieves 80.57% ± 0.12pp on CIFAR-10 (gap −5.67pp vs Linear 86.24%)".**
- **step979** (5060ti_cuda): DONE. **CIFAR-10 multi-seed T1 — variance confirmed.** 5 seeds × canonical DeltaW (N=2048, K_iter=5), 75ep T1, 50% data. SGNNET T1: **77.43% ± 0.31pp** (seeds: s0=77.04%, s1=77.56%, s42=77.93%, s123=77.19%, s2024=77.45%). Gap vs Linear at T1: −8.81pp (T1 underfit; paper number = step882 T2 = −5.55pp). Seed variance ±0.31pp paper-usable. Note: seed=42 highest — consistent with step882 T2 seed42=80.69%.
- **step916** (5060ti_cuda): DONE. **CIFAR-10 K_iter sweep T0 — CATASTROPHIC collapse at K_iter>5.** Ref_k5=75.84% (T0 expected underfit vs step882 80.69% T2). A_k3=73.51% (−2.33pp KILL). B_k10=60.34% (−15.50pp CATASTROPHIC). C_k15=16.31% (−59.53pp NEAR-RANDOM). **Finding: K_iter=5 CIFAR-10 ceiling; more iterations cause collapse. Opposite of Imagenette (where K_iter primary capacity knob). HYPOTHESIS: CIFAR-10 higher-dimensional noisier features cause over-smoothing/instability at K_iter≥10. K_iter=5 CONFIRMED default.**
- **step978** (5060ti_cuda): DONE. **Random features + linear baseline — Claim 3 CONFIRMED.** Lin_VGG=96.76% (full features→linear, ref). RandProj+meanpool (K_in=25, N=2048, D=16) = **13.35%** ≈ chance. RandProj+meanpool D=64 = 17.12%. RandProj_concat (N=256, D=16, concat 4096-dim) = **95.75%** (40K params). **Key finding:** mean-pool of random projections = chance (13-18%); concat without pooling recovers 95.75%; SGNNET routing + mean-pool = 97.30% — routing transforms worthless mean-pooled representation into discriminative one. Confirms Claim 3: iterative routing load-bearing, not projection topology.

**Completed this session (session 33):**
- **step972** (5060ti_cuda): DONE. **KD α-sweep T0** — Ref(α=0,T=1 pure KD)=**93.30%** > D_a1(pure hard CE)=93.17% (−0.13pp NEUTRAL) > all T=4 variants (−0.82 to −1.22pp). **Hard CE matches soft KD within noise.** Combined with step976, VGG soft-label distillation at T=1 contributes nothing measurable on Imagenette. Paper can simplify: "trained against VGG soft labels but hard CE equivalent."
- **step977** (5060ti_cuda): DONE. **Multi-seed KD vs CE T1 — NEUTRAL CONFIRMED.** 5 seeds × 2 configs, 75ep T1, BATCH=512. mean Δ(KD−CE)=**+0.06pp**, σ=**0.15pp** — both well inside |0.3pp|/0.5pp thresholds. Per-seed: s0=+0.05, s1=+0.33, s42=−0.05, s123=+0.08, s2024=−0.10. **Paper: soft-KD at T=1 and hard-CE equivalent. May describe as "cross-entropy against VGG16 soft labels (T=1 ≈ hard CE)."**
- **step970** (5060ti_cuda): DONE. Layerwise isolation — best=0.8316. See log for per-config breakdown.
- **step976** (5060ti_cuda): DONE. **KD temperature sweep settles "rewrite story" concern NEGATIVE.** step199 full stack (N=2048 D=16 K_hh=2 K_iter=5), 20ep T0, 50% data. T=1: **90.29%** (current baseline). T=2: 90.62% (+0.33pp, neutral). T=4: 88.48% (−1.81pp, hurt). T=8: 74.62% (−15.67pp, hurt). Monotonic decrease beyond T=2. **CONFIRMED: current T=1 VGG soft labels already near-optimal.** Root cause: VGG pretrained on ImageNet (Imagenette ⊂ ImageNet) → high-quality confident labels; tempering spreads mass onto wrong classes, adds noise. Classic KD gains require teacher-student capacity mismatch or hard dataset — neither applies here. **Paper numbers stand; no rerun needed.**
- **step971** (5060ti_cuda): KILLED. Quantization scout stuck at 10% — bare SmallWorld ceiling + 20ep OneCycleLR decays before model learns. Needs full Resonant+AH stack.
- **step973** (5060ti_cuda): CLOSED. Adiabatic+GTF scout on bare SmallWorld. Ref=46.73% (bare SmallWorld ceiling). Best adiabatic=27.54% (Adiab_K50), best GTF=24.71% — all configs <80% of Ref. Root cause: (1) bare SmallWorld ≠ Resonant+AH stack (ceiling ~47% not 95.52%), (2) W_pos coupled routing-geometry tensor — sparse K=50 updates (0.15% coverage) destroy K-NN geometry. DIRECTION CLOSED for SGNNET. Adiabatic may still viable for Lever 7 (FC block distillation).

**Completed this session (session 32):**
- **step960 T0** (mini_mps): DONE. ESC-50 low-K_in sweep. Structural failure confirmed.
- **step961 T0** (mini_cpu): DONE. ESC-50 N-sweep at K_in=1. N2048=0.335, −14pp vs Linear. No crossover.
- **step961b T0** (mini_mps): DONE. Large-N extension N∈{2048→16384}. Peak N4096=0.3475, then collapse. Non-monotonic.
- **step962 T0** (mini_mps): DONE. Dense seed projection. C_N256_dp=0.5650 (+14.75pp vs Linear). **CROSSOVER.** D_N512_dp=0.5725.
- **step963 T0** (mini_cpu): DONE. Subspace routing B-sweep. E_b96_s4=0.3125 (−10.5pp vs Linear). Trend: more blocks = better.
- **step964 T0** (mini_mps): DONE. Routing ablation. SGNNET_K0=0.6075 (+19pp). K5 routing HURTS by −4.75pp. **Routing degrades unstructured dense embeddings.** MLP_matched=0.5450 (routing beats MLP by +4.25pp from K0).
- **step965 T0** (mini_cpu): DONE. Finer subspace routing. D_b192_d32=0.4350 (+1.75pp vs Linear = EXCEEDS_LINEAR). Richer D_node + more blocks key.
- **diag_step967** (mini_mps): DONE. **Zero pathway specialization in canonical SGNNET.** mean_act_frac=1.0 (all nodes active), intra_jaccard=1.0, inter_jaccard=1.0, separation=0.0. **Finding: No class-specific routing — all nodes fire for all classes identically. Canonical SGNNET = dense, non-selective routing network. Consistent with routing_gain<0 in Haki: routing = spatial smoothing, not selective pathway activation.**

**Scripted and READY to launch:**
- **step988 T0** (5060ti_cuda): **KILLED**. All configs ~11-13% (random). Ref=91.57%, A_softmax_gate=11.95% (−79.62pp), B_softmax_gate_dw=13.35% (−78.22pp), C_temperature=11.92% (−79.65pp). Root cause: per-neighbor softmax gate over K_hh=2 collapses to uniform weighting after L2 normalization — same symmetry-breaking issue as step985 but via different path. K_hh=2 too small for softmax temperature signal; gate entropy saturates near max (ln2=0.693). **Direction: pure learned routing weights CLOSED (both single-node and per-neighbor variants).**
- **step987 T0** (5060ti_cuda): **KILLED**. Ref=91.57%, A_compound_mul=13.99% (−77.58pp), B_compound_add=14.29% (−77.28pp), C_gate_only_perneighbor=13.50% (−78.07pp). Even compound-with-ΔW-proj-anchor collapses to random. **PhaseGate direction fully CLOSED across all 4 forms (step985 v1 symmetric, v2 asymmetric per-node, step987 compound, step988 per-neighbor softmax). 16th+ multiplicative/learned-gate kill — consistent with gate-death + step852 dynamic-routing closure. CONFIRMED.**
- step966: [KILLED — session 34. Training/eval path mismatch, all configs ~13% random chance. Direction CLOSED.]

**Completed session 36 (step985 T0 v1+v2 KILLED, step982/step986 final status):**
- **step985 T0 v1** (5060ti_cuda): KILLED (BUG). gate=[s_i,−s_i]+softmax always 50/50 — zero routing signal. Symmetric by construction.
- **step985 T0 v2** (5060ti_cuda): KILLED. Corrected asymmetric [relu(s), alpha*relu(-s)] gate — A_phasegate=0.1340 (−0.78pp), B_phasegate_norandom=0.1343 (−0.78pp), C_phasegate_learned=0.1361 (−0.78pp). Root cause: per-NODE dot(Z[i], w_n[i]) degenerate — same-node alignment symmetry-breaks after L2 norm. **→ step987 pivots to per-NEIGHBOR dot(Z[i], w_n[j]) which is structurally asymmetric.**

**Completed this session (session 34, cont.):**
- **step954 T1** (5060ti_cuda): DONE. **K_in=60 T1 Imagenette — BORDERLINE NEUTRAL.** Ref_k25=95.46%, D_kin40=95.75% (+0.28pp), E_kin60=**95.95% (+0.48pp)**. K_in=60 reaches efficiency-champion accuracy (step605=95.95%) but +0.48pp below ≥+0.5pp T2 advance threshold. **VERDICT: NEUTRAL — K_in=60 confirms efficiency champion parity, insufficient to advance T2. Default stays K_in=25.**
- **step969 T0** (mini_cpu): KILLED. **Gradient threshold firing catastrophic on all configs.** GTF_sel_lo/hi/full_lo/hi: 14.70% (random), fire_pct=0.68% (essentially never fires). GTF_sel_adaptive: 10.19% (random), fire_pct=0.0% (never fires, trapped). Ref (no GTF): 38.7% (severely underfit for 20ep). Root cause: threshold calibrated from gradient norms prevents >99% of updates — network cannot learn. **ALL KILLED.**
- **step968 T0** (5060ti_cuda): KILLED. **Adiabatic FP4 long-horizon — FP4 fails completely.** Ref (no FP4, 500ep): 57.32% (underfit — bare SmallWorld below Resonant+AH ceiling). FP4 (2400ep): 11.18% stuck at random after 2400 epochs — NEVER learns. **Root cause: FP4 quantization destroys gradient signal entirely. FP4 direction CLOSED for SGNNET.**

**Completed this session (session 31):**
- **step952 T0** (mini_mps): Grouped input projection. C2_grouped16=+0.13pp, C_grouped64=+0.10pp NEUTRAL. E_multiout KILL (−1.76pp). No config clearly advances; grouped proj adds params for marginal gain.
- **step940 T0** (mini_cpu): Input-conditioned edge gate (tau=1,3,lr). All NEUTRAL (±0.03pp). No signal.
- **step965 T0** (mini_mps): Adiabatic update — launched.
- **step957 T0** (mini_cpu): Matryoshka D-nesting — launched.

**Next launches (when slots free, 2026-05-13):**
- When step985 finishes: advance configs with A/B/C >= Ref+0.5pp → T1.
- When step982 T2 finishes: paper claim if A_aug >= Ref+0.5pp at T2.
- When step986 T1 finishes: if N=16384 >= 83.55% → extend CIFAR-10 scaling curve; else ceiling confirmed.

**Vision-debt T0 batch (see learnings/VISION_DEBT.md + VISION_REVIEW_2026-06-10.md):**
- **step989 T0**: **DONE-KILLED-CONFIRMED (2026-06-11)**. Transformer FFN distillation. Ref_mlp: cos_sim=0.581. A_sgnnet_d16: cos_sim=0.187 (KILL). B_sgnnet_d32: cos_sim=0.213 (KILL). Threshold: cos_sim≥0.5. Both massively below. **SGNNET cannot replicate GPT-2 FFN. Founding vision retired. Paper scope = VGG FC replacement only.** VISION_DEBT fully DONE.
- **step990 T0**: **DONE-ADVANCE** — additive dynamic connectivity. Ref=61.27%, A=+0.67pp, B=+1.51pp, C=+1.15pp. ALL ADVANCE. **step993 T1 RUNNING (mini_mps).**
- **step991 T0**: **KILLED-CONFIRMED** — Hebbian prune-grow. Ref=85.40%, A=−38.19pp, B=−14.29pp, C=−49.81pp. Epoch-boundary rewiring destroys routing. Brief §9 retired.
- **step992 T0**: **KILLED-CONFIRMED** — K-means init. Ref=85.58%, A=−1.83pp, B=−3.11pp. Random init wins. Brief §6.1 retired.
- **step993 T1**: **RUNNING (mini_mps)** — additive dynamic connectivity, 75ep, 50% data, N=512. Advance: ≥ Ref + 0.5pp → defaults update.
- Rule: each gets ONE T0. Kill → CONFIRMED close in architecture_dead_ends.md. No re-attack without named new variable.

**step929 status (CIFAR-10 hflip-aug T1):** aug file extraction DONE (2.14GB, 100K train). Crashed OOM. Superseded by step982 which uses sequential loading (fixed). step929 CLOSED.

**Scripted and queued (smoke-tested):**
- step953: piecewise N_in→D seed T0 — `scripts/train_step953_piecewise_seed_t0.py` [PARKED — mechanism search closed per meditation 003]
- step955: W_edge + ΔW-proj T0 — `scripts/train_step955_wedge_dwproj_t0.py` [PARKED — mechanism search closed per meditation 003]
- step956: refractory B+C T1 — `scripts/train_step956_refractory_t1.py` ✓ DONE (session 35)
- step934: p-RoPE dim split T0 — `scripts/train_step934_prope_dim_split_t0.py` [PARKED — mechanism search closed per meditation 003]
- step935: MoE topology T0 — `scripts/train_step935_moe_topology_t0.py` [PARKED — mechanism search closed per meditation 003]
- step936: per-iter W_pos T0 — `scripts/train_step936_per_iter_wpos_t0.py` [PARKED — mechanism search closed per meditation 003]
- step957: Matryoshka D-nesting T0 — `scripts/train_step957_matryoshka_d_t0.py` ✓ DONE (session 34)
- step958: LayerNorm routing T0 — `scripts/train_step958_layernorm_routing_t0.py` ✓ DONE (session 34)
- step933: local/global K_iter T0 — `scripts/train_step933_local_global_kiter_t0.py` [PARKED — mechanism search closed per meditation 003]

**Physics of Deep Learning diagnostics (decided 2026-04-21):**
- step965: Adiabatic update T0 — `scripts/train_step965_adiabatic_update_t0.py` ✓ DONE (results/train_step965_adiabatic_update_t0_seed42__mini_mps.json)
- step964: Routing field irrotationality — `scripts/diag_step964_routing_curl.py` ✓ DONE (results/diag_step964_routing_curl_seed42__local.json)
- HAKI v2 — `src/sgnnet/haki.py` ✓ DONE (2026-04-21)
  Standalone module. New metrics: pr_seed, pr_gain, routing_entropy, effective_k,
  routing_invariance (CKA), convergence_deltas per step, node_utilization.
  Import: `from src.sgnnet.haki import HAKI`. Run standalone on any checkpoint.
  Paper 1 tooling contribution.

**Paper 1 expansion — audio gap CLOSED (2026-04-21 session 32):**
- step960/961/961b DONE: ΔW-proj structural failure on compressed Whisper embeddings confirmed.
- step962 DONE: Dense seed (SGNNET_K0=0.6075, +19pp vs Linear). Root cause: seeding, not routing.
- step963 DONE: Subspace routing best = 0.3125 (−10.5pp vs Linear). Structure matters.
- step964 DONE: Routing HURTS for unstructured embeddings (−4.75pp). K0 > K5. L2 norm key bias.
- step965 DONE: Subspace routing D_b192_d32=0.435, EXCEEDS Linear. Two mechanisms beat Linear.
- **step966 T0**: Backward reward scoring — `scripts/train_step966_backward_reward_t0.py` ✓ SCRIPTED
- **step967 diagnostic**: Path sparsity — `scripts/diag_step967_path_sparsity.py` ✓ RUNNING (mini_mps).
- step963 (vision): Scaling ceiling — N∈{2048,4096,8192}, D∈{16,32}, K_in∈{25,60}, T2 on Imagenette.
- **ts_step030 T0**: **DONE-NEGATIVE (2026-06-11)**. All models dir_acc≈50%, sharpe<−97. Linear=50.2%/−152, MLP_256=49.8%/−98, SGNNET_K3=50.1%/−164, SGNNET_K0=49.8%/−191. **Honest negative: NIFTY50 next-day return prediction near-random. Directional loss fix confirmed not the issue — task itself is efficient-market limited.** SGNNET provides no advantage. Paper = honest negative result.

**Recently completed (session 30):**
- **step939** (studio_cpu T0 → synced): abs() in ΔW-proj LOAD-BEARING. A_signed=−0.64pp, B_signed_clamp=−0.64pp. BOTH KILLED. abs() confirmed essential.
- **step950** (studio_mps T0 → synced): Per-edge W_edge[N,K_hh,D,D] rotation. ALL KILLED. Root cause: gather-sum without ΔW-proj causes Z collapse (loss=13.19 at ep1). 1M-param W_edge cannot rescue structural smoothing failure. Direction reformulated as step955 (W_edge ON TOP of ΔW-proj).
- **step951** (mini_cpu T0): K_in sweep with Haki diagnostics. Signal purity hypothesis DISPROVED. Coverage dominates: K_in=60 (+1.15pp), K_in=40 (+0.56pp) → both ADVANCE. K_in=5,10,15 KILLED. **New insight: routing_gain ALWAYS NEGATIVE (routing = spatial smoothing, not amplifier). seed_Fisher dominant factor.**
- **step922** (5060ti_cuda T2): N=8192 CIFAR-10 multi-seed. mean=83.55% ±0.014pp. Gap vs Linear=−2.69pp (tight). N-scaling: 80.69→82.53→83.55%.

**Recently completed (session 29):**
- **step938** (mini_cpu T0): Refractory neurons. Ref=93.89%. A_β07_αr2=−1.94pp KILL. B_β05_αr1=−0.03pp NEUTRAL. C_β09_αr1=−0.28pp NEUTRAL. T1 queued for B+C.
- **step952 smoke test** (local 2ep): Init correct — Ref/A_shared/D_node all start ~0.86, PR=2.29 stable. Script ready.

**Recently completed (session 24→25):**
- **step928** (studio_mps T0): ESC-50 architecture tuning. **A_D8 (D=8,K_in=25)=36.25% best** (−11.5pp vs Linear 47.75%). D=4 KILL (−16pp), K_in=50 KILL (−18pp). D=8 narrows gap 3pp vs D=16 but gap remains huge. Audio negative confirmed; vision scope stands.
- **step925** (studio_mps T0, re-run for A_k15): **K_in=15 WINS at N=8192: A_k15=78.84% vs Ref_k25=77.97% (+0.87pp).** step914 N=8192 table CONFIRMED VALID (used K_in=15). CIFAR-10 K_in crossover: K_in=25 best at N=2048, NEUTRAL at N=4096, K_in=15 best at N=8192. Same pattern as Imagenette but shifted higher.

---

**Historical archive (continued):** [EXPERIMENT_QUEUE_history_part3.md](EXPERIMENT_QUEUE_history_part3.md) and [EXPERIMENT_QUEUE_history_part4.md](EXPERIMENT_QUEUE_history_part4.md) — Priority Queue, CNN, Dynamic Routing, AH Era archived 2026-05-13.
