# SGNNET Experiment Queue — Historical Archive Part 4

*Archived from EXPERIMENT_QUEUE.md on 2026-05-13 (session 36).*
*Sections: CNN Distillation Track, Parked experiments, Dynamic Routing Revival, AH Era Revival.*

---

## CNN Distillation Track — `scripts/cnn_distiller/` (2026-04-18)

**Objective:** Distill VGG16 full conv+pool stack (not just FC) into EfficientVGG (~420K params, 2.9% of VGG16 conv 14.7M).
**Architecture:** Depthwise-separable ConvNeXt blocks + side-branch channel attention + CReLU + cosine feature distillation.
**Output:** [B, 25088] features (compatible with SGNNET for compositional eval) + [B, 10] logits.
**Teacher:** VGG16 pool5 features from `data/store.h5` (pre-extracted, aligned with ImageFolder order).
**Target:** ≥95% Imagenette accuracy at <3% of VGG16 conv params.

| Step | Description | Script | Status |
|------|-------------|--------|--------|
| **cnn_step001** | T0 scout (20ep, 50% data). Results: Ref=70.34%, A_no_side=65.89%(-4.46pp side branch huge!), B_vanilla=in-progress. Side branch confirmed critical at 419K scale. | `scripts/cnn_distiller/train_cnn_step001_t0.py` | **DONE** |
| **cnn_step002** | Pareto sweep T0 (20ep, 50% data). 8 configs. F_wide=72.7%, Ref=70.3%, D_small_s=61.0%, B_tiny_k7=41.0%. Kill: E_expand4, A_tiny, G_ultra. | `scripts/cnn_distiller/train_cnn_step002_pareto_t0.py` | **DONE** |
| **cnn_step003** | T1 calibration (75ep, 50% data). **DONE**: Ref=73.17%@ep30, F_wide=74.42%@ep20(+1.25pp), D_small_s=70.96%@ep69(−2.22pp), B_tiny_k7=63.44%@ep71(KILL). C_small: ~63% @ep20 (CPU, running until ~midnight, results TBD). | `scripts/cnn_distiller/train_cnn_step003_t1.py` | **PARTIAL** (mini_cpu: C_small RUNNING ~4h total) |
| **cnn_step004** | T2 validation (150ep, 100% data). Ref crashed @ep111 (permission bug, fixed). **Re-launched mini_mps (Ref+D_small_s)**. F_wide: queue to mini_cpu after C_small + F_wide T2 script written. Previous partial: Ref best=77.35% @ep30 (flat through ep110 — probably converged). | `scripts/cnn_distiller/train_cnn_step004_t2.py` | **RUNNING** (mini_mps: Ref+D_small_s; MPS Metal JIT ~20min startup) |
| **cnn_step005** | F_wide T2 (150ep, 100% data, mini_cpu after C_small). Expected ~3.6h. If F_wide T2 ≥ 77.5%: beats Ref T2 (paper accuracy claim; 825K params at 3× MACs). | TBD (write script or use step004 with `--configs F_wide`) | **TODO** (when mini_cpu free ~midnight) |

**Tier plan:** T0 on mini → Pareto winners (≥85% at best MACs/acc) → T1 (75ep, mini_mps) → T2 (150ep, mini_mps/cpu — CNN mini-only).

---

## Parked — Resume after arch experiments complete

### bench_step840: Concurrent users / hardware democratization claim
**Parked (2026-04-15). Resume after arch experiments.**

Original simple framing (max batch size before OOM → max concurrent users) superseded by more powerful architectural vision:

**Vision (user directive):** Break SGNNET into layers where each K_iter routing step = own microservice. Workflow manager dispatches concurrent requests across steps; all transfers stay within GPU memory. Pipeline-parallel inference — each routing "layer" processes different request at every clock cycle, multiplying effective throughput by K_iter.

**Why matters for paper:**
- VGG16 full model = ~528MB. Head sizes negligible (SGNNET 0.27MB vs FC 494MB). Head memory not bottleneck.
- Real advantage: SGNNET K_iter routing steps structurally identical and independently schedulable — ideal for pipeline parallelism. Standard FC has no such decomposition.
- Claim: "SGNNET homogeneous routing steps enable pipeline-parallel inference on commodity 16GB hardware, multiplying concurrent user capacity by K_iter without additional memory overhead."

**What needs built for validation:**
1. Pipeline-parallel inference harness: K_iter=5 steps as 5 stages, each stage separate forward kernel call
2. Workflow manager: routes batch[i] to stage[i % K_iter], maintains ring buffer of in-flight requests
3. Benchmark: max sustained throughput (requests/sec) vs VGG16-FC and MLP_37 at 16GB budget
4. Script: `bench_step840_pipeline_concurrent.py` — TODO (write after arch experiments done)

---

## Dynamic Routing Revival — 2026-04-18

**Context:** User confirmed stubborn interest in parameter-efficient dynamic routing. Analysis in `learnings/concepts/dynamic_routing_analysis.md`.

| Step | Description | Script | Slot | Status |
|------|-------------|--------|------|--------|
| **step894** | Z-dot + AH softmax routing T0. Ref_dw=94.04%, A_zdot_only=14.09%(−79.95pp COLLAPSE — D=16 noise FM3 confirmed), B_ah_softmax=75.75%(−18.29pp), C_zdot_ah_t10=75.80%(−18.24pp), D_zdot_ah_t03=75.95%(−18.09pp). **ALL KILLED. AH-softmax routing loses −18pp regardless of Z-dot.** Softmax-weight selection fundamentally weaker than ΔW-proj magnitude gating at K_hh=2. | `scripts/train_step894_zdot_soft_routing_t0.py` | 5060ti_cuda | **DONE — KILLED** |
| **step895** | Parameter-free routing T0. Ref_dw=93.99%, A_norm_weighted=15.90%(−78.09pp), B_shared_query=17.78%(−76.20pp), C_factored_attn=15.52%(−78.47pp), D_learned_temp=66.57%(−27.41pp). **ALL KILLED.** A/B/C collapse near-random — softmax over K_hh neighbors catastrophically unstable without structural anchor. D partial recovery (learned temp) still −27pp. Parameter-free routing dead end. | `scripts/train_step895_paramfree_routing_t0.py` | studio_cpu | **DONE — KILLED** |
| **step896** | Biased softmax routing T0. Ref_dw=93.96%, A_bias_khh=74.34%(−19.62pp), B_bias_step=74.34%(−19.62pp), C_bias_full=74.34%(−19.62pp), D_temp_only=74.34%(−19.62pp), E_compound=92.00%(−1.96pp), F_bias_per_node=74.34%(−19.62pp). **ALL KILLED.** A-D-F converge identical 74.34% — bias terms converge to zero (initialized 0, no gradient to break symmetry at fixed point). E_compound partially recovers via ΔW-proj but adds net −1.96pp. LeakyReLU no help. **Softmax routing direction DEFINITIVELY CLOSED across step894/895/896.** | `scripts/train_step896_biased_soft_routing_t0.py` | 5060ti_cuda | **DONE — KILLED** |
| **step897** | Dense cosine gate T0. v1 (W_pos key): Ref=77.45%, A_global_b=16.28%(−61.17pp), D_topk_eval=9.91%(−67.54pp). **KILLED — FM5+FM8.** v1 hijacks W_pos geometry (FM5); v2 fixed W_pos via separate W_key but still collapsed by FM8 (O(N) gradient dominance on shared recurrent Z). C/B configs not run — killed after A_global_b collapse. | `scripts/train_step897_dyn_cosine_gate_t0.py` | 5060ti_cuda | **DONE — KILLED** |
| **step898** | K_hh cosine gate + ΔW-proj additive T0. Division-bug fix required (LeakyReLU negative + clamp explosion). Ref=93.96%, A_khh_gate=93.58%(−0.38pp), B_per_neuron_b=93.66%(−0.31pp), **C_gate_only=15.36%(−78.60pp KILL)**, D_per_edge_b=93.91%(−0.05pp). C_gate_only KILL proves gate has no structural prior without ΔW-proj — fully parasitic. A/B/D neutral with 32–37K extra params = FM5 co-adaptation (same signal path). **Dynamic routing direction CLOSED (all mechanisms 2026-04-19).** | `scripts/train_step898_khh_cosine_gate_t0.py` | local | **DONE — KILLED** |
| **step526** | INT8 QAT. Part A: fp32 train → quant eval (saturate/modular/crt × W-only/W+Z). Hypothesis: L2-norm sphere bounds W_pos ±25/128 → int8 wrap never fires. Part B: QAT from scratch with grad_accum ∈ {1,4,16,64} — tests whether gradient accumulation crosses bin boundaries to recover fp32 accuracy. Script ready. | `scripts/train_step526_int8_qat.py` | 5060ti_cuda | **DONE** (step526/527 results exist from mini_mps) |
| **step901** | Ephemeral Teleportation T1. Pre-killed: step899 closed direction (all K_ep=1 configs 18–56%). | `scripts/train_step901_ephemeral_t1.py` | — | **KILLED (pre-kill, step899 → direction closed)** |
| **step902** | D-scaling T1. Pre-killed: step900 closed direction (C_d24_dw=−0.35pp, D_d32_dw=−0.15pp vs D=16+dw). | `scripts/train_step902_d_scaling_t1.py` | — | **KILLED (pre-kill, step900 → direction closed)** |
| **step903** | **Epoch-Topology Dynamic Slot T0** FINAL: Ref=94.17%, A_std_lr=93.32%(−0.84pp), B_half_lr=91.97%(−2.19pp KILL), C_tenth_lr=85.99%(−8.18pp KILL), D_warmup10=93.17%(−0.99pp). None advance. Epoch topology rebuilding consistently hurts — KNN disrupts learned geometry. **Direction CLOSED.** | `scripts/train_step903_epoch_topology_t0.py` | 5060ti_cpu | **DONE — KILLED** |
| **step904** | **Node-Level Input-Conditioned Gating T0** FINAL: Ref_dw=94.06%, A_tau1=92.51%(−1.55pp), B_tau3=88.38%(−5.68pp KILL), C_tau_learnt=92.51%(−1.55pp), D_se_bn=93.55%(−0.51pp), E_wpos_geo=93.78%(−0.28pp), F_per_iter=92.46%(−1.61pp). Gates ARE dynamic (H=0.4-0.7) but still hurt. NOT gate-death; information-bottleneck from gating out useful nodes. None advance (≥+0.5pp threshold not met). **Direction CLOSED (bottleneck, not collapse).** | `scripts/train_step904_node_gating_t0.py` | 5060ti_cuda | **DONE — KILLED** |
| **step905** | **CIFAR-100 Rigor Baseline T1** (75ep, 50% data). Linear=64.78%, MLP_256=64.24%, MLP_512=64.73%, SGNNET_can=35.40%(−29.38pp), SGNNET_N4096=40.57%(−24.21pp). **CATASTROPHIC GAP.** Root cause: N_out=100, C_ho readout gives ~18 nodes/class (vs ~184 for Imagenette 10-class) — capacity-limited. Paper scope CONFIRMED Imagenette only. | `scripts/train_step905_cifar100_rigor_t1.py` | studio_cpu | **DONE — KILLED. Paper scope = Imagenette only.** |
| **step909** | **CIFAR-10 N-scaling** — close cross-dataset gap. B_N4096(150ep) DONE: 82.53% (+1.84pp vs N=2048, gap closes −5.55→−3.71pp vs Linear 86.15%). Ref_linear=86.15% (consistent). A_N2048_T3(200ep) still running on 5060ti_cpu. H1: +1.84pp (just under ≥2pp) — capacity scaling helps but no close gap fully. | `scripts/train_step909_cifar10_n_scaling.py` | 5060ti_cpu RUNNING | **PARTIAL** |
| **step914** | **CIFAR-10 N=8192 T2** (150ep, 100% data). **FINAL: C_N8192=83.58% best @ep130, gap=−2.66pp vs Linear. N-scaling: N2048=80.69% → N4096=82.53% → N8192=83.58% (closing ~1.4pp per 2×N, diminishing returns). Multi-seed validation → step922.** | `scripts/train_step914_cifar10_n8192_t2.py` | 5060ti_cuda | **DONE — N-scaling curve confirmed** |
| **step915** | **CIFAR-10 ΔW-proj Ablation T1** (75ep, 50% data). **FINAL: Ref_dw=78.37%, A_no_dw=15.96% (−62.41pp COLLAPSE). ΔW-proj load-bearing on CIFAR-10. Without it: plain sum averaging → trivial diffusion fixed point → near-random (10% chance = 10 classes). Cross-dataset claim STRONGLY CONFIRMED.** | `scripts/train_step915_cifar10_dwproj_ablation_t1.py` | studio_mps | **DONE — ΔW-proj ESSENTIAL cross-dataset** |
| **step916** | **CIFAR-10 K_iter Sweep T0** (20ep, 50% data). **FINAL: Ref_k5=75.72%, A_k3=−2.12pp(KILL), B_k10=−15.54pp(KILL), C_k15=−59.97pp(KILL/collapse). K_iter=5 optimal — over-smoothing at K_iter>5 (GNN over-smoothing analogue). Paper finding: K_iter sweet-spot at 5; direction CLOSED.** | `scripts/train_step916_cifar10_kiter_sweep_t0.py` | studio_cpu | **DONE — K_iter=5 universal sweet-spot** |
| **step917** | **CIFAR-10 α_reflect Sweep T0** (20ep, 50% data). **FINAL: Ref_a05=75.77%, A_a00=−0.49pp(NEUTRAL), B_a025=−0.51pp(KILL), C_a075=+0.52pp(ADVANCE→T1), D_a10=−2.01pp(KILL). α=0.75 beats canonical α=0.5 on CIFAR-10 by +0.52pp. α=0.5 not cross-dataset optimal.** → step918 T1. | `scripts/train_step917_cifar10_alpha_reflect_t0.py` | studio_mps | **DONE — C_a075 ADVANCE** |
| **step918** | **CIFAR-10 α_reflect T1** (75ep, 50% data). **FINAL: Ref_a05=78.69%, C_a075=77.88% (−0.81pp). T0 artifact confirmed. α=0.5 canonical cross-dataset. Direction CLOSED.** | `scripts/train_step918_cifar10_alpha_reflect_t1.py` | studio_mps | **DONE — T0 artifact, α=0.5 CANONICAL** |
| **step919** | **Imagenette K_iter Sweep T0** (20ep, 50% data). **FINAL: Ref_k5=94.09%, A_k3=−0.66pp, B_k8=−4.13pp, C_k10=−9.35pp, D_k15=−74.78pp(COLLAPSE). Universal over-smoothing CONFIRMED. Same collapse profile as CIFAR-10 (step916). PAPER CLAIM: K_iter=5 sweet-spot on both datasets; over-smoothing dataset-independent.** | `scripts/train_step919_imagenette_kiter_sweep_t0.py` | studio_cpu | **DONE — UNIVERSAL OVER-SMOOTHING CONFIRMED** |
| **step920** | **CIFAR-10 K_in Sweep T0** (20ep, 50% data). **FINAL: A_k15=76.03%, Ref_k25=75.32% — run-order artifact (A_k15 ran before Ref, delta_vs_ref=null). B_k50=74.91%(−0.41pp), C_k100=74.81%(−0.51pp). Apparent advance was noise — step923 T1 REVERTED (K_in=15=−0.88pp).** | `scripts/train_step920_cifar10_kin_sweep_t0.py` | studio_cpu | **DONE — T0 artifact, K_in=25 stays CIFAR-10 default** |
| **step923** | **CIFAR-10 K_in=15 T1** (75ep, 50% data, N=2048, seed=42). **FINAL: Ref_k25=78.59% @ep60, A_k15=77.71% @ep74, Δ=−0.88pp. REVERT. step920 T0 advance run-order artifact. K_in=25 CIFAR-10 default at N=2048. K_in cost: CIFAR-10 −0.88pp vs Imagenette −0.33pp — harder task penalizes sparse seeding more.** → step924: probe K_in at N=4096 CIFAR-10. | `scripts/train_step923_cifar10_kin15_t1.py` | studio_cpu | **DONE — REVERT, K_in=25 stays** |
| **step918** | **CIFAR-10 N=4096 underfitting probe** (200ep, 100% data). Controls for underfitting at N=4096. Tests whether −3.71pp gap vs Linear epoch-limited or architectural. | `scripts/train_step918_cifar10_n4096_200ep.py` | studio_cpu RUNNING | **RUNNING** |
| **step924** | **CIFAR-10 K_in=25 vs K_in=15 @ N=4096 T0** (20ep, 50% data). **NEUTRAL: Ref_k25=77.82%, A_k15=77.75% (−0.07pp)**. Crossover between N=2048 (K_in=25 wins) and N=8192 (K_in=15 wins) confirmed. step909 82.53% VALID. | `scripts/train_step924_cifar10_kin_n4096_t0.py` | studio_cpu | **DONE — NEUTRAL** |
| **step925** | **CIFAR-10 K_in=25 vs K_in=15 @ N=8192 T0** (20ep, 50% data). **K_in=15 WINS: A_k15=78.84% vs Ref_k25=77.97% (+0.87pp)**. step914 83.58% (K_in=15) CONFIRMED VALID. K_in crossover between N=2048 and N=8192, mirrors Imagenette exactly. | `scripts/train_step925_cifar10_kin_n8192_t0.py` | studio_mps | **DONE — K_in=15 wins at N=8192** |
| **step926** | **ESC-50 Audio Robustness T0** (20ep, 50% data). Canonical ΔW-proj. **AUDIO GAP CONFIRMED: Linear=47.75%, N512=18.5%, N1024=27.5%, N2048=32.0% (−15.75pp)**. ΔW-proj no close audio gap. Whisper features lack spatial geometry. Paper scope = vision only. | `scripts/train_step926_esc50_robustness_t0.py` | studio_cpu | **DONE — AUDIO GAP CONFIRMED** |
| **step927** | **CIFAR-100 N=8192 T0** (20ep, 50% data). Extends step905 N-scaling (N2048=35.40%, N4096=40.57%). Tests if N=8192 closes −24pp gap. Advance: gap ≤20pp → T1 (step930). Stagnate → architectural failure confirmed. | `scripts/train_step927_cifar100_n8192_t0.py` | studio_mps | **RUNNING** |
| **step929** | **CIFAR-10 hflip-aug T1** (75ep, 50% aug data). Follows step924 T0 winners. Tests if aug consistently closes gap at calibration level. Advance rule: ≥+0.5pp → T2 (step930). PREREQUISITE: store_cifar10_aug.h5 + step924 T0 result. Script: train_step925_cifar10_hflip_aug_t1.py (naming quirk). Target slot: 5060ti_cuda. | `scripts/train_step925_cifar10_hflip_aug_t1.py` | 5060ti_cuda | **QUEUED — after step924 T0 AND store_cifar10_aug.h5** |
| **step930** | **CIFAR-10 hflip-aug T2** (150ep, 100% aug data, best T1 config). Paper claim: N=8192+aug vs gap −2.66pp. Target: gap ≤−1.5pp. OR: CIFAR-100 N=8192 T1 if step927 shows gap ≤20pp. Script TBD. | TBD | 5060ti_cuda | **QUEUED — after step929 T1 OR step927 T0** |
| **step928** | **ESC-50 Architecture Tuning T0** (20ep, 50% data). D×K_in sweep: {D=4,8,16}×{K_in=25,50}. **FINAL: A_D8=36.25%(−11.5pp best), D=4 KILL(−16pp), K_in=50 KILL(−18pp). Audio gap confirmed not closeable by arch. Paper scope = vision only.** | `scripts/train_step928_esc50_arch_tuning_t0.py` | studio_mps | **DONE — AUDIO GAP CONFIRMED, all fail >10pp** |
| **step906** | **Top-K Activation Sparsity T0** (20ep, 50% data). Ref_dw=94.11%, A_top75=79.06%(−15.06pp), B_top50=78.19%(−15.92pp), C_top25=80.46%(−13.66pp), D_top10=78.80%(−15.31pp). Hard top-K ALL CATASTROPHIC. E_soft_tau1=92.31%(−1.81pp), F_soft_tau3=88.25%(−5.86pp). Soft marginal/significant hurt. Gate entropy=0.62 (near-max, non-discriminative). Same failure mode as step904 — gating Z before K_iter destroys collective computation. **ALL KILLED. FGSEGNet-style gating before MP definitively closes.** | `scripts/train_step906_topk_activation_t0.py` | studio_mps | **DONE — KILLED** |
| **step907** | **Readout-Gate Input-Conditioned T0** (20ep, 50% data). Ref_dw=94.04%, A_ro_tau1=+0.36pp(NEUTRAL), B_ro_tau3=−1.04pp, C_ro_tau_lrn=**+0.54pp ADVANCE**, D_ro_topk25=−3.49pp(KILL), **E_ro_geo=+1.12pp ADVANCE**, F_ro_norm=−0.74pp. KEY FINDING: readout-level gating WORKS (+1.12pp) while pre-K_iter gating fails (FM10). Geometric gate (W_pos cosine vs Z_mean) strongest. → T1 (step910). | `scripts/train_step907_readout_gate_t0.py` | studio_mps | **DONE — E_ro_geo +1.12pp, C_ro_tau_lrn +0.54pp → step910 T1** |
| **step910** | **Readout Gate T1** (75ep, 50% data). Ref_dw=95.41%, A_ro_tau1=+0.66pp, C_ro_tau_lrn=+0.69pp, E_ro_geo=+0.61pp. **ALL THREE ADVANCE.** All within 0.08pp of each other. → T2 (step911). | `scripts/train_step910_readout_gate_t1.py` | studio_mps | **DONE — all three advance to T2** |
| **step911** | **Readout Gate T2** (150ep, 100% data). **FINAL: Ref=96.74%, A_ro_tau1=+0.15pp, C_ro_tau_lrn=+0.25pp, E_ro_geo=+0.18pp. ALL below ≥+0.5pp threshold. T1 artifact confirmed — gain compresses from +0.61–0.69pp (T1) to +0.15–0.25pp (T2). Paper claim NOT supported.** Direction CLOSED. | `scripts/train_step911_readout_gate_t2.py` | studio_mps | **DONE — T1 artifact, direction CLOSED** |
| **step912** | **CIFAR-10 Readout Gate T0** (20ep, 50% data). Ref=76.03%, A_ro_tau1=+0.40pp(NEUTRAL), **E_ro_geo=−0.60pp(KILL)**. Readout gate NOT generalize to CIFAR-10. Gain Imagenette-specific. | `scripts/train_step912_cifar10_readout_gate_t0.py` | studio_cpu | **DONE — gate Imagenette-specific, no CIFAR-10 generalization** |
| **step899** | Ephemeral teleportation T0. Ref_dw=93.96%, A_local1_dw=92.10%(−1.86pp). All K_ep=1 configs: 18–56% (−38 to −76pp KILL). Random per-step reconnection introduces gradient noise identical to FM9 — complete training collapse. **ALL KILLED. Topology diameter direction CLOSED.** step901 (T1) not triggered. | `scripts/train_step899_ephemeral_teleport_t0.py` | 5060ti_cuda | **DONE — KILLED** |

### AH Era Revival — Mechanisms Not Tested on ΔW-proj (2026-04-20)

Gained significant ground in AH era (D=64/N=1024/K_iter=12) but never ported to current efficiency config (N=2048/D=16/K_iter=5/ΔW-proj). Full analysis: `learnings/concepts/ah_era_untested_on_dw.md`.
**Caveat:** AH arch had D=64 — expect gains smaller at D=16. Treat as directional signal only.

| Step | Description | AH Era Gain | Script | Slot | Status |
|------|-------------|------------|--------|------|--------|
| **step937** | **Z-bias per K_iter T0** (20ep, 50% data). Port of step106 to ΔW-proj. `Z_t += emb[t]` at each K_iter=5 steps. 80 extra params. Configs: Ref / A_zbias_init0 / B_zbias_initrand. | +7.42pp (D=64/N=1024) | Ref=94.19%, A_z0=94.19% (+0.00pp), B_zrnd=94.32% (+0.13pp) | mini_cpu | **DONE — NEUTRAL** (best +0.13pp < 0.50pp threshold; regime mismatch: AH era D=64/K_iter=12 vs current D=16/K_iter=5) |
| **step938** | **Refractory neurons T0** (20ep, 50% data). Port of step16E. `Z_t = Z_t − α_r × max(0, β × |Z_{t-1}|)`. Configs: Ref / A_β07_αr2 / B_β05_αr1 / C_β09_αr1 (sweep β×α_r). Ref=93.89%. A_β07_αr2=−1.94pp KILL. B_β05_αr1=−0.03pp NEUTRAL. C_β09_αr1=−0.28pp NEUTRAL. B+C advance to T1. | +4.31pp (D=16/N=512 wave arch) | Ref=93.89% | mini_cpu | **DONE — B/C NEUTRAL, advance to T1** |
| **step939** | **Signed ΔW routing T0** (20ep, 50% data). Remove abs() from proj_coeff. Ref=94.09%. A_signed=−0.64pp KILL. B_signed_clamp=−0.64pp KILL. **abs() load-bearing — anti-aligned negative signals cancel gradient coherence.** Direction CLOSED. | +3.97pp (D=64, ConcatReLU — different regime) | Ref=94.09%, A/B=−0.64pp | studio_cpu | **DONE — KILLED** |
| **step940** | **Input-conditioned edge reweighting T0** (20ep, 50% data). Port of step36. `w_ij = sigmoid(x_in_i · x_in_j / tau)` for each conn_hh edge; scale Z_agg by w_ij. Uses fixed input features x (not Z state — safe from gate-death). Configs: Ref / A_tau1 / B_tau3 / C_tau_learnable. | +2.34pp (D=64 without AH) | TBD | any | **QUEUED** |
| **step941** | **Sparse beam active set T0** (20ep, 50% data). Port of step25. At each K_iter step, only top-K active nodes by ‖Z‖ route; others hold Z_prev. Efficiency mechanism. Configs: Ref / A_top_half / B_top_quarter / C_top_eighth. step25 found top-1/8 best. | +2.88pp + 53× speedup (D=16/N=512) | TBD | any | **QUEUED** |
| **step942** | **Max-pool readout T0** (20ep, 50% data). Replace mean-pool over N nodes with max-pool. Minimal change. Note: step907/911 readout gate T1 artifact suggests readout changes noisy; max-pool different (permutation-invariant aggregation, not gating). Configs: Ref / A_maxpool / B_topk_mean (top-32 mean). | Not run in AH era | TBD | any | **QUEUED** |
| **step950** | **Per-edge channel rotation T0** (20ep, 50% data). NO ΔW-proj. Ref=gather-sum. Ref=19.52%, A_shared=+0.03pp, B_peredge(1M params)=+0.03pp, C_small_D8=−5.99pp. **STRUCTURAL FAILURE: gather-sum without ΔW-proj collapses Z (all nodes average to correlated direction → extreme wrong predictions at init). W_edge identity init can't escape. 1M params = zero gain.** Q answered: W_edge CANNOT substitute for ΔW-proj. Reformulation: test W_edge ON TOP of ΔW-proj (step955). | New mechanism | `scripts/train_step950_per_edge_rotation_t0.py` | studio_mps | **DONE — KILLED. ΔW-proj load-bearing.** |
| **step951** | **K_in Signal Purity Sweep T0 + Observational Haki** (20ep, 50% data). Tests whether K_in affects signal quality in seed scalar Z[i,0]=sum(x[conn_in[i]]). Configs: Ref(K_in=25) / A_kin5 / B_kin10 / C_kin15 / D_kin40 / E_kin60. All 34,976 params. Haki metrics: seed_Fisher, final_Fisher, routing_gain, PR, dead_frac logged at ep 1/5/10/20. | — | `scripts/train_step951_kin_purity_t0.py` | mini_cpu | **RUNNING** (launched 2026-04-20) |
| **step952** | **Grouped Input Projection T0 + Haki** (20ep, 50% data). Replace parameter-free scatter sum with learned per-group weighted projection. Configs: Ref/A_shared/C2_grouped16/C_grouped64/B_grouped256/D_node/E_multiout. Raw linear weights init=1 (= sum baseline at ep0). Smoke test passed: Ref=0.8601, A_shared/D_node=0.8614, PR=2.29 stable. | — | `scripts/train_step952_grouped_input_proj_t0.py` | mini_cpu (after step951) | **QUEUED** |
| **step900** | D-scaling T0. Script bug: SGNNET_DRef (no ΔW-proj) collapses — Ref_d16=19.57%, A_d24=18.96%, B_d32=16.64% (all meaningless). ΔW variants vs D=16+dw ref (93.96%): C_d24_dw=93.61%(−0.35pp), D_d32_dw=93.81%(−0.15pp). ΔW-proj saturates D-dimensional geometry at D=16; larger D adds no benefit. **D-scaling direction CLOSED. step902 not triggered.** | `scripts/train_step900_d_scaling_t0.py` | studio_cpu | **DONE — KILLED** |
## Older completed sessions (session 17→23) — archived 2026-05-13

**Recently completed (session 22→23):**
- **step924/926** (studio_cpu T0): CIFAR-10 K_in=25 vs K_in=15 @ N=4096. **NEUTRAL: Ref_k25=77.82%, A_k15=77.75% (−0.07pp)**. Crossover confirmed between N=2048 and N=4096. step909 82.53% VALID.
- **step926** (studio_cpu T0): ESC-50 audio robustness, canonical ΔW-proj. **AUDIO GAP CONFIRMED: N2048=32.0% vs Linear=47.75% (−15.75pp)**. ΔW-proj no help audio. Vision scope only confirmed.
- **step923** (studio_cpu T1): CIFAR-10 K_in=15 vs K_in=25 @ N=2048. **REVERT: K_in=25=78.59%, K_in=15=77.71% (−0.88pp)**. K_in=25 CIFAR-10 default at N=2048.
- **step921** (studio_mps 200ep): CIFAR-10 N=4096 extended. **PLATEAU: best=83.08% @ep186 (+0.55pp vs 150ep)**. 150ep correct budget.
- **cnn_step003 T1** (mini_mps): EfficientVGG T1 Imagenette. **Ref=73.17% (183M MACs, 419K), F_wide=74.42% (+1.25pp, 560M MACs, 825K), D_small_s=70.96% (−2.22pp, 57.5M MACs, 150K)**. B_tiny_k7 KILLED (63.44%). All 3 advance to T2.

**Recently completed (session 21→22):**
- **step921** (studio_mps): CIFAR-10 N=4096 200ep — **best=83.08% @ep186, Δ=+0.55pp vs step909 (150ep)**. PLATEAU confirmed: 150ep already converged; gap architectural. 200ep gives marginal improvement within seed variance.
- **step923** (studio_cpu): CIFAR-10 K_in=15 T1 @ N=2048 — **REVERT. Ref_k25=78.59%, A_k15=77.71%, delta=−0.88pp**. step920 T0 advance was run-order artifact. K_in=25 confirmed CIFAR-10 default at N=2048.

**Recently completed (session 17→18):**
- **step877** (5060ti_cpu): BFS+Hub T0 — KILLED. BFS diverged ep1 (loss=14.17). Direction CLOSED.
- **step878** (5060ti_cuda): K_hh=1 T1: Ref=0.9536, K_hh=1=0.9496 **Δ=-0.41pp STRONG**. Advancing to T2 (step885).
- **step881** (studio_mps): Multi-seed T2 non-canonical (67K params). seed0=96.20%, seed1=96.31%, seed42=96.82%. **Mean=96.44%, std=±0.26pp** — variance estimate for paper.
- **step883** (mini_cpu): ΔW-proj ablation T0. **D_rand_dir=-76.56pp (geometry ESSENTIAL)**. B_no_ref=-1.32pp, A_sign=-1.83pp, C_no_theta=-0.71pp. All components load-bearing at T0. T1 confirmation running (step886).
- **step886** (mini_cpu): ΔW-proj ablation T1. A_sign=-0.59pp **LOAD-BEARING**. B_no_ref=-0.51pp **LOAD-BEARING**. C_no_theta=+0.15pp **NEUTRAL** — theta NOT needed, flipped from T0 as predicted. **Paper: 2 load-bearing components; theta simplifies out.**
- **step884 (studio)** (studio_cpu): NON-CANONICAL. Canonical re-run on 5060ti_cpu.
- **step885** (5060ti_cuda): K_hh=1 T2. Ref=0.9664, A_khh1=0.9590. **Δ=-0.74pp MARGINAL** (between -0.5 and -1.0pp). Paper: "50% routing MACs at -0.74pp, mention with caveat." Advancing canonical multi-seed (step887).
- **step884_canonical** (5060ti_cpu): K_hh=1+K_in=15 compound T0 canonical. C_compound=-1.89pp → **KILL by T0 criterion**. NOTE: B_kin15 T0=-1.22pp is T0 artifact (step632 T2=-0.33pp). Expected compound T2≈-1.07pp (viable). C_compound T1 queued for later. Components reported separately in paper.
- **step887** (5060ti_cuda): Canonical multi-seed T2 (34,976 params). seed0=96.23% @ep150, seed1=96.28% @ep71, seed42=96.64% @ep120. **Mean=96.38% ± 0.18pp. CONFIRMED for paper.** Replaces non-canonical step881. Paper claim: "96.38% ± 0.18pp (canonical 34,976 params)".
- **step888** (mini_mps): K_hh=1+K_in=15 compound T1. Ref=95.26%, C_compound=94.29% (-0.97pp). **VIABLE (within -1.5pp).** 43% total FLOPs reduction (1.31M vs 2.29M). Advancing to T2 (step889 on 5060ti_cuda). T0=-1.89pp confirmed artifact.
- **step855** (mini_cpu): Sparse BFS T0. A_fixed_M16=-2.10pp, B_cascade=-2.23pp, C_quiet_zero=-2.20pp, D_readout_active=-2.05pp. **ALL KILLED. Direction CLOSED.**
- **step859** (5060ti_cpu): Soft routing T0. B_soft_anneal=+0.99pp T0 artifact (step861 T1=0.0pp confirmed). D_soft_dwproj=-60.51pp CATASTROPHIC. C_soft_ah=-0.10pp. **Soft routing CLOSED.** AH cancels soft gain; ΔW+soft collapses.
- **step862** (mini_cpu): CIFAR-10 cross-dataset T0 — **CONFIG BUG** (N=1024/D=8, 9K params). Ref_SGNNET=-29.39pp (invalid). Re-run with canonical config as step890 on 5060ti_cpu.
- **step889** (5060ti_cuda): K_hh=1+K_in=15 compound T2. Ref=96.64%, C_compound=95.21% (-1.43pp). **CONFIRMED — 0.57× FLOPs at -1.43pp. PAPER CLAIM VALID.**
- **step890** (5060ti_cpu): CIFAR-10 cross-dataset T0 canonical. SGNNET=75.77% vs Linear=86.47%, **Δ=-10.70pp → KILLED by T0 criterion** (barely over -10pp threshold). T0 underfit confirmed by step882 T2 below.
- **step882** (mini_mps): CIFAR-10 cross-dataset T2 (150ep, 100% data, canonical 34,976 params). **Linear=86.24%, SGNNET=80.69%, Δ=-5.55pp — MARGINAL (paper-presentable).** 7.4× fewer params at -5.55pp cost. Queue was stale — result already existed from session ~Apr 13.
- **step891** (5060ti_cuda): CIFAR-10 MLP matched-params T2. Ref_linear=86.14%, MLP_h1(25K)=14.31%(-71.83pp!), MLP_h2(50K)=17.05%(-69.09pp!), Ref_SGNNET(35K)=**80.42%**. **+66.11pp SGNNET vs matched-params MLP — bottleneck confirmed.** Paper claim: "N_in=25088 information bottleneck collapses MLP h=1,2; SGNNET sparse graph achieves 80.42%."
- **step892** (5060ti_cpu): CIFAR-10 MLP h-sweep T1 (75ep, 50% data). h=4(40.5%), h=6(45.1%), h=8(45.5%), **h=16(81.3%)** ← CROSSOVER (+0.60pp vs SGNNET), h=32(85.0%), h=64(84.2%). Cliff confirmed at h=8→h=16 (35pp jump). T2 confirmed as step893.
- **step893** (5060ti_cuda): CIFAR-10 MLP crossover T2 (150ep, 100% data). h=8=45.8%(-34.6pp), h=12=67.1%(-13.3pp), **h=16=80.75%(+0.33pp)** ← T2 CROSSOVER CONFIRMED at 11.5× SGNNET. Paper claim validated.


