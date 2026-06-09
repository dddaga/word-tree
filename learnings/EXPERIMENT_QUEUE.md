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

## Currently Running (updated 2026-06-10 session 38)

**Slot policy:** 5060ti first, then Mini. Studio excluded.

| Machine:Device | Status | Note |
|---------|--------|------|
| mini:mps | RUNNING | step993 T1 — additive dynamic connectivity 75ep |
| mini:cpu | FREE | — |
| 5060ti:cuda | RUNNING | step982 T2 — CIFAR-10 aug paper claim (expandable_segments fix) |

**Completed this session (session 38):**
- **step990 T0 v2** (mini_mps): DONE. **Additive dynamic connectivity ALL ADVANCE TO T1.** Ref=61.27% (N=512). A_additive_10=61.94% (+0.67pp), **B_additive_05=62.78% (+1.51pp, best)**, C_learned_alpha=62.42% (+1.15pp). All 3 configs advance. First positive vision-debt result. T1 scripted: `scripts/train_step990_additive_t1.py`. **Vision debt: additive dynamic connectivity (brief §3.5) ADVANCES.**
- **step991 T0 v2** (mini_cpu): DONE. **Hebbian prune-grow KILLED-CONFIRMED.** Ref=85.40%. A_hebbian_random=47.21% (−38.19pp), B_hebbian_wpos=71.11% (−14.29pp), C_hebbian_fast=35.59% (−49.81pp). All massively below Ref. Hebbian prune-grow epoch-boundary rewiring catastrophically disrupts learned ΔW-proj routing. **Vision debt: Hebbian prune-grow (brief §9) RETIRED.**
- **step992 T0** (5060ti_cuda): DONE. **K-means init KILLED-CONFIRMED.** Ref=85.58%. A_kmeans=83.75% (−1.83pp), B_kmeans_classaware=82.47% (−3.11pp). Random init superior to both K-means variants. **Vision debt: K-means init (brief §6.1) RETIRED.**
- **step990 T0 v1** (mini_mps): INVALID. Ref=11.6% (bare SmallWorld, AH chain broken in v1 script). v2 relaunched (see RUNNING above).
- **step991 T0 v1** (mini_cpu): INVALID. Ref=14% (bare SmallWorld, AH chain broken in v1 script). v2 superseded above.
- **step986 T1** (5060ti_cuda or mini): DONE. N=16384 CIFAR-10 T1 — 82.85% @ep69 (75ep, 50%). Scaling curve continues. **Consider T2 (150ep) for paper scaling section.** N-scaling: 80.57→82.53→83.55% (N=2048→4096→8192 T2) + 82.85% T1 at N=16384 (T2 pending).
- **step982 T2** (5060ti_cuda): PARTIAL. Ref=80.79% @ep140 (471s). A_aug crashed silently (OOM, loading 2.3GB `store_cifar10_aug.h5`). **Relaunch needed after 5060ti free + extraction done.**

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
- **step989 T0**: Transformer FFN distillation — GPT-2-small layer-6 FFN, x_ffn→y_ffn MSE. Extraction OOM'd on 5060ti (GPU full, teammates). Launch `train_step989_ffn_distil_t0.py` when GPU frees. Advance: val_mse ≤ 2× Ref_mlp AND cos_sim ≥ 0.5. Decides Paper 2 spine.
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
- ts_step030-032: SGNNET-TS time series (financial forecasting, see learnings/ts/QUEUE.md). All PENDING.

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
