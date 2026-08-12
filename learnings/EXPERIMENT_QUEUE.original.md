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

## Currently Running (updated 2026-04-22 session 34)

**Slot policy:** Studio excluded (user opt-out). Active slots: mini_mps, mini_cpu, 5060ti_cuda only.

| Machine:Device | Status | Note |
|---------|--------|------|
| mini:mps | RUNNING | step956 T1 (refractory B+C, 75ep) — ep74 Ref done, B_β05 in progress |
| mini:cpu | FREE | step957 T0 DONE |
| studio:mps | FREE | |
| studio:cpu | FREE | |
| 5060ti:cuda | FREE | step958 T0 DONE, step966 T0 DONE |

**Completed this session (session 34, cont. 2):**
- **step958 T0** (5060ti_cuda): DONE. **LayerNorm routing — L2-normalize is load-bearing, alternatives kill.** Ref(L2)=93.94%. A_layernorm=91.36% (−2.57pp KILL, PR↑3.55). B_rmsnorm=90.80% (−3.13pp KILL, PR↑2.71). C_rmsnorm_learned=90.80% (−3.13pp KILL). D_scaledl2=93.94% (±0.00pp NEU). **Finding: LayerNorm raises PR 1.83→3.55 (activates more dims) but hurts −2.57pp. L2-norm's angular structure is geometrically essential for ΔW-proj routing — it's not a limitation, it's load-bearing. PR metric does not predict accuracy gains.** All norm variants KILLED. L2-normalize confirmed as canonical.
- **step957 T0** (mini_cpu): DONE. **Matryoshka D-nesting — MRL raises PR but doesn't help accuracy.** Ref=94.04% (PR=1.83). A_uniform(d=[2,4,8], α=1.0)=93.96% (−0.08pp NEU, PR→2.37). B_decay2(α=0.5)=94.01% (−0.03pp NEU, PR→2.32). C_d48only=93.89% (−0.15pp NEU, PR→2.18). **Finding: All MRL configs raise PR (more dims used) but accuracy flat or slightly down. Low PR (≈2.3) is a property of the representation, not a bug — forcing more dims with aux losses doesn't help. D=16 may genuinely encode in ~2 effective dims.** All NEUTRAL. No T1.
- **step966 T0** (5060ti_cuda): KILLED. **Backward reward scoring — training/eval path mismatch, all configs at ~13% (random chance).** Custom `forward_with_Z` loop trains different pathway than `model(x)` eval. Root cause: model trained but weights learned through custom BFS routing path that doesn't align with standard Resonant eval. Also moot: step967 confirmed zero pathway specialization (no class-selective routing exists to reward). **Direction CLOSED.**

**Completed this session (session 34):**
- **step980** (5060ti_cuda): DONE. **CIFAR-10 multi-seed T2 — authoritative paper variance.** 3 seeds × canonical DeltaW (N=2048, K_iter=5), 150ep T2, 100% data. SGNNET T2: **80.57% ± 0.12pp** (s0=80.43%, s1=80.56%, s42=80.73%). Gap vs Linear 86.24%: **−5.67pp**. Variance ±0.12pp (tighter than T1 ±0.31pp, consistent with step887 ±0.18pp pattern). Seed42 T2=80.73% vs step882 seed42=80.69% — **bit-consistent**. **Paper claim: "SGNNET achieves 80.57% ± 0.12pp on CIFAR-10 (gap −5.67pp vs Linear 86.24%)".**
- **step979** (5060ti_cuda): DONE. **CIFAR-10 multi-seed T1 — variance confirmed.** 5 seeds × canonical DeltaW (N=2048, K_iter=5), 75ep T1, 50% data. SGNNET T1: **77.43% ± 0.31pp** (seeds: s0=77.04%, s1=77.56%, s42=77.93%, s123=77.19%, s2024=77.45%). Gap vs Linear at T1: −8.81pp (T1 underfit; paper number is step882 T2 = −5.55pp). Seed variance ±0.31pp paper-usable. Note: seed=42 highest — consistent with step882 T2 seed42=80.69%.
- **step916** (5060ti_cuda): DONE. **CIFAR-10 K_iter sweep T0 — CATASTROPHIC collapse at K_iter>5.** Ref_k5=75.84% (T0 expected underfit vs step882 80.69% T2). A_k3=73.51% (−2.33pp KILL). B_k10=60.34% (−15.50pp CATASTROPHIC). C_k15=16.31% (−59.53pp NEAR-RANDOM). **Finding: K_iter=5 is the CIFAR-10 ceiling; more iterations cause collapse. Opposite of Imagenette (where K_iter is the primary capacity knob). HYPOTHESIS: CIFAR-10's higher-dimensional noisier features cause over-smoothing/instability at K_iter≥10. K_iter=5 CONFIRMED as default.**
- **step978** (5060ti_cuda): DONE. **Random features + linear baseline — Claim 3 CONFIRMED.** Lin_VGG=96.76% (full features→linear, ref). RandProj+meanpool (K_in=25, N=2048, D=16) = **13.35%** ≈ chance. RandProj+meanpool D=64 = 17.12%. RandProj_concat (N=256, D=16, concat 4096-dim) = **95.75%** (40K params). **Key finding:** mean-pool of random projections = chance (13-18%); concat without pooling recovers 95.75%; SGNNET routing + mean-pool = 97.30% — routing transforms worthless mean-pooled representation into discriminative one. Confirms Claim 3: iterative routing is load-bearing, not the projection topology.

**Completed this session (session 33):**
- **step972** (5060ti_cuda): DONE. **KD α-sweep T0** — Ref(α=0,T=1 pure KD)=**93.30%** > D_a1(pure hard CE)=93.17% (−0.13pp NEUTRAL) > all T=4 variants (−0.82 to −1.22pp). **Hard CE matches soft KD within noise.** Combined with step976, VGG soft-label distillation at T=1 contributes nothing measurable on Imagenette. Paper can simplify training story: "trained against VGG soft labels but hard CE equivalent."
- **step977** (5060ti_cuda): DONE. **Multi-seed KD vs CE T1 — NEUTRAL CONFIRMED.** 5 seeds × 2 configs, 75ep T1, BATCH=512. mean Δ(KD−CE)=**+0.06pp**, σ=**0.15pp** — both well inside |0.3pp|/0.5pp thresholds. Per-seed: s0=+0.05, s1=+0.33, s42=−0.05, s123=+0.08, s2024=−0.10. **Paper training story: soft-KD at T=1 and hard-CE are equivalent. Paper may describe training as "cross-entropy against VGG16 soft labels (T=1 ≈ hard CE)."**
- **step970** (5060ti_cuda): DONE. Layerwise isolation — best=0.8316. See log for per-config breakdown.
- **step976** (5060ti_cuda): DONE. **KD temperature sweep settles the "rewrite story" concern NEGATIVE.** step199 full stack (N=2048 D=16 K_hh=2 K_iter=5), 20ep T0, 50% data. T=1: **90.29%** (current baseline). T=2: 90.62% (+0.33pp, neutral). T=4: 88.48% (−1.81pp, hurt). T=8: 74.62% (−15.67pp, hurt). Monotonic decrease beyond T=2. **CONFIRMED: current T=1 VGG soft labels already near-optimal.** Root cause: VGG pretrained on ImageNet (Imagenette ⊂ ImageNet) → high-quality confident labels; tempering spreads mass onto wrong classes, adds noise. Classic KD gains require teacher-student capacity mismatch or hard dataset — neither applies here. **Paper numbers stand as-is; no rerun of efficiency story required.**
- **step971** (5060ti_cuda): KILLED. Quantization scout stuck at 10% — bare SmallWorld ceiling + 20ep OneCycleLR decays before model learns. Needs full Resonant+AH stack.
- **step973** (5060ti_cuda): CLOSED. Adiabatic+GTF scout on bare SmallWorld. Ref=46.73% (bare SmallWorld ceiling). Best adiabatic=27.54% (Adiab_K50), best GTF=24.71% — all configs <80% of Ref. Root cause: (1) bare SmallWorld ≠ Resonant+AH stack (ceiling ~47% not 95.52%), (2) W_pos is coupled routing-geometry tensor — sparse K=50 updates (0.15% coverage) destroy K-NN geometry. DIRECTION CLOSED for SGNNET. Adiabatic may still be viable for Lever 7 (FC block distillation).

**Completed this session (session 32):**
- **step960 T0** (mini_mps): DONE. ESC-50 low-K_in sweep. Structural failure confirmed.
- **step961 T0** (mini_cpu): DONE. ESC-50 N-sweep at K_in=1. N2048=0.335, −14pp vs Linear. No crossover.
- **step961b T0** (mini_mps): DONE. Large-N extension N∈{2048→16384}. Peak N4096=0.3475, then collapse. Non-monotonic.
- **step962 T0** (mini_mps): DONE. Dense seed projection. C_N256_dp=0.5650 (+14.75pp vs Linear). **CROSSOVER.** D_N512_dp=0.5725.
- **step963 T0** (mini_cpu): DONE. Subspace routing B-sweep. E_b96_s4=0.3125 (−10.5pp vs Linear). Trend: more blocks = better.
- **step964 T0** (mini_mps): DONE. Routing ablation. SGNNET_K0=0.6075 (+19pp). K5 routing HURTS by −4.75pp. **Routing degrades unstructured dense embeddings.** MLP_matched=0.5450 (routing beats MLP by +4.25pp from K0).
- **step965 T0** (mini_cpu): DONE. Finer subspace routing. D_b192_d32=0.4350 (+1.75pp vs Linear = EXCEEDS_LINEAR). Richer D_node + more blocks key.
- **diag_step967** (mini_mps): DONE. **Zero pathway specialization in canonical SGNNET.** mean_act_frac=1.0 (all nodes active), intra_jaccard=1.0, inter_jaccard=1.0, separation=0.0. **Finding: No class-specific routing — all nodes fire for all classes identically. Canonical SGNNET is a dense, non-selective routing network. Consistent with routing_gain<0 in Haki: routing is spatial smoothing, not selective pathway activation.**

**Scripted and READY to launch:**
- step966: Backward reward scoring T0 — `scripts/train_step966_backward_reward_t0.py` ✓ SCRIPTED
  Configs: Ref, A_d05, B_d07, C_d085, D_d07_noAH, E_d07_lam001.
  Launch on 5060ti_cuda (step967 is done).
  Dual-axis advance: accuracy ≥−0.5pp AND separation improvement >+0.05.

**Completed this session (session 34, cont.):**
- **step954 T1** (5060ti_cuda): DONE. **K_in=60 T1 Imagenette — BORDERLINE NEUTRAL.** Ref_k25=95.46%, D_kin40=95.75% (+0.28pp), E_kin60=**95.95% (+0.48pp)**. K_in=60 reaches efficiency-champion accuracy (step605=95.95%) but +0.48pp is below ≥+0.5pp T2 advance threshold. **VERDICT: NEUTRAL — K_in=60 confirms efficiency champion parity, but insufficient to advance to T2. Default stays K_in=25.**
- **step969 T0** (mini_cpu): KILLED. **Gradient threshold firing catastrophic on all configs.** GTF_sel_lo/hi/full_lo/hi: 14.70% (random), fire_pct=0.68% (essentially never fires). GTF_sel_adaptive: 10.19% (random), fire_pct=0.0% (never fires, trapped). Ref (no GTF): 38.7% (severely underfit for 20ep). Root cause: threshold calibrated from gradient norms prevents >99% of updates — network cannot learn. **ALL KILLED.**
- **step968 T0** (5060ti_cuda): KILLED. **Adiabatic FP4 long-horizon — FP4 fails completely.** Ref (no FP4, 500ep): 57.32% (underfit — bare SmallWorld below Resonant+AH ceiling). FP4 (2400ep): 11.18% stuck at random after 2400 epochs — NEVER learns. **Root cause: FP4 quantization destroys gradient signal entirely. FP4 direction CLOSED for SGNNET.**

**Completed this session (session 31):**
- **step952 T0** (mini_mps): Grouped input projection. C2_grouped16=+0.13pp, C_grouped64=+0.10pp NEUTRAL. E_multiout KILL (−1.76pp). No config clearly advances; grouped proj adds params for marginal gain.
- **step940 T0** (mini_cpu): Input-conditioned edge gate (tau=1,3,lr). All NEUTRAL (±0.03pp). No signal.
- **step965 T0** (mini_mps): Adiabatic update — launched.
- **step957 T0** (mini_cpu): Matryoshka D-nesting — launched.

**Next launches (2026-04-22 session 34, cont.):**
- 5060ti_cuda → **step966 T0** (backward reward scoring, 20ep) — **RUNNING** (launched 19:38)
- mini_mps → **step956 T1** (refractory B+C, 75ep) — **RUNNING** (re-launch after T9 unmount crash, launched 19:42)
- mini_cpu → **step957 T0** (Matryoshka D-nesting, 20ep) — **RUNNING** (launched 19:42)
- studio_mps or studio_cpu → step958 T0 (LayerNorm routing, 20ep) — pending Studio access

**step929 status (CIFAR-10 hflip-aug T1):** aug file extraction DONE (2.14GB, 100K train). step929 crashed silently (OOM loading 2 h5 files ×25088 features simultaneously). Fix needed: rework script to avoid loading clean+aug into RAM at same time. Deferred.

**step966 fix (2026-04-22):** Script had broken torchvision import + wrong N_IN=512. Fixed to use pre-extracted store.h5 features (N_IN=25088) + make_subset_loader added to dataset.py.

**Scripted and queued (smoke-tested):**
- step953: piecewise N_in→D seed T0 — `scripts/train_step953_piecewise_seed_t0.py` ⚠️ expect KILL (K_in/D ratio)
- step955: W_edge + ΔW-proj T0 — `scripts/train_step955_wedge_dwproj_t0.py`
- step956: refractory B+C T1 — `scripts/train_step956_refractory_t1.py`
- step934: p-RoPE dim split T0 — `scripts/train_step934_prope_dim_split_t0.py` (p∈{0.25,0.50,0.75})
- step935: MoE topology T0 — `scripts/train_step935_moe_topology_t0.py` ⚠️ ep1 collapse, needs 20ep to assess
- step936: per-iter W_pos T0 — `scripts/train_step936_per_iter_wpos_t0.py` (proj/scale/residual; clean init)
- step957: Matryoshka D-nesting T0 — `scripts/train_step957_matryoshka_d_t0.py` ✓ PR already rising ep1 (1.84→2.22)
- step958: LayerNorm routing T0 — `scripts/train_step958_layernorm_routing_t0.py` ⚠️ ep1 −10pp, needs 20ep
- step933: local/global K_iter T0 — `scripts/train_step933_local_global_kiter_t0.py` ⚠️ ep1 −65pp (random dw noise); consider fix: global rounds skip ΔW-proj, use uniform weights instead

**Physics of Deep Learning diagnostics (decided 2026-04-21):**
- step965: Adiabatic update T0 — `scripts/train_step965_adiabatic_update_t0.py` ✓ SCRIPTED
  Selective W_pos gradient masking: top-1%/5%/20% by |grad| + quantum-step + accum4.
  Fundamental: fp32 is discrete → adiabatic = one discrete jump at a time → let network digest.
  Advance: ≥+0.5pp OR faster convergence (ep98 < 14) → T1.
  Gradient staleness is the main risk. Run on mini_cpu (no CUDA needed).
- step964: Routing field irrotationality — `scripts/diag_step964_routing_curl.py` ✓ SCRIPTED
  Measures discrete curl of ΔW-proj routing field at each K_iter step.
  If curl_ratio < 0.05 → routing is conservative → K_iter has analytic fixed point → step965.
  Run on: trained checkpoint (step887 seed42) AND random init. Compare.
  Device: any (CPU fine, diagnostic only). No training.
- HAKI v2 — `src/sgnnet/haki.py` ✓ DONE (2026-04-21)
  Standalone module. New metrics: pr_seed, pr_gain, routing_entropy, effective_k,
  routing_invariance (CKA), convergence_deltas per step, node_utilization.
  Import: `from src.sgnnet.haki import HAKI`. Run standalone on any checkpoint.
  Paper 1 tooling contribution.

**Paper 1 expansion — audio gap CLOSED (2026-04-21 session 32):**
- step960/961/961b DONE: ΔW-proj structural failure on compressed Whisper embeddings confirmed.
- step962 DONE: Dense seed (SGNNET_K0=0.6075, +19pp vs Linear). Root cause: seeding, not routing.
- step963 DONE: Subspace routing best = 0.3125 (−10.5pp vs Linear). Structure matters.
- step964 DONE: Routing HURTS for unstructured embeddings (−4.75pp). K0 > K5. L2 norm is the key bias.
- step965 DONE: Subspace routing D_b192_d32=0.435, EXCEEDS Linear. Two mechanisms beat Linear.
- **step966 T0**: Backward reward scoring — `scripts/train_step966_backward_reward_t0.py` ✓ SCRIPTED
- **step967 diagnostic**: Path sparsity — `scripts/diag_step967_path_sparsity.py` ✓ RUNNING (mini_mps).
- step963 (vision): Scaling ceiling — N∈{2048,4096,8192}, D∈{16,32}, K_in∈{25,60}, T2 on Imagenette.
- ts_step030-032: SGNNET-TS time series (financial forecasting, see learnings/ts/QUEUE.md). All PENDING.

**Recently completed (session 30):**
- **step939** (studio_cpu T0 → synced): abs() in ΔW-proj is LOAD-BEARING. A_signed=−0.64pp, B_signed_clamp=−0.64pp. BOTH KILLED. abs() confirmed essential.
- **step950** (studio_mps T0 → synced): Per-edge W_edge[N,K_hh,D,D] rotation. ALL KILLED. Root cause: gather-sum without ΔW-proj causes Z collapse (loss=13.19 at ep1). 1M-param W_edge cannot rescue structural smoothing failure. Direction reformulated as step955 (W_edge ON TOP of ΔW-proj).
- **step951** (mini_cpu T0): K_in sweep with Haki diagnostics. Signal purity hypothesis DISPROVED. Coverage dominates: K_in=60 (+1.15pp), K_in=40 (+0.56pp) → both ADVANCE. K_in=5,10,15 KILLED. **New insight: routing_gain is ALWAYS NEGATIVE (routing = spatial smoothing, not amplifier). seed_Fisher is the dominant factor.**
- **step922** (5060ti_cuda T2): N=8192 CIFAR-10 multi-seed. mean=83.55% ±0.014pp. Gap vs Linear=−2.69pp (tight). N-scaling: 80.69→82.53→83.55%.

**Recently completed (session 29):**
- **step938** (mini_cpu T0): Refractory neurons. Ref=93.89%. A_β07_αr2=−1.94pp KILL. B_β05_αr1=−0.03pp NEUTRAL. C_β09_αr1=−0.28pp NEUTRAL. T1 queued for B+C.
- **step952 smoke test** (local 2ep): Init correct — Ref/A_shared/D_node all start at ~0.86, PR=2.29 stable. Script ready.

**Recently completed (session 24→25):**
- **step928** (studio_mps T0): ESC-50 architecture tuning. **A_D8 (D=8,K_in=25)=36.25% best** (−11.5pp vs Linear 47.75%). D=4 KILL (−16pp), K_in=50 KILL (−18pp). D=8 narrows gap 3pp vs D=16 but gap remains huge. Audio negative confirmed; vision scope stands.
- **step925** (studio_mps T0, re-run to get A_k15): **K_in=15 WINS at N=8192: A_k15=78.84% vs Ref_k25=77.97% (+0.87pp).** step914 N=8192 table CONFIRMED VALID (used K_in=15). CIFAR-10 K_in crossover: K_in=25 best at N=2048, NEUTRAL at N=4096, K_in=15 best at N=8192. Same pattern as Imagenette but shifted higher.

**Recently completed (session 22→23):**
- **step924/926** (studio_cpu T0): CIFAR-10 K_in=25 vs K_in=15 @ N=4096. **NEUTRAL: Ref_k25=77.82%, A_k15=77.75% (−0.07pp)**. Crossover confirmed between N=2048 and N=4096. step909's 82.53% VALID.
- **step926** (studio_cpu T0): ESC-50 audio robustness, canonical ΔW-proj. **AUDIO GAP CONFIRMED: N2048=32.0% vs Linear=47.75% (−15.75pp)**. ΔW-proj does NOT help audio. Vision scope only confirmed.
- **step923** (studio_cpu T1): CIFAR-10 K_in=15 vs K_in=25 @ N=2048. **REVERT: K_in=25=78.59%, K_in=15=77.71% (−0.88pp)**. K_in=25 is CIFAR-10 default at N=2048.
- **step921** (studio_mps 200ep): CIFAR-10 N=4096 extended. **PLATEAU: best=83.08% @ep186 (+0.55pp vs 150ep)**. 150ep is the correct budget.
- **cnn_step003 T1** (mini_mps): EfficientVGG T1 Imagenette. **Ref=73.17% (183M MACs, 419K), F_wide=74.42% (+1.25pp, 560M MACs, 825K), D_small_s=70.96% (−2.22pp, 57.5M MACs, 150K)**. B_tiny_k7 KILLED (63.44%). All 3 advance to T2.

**Recently completed (session 21→22):**
- **step921** (studio_mps): CIFAR-10 N=4096 200ep — **best=83.08% @ep186, Δ=+0.55pp vs step909 (150ep)**. PLATEAU confirmed: 150ep already converged; gap is architectural. 200ep gives marginal improvement but within seed variance.
- **step923** (studio_cpu): CIFAR-10 K_in=15 T1 @ N=2048 — **REVERT. Ref_k25=78.59%, A_k15=77.71%, delta=−0.88pp**. step920 T0 advance was run-order artifact. K_in=25 confirmed CIFAR-10 default at N=2048.

**Recently completed (session 17→18):**
- **step877** (5060ti_cpu): BFS+Hub T0 — KILLED. BFS diverged ep1 (loss=14.17). Direction CLOSED.
- **step878** (5060ti_cuda): K_hh=1 T1: Ref=0.9536, K_hh=1=0.9496 **Δ=-0.41pp STRONG**. Advancing to T2 (step885).
- **step881** (studio_mps): Multi-seed T2 non-canonical (67K params). seed0=96.20%, seed1=96.31%, seed42=96.82%. **Mean=96.44%, std=±0.26pp** — variance estimate for paper.
- **step883** (mini_cpu): ΔW-proj ablation T0. **D_rand_dir=-76.56pp (geometry ESSENTIAL)**. B_no_ref=-1.32pp, A_sign=-1.83pp, C_no_theta=-0.71pp. All components load-bearing at T0. T1 confirmation running (step886).
- **step886** (mini_cpu): ΔW-proj ablation T1. A_sign=-0.59pp **LOAD-BEARING**. B_no_ref=-0.51pp **LOAD-BEARING**. C_no_theta=+0.15pp **NEUTRAL** — theta NOT needed, flipped from T0 as predicted. **Paper: 2 load-bearing components; theta simplifies out.**
- **step884 (studio)** (studio_cpu): NON-CANONICAL. Canonical re-run on 5060ti_cpu.
- **step885** (5060ti_cuda): K_hh=1 T2. Ref=0.9664, A_khh1=0.9590. **Δ=-0.74pp MARGINAL** (between -0.5 and -1.0pp). Paper: "50% routing MACs at -0.74pp, mention with caveat." Advancing canonical multi-seed (step887).
- **step884_canonical** (5060ti_cpu): K_hh=1+K_in=15 compound T0 canonical. C_compound=-1.89pp → **KILL by T0 criterion**. NOTE: B_kin15 T0=-1.22pp is a T0 artifact (step632 T2=-0.33pp). Expected compound T2≈-1.07pp (viable). C_compound T1 queued for later. Components reported separately in paper.
- **step887** (5060ti_cuda): Canonical multi-seed T2 (34,976 params). seed0=96.23% @ep150, seed1=96.28% @ep71, seed42=96.64% @ep120. **Mean=96.38% ± 0.18pp. CONFIRMED for paper.** Replaces non-canonical step881. Paper claim: "96.38% ± 0.18pp (canonical 34,976 params)".
- **step888** (mini_mps): K_hh=1+K_in=15 compound T1. Ref=95.26%, C_compound=94.29% (-0.97pp). **VIABLE (within -1.5pp).** 43% total FLOPs reduction (1.31M vs 2.29M). Advancing to T2 (step889 on 5060ti_cuda). T0=-1.89pp was confirmed artifact.
- **step855** (mini_cpu): Sparse BFS T0. A_fixed_M16=-2.10pp, B_cascade=-2.23pp, C_quiet_zero=-2.20pp, D_readout_active=-2.05pp. **ALL KILLED. Direction CLOSED.**
- **step859** (5060ti_cpu): Soft routing T0. B_soft_anneal=+0.99pp T0 artifact (step861 T1=0.0pp confirmed). D_soft_dwproj=-60.51pp CATASTROPHIC. C_soft_ah=-0.10pp. **Soft routing CLOSED.** AH cancels soft gain; ΔW+soft collapses.
- **step862** (mini_cpu): CIFAR-10 cross-dataset T0 — **CONFIG BUG** (N=1024/D=8, 9K params). Ref_SGNNET=-29.39pp (invalid). Re-run with canonical config as step890 on 5060ti_cpu.
- **step889** (5060ti_cuda): K_hh=1+K_in=15 compound T2. Ref=96.64%, C_compound=95.21% (-1.43pp). **CONFIRMED — 0.57× FLOPs at -1.43pp. PAPER CLAIM VALID.**
- **step890** (5060ti_cpu): CIFAR-10 cross-dataset T0 canonical. SGNNET=75.77% vs Linear=86.47%, **Δ=-10.70pp → KILLED by T0 criterion** (just barely over -10pp threshold). T0 underfit confirmed by step882 T2 below.
- **step882** (mini_mps): CIFAR-10 cross-dataset T2 (150ep, 100% data, canonical 34,976 params). **Linear=86.24%, SGNNET=80.69%, Δ=-5.55pp — MARGINAL (paper-presentable).** 7.4× fewer params at -5.55pp cost. Queue was stale — result already existed from session ~Apr 13.
- **step891** (5060ti_cuda): CIFAR-10 MLP matched-params T2. Ref_linear=86.14%, MLP_h1(25K)=14.31%(-71.83pp!), MLP_h2(50K)=17.05%(-69.09pp!), Ref_SGNNET(35K)=**80.42%**. **+66.11pp SGNNET vs matched-params MLP — bottleneck confirmed.** Paper claim: "N_in=25088 information bottleneck collapses MLP h=1,2; SGNNET sparse graph achieves 80.42%."
- **step892** (5060ti_cpu): CIFAR-10 MLP h-sweep T1 (75ep, 50% data). h=4(40.5%), h=6(45.1%), h=8(45.5%), **h=16(81.3%)** ← CROSSOVER (+0.60pp vs SGNNET), h=32(85.0%), h=64(84.2%). Cliff confirmed at h=8→h=16 (35pp jump). T2 confirmed as step893.
- **step893** (5060ti_cuda): CIFAR-10 MLP crossover T2 (150ep, 100% data). h=8=45.8%(-34.6pp), h=12=67.1%(-13.3pp), **h=16=80.75%(+0.33pp)** ← T2 CROSSOVER CONFIRMED at 11.5× SGNNET. Paper claim validated.

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
| **step856** | **C_ho sparsity=0.98 multi-seed T1** (5 seeds, 75ep, 50% data). Ref mean=93.92%, D_very mean=93.97% (+0.05pp). Paired per-seed deltas: +0.28, +0.40, -0.38, +0.07, -0.11pp. **D_very is NEUTRAL — single-seed T1 +0.36pp was lucky seed42. Sparsity default stays at 0.90.** | **DONE — D_very NEUTRAL (not confirmed at multi-seed)** |
| **step855** | **Sparse BFS routing T0** (5 configs, 20ep, 50% data). ALL KILLED: A=-2.10pp, B=-2.23pp, C=-2.20pp, D=-2.05pp. Direction CLOSED. | **DONE — KILLED** |
| **step859** | **Soft distance-weighted routing T0** (5 configs, 20ep, 50% data). Tests softmax over static K_hh neighbor positions (W_pos distance). β annealing 0.5→3.0. Configs: Ref/A_soft_β1/B_soft_anneal/C_soft_ah/D_soft_dwproj — tests both AH and ΔW-proj on soft routing. HYPOTHESIS: W_pos gradient through distance term improves topology. Script: `scripts/train_step859_soft_routing.py`. | **DONE — KILLED (B_soft_anneal +0.99pp was T0 artifact; step861 T1=0.0pp confirmed; soft routing CLOSED)** |

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
| **step411** | AG News config sweep. Linear=91.18%, MLP_64=92.53%. All SGNNET configs trail Linear: Ref_orig=90.34%(-0.84pp), B_low_kin=90.34%(-0.84pp), C_low_kiter=90.34%(-0.84pp), A_small=89.16%(-2.02pp), D_combined=88.67%(-2.51pp). Text gap CONFIRMED — SGNNET loses monotonically on text. Paper scope = vision only. | **DONE — text gap CONFIRMED** |
| **step892** | CIFAR-10 MLP h-sweep crossover (T1: 75ep, 50%, seed=42). h=4(2.9×)=40.5%, h=6(4.3×)=45.1%, h=8(5.7×)=45.5%, **h=16(11.5×)=81.3% ← CROSSOVER**, h=32(23.0×)=85.0%, h=64(45.9×)=84.2%. Cliff at h=8→h=16 (35pp jump). T2 confirmed by step893. | **DONE** |
| **step893** | CIFAR-10 MLP crossover T2 (150ep, 100% data, seed=42). h=8=45.8%(-34.6pp), h=12=67.1%(-13.3pp), **h=16=80.75%(+0.33pp) ← T2 CROSSOVER**. Paper claim CONFIRMED: SGNNET needs 11.5× fewer params than min-viable MLP on CIFAR-10. | **DONE** |
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
| **step523** | **Alternating W_pos / edge training cycle.** Ref=77.68%, A_low=74.85% (−2.83pp), B_high=75.03% (−2.65pp). Both fail. Edge rewiring destroys weights; recovery incomplete. **KILLED — dynamic connectivity direction CONFIRMED dead (7/7 negative including step511-514).** | **DONE** |
| **step521** | Deep supervision on K_iter routing. Ref=94.85%, A_ds_3to5=88.36% (-6.5pp), B_ds_2to5=88.87% (-6.0pp). **KILLED — routing disrupted by intermediate supervision.** C/D still running but expected dead. | **DONE — KILLED** |
| **step522** | Muon optimizer vs AdamW at N=2048 K=5 ΔW proj. Ref_adamw=95.64% (ep_to_95=38), A_muon=95.39% (ep_to_95=42). Muon −0.25pp vs AdamW at T1. No convergence benefit. | **DONE — KILLED. AdamW remains default.** |
| **direct K=4 wall-clock bench** | Measure SGNNET K=4 inference latency directly (currently projected 0.224ms based on 20% reduction). Add to bench_step811 variant list. | TODO |
| **step524** | **Edge-SHIFT probes** (post-step523 follow-up). 6 configs at N=1024 T0: Ref / P1 step523+Adam-reset / P2 alt-schedule+0%cap / S1 edge-β scalar / S2 W_pos-passive-rebind / S4 cyclic-shift-null-control. Tests H1-H2-H5 of step523 failure + 2 continuous-parameterization alternatives + 1 null control. ~4h one slot. Design in LEARNINGS_design_2026_04_15.md. | TODO (script) |
| **step526** | **INT8 QAT + inference impact + grad-accum sweep.** Part A: fp32 train → quant eval for 3 modes (saturate/modular/crt) × {W only, W+Z}. Tests hypothesis: L2-norm at D=16 bounds components to ±0.25 × scale=100 → int8 range ±25, so wrap never fires. Part B: QAT from scratch (fake-quant+STE forward, fp32 master), grad_accum ∈ {1,4,16,64}. Telemetry: wrap_rate per forward. Script ready: `train_step526_int8_qat.py`. | TODO (launch) |
| **Param count reconciliation** | Bench reports SGNNET=34,976; training reports 67,744. Diff ≈ 32K. Find missing component (likely K_in=25 seed projection). Resolve before paper. | TODO |
| **bench_step832** | PyG scatter vs fancy-index. V_ref_c (compiled) = **0.151ms** (6.55× over eager). Scatter_add compiled = 0.217ms — fancy-index wins. CUDA Graph approach (0.896ms) slower. Best: `torch.compile(max-autotune)` on fancy-index. | **DONE** |
| **bench_step830** | K=4 vs K=5 wall-clock. K=4 compiled=0.140ms vs K=5=0.154ms → **1.10× compiled, 1.18× eager**. Paper claim updated: "10–18% latency reduction" (was "20% projected"). | **DONE** |

## Meditation P0 — 2026-04-17 (step860–863)

From meditation 001 (step267→step859). Scripts written and smoke-tested.

| Step | Description | Script | Slot | Status |
|------|-------------|--------|------|--------|
| **step860** | K=1 KD student @ N=4096 T0. Ref_k5=89.81%, A_k1_scratch=34.37% (−55.44pp!), B_k1_kd=21.10% (−68.71pp!). Routing degenerates completely at large N+K=1. **KILLED.** | `scripts/train_step860_k1_n4096_t0.py` | 5060ti_cuda | **DONE — KILLED** |
| **step861** | Soft routing T1 — NEGATIVE. B_soft_anneal=0.9381=Ref (0.0pp). T0 +1.22pp was early-epoch artifact. D_soft_dwproj=0.3659 (catastrophic). Soft routing KILLED. | `scripts/train_step861_soft_routing_t1.py` | 5060ti_cuda | **DONE — KILLED** |
| **step862** | CIFAR-10 cross-dataset T0. Paper requirement (≥2 datasets). VGG pool5 512-dim features. Configs: Linear, MLP_37, MLP_256, Ref_SGNNET. N=512, D=8. | `scripts/train_step862_cifar10_crossdataset.py` | 5060ti_cpu | QUEUED (fringe slot) |
| **step863** | D probe T0: Ref_D16=91.75%, A_D8=89.89% (−1.86pp marginal), B_D8_K10=75.29% (KILLED), C_D12=91.46% (−0.28pp ADVANCES). D=12 advances to T1 (step871). D floor = D=12. | `scripts/train_step863_d8_efficiency.py` | 5060ti_cuda | **DONE** |
| **step864** | D floor+beam ablation T0: D=6 KILLED (−6.93pp), D=4 KILLED (−12.76pp), D=4+K=10 catastrophic. Rbeam_M8=Ref (insensitive), Rbeam_M32=+0.05pp (insensitive). BFS_M32=+0.15pp ADVANCES. BFS_M16/M64 neutral. D=12 confirmed floor. | `scripts/train_step864_d_floor_beam_m.py` | 5060ti_cpu | **DONE** |

**Parking lot (wait for above results):**
- step865: K_hh=1 probe — low priority (5060ti_cpu candidate)
- step867: K=1 + soft routing — DEAD (step861 killed soft routing)

| **step866** | HNSW eval-mode T0. 4 configs: Ref_dw(AH)=91.72%, A_soft_static=91.92%(+0.20pp), B_beam_topk=91.87%(+0.15pp), C_beam_wider=91.90%(+0.18pp), D_beam_train=91.80%(+0.08pp). All advance vs AH but ALL below ΔW baseline (~93.96%). Beam/soft routing is not competitive with ΔW-proj. | `scripts/train_step866_hnsw_eval_mode.py` | mini_mps | **DONE — KILLED vs ΔW** |
| **step868** | Z-memory retention T0. Ref_dw=93.96%, A_g03=94.04%(+0.08pp neutral), B_g05=94.04%(+0.08pp neutral), **C_g08=94.29% (+0.33pp ADVANCES)**, D_g09=93.63%(-0.33pp KILLED). gamma=0.8 is the sweet spot. | `scripts/train_step868_zmem_retention_t0.py` | mini_cpu | **DONE — C_g08 ADVANCES** |
| **step869** | Hub aggregation T0: Ref=93.91%, A_hub005=94.24% (+0.33pp ADVANCES), B_hub03=93.27% (-0.64pp), C_hub10=88.00% (catastrophic), D_hub_beam=92.61% (-1.30pp). alpha=0.05 only sweet spot — larger values homogenize representations. | `scripts/train_step869_hub_aggregation_t0.py` | 5060ti_cuda | **DONE** |
| **step872** | Hub aggregation T1 (75ep, 50% data). Ref_dw=95.36%, **A_hub005=95.54% (+0.18pp VIABLE)**. Light global context consistently helps but below the +0.2pp STRONG threshold. Advances to T2. | `scripts/train_step872_hub_t1.py` | 5060ti_cuda | **DONE — VIABLE** |
| **step870** | D_very+ΔW compound T2 (150ep, 100% data). Ref_dw=96.92%, C_compound=96.43% (**-0.48pp KILL**). T1 synergy (+0.18pp) did NOT hold at T2. ΔW alone is the base. D_very+ΔW compound direction closed. | `scripts/train_step870_dvery_dw_compound_t2.py` | studio_mps | **DONE — KILLED** |
| **step871** | D=12+ΔW T1. Ref_dw=95.34%, A_d12=93.91%(-1.43pp), **B_d12_dw=93.63%(-1.71pp KILLED)**, C_d12_comp=93.86%(-1.48pp). ΔW-proj WORSENS D=12 (projection direction has insufficient info). D=16 is hard floor for ΔW-proj family. | `scripts/train_step871_d12_dw_t1.py` | studio_cpu | **DONE — KILLED** |
| **step873** | BFS M=32 T1. Ref=95.36%, A_bfs_m32=70.96% (**-24.41pp CATASTROPHIC KILL**). Completely collapses at T1 — dynamic top-M selection creates unstable routing gradients. BFS direction CLOSED. | `scripts/train_step873_bfs_m32_t1.py` | 5060ti_cuda | **DONE — KILLED** |
| **step874** | Z-memory T1. Ref=95.29%, A_g08=95.29% (**0.00pp NEUTRAL**). T0 +0.33pp was early-epoch artifact. Z-mem does NOT advance to T2. Direction CLOSED. | `scripts/train_step874_zmem_t1.py` | mini_mps | **DONE — NEUTRAL** |
| **step875** | Hub T2. Ref=96.59%, A_hub005=96.61% (**+0.03pp NEUTRAL**, non-canonical params 67744). T1 +0.18pp did not hold at full training. Hub direction CLOSED. NOTE: MPS path double-counts W_pos (67744 vs 34976); inflated baseline explains ref>95.52%. | `scripts/train_step875_hub_t2.py` | studio_mps | **DONE — NEUTRAL** |
| **step865** | K_hh=1 efficiency probe T0. Ref=93.73%, A_khh1=93.25% (**-0.48pp VIABLE**). 50% routing MACs reduction advances to T1 (step878). | `scripts/train_step865_khh1_t0.py` | mini_cpu | **DONE — VIABLE** |
| **step876** | Hub+Z-mem compound T0. Ref=93.96%, A_hub005=94.11%(+0.15pp), B_zmem_g08=94.42%(+0.46pp), C_compound=93.76%(**-0.20pp CANCEL**). Hub+Z-mem interact negatively — DO NOT compound. Run each independently. | `scripts/train_step876_hub_zmem_compound_t0.py` | studio_cpu | **DONE — CANCEL** |
| **step877** | BFS+Hub compound T0 (20ep, 50% data). 4 configs: Ref, bfs32, hub005, compound. Tests same-path cancellation. | `scripts/train_step877_bfs_hub_compound_t0.py` | 5060ti_cpu | **DONE — KILLED (BFS diverged ep1)** |
| **step878** | K_hh=1 T1 (75ep, 50% data). Ref=95.36%, A_khh1=95.36% (−0.41pp STRONG). Advancing to T2 (step885). | `scripts/train_step878_khh1_t1.py` | 5060ti_cuda | **DONE** |
| **step889** | K_hh=1+K_in=15 compound T2 (150ep, 100% data). Ref=96.64%, C_compound=95.21% (-1.43pp @ 0.57× FLOPs). **CONFIRMED — PAPER CLAIM VALID.** | `scripts/train_step889_compound_t2.py` | 5060ti_cuda | **DONE** |
| **step862** | CIFAR-10 cross-dataset T0 (20ep, 50% data). Paper requirement ≥2 datasets. VGG16 pool5 features N_in=25088. Configs: Linear, MLP_37, MLP_256, Ref_SGNNET (N=1024, D=8). | `scripts/train_step862_cifar10_crossdataset.py` | mini_cpu | **DONE — superseded by step882 T2 (Linear=86.24%, SGNNET=80.69%, -5.55pp MARGINAL)** |
| **step879** | Z-mem gamma fine-scan T0. γ=0.80 confirmed peak (+0.54pp T0). Curve: g070=+0.25, g075=+0.31, **g080=+0.54**, g085=+0.33. Peak at 0.80 but T1 is 0.00pp — Z-mem direction CLOSED. | `scripts/train_step879_zmem_gamma_scan_t0.py` | mini_cpu | **DONE — g080 T0 peak but T1 NEUTRAL → Z-mem CLOSED** |
| **step880** | K_hh=3 probe T0. Ref=93.91%, A_khh3=93.78% (**-0.13pp WORSE**). More local edges hurt. K_hh=2 confirmed Pareto optimal. K_hh curve: K_hh=1(-0.48pp) < K_hh=2(ref) > K_hh=3(-0.13pp). | `scripts/train_step880_khh3_t0.py` | studio_cpu | **DONE — K_hh=2 confirmed optimal** |
| **step881** | ΔW-proj T2 multi-seed seeds=[0,1,42] for paper error bars. Non-canonical (studio, 67744 params). Use for variance estimation; paper numbers need CUDA re-run. | `scripts/train_step881_multiseed_t2.py` | studio_mps | **DONE — non-canonical Mean=96.44% ±0.26pp; step887 canonical (34,976 params) replaces for paper** |
| **step882** | CIFAR-10 cross-dataset T2 (150ep, 100% data). Linear=86.24%, SGNNET=80.69% (Δ=−5.55pp). 7.4× fewer params at -5.55pp cost. MARGINAL — paper-presentable as honest cross-dataset. | `scripts/train_step882_cifar10_cross_t2.py` | mini_mps | **DONE — MARGINAL** |
| **step883** | ΔW-proj component ablation T0 (20ep, 50% data). 5 configs: Ref_dw/A_sign(clamp0)/B_no_ref(α_r=0)/C_no_theta(θ=0)/D_rand_dir. Paper ablation table: which components are essential? | `scripts/train_step883_dwproj_ablation_t0.py` | mini_cpu | **DONE — D_rand_dir=-76.56pp (geometry ESSENTIAL); T1 confirmed in step886** |
| **step884** | K_hh=1 + K_in=15 compound efficiency probe T0 (20ep, 50% data). Configs: Ref_dw/A_khh1/B_kin15/C_compound. Tests ~45% total FLOPs reduction. SUCCESS: C_compound within -1.5pp → ultra-efficient config for paper. | `scripts/train_step884_khh1_kin15_compound_t0.py` | studio_cpu | **DONE — non-canonical; step884_canonical (5060ti_cpu) C_compound=-1.89pp T0 artifact; step889 T2 CONFIRMED -1.43pp** |

---

## CNN Distillation Track — `scripts/cnn_distiller/` (2026-04-18)

**Objective:** Distill VGG16's full conv+pool stack (not just FC) into EfficientVGG (~420K params, 2.9% of VGG16 conv 14.7M).
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

**Tier plan:** T0 on mini → Pareto winners (≥85% at best MACs/acc) → T1 (75ep, mini_mps) → T2 (150ep, mini_mps/cpu — CNN is mini-only).

---

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

---

## Dynamic Routing Revival — 2026-04-18

**Context:** User confirmed stubborn interest in parameter-efficient dynamic routing. Analysis in `learnings/concepts/dynamic_routing_analysis.md`.

| Step | Description | Script | Slot | Status |
|------|-------------|--------|------|--------|
| **step894** | Z-dot + AH softmax routing T0. Ref_dw=94.04%, A_zdot_only=14.09%(−79.95pp COLLAPSE — D=16 noise FM3 confirmed), B_ah_softmax=75.75%(−18.29pp), C_zdot_ah_t10=75.80%(−18.24pp), D_zdot_ah_t03=75.95%(−18.09pp). **ALL KILLED. AH-softmax routing loses −18pp regardless of Z-dot.** Softmax-weight selection is fundamentally weaker than ΔW-proj magnitude gating at K_hh=2. | `scripts/train_step894_zdot_soft_routing_t0.py` | 5060ti_cuda | **DONE — KILLED** |
| **step895** | Parameter-free routing T0. Ref_dw=93.99%, A_norm_weighted=15.90%(−78.09pp), B_shared_query=17.78%(−76.20pp), C_factored_attn=15.52%(−78.47pp), D_learned_temp=66.57%(−27.41pp). **ALL KILLED.** A/B/C collapse to near-random — softmax over K_hh neighbors is catastrophically unstable without structural anchor. D partial recovery (learned temp) still −27pp. Parameter-free routing dead end. | `scripts/train_step895_paramfree_routing_t0.py` | studio_cpu | **DONE — KILLED** |
| **step896** | Biased softmax routing T0. Ref_dw=93.96%, A_bias_khh=74.34%(−19.62pp), B_bias_step=74.34%(−19.62pp), C_bias_full=74.34%(−19.62pp), D_temp_only=74.34%(−19.62pp), E_compound=92.00%(−1.96pp), F_bias_per_node=74.34%(−19.62pp). **ALL KILLED.** A-D-F all converge to identical 74.34% — bias terms converge to zero (initialized 0, no gradient to break symmetry at that fixed point). E_compound partially recovers via ΔW-proj but adds net −1.96pp. LeakyReLU did not help. **Softmax routing direction DEFINITIVELY CLOSED across step894/895/896.** | `scripts/train_step896_biased_soft_routing_t0.py` | 5060ti_cuda | **DONE — KILLED** |
| **step897** | Dense cosine gate T0. v1 (W_pos key): Ref=77.45%, A_global_b=16.28%(−61.17pp), D_topk_eval=9.91%(−67.54pp). **KILLED — FM5+FM8.** v1 hijacks W_pos geometry (FM5); v2 fixed W_pos via separate W_key but still collapsed by FM8 (O(N) gradient dominance on shared recurrent Z). C/B configs not run — killed after A_global_b collapse. | `scripts/train_step897_dyn_cosine_gate_t0.py` | 5060ti_cuda | **DONE — KILLED** |
| **step898** | K_hh cosine gate + ΔW-proj additive T0. Division-bug fix required (LeakyReLU negative + clamp explosion). Ref=93.96%, A_khh_gate=93.58%(−0.38pp), B_per_neuron_b=93.66%(−0.31pp), **C_gate_only=15.36%(−78.60pp KILL)**, D_per_edge_b=93.91%(−0.05pp). C_gate_only KILL proves gate has no structural prior without ΔW-proj — fully parasitic. A/B/D neutral with 32–37K extra params = FM5 co-adaptation (same signal path). **Dynamic routing direction CLOSED (all mechanisms 2026-04-19).** | `scripts/train_step898_khh_cosine_gate_t0.py` | local | **DONE — KILLED** |
| **step526** | INT8 QAT. Part A: fp32 train → quant eval (saturate/modular/crt × W-only/W+Z). Hypothesis: L2-norm sphere bounds W_pos to ±25/128 → int8 wrap never fires. Part B: QAT from scratch with grad_accum ∈ {1,4,16,64} — tests whether gradient accumulation crosses bin boundaries to recover fp32 accuracy. Script ready. | `scripts/train_step526_int8_qat.py` | 5060ti_cuda | **DONE** (step526/527 results exist from mini_mps) |
| **step901** | Ephemeral Teleportation T1. Pre-killed: step899 closed the direction (all K_ep=1 configs 18–56%). | `scripts/train_step901_ephemeral_t1.py` | — | **KILLED (pre-kill, step899 → direction closed)** |
| **step902** | D-scaling T1. Pre-killed: step900 closed the direction (C_d24_dw=−0.35pp, D_d32_dw=−0.15pp vs D=16+dw). | `scripts/train_step902_d_scaling_t1.py` | — | **KILLED (pre-kill, step900 → direction closed)** |
| **step903** | **Epoch-Topology Dynamic Slot T0** FINAL: Ref=94.17%, A_std_lr=93.32%(−0.84pp), B_half_lr=91.97%(−2.19pp KILL), C_tenth_lr=85.99%(−8.18pp KILL), D_warmup10=93.17%(−0.99pp). None advance. Epoch topology rebuilding consistently hurts — KNN disrupts learned geometry. **Direction CLOSED.** | `scripts/train_step903_epoch_topology_t0.py` | 5060ti_cpu | **DONE — KILLED** |
| **step904** | **Node-Level Input-Conditioned Gating T0** FINAL: Ref_dw=94.06%, A_tau1=92.51%(−1.55pp), B_tau3=88.38%(−5.68pp KILL), C_tau_learnt=92.51%(−1.55pp), D_se_bn=93.55%(−0.51pp), E_wpos_geo=93.78%(−0.28pp), F_per_iter=92.46%(−1.61pp). Gates ARE dynamic (H=0.4-0.7) but still hurt. NOT gate-death; information-bottleneck from gating out useful nodes. None advance (≥+0.5pp threshold not met). **Direction CLOSED (bottleneck, not collapse).** | `scripts/train_step904_node_gating_t0.py` | 5060ti_cuda | **DONE — KILLED** |
| **step905** | **CIFAR-100 Rigor Baseline T1** (75ep, 50% data). Linear=64.78%, MLP_256=64.24%, MLP_512=64.73%, SGNNET_can=35.40%(−29.38pp), SGNNET_N4096=40.57%(−24.21pp). **CATASTROPHIC GAP.** Root cause: with N_out=100, C_ho readout gives ~18 nodes/class (vs ~184 for Imagenette 10-class) — capacity-limited. Paper scope CONFIRMED to Imagenette only. | `scripts/train_step905_cifar100_rigor_t1.py` | studio_cpu | **DONE — KILLED. Paper scope = Imagenette only.** |
| **step909** | **CIFAR-10 N-scaling** — close cross-dataset gap. B_N4096(150ep) DONE: 82.53% (+1.84pp vs N=2048, gap closes −5.55→−3.71pp vs Linear 86.15%). Ref_linear=86.15% (consistent). A_N2048_T3(200ep) still running on 5060ti_cpu. H1: +1.84pp (just under ≥2pp) — capacity scaling helps but doesn't close gap fully. | `scripts/train_step909_cifar10_n_scaling.py` | 5060ti_cpu RUNNING | **PARTIAL** |
| **step914** | **CIFAR-10 N=8192 T2** (150ep, 100% data). **FINAL: C_N8192=83.58% best @ep130, gap=−2.66pp vs Linear. N-scaling: N2048=80.69% → N4096=82.53% → N8192=83.58% (closing ~1.4pp per 2×N, diminishing returns). Multi-seed validation → step922.** | `scripts/train_step914_cifar10_n8192_t2.py` | 5060ti_cuda | **DONE — N-scaling curve confirmed** |
| **step915** | **CIFAR-10 ΔW-proj Ablation T1** (75ep, 50% data). **FINAL: Ref_dw=78.37%, A_no_dw=15.96% (−62.41pp COLLAPSE). ΔW-proj is load-bearing on CIFAR-10. Without it: plain sum averaging → trivial diffusion fixed point → near-random (10% chance = 10 classes). Cross-dataset claim STRONGLY CONFIRMED.** | `scripts/train_step915_cifar10_dwproj_ablation_t1.py` | studio_mps | **DONE — ΔW-proj ESSENTIAL cross-dataset** |
| **step916** | **CIFAR-10 K_iter Sweep T0** (20ep, 50% data). **FINAL: Ref_k5=75.72%, A_k3=−2.12pp(KILL), B_k10=−15.54pp(KILL), C_k15=−59.97pp(KILL/collapse). K_iter=5 is optimal — over-smoothing at K_iter>5 (GNN over-smoothing analogue). Paper finding: K_iter has sweet-spot at 5; direction CLOSED.** | `scripts/train_step916_cifar10_kiter_sweep_t0.py` | studio_cpu | **DONE — K_iter=5 universal sweet-spot** |
| **step917** | **CIFAR-10 α_reflect Sweep T0** (20ep, 50% data). **FINAL: Ref_a05=75.77%, A_a00=−0.49pp(NEUTRAL), B_a025=−0.51pp(KILL), C_a075=+0.52pp(ADVANCE→T1), D_a10=−2.01pp(KILL). α=0.75 beats canonical α=0.5 on CIFAR-10 by +0.52pp. α=0.5 not cross-dataset optimal.** → step918 T1. | `scripts/train_step917_cifar10_alpha_reflect_t0.py` | studio_mps | **DONE — C_a075 ADVANCE** |
| **step918** | **CIFAR-10 α_reflect T1** (75ep, 50% data). **FINAL: Ref_a05=78.69%, C_a075=77.88% (−0.81pp). T0 artifact confirmed. α=0.5 canonical cross-dataset. Direction CLOSED.** | `scripts/train_step918_cifar10_alpha_reflect_t1.py` | studio_mps | **DONE — T0 artifact, α=0.5 CANONICAL** |
| **step919** | **Imagenette K_iter Sweep T0** (20ep, 50% data). **FINAL: Ref_k5=94.09%, A_k3=−0.66pp, B_k8=−4.13pp, C_k10=−9.35pp, D_k15=−74.78pp(COLLAPSE). Universal over-smoothing CONFIRMED. Same collapse profile as CIFAR-10 (step916). PAPER CLAIM: K_iter=5 sweet-spot on both datasets; over-smoothing is dataset-independent.** | `scripts/train_step919_imagenette_kiter_sweep_t0.py` | studio_cpu | **DONE — UNIVERSAL OVER-SMOOTHING CONFIRMED** |
| **step920** | **CIFAR-10 K_in Sweep T0** (20ep, 50% data). **FINAL: A_k15=76.03%, Ref_k25=75.32% — run-order artifact (A_k15 ran before Ref, delta_vs_ref=null). B_k50=74.91%(−0.41pp), C_k100=74.81%(−0.51pp). Apparent advance was noise — step923 T1 REVERTED (K_in=15=−0.88pp).** | `scripts/train_step920_cifar10_kin_sweep_t0.py` | studio_cpu | **DONE — T0 artifact, K_in=25 stays CIFAR-10 default** |
| **step923** | **CIFAR-10 K_in=15 T1** (75ep, 50% data, N=2048, seed=42). **FINAL: Ref_k25=78.59% @ep60, A_k15=77.71% @ep74, Δ=−0.88pp. REVERT. step920 T0 advance was run-order artifact. K_in=25 is CIFAR-10 default at N=2048. K_in cost: CIFAR-10 −0.88pp vs Imagenette −0.33pp — harder task penalizes sparse seeding more.** → step924: probe K_in at N=4096 CIFAR-10. | `scripts/train_step923_cifar10_kin15_t1.py` | studio_cpu | **DONE — REVERT, K_in=25 stays** |
| **step918** | **CIFAR-10 N=4096 underfitting probe** (200ep, 100% data). Controls for underfitting at N=4096. Tests whether the −3.71pp gap vs Linear is epoch-limited or architectural. | `scripts/train_step918_cifar10_n4096_200ep.py` | studio_cpu RUNNING | **RUNNING** |
| **step924** | **CIFAR-10 K_in=25 vs K_in=15 @ N=4096 T0** (20ep, 50% data). **NEUTRAL: Ref_k25=77.82%, A_k15=77.75% (−0.07pp)**. Crossover between N=2048 (K_in=25 wins) and N=8192 (K_in=15 wins) confirmed. step909 82.53% VALID. | `scripts/train_step924_cifar10_kin_n4096_t0.py` | studio_cpu | **DONE — NEUTRAL** |
| **step925** | **CIFAR-10 K_in=25 vs K_in=15 @ N=8192 T0** (20ep, 50% data). **K_in=15 WINS: A_k15=78.84% vs Ref_k25=77.97% (+0.87pp)**. step914's 83.58% (K_in=15) CONFIRMED VALID. K_in crossover between N=2048 and N=8192, mirrors Imagenette exactly. | `scripts/train_step925_cifar10_kin_n8192_t0.py` | studio_mps | **DONE — K_in=15 wins at N=8192** |
| **step926** | **ESC-50 Audio Robustness T0** (20ep, 50% data). Canonical ΔW-proj. **AUDIO GAP CONFIRMED: Linear=47.75%, N512=18.5%, N1024=27.5%, N2048=32.0% (−15.75pp)**. ΔW-proj does NOT close audio gap. Whisper features lack spatial geometry. Paper scope = vision only. | `scripts/train_step926_esc50_robustness_t0.py` | studio_cpu | **DONE — AUDIO GAP CONFIRMED** |
| **step927** | **CIFAR-100 N=8192 T0** (20ep, 50% data). Extends step905 N-scaling (N2048=35.40%, N4096=40.57%). Tests if N=8192 closes the −24pp gap. Advance: gap ≤20pp → T1 (step930). Stagnate → architectural failure confirmed. | `scripts/train_step927_cifar100_n8192_t0.py` | studio_mps | **RUNNING** |
| **step929** | **CIFAR-10 hflip-aug T1** (75ep, 50% aug data). Follows step924 T0 winners. Tests if aug consistently closes gap at calibration level. Advance rule: ≥+0.5pp → T2 (step930). PREREQUISITE: store_cifar10_aug.h5 + step924 T0 result. Script: train_step925_cifar10_hflip_aug_t1.py (naming quirk). Target slot: 5060ti_cuda. | `scripts/train_step925_cifar10_hflip_aug_t1.py` | 5060ti_cuda | **QUEUED — after step924 T0 AND store_cifar10_aug.h5** |
| **step930** | **CIFAR-10 hflip-aug T2** (150ep, 100% aug data, best T1 config). Paper claim: N=8192+aug vs gap −2.66pp. Target: gap ≤−1.5pp. OR: CIFAR-100 N=8192 T1 if step927 shows gap ≤20pp. Script TBD. | TBD | 5060ti_cuda | **QUEUED — after step929 T1 OR step927 T0** |
| **step928** | **ESC-50 Architecture Tuning T0** (20ep, 50% data). D×K_in sweep: {D=4,8,16}×{K_in=25,50}. **FINAL: A_D8=36.25%(−11.5pp best), D=4 KILL(−16pp), K_in=50 KILL(−18pp). Audio gap confirmed not closeable by arch. Paper scope = vision only.** | `scripts/train_step928_esc50_arch_tuning_t0.py` | studio_mps | **DONE — AUDIO GAP CONFIRMED, all fail >10pp** |
| **step906** | **Top-K Activation Sparsity T0** (20ep, 50% data). Ref_dw=94.11%, A_top75=79.06%(−15.06pp), B_top50=78.19%(−15.92pp), C_top25=80.46%(−13.66pp), D_top10=78.80%(−15.31pp). Hard top-K ALL CATASTROPHIC. E_soft_tau1=92.31%(−1.81pp), F_soft_tau3=88.25%(−5.86pp). Soft marginal/significant hurt. Gate entropy=0.62 (near-max, non-discriminative). Same failure mode as step904 — gating Z before K_iter destroys collective computation. **ALL KILLED. FGSEGNet-style gating before MP definitively closes.** | `scripts/train_step906_topk_activation_t0.py` | studio_mps | **DONE — KILLED** |
| **step907** | **Readout-Gate Input-Conditioned T0** (20ep, 50% data). Ref_dw=94.04%, A_ro_tau1=+0.36pp(NEUTRAL), B_ro_tau3=−1.04pp, C_ro_tau_lrn=**+0.54pp ADVANCE**, D_ro_topk25=−3.49pp(KILL), **E_ro_geo=+1.12pp ADVANCE**, F_ro_norm=−0.74pp. KEY FINDING: readout-level gating WORKS (+1.12pp) while pre-K_iter gating fails (FM10). Geometric gate (W_pos cosine vs Z_mean) is strongest. → T1 (step910). | `scripts/train_step907_readout_gate_t0.py` | studio_mps | **DONE — E_ro_geo +1.12pp, C_ro_tau_lrn +0.54pp → step910 T1** |
| **step910** | **Readout Gate T1** (75ep, 50% data). Ref_dw=95.41%, A_ro_tau1=+0.66pp, C_ro_tau_lrn=+0.69pp, E_ro_geo=+0.61pp. **ALL THREE ADVANCE.** All within 0.08pp of each other. → T2 (step911). | `scripts/train_step910_readout_gate_t1.py` | studio_mps | **DONE — all three advance to T2** |
| **step911** | **Readout Gate T2** (150ep, 100% data). **FINAL: Ref=96.74%, A_ro_tau1=+0.15pp, C_ro_tau_lrn=+0.25pp, E_ro_geo=+0.18pp. ALL below ≥+0.5pp threshold. T1 artifact confirmed — gain compresses from +0.61–0.69pp (T1) to +0.15–0.25pp (T2). Paper claim NOT supported.** Direction CLOSED. | `scripts/train_step911_readout_gate_t2.py` | studio_mps | **DONE — T1 artifact, direction CLOSED** |
| **step912** | **CIFAR-10 Readout Gate T0** (20ep, 50% data). Ref=76.03%, A_ro_tau1=+0.40pp(NEUTRAL), **E_ro_geo=−0.60pp(KILL)**. Readout gate does NOT generalize to CIFAR-10. Gain is Imagenette-specific. | `scripts/train_step912_cifar10_readout_gate_t0.py` | studio_cpu | **DONE — gate Imagenette-specific, no CIFAR-10 generalization** |
| **step899** | Ephemeral teleportation T0. Ref_dw=93.96%, A_local1_dw=92.10%(−1.86pp). All K_ep=1 configs: 18–56% (−38 to −76pp KILL). Random per-step reconnection introduces gradient noise identical to FM9 — complete training collapse. **ALL KILLED. Topology diameter direction CLOSED.** step901 (T1) not triggered. | `scripts/train_step899_ephemeral_teleport_t0.py` | 5060ti_cuda | **DONE — KILLED** |

### AH Era Revival — Mechanisms Not Tested on ΔW-proj (2026-04-20)

These gained significant ground in the AH era (D=64/N=1024/K_iter=12) but were never ported to the current efficiency config (N=2048/D=16/K_iter=5/ΔW-proj). Full analysis: `learnings/concepts/ah_era_untested_on_dw.md`.
**Caveat:** AH arch had D=64 — expect gains to be smaller at D=16. Treat as directional signal only.

| Step | Description | AH Era Gain | Script | Slot | Status |
|------|-------------|------------|--------|------|--------|
| **step937** | **Z-bias per K_iter T0** (20ep, 50% data). Port of step106 to ΔW-proj. `Z_t += emb[t]` at each of K_iter=5 steps. 80 extra params. Configs: Ref / A_zbias_init0 / B_zbias_initrand. | +7.42pp (D=64/N=1024) | Ref=94.19%, A_z0=94.19% (+0.00pp), B_zrnd=94.32% (+0.13pp) | mini_cpu | **DONE — NEUTRAL** (best +0.13pp < 0.50pp threshold; regime mismatch: AH era D=64/K_iter=12 vs current D=16/K_iter=5) |
| **step938** | **Refractory neurons T0** (20ep, 50% data). Port of step16E. `Z_t = Z_t − α_r × max(0, β × |Z_{t-1}|)`. Configs: Ref / A_β07_αr2 / B_β05_αr1 / C_β09_αr1 (sweep β×α_r). Ref=93.89%. A_β07_αr2=−1.94pp KILL. B_β05_αr1=−0.03pp NEUTRAL. C_β09_αr1=−0.28pp NEUTRAL. B+C advance to T1. | +4.31pp (D=16/N=512 wave arch) | Ref=93.89% | mini_cpu | **DONE — B/C NEUTRAL, advance to T1** |
| **step939** | **Signed ΔW routing T0** (20ep, 50% data). Remove abs() from proj_coeff. Ref=94.09%. A_signed=−0.64pp KILL. B_signed_clamp=−0.64pp KILL. **abs() is load-bearing — anti-aligned negative signals cancel gradient coherence.** Direction CLOSED. | +3.97pp (D=64, ConcatReLU — different regime) | Ref=94.09%, A/B=−0.64pp | studio_cpu | **DONE — KILLED** |
| **step940** | **Input-conditioned edge reweighting T0** (20ep, 50% data). Port of step36. `w_ij = sigmoid(x_in_i · x_in_j / tau)` for each conn_hh edge; scale Z_agg by w_ij. Uses fixed input features x (not Z state — safe from gate-death). Configs: Ref / A_tau1 / B_tau3 / C_tau_learnable. | +2.34pp (D=64 without AH) | TBD | any | **QUEUED** |
| **step941** | **Sparse beam active set T0** (20ep, 50% data). Port of step25. At each K_iter step, only top-K active nodes by ‖Z‖ route; others hold Z_prev. Efficiency mechanism. Configs: Ref / A_top_half / B_top_quarter / C_top_eighth. step25 found top-1/8 best. | +2.88pp + 53× speedup (D=16/N=512) | TBD | any | **QUEUED** |
| **step942** | **Max-pool readout T0** (20ep, 50% data). Replace mean-pool over N nodes with max-pool. Minimal change. Note: step907/911 readout gate T1 artifact suggests readout changes are noisy; max-pool is different (permutation-invariant aggregation, not gating). Configs: Ref / A_maxpool / B_topk_mean (top-32 mean). | Not run in AH era | TBD | any | **QUEUED** |
| **step950** | **Per-edge channel rotation T0** (20ep, 50% data). NO ΔW-proj. Ref=gather-sum. Ref=19.52%, A_shared=+0.03pp, B_peredge(1M params)=+0.03pp, C_small_D8=−5.99pp. **STRUCTURAL FAILURE: gather-sum without ΔW-proj collapses Z (all nodes average to correlated direction → extreme wrong predictions at init). W_edge identity init can't escape. 1M params = zero gain.** Q answered: W_edge CANNOT substitute for ΔW-proj. Reformulation: test W_edge ON TOP of ΔW-proj (step955). | New mechanism | `scripts/train_step950_per_edge_rotation_t0.py` | studio_mps | **DONE — KILLED. ΔW-proj is load-bearing.** |
| **step951** | **K_in Signal Purity Sweep T0 + Observational Haki** (20ep, 50% data). Tests whether K_in affects signal quality in seed scalar Z[i,0]=sum(x[conn_in[i]]). Configs: Ref(K_in=25) / A_kin5 / B_kin10 / C_kin15 / D_kin40 / E_kin60. All 34,976 params. Haki metrics: seed_Fisher, final_Fisher, routing_gain, PR, dead_frac logged at ep 1/5/10/20. | — | `scripts/train_step951_kin_purity_t0.py` | mini_cpu | **RUNNING** (launched 2026-04-20) |
| **step952** | **Grouped Input Projection T0 + Haki** (20ep, 50% data). Replace parameter-free scatter sum with learned per-group weighted projection. Configs: Ref/A_shared/C2_grouped16/C_grouped64/B_grouped256/D_node/E_multiout. Raw linear weights init=1 (= sum baseline at ep0). Smoke test passed: Ref=0.8601, A_shared/D_node=0.8614, PR=2.29 stable. | — | `scripts/train_step952_grouped_input_proj_t0.py` | mini_cpu (after step951) | **QUEUED** |
| **step900** | D-scaling T0. Script bug: SGNNET_DRef (no ΔW-proj) collapses — Ref_d16=19.57%, A_d24=18.96%, B_d32=16.64% (all meaningless). ΔW variants vs D=16+dw ref (93.96%): C_d24_dw=93.61%(−0.35pp), D_d32_dw=93.81%(−0.15pp). ΔW-proj saturates D-dimensional geometry at D=16; larger D adds no benefit. **D-scaling direction CLOSED. step902 not triggered.** | `scripts/train_step900_d_scaling_t0.py` | studio_cpu | **DONE — KILLED** |
