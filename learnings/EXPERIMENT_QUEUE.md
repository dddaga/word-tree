# SGNNET Experiment Queue (Live)

---

## FINAL EFFICIENCY CONFIG (2026-04-11) — step199

**95.52% @ 0.98M FLOPs — both ≤1% FLOPs AND ≤1% params criteria met simultaneously.**

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
| Params | 67K | **0.05%** |

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

## Current Architecture Defaults (N=4096, patched arch)

| Param | Value | Source |
|-------|-------|--------|
| AH alpha | 1.05 (step133 Tier-1: B=97.27% > A=97.20% > Ref=96.48%) | step133 confirmed |
| K_hh | 4 | step86 (+0.56pp, −18% FLOPs) |
| K_iter | 12 | step71/step89 confirmed |
| turing | 0.0 | step70 confirmed |
| reflect | 0.5 | step69 confirmed |
| n_groups | max(8, N//8) | step86/feedback |
| Project best | **97.86%** | step89-A (K_iter=12+K_hh=4), 150ep, best_ep=134 |

### Revised Budget (2026-04-10)

| Metric | VGG16 FC | Budget | Current best |
|--------|----------|--------|-------------|
| Params | 123.6M | **≤2.47M (2%)** | 529K (0.43%) ✅ |
| FLOPs | 123.6M | **≤6.18M (5%)** | 38.8M (31.4%) ❌ |

### Confirmed N=1024 Winners (valuable for efficiency track)

| Mechanism | Δ at N=1024 | Params | Step |
|-----------|-------------|--------|------|
| weighted_neg β=0.3 | +3.97pp (Tier-1) | 0 | step131-A |
| W_proj [D,D] | +5.48pp (Tier-1) | +4K | step131-B |
| RigL topology (every 5ep) | +6.82pp (Tier-0) | 0 | step124-B |
| α=1.10 | +1.53pp (Tier-0 N=4096) | 0 | step125-B |
| twopop_weight (het neurons) | +6.42pp (Tier-1) | ~+2K | step143-B |
| twopop_theta (het neurons) | +5.27pp (Tier-1) | 0 | step143-A |
| curriculum K_iter 2→4→8→12 | +2.85pp (Tier-1) | 0 | step142-C |

---

## Currently Running (5 slots, updated 2026-04-14)

| Machine:Device | Session | Step | Status | Note |
|---------|---------|------|--------|------|
| mini:mps | step306 | Activation retention N=1024 | RUNNING | 20ep × 7 configs, relaunched after bug fix |
| mini:cpu | step300 | Polarizer α sweep Tier-1 | RUNNING | 75ep × 6 configs, α={0.5,1.0,1.5,2.0,2.5}, ep~30 |
| studio:mps | step320 | F.normalize ablation | RUNNING | 20ep × 3 configs, paper-critical |
| studio:cpu | step321 | AH α sweep | RUNNING | 20ep × 7 configs, paper-critical |
| 5060ti:cuda | step301 | ΔW × AH interaction | RUNNING | 8 configs, ΔW proj + weak AH sweep |

**bench_hw DONE 2026-04-14:** RTX 5060Ti beats Mac Mini MPS on SGNNET training (23.7ms vs 70.5ms). SGNNET outperforms VGG FC at bs≤32 on CUDA. Results: `results/bench_hardware_5060ti_cuda.json`
**ga_v2 DONE 2026-04-13:** GA converged to (N=2048 D=16 K_hh=2 K_iter=5, alpha_ahebb=0.0, delta_proj, pa=1.5) at 96.31% (gen 4).

---

## Priority Queue (Restructured 2026-04-11 — Compound Winners + Polarizer Routing)

### P-PAPER — Paper-blocking ablations and synthesis experiments (2026-04-14)

| Step | Description | Scale | FLOPs | Script | Status |
|------|-------------|-------|-------|--------|--------|
| **step300** | **Polarizer α Tier-1 sweep** — α={0.5,1.0,1.5,2.0,2.5}. step217b showed α=1.5 optimal at Tier-1 (95.92%); GA v2 converged pa=1.5. Confirms monotonic trend. | N=2048 D=16 | ~2M | ✅ | **RUNNING (mini:mps)** |
| **step301** | **ΔW × AH interaction** — user hypothesis: weak AH (α<1) may serve as small-ΔW edge mask. Configs: A_proj (no AH), B_a{025,050,075,100} (ΔW proj + AH sweep), C_rot, C_rot_a050. | N=2048 D=16 | 0.98M | ✅ | **RUNNING (5060ti:cuda)** |
| **step306** | **Activation retention mechanism** — user-proposed: keep portion of Z_fwd on sender. Static/decay/norm-conserving/reinject variants at **low N first (N=1024)**. | N=1024 D=16 | ~0.5M | ✅ | **RUNNING (mini:cpu)** |
| **step320** | **F.normalize ablation (paper-critical)** — rerun step129 at N=2048 efficiency base. Tests "F.normalize is load-bearing" claim. A=no norm (clamp only), B=RMSNorm. | N=2048 D=16 | 0.98M | ✅ | **RUNNING (studio:mps)** |
| **step321** | **AH α sweep (paper-critical)** — α={0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50}. Validates "AH is load-bearing" claim. α=0 collapse reproduces step218. | N=2048 D=16 | 0.98M | ✅ | **RUNNING (studio:cpu)** |
| **step400** | **CIFAR-10 generalization (paper-critical)** — raw pixel input (N_IN=3072). Linear baseline + N={512,2048,4096} SGNNET. 30ep full data. Cross-dataset proof. | N=2048 D=16 | ~1M | ✅ | QUEUED |
| **step401** | **Baselines: pruned VGG + random-proj SGNNET (paper-critical)** — Lin_direct / MLP_2 / MLP_3 / SGNNET_RandProj. Contextualize efficiency claim. 20ep Tier-0. | N=2048 | 0.98M | ✅ | QUEUED |
| **step402** | **N-scaling law curve** — N={256,512,1024,2048,4096,8192}, D=16, K_hh=2, K_iter=5, AH=1.0. 75ep 50% data. Fit log-linear curve for paper. | various | various | ✅ | QUEUED |
| **step403** | **ΔW Tier-1 validation + ΔW+Polarizer combo** — Ref/A_proj/A_proj_pa15/A_proj_a025. Confirms step234 breakthrough at 75ep; tests ΔW+polarizer compound. | N=2048 D=16 | 0.98M | ✅ | QUEUED |

### P-CUDA — CUDA optimization for realizing SGNNET's 116x FLOPs advantage (2026-04-14)

Motivation: Despite 116x fewer FLOPs, SGNNET is only 1.21x faster than VGG FC on CUDA. GPU utilization = 2.6% (memory-bandwidth-bound due to random gather indices). Research (VLDB 2025, DGL blog, PyTorch CUDA graphs) identifies 4 stacked optimizations giving 3-8x combined speedup.

| Step | Description | Effort | Expected speedup | Script | Status |
|------|-------------|--------|------------------|--------|--------|
| **step500** | **torch.compile + CUDA graphs bench** — V0_eager / V1_compile / V2_cuda_graph / V3_fused. Also adds `src/sgnnet/model_resonant_cuda.py` dropin. CUDA-only. | 30min-2h | 1.4-2x | 🔨 agent writing | QUEUED |
| **step520** | **RCM index reordering** — Reverse Cuthill-McKee permutation of node IDs for L2 cache locality. Mathematically identical but reorders memory layout. Also retests reordering mid-training. | 4h | 1.3-2.4x | 🔨 agent writing | QUEUED |
| **step530** | **Triton fused kernel (gather+mul+sum)** — custom kernel eliminating [B,N,K_hh,D] intermediate tensor. Register-tiled for D=16. | 1-2d | 2-4x | DEFERRED | QUEUED |

### P-KITER — Reducing sequential K_iter passes (biggest FLOPs lever) (2026-04-14)

Motivation: K_iter=5 is the dominant latency factor (5 serial passes can't parallelize, theoretical max speedup = 116/5 = 23x).

**Prior findings (STALE, not on current efficiency config):**
- step163 warm-start (K=12 teacher → K=8 student weights): **+7.82pp** at N=1024 D=16 K_hh=8 (known winner)
- step142 C curriculum LOW→HIGH (2→4→8→12): **+2.85pp** at N=1024
- step127 distillation: **HURT** — distillation objective interferes with warm-start gain
- step216 curriculum at N=2048 D=16: **KILLED −58pp** ("curriculum is N=1024-only capacity crutch")
- step173 warm-start at N=2048 D=32 K_hh=4: HURT −0.23pp

**Hypothesis:** warm-start/annealing may still work at efficiency config (N=2048 D=16 **K_hh=2**) — never tested at this exact config. step173 failure was at different K_hh, different D.

| Step | Description | Scale | FLOPs | Script | Status |
|------|-------------|-------|-------|--------|--------|
| **step610** | **K_iter annealing at efficiency config** — 6 schedules tested: Ref/up/down_gentle/down_aggr/warm_switch_k5/warm_switch_k3. 75ep, 50% data. Retests step142/164 findings at current config. | N=2048 D=16 K_hh=2 | varies | 🔨 agent writing | QUEUED |
| **step611** | **K_iter warm transfer** — teacher K=12 for 40ep, student K=5/K=3 for 35ep (reproduces step163 mechanism at efficiency config). Configs: A_12to5/B_12to3/C_8to5/D_16to5. | N=2048 D=16 K_hh=2 | final varies | 🔨 agent writing | QUEUED |
| **step612** | **K_iter distillation at efficiency config** — retest step127 (KL distillation) on current config to confirm distillation still hurts. | N=2048 D=16 K_hh=2 | 0.60M (K=3) | 🔨 agent writing | QUEUED |

### P-NEW — Active experiments on step199 final config

| Step | Description | Scale | FLOPs | Script | Status |
|------|-------------|-------|-------|--------|--------|
| **step216** | Compound winners — ALL KILLED. twopop_weight -1.81pp, twopop_theta -2.45pp, curriculum -58pp. α=1.05 neutral. | N=2048 | 0.98M | ✅ | **DONE** |
| **step217** | Polarizer routing — **WINNER: full polarizer +1.27pp**, partial +0.89pp. Rotation neutral. | N=2048 | ~2M | ✅ | **DONE → Tier-1** |
| **step217b** | Polarizer Tier-1 — **WINNER: over-polarizer α=1.5 = 95.92% (+1.91pp)**. Full α=1.0 = 95.18% (+1.17pp). Monotonic alpha trend. | N=2048 | ~2M | ✅ | **DONE** |
| **step218** | Random projection ablation — Without AH: 91.5%→18.8% collapse. AH is prerequisite. Freezing bug, needs rerun. | N=2048 | 0.98M | ✅ | **DONE (partial)** |
| **step219** | Topology comparison — barely matters. Anti-pref +0.15pp, uniform random -0.84pp. | N=2048 | 0.98M | ✅ | **DONE** |
| **step220** | Heterogeneous K_hh — **ALL KILLED**. Uniform K_hh=2 beats all variants. Best -0.28pp, worst -2.40pp. | N=2048 | ~1M | ✅ | **DONE** |
| **step221** | Output-assigned topology — **ALL KILLED**. Input-assigned better. Best -1.30pp, worst -2.45pp. JSON bug. | N=2048 | ~1M | ✅ | **DONE** |
| **step222** | Paper baselines — MLP/random proj/linear all stuck ~9%. Trainer incompatibility bug. | N=2048 | varies | ✅ | **DONE (broken)** |
| **step223** | CIFAR-10 cross-dataset generalization | N=2048 | 0.98M | ✅ | QUEUED |
| **step224** | Scored dynamic topology — ALL NEUTRAL ±0.5pp. No benefit from per-epoch edge replacement. | N=2048 | 0.98M | ✅ | **DONE** |
| **step225** | Equilibrium propagation pilot — train without backprop | N=2048 | 0.98M | ✅ | QUEUED |
| **step226** | Skip connections — A/B KILLED (−10 to −14pp). C learned gate α=0 (explicitly rejected skips). | N=2048 | 0.98M | ✅ | **DONE** |
| **step232** | Scalar edge weights — ALL ~18% (broken forward pass, gradients didn't flow). Needs redesign. | N=2048 | 0.98M | ✅ | **DONE (broken)** |
| **step233** | θ-edge precision ablation — **ALL DEAD: best 75.2% (fp32), 62-75% across all dtypes**. Sinusoidal parameterization insufficient as AH replacement. Optimizer injection fix confirmed θ learns (std=0.115) but still fails. | N=2048 | 0.98M | ✅ | **DONE — KILLED** |
| **step234** | ΔW-vector polarizer — **BREAKTHROUGH: Config A (ΔW proj, NO AH) = 95.44% (+3.77pp)**. ΔW proj > W_pos proj > ΔW rot. Adding AH hurts (-1.2pp). | N=2048 | ~2M | ✅ | **DONE** |
| **step235** | ΔW-rot ± AH ± Augmentation (150ep full) — **WINNER: A_aug (ΔW-rot no-AH, aug) = 97.30%**. A_100 (no aug) = 96.97%. B configs (with AH) regress to 96.79-96.82%. Confirms: ΔW-rot > AH, NO synergy when combined. Augmentation helps. | N=2048 | 0.98M | ✅ | **DONE — ΔW-rot validated** |
| **step235** | ΔW rotation ± AH ± augmentation — **A_100 (ΔW proj, no AH, 100%/150ep) = 96.97% @ ep129. NEW EFFICIENCY RECORD at 0.98M FLOPs (+1.45pp over step199)**. Ref_100=95.29%, A_50=96.10%. A_aug running. | N=2048 | 0.98M | ✅ | RUNNING (Mac Studio CPU, A_aug) |
| **step238** | Gradient-safe θ parameterizations — cos_shifted=62%, phase_delta=75%, triangle=~65%. **ALL DEAD.** θ-edge direction confirmed killed. | N=2048 | 0.98M | ✅ | **DONE — KILLED** |
| **ga_v2** | GA autorun v2 — pool=50 (20 elite, 20 crossbreed, 10 mutation). gen 1, individual 9/50. Early: N=4096 D=16 K_hh=2 K_iter=5 = 94.98% (strong scout). | varies | varies | ✅ | RUNNING (Mac Studio MPS) |

### P0 — Critical path: structural experiments addressing architecture ceiling

| Step | Description | Scale | FLOPs | Script |
|------|-------------|-------|-------|--------|
| **step156** | **LayerNorm at N=4096 D=64** — validate step116-C winner (+2.24pp N=1024) at scale. 20ep scout (Ref/A/B/C). If +pp → Tier-1 | N=4096 | 38.8M | ✅ |
| **step140** | **N×K connectivity tradeoff** — N dominates; more K HURTS at D=16. KILLED hypothesis | Multi-N | 2.6-5.6M | DONE |
| **step141** | **Split-D (12act+4pos) + residual hypersphere** — fix input collapse + oversmoothing | N=1024 | ~3.1M | ✅ |
| **step116** | **RMSNorm / normalization ablation** — RMSNorm, LayerNorm, pre-norm, delayed norm, no-norm | N=1024 | ~3.1M | DONE — C=LayerNorm+2.24pp WINNER; A=RMSNorm−11.49pp, B=pre_route−6.50pp |
| **step152** | **Constraint discovery** — nuclear norm, bottleneck, dim gating, L1 sparsity, contrastive loss | N=1024 | ~3.1M | DONE — ALL KILLED. B=bottleneck near-neutral (−0.05pp). Nuc norm/contrastive −13-15pp |
| **step153** | **Progressive capacity reduction** — start large (N/D), prune by contribution (Matformer/GMP) | Multi-N/D | varies | DONE — ALL KILLED. A=prune_dims −0.62pp best. Nested dropout/matformer/prune_neurons −10-15pp |

### P0.5 — Efficiency track: compound stacking on step165-B

| Step | Description | Scale | FLOPs | Script |
|------|-------------|-------|-------|--------|
| **step166** ✗ | **ALL KILLED** — Ref=91.59%, E=90.09%(−1.50pp twopop), F=87.92%(−3.67pp curric), G=89.86%(−1.73pp spec_reg), H=86.47%(−5.12pp). warm+W_proj is local max. | N=1024 D=16 | ~3.1M | DONE |
| **step167** | **warm+W_proj Tier-2 validation** — full data 100%/150ep. Ref ep70=84.92%. B pending. | N=1024 D=16 | ~3.1M | ✅ RUNNING |
| **step168** ✓ | **warm+W_proj D=32 Tier-1** — Ref=81.96%, A=87.62%(+5.66pp warm-only), **B=93.20%(+11.24pp warm+W_proj)**. Near step144-C (93.50% W_proj+RigL). | N=1024 D=32 | ~6.1M | DONE |
| **step169** ✓ | **warm+W_proj D=32 full data 150ep** — **94.01% (best_ep=134)**. Phase exit NOT achieved (gap=0.99pp). D=32 ceiling confirmed ~94.0%. | N=1024 D=32 | ~6.1M | DONE |
| **step170** | **warm+W_proj N=2048 D=16 Tier-1** — N-scaling path (step72 N=2048 scratch=93.38%). Teacher training in progress (ep60=86.83%) | N=2048 D=16 | ~6.2M | ✅ RUNNING |
| **step171** | **warm+W_proj D=32 K_hh=4 Tier-1** — K_hh=4 was +0.56pp at N=4096; tests if sparser connectivity lifts 94.01% ceiling. Configs: Ref/A(α=1.0)/B(α=1.05)/C(K_iter=12). FLOPs ~3.1-4.6M | N=1024 D=32 K_hh=4 | ~3.1-4.6M | ✅ RUNNING |
| **step172** | **warm+W_proj D=48 K_hh=4 Tier-1** — D capacity test: D=32 ceiling ~94.0%, D=48 sits between D=32 and D=64 (accuracy track). FLOPs ~4.65M ≤ 6.18M ✓. Configs: Ref/B(α=1.0)/C(α=1.05) | N=1024 D=48 K_hh=4 | ~4.65M | ✅ RUNNING |
| **step175** | **warm+W_proj N=1024 D=32 K_hh=4 Tier-2** — reuses step171 teacher (K_hh=4 topology required). Tests if full data 150ep matches step169 (94.01% @6.1M) at HALF FLOPs (~3.1M). step171-B=93.35% Tier-1 → expected ~94.1-94.5%. | N=1024 D=32 K_hh=4 | ~3.1M | ✅ RUNNING |
| **step173-Ref** ✓ | **scratch N=2048 D=32 K_hh=4 Tier-1 (50%/75ep)**: 94.93%. Warm-start REVERSAL confirmed: B=94.70% < Ref. | N=2048 D=32 K_hh=4 | ~6.1M | DONE |
| **step174** ✓ | **PHASE EXIT CONFIRMED: 95.82% best_ep=118** — N=2048 D=16 K_hh=8 warm+W_proj full data 150ep. | N=2048 D=16 K_hh=8 | ~6.2M | DONE |
| **step175** ✓ | **N=1024 D=32 K_hh=4 warm+W_proj Tier-2**: 94.62% best_ep=138. No phase exit. N=1024 ceiling confirmed. | N=1024 D=32 K_hh=4 | ~3.1M | DONE |
| **step176-A** ✓ | **PHASE EXIT CONFIRMED: 96.18% best_ep=125** — N=2048 D=32 K_hh=4 scratch Tier-2 full data. BEST efficiency point. | N=2048 D=32 K_hh=4 | ~6.1M | DONE |
| **step176-B** | α=1.05 calibration, ep110=93.96%, ~40ep left | N=2048 D=32 K_hh=4 | ~6.1M | ✅ RUNNING |
| **step177** ✓ | **NEW MIN-FLOPs RECORD: 95.13% best_ep=133** — N=1024 D=48 K_hh=4 warm+W_proj Tier-2. Phase exit @ ~4.72M FLOPs (24% below prior records). | N=1024 D=48 K_hh=4 | ~4.72M | DONE |
| **step180** ✓ | **94.14% best_ep=64** — N=2048 D=20 scratch Tier-1 @~3.93M FLOPs. Tier-2 projected ~95.2-95.4%. | N=2048 D=20 K_hh=4 | ~3.93M | DONE |
| **step178** ✓ | **94.57% best_ep=73** — N=2048 D=24 scratch Tier-1 @~4.72M. Tier-2 → step184. | N=2048 D=24 K_hh=4 | ~4.72M | DONE |
| **step179** ✓ | **94.24% best_ep=57** — N=2048 D=28 scratch Tier-1 @~5.51M. Tier-2 → step183. | N=2048 D=28 K_hh=4 | ~5.51M | DONE |
| **step183** ✓ | **PHASE EXIT: 95.59% best_ep=122** — N=2048 D=28 Tier-2 @~5.51M FLOPs. Above 5.51M floor. | N=2048 D=28 K_hh=4 | ~5.51M | DONE |
| **step180** ✓ | **94.14% best_ep=64** — N=2048 D=20 scratch Tier-1 @~3.93M. Tier-2 → step181. | N=2048 D=20 K_hh=4 | ~3.93M | DONE |
| **step181** ✓ | **PHASE EXIT: 96.03% best_ep=147** — N=2048 D=20 scratch Tier-2 @ ~3.93M FLOPs. Min-FLOPs record (beat step177 4.72M). | N=2048 D=20 K_hh=4 | ~3.93M | DONE |
| **step182** ✓ | **93.96% best_ep=74** — N=2048 D=16 scratch Tier-1 @~3.15M. Tier-2 → step185. | N=2048 D=16 K_hh=4 | ~3.15M | DONE |
| **step183** | N=2048 D=28 scratch Tier-2 — projected ~95.3-95.5% @ ~5.51M. ep30=92.74%. | N=2048 D=28 K_hh=4 | ~5.51M | ✅ RUNNING |
| **step184** | N=2048 D=24 scratch Tier-2 — projected ~95.7% @ ~4.72M. ep30=93.86%. | N=2048 D=24 K_hh=4 | ~4.72M | ✅ RUNNING |
| **step185** ✓ | **PHASE EXIT: 95.87% best_ep=140** — N=2048 D=16 scratch Tier-2 @ ~3.15M FLOPs. NEW MIN-FLOPs RECORD (beats step181 @ 3.93M). | N=2048 D=16 K_hh=4 | ~3.15M | DONE |
| **step186** ✓ | **93.10% best_ep=75** — N=2048 D=12 Tier-1 @~2.36M. Tier-2 → step188 (borderline). | N=2048 D=12 K_hh=4 | ~2.36M | DONE |
| **step187** ✓ | **91.26% best_ep=72** — N=2048 D=8 Tier-1 @~1.57M. D=8 Tier-2 RULED OUT (max ~93.2%). | N=2048 D=8 K_hh=4 | ~1.57M | DONE |
| **step188** ✗ | **94.62% best_ep=138** — N=2048 D=12 Tier-2 @~2.36M. NO PHASE EXIT (0.38pp short). D-reduction floor confirmed at D=16 @ 3.15M. | N=2048 D=12 K_hh=4 | ~2.36M | DONE |
| **step189** ✓ | **92.25% best_ep=73** — N=2048 D=10 Tier-1 @~1.97M. Tier-2 ruled out (max ~94.2%). | N=2048 D=10 K_hh=4 | ~1.97M | DONE |
| **step190** ✓ | **93.86% best_ep=75** — N=2048 D=16 K_hh=2 Tier-1 @~1.57M. CRITICAL: beats D=8 K_hh=4 by +2.6pp at same FLOPs. D=16 dimensionality >> K_hh. Tier-2 → step193. | N=2048 D=16 K_hh=2 | ~1.57M | DONE |
| **step191** ✓ | **94.68% best_ep=71** — N=2048 D=16 K_hh=3 Tier-1 @~2.36M. Already above D=12 Tier-2 (94.62%) at same FLOPs. Tier-2 proj ~96%. → step192. | N=2048 D=16 K_hh=3 | ~2.36M | DONE |
| **step192** ✓ | **PHASE EXIT: 95.90% best_ep=105** — N=2048 D=16 K_hh=3 Tier-2 @~2.36M. NEW MIN-FLOPs record at time. | N=2048 D=16 K_hh=3 | ~2.36M | DONE |
| **step193** ✓ | **PHASE EXIT: 95.67% best_ep=135** — N=2048 D=16 K_hh=2 Tier-2 @~1.57M. NEW MIN-FLOPs record (50% below 3.15M). | N=2048 D=16 K_hh=2 | ~1.57M | DONE |
| **step194** ✓ | **94.88% best_ep=71** — N=2048 D=16 K_hh=2 K_iter=6 Tier-1 @~1.18M. Extraordinary: +1.02pp vs K_iter=8 Tier-1 despite 25% fewer FLOPs. Tier-2 → step195. | N=2048 D=16 K_hh=2 | ~1.18M | DONE |
| **step195** ✓ | **PHASE EXIT: 96.08% best_ep=106** — N=2048 D=16 K_hh=2 K_iter=6 Tier-2 @ 1.18M FLOPs. **FULL EFFICIENCY CRITERION MET: ≤1% FLOPs + ≥95% accuracy.** | N=2048 D=16 K_hh=2 | ~1.18M | DONE |
| **step196** ✗ | **92.74% best_ep=72** — N=2048 D=16 K_hh=2 K_iter=4 Tier-1 @~0.79M. KILLED: −2.14pp vs K_iter=6. Too few routing steps. | N=2048 D=16 K_hh=2 | ~0.79M | DONE |
| **step197** ✓ | **93.96% best_ep=72** — N=2048 D=16 K_hh=2 K_iter=5 Tier-1 @~0.98M. ≥93% threshold met. Tier-2 → step199. | N=2048 D=16 K_hh=2 | ~0.98M | DONE |
| **step198** ✗ | **88.92% best_ep=62** — N=1024 D=16 K_hh=2 K_iter=6 Tier-1 @~0.59M. KILLED: −6pp vs N=2048. N=2048 is floor for D=16 config. | N=1024 D=16 K_hh=2 | ~0.59M | DONE |
| **step199** ✓ | **SUB-1% EXIT: 95.52% best_ep=136** — N=2048 D=16 K_hh=2 K_iter=5 Tier-2 @ 0.98M FLOPs. NEW MINIMUM: 0.79% of VGG16 FC FLOPs + ≥95%. | N=2048 D=16 K_hh=2 | ~0.98M | DONE |
| **step200** ✗ | **90.52% best_ep=72** — N=2048 D=16 K_hh=1 K_iter=8 Tier-1 @ 0.79M. KILLED. K_hh=1 breaks connectivity; 8 steps insufficient. K_hh=2 is minimum viable. | N=2048 D=16 K_hh=1 | ~0.79M | DONE |
| **step201** | N=4096 D=16 K_hh=2 K_iter=6 Tier-1 @ 2.36M — N-scaling probe. N=1024→88.92%, N=2048→96.08%, N=4096→? | N=4096 D=16 K_hh=2 | ~2.36M | ✅ RUNNING |
| **step202** ✗ | **89.25% best_ep=74** — N=2048 D=16 K_hh=2 K_iter=3 T1 @ 0.59M. KILLED. K_iter floor: 3→89%, 4→92.74%, 5→95.52%. Min viable = K_iter=5. | N=2048 D=16 K_hh=2 | ~0.59M | DONE |
| **step203** ✓ | **96.08% best_ep=69** — N=4096 D=16 K_hh=2 K_iter=5 T1 @ 1.97M. HIGHEST T1 EVER at D=16. N-scaling: N=2048→93.96%, N=4096→96.08% (+2.12pp). T2 → step205. | N=4096 D=16 K_hh=2 | ~1.97M | DONE |
| **step201** ✓ | **95.64% best_ep=66** — N=4096 D=16 K_hh=2 K_iter=6 T1 @ 2.36M. T1 phase exit hit. Tier-2 → step204. | N=4096 D=16 K_hh=2 | ~2.36M | DONE |
| **step204** ✓ | **PHASE EXIT: 97.15% best_ep=71** — N=4096 D=16 K_hh=2 K_iter=6 T2 @ 2.36M. vs N=2048 T2: +1.07pp. vs D=64: −0.71pp. N-scaling law confirmed. | N=4096 D=16 K_hh=2 | ~2.36M | DONE |
| **step205** ✓ | **PHASE EXIT: 97.17% best_ep=118** — N=4096 D=16 K_hh=2 K_iter=5 T2 @ 1.97M. **NEW D=16 RECORD**. Only −0.69pp from D=64 record (97.86%). | N=4096 D=16 K_hh=2 | ~1.97M | DONE |
| **step206** | **95.11% best_ep=72** — N=8192 D=16 K_hh=2 K_iter=6 T1 @ 4.72M. T1 regression vs N=4096 (−0.53pp). N-scaling breaks at T1 for N=8192. T2 → step207. | N=8192 D=16 K_hh=2 | ~4.72M | DONE |
| **step207** ✗ | **96.20%@ep54** — N=8192 D=16 K_hh=2 K_iter=6 T2 @ 4.72M. N-SCALING BREAKS: −0.95pp vs N=4096 T2 (97.15%). Plateau 95.75-95.90% after ep54. D=16 bottleneck confirmed for K_iter=6 at N=8192. | N=8192 D=16 K_hh=2 | ~4.72M | DONE |
| **step208** ✓ | **95.77% best_ep=51** — N=8192 D=16 K_hh=2 K_iter=5 T1 @ 3.93M. K_iter=5 beats K_iter=6 at N=8192 (+0.66pp). −0.31pp vs N=4096 K5 T1. T2 → step209. | N=8192 D=16 K_hh=2 | ~3.93M | DONE |
| **step209** | N=8192 D=16 K_hh=2 K_iter=5 Tier-2 @ 3.93M — ep60=95.97% (strong, well above step207 ep60=95.64%). Proj ~97%+. | N=8192 D=16 K_hh=2 | ~3.93M | ✅ RUNNING |
| **step210** | **95.49%@ep65** — N=8192 D=16 K_hh=2 K_iter=4 T1 @ 3.15M. K_iter=4 VIABLE at N=8192 (KILLED at N=2048!). K_iter axis T1: K6=95.11% < K4=95.49% < K5=95.77%. Optimal K_iter decreases with N. | N=8192 D=16 K_hh=2 | ~3.15M | DONE |
| **step211** | **94.93%@ep72** — N=8192 D=16 K_hh=2 K_iter=3 T1 @ 2.36M. Just below 95% (−0.07pp). +5.68pp vs N=2048 K_iter=3. K_iter=3 floor at N=8192 borderline. → step212 (N=16384). | N=8192 D=16 K_hh=2 | ~2.36M | DONE |
| **step209** ✓ | **97.17%@ep115** — N=8192 D=16 K_hh=2 K_iter=5 T2 @ 3.93M. EXACTLY matches N=4096 K5 T2 (97.17%). **D=16 K_hh=2 ceiling at 97.17% CONFIRMED**. N-scaling holds flat at N=8192. | N=8192 D=16 K_hh=2 | ~3.93M | DONE |
| **step212** | N=16384 D=16 K_hh=2 K_iter=3 T1 @ 4.72M — N-scaling continuation. K_iter=3 at N=8192=94.93% (−0.07pp). At N=16384 may clear 95%. BATCH=32. | N=16384 D=16 K_hh=2 | ~4.72M | ✅ RUNNING |
| **step213** | N=8192 D=16 K_hh=3 K_iter=5 T1 @ 5.90M — K_hh axis at N=8192. K_hh=2 ceiling=97.17%. Does K_hh=3 break it? At N=2048 K_hh=3 was +0.82pp vs K_hh=2 T1. | N=8192 D=16 K_hh=3 | ~5.90M | ✅ RUNNING |

### P1 — High value: efficiency gains + input pipeline + confirmed stacking

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

### P1.5 — Regularization + remaining transformer techniques

| Step | Description | Scale | Script |
|------|-------------|-------|--------|
| **step158** | **DropMessage regularization** — stochastic edge dropping in K_iter routing, combat over-smoothing. Configs: drop=0.1/0.2/0.3 + late-only variant. 75ep Tier-0 | N=1024 | ✅ |
| **step121** | **Spectral norm / regularization on W_pos** — spectral norm, soft reg, re-orthogonalize | N=1024 | ✅ |
| **step120** | **High K_iter (16-24) + Z-bias + grad checkpoint** at N=4096 | N=4096 | ✅ |
| step72 | **N-scaling patched arch** — full curve N={256-8192} | Multi-N | ✅ |
| step108-C | **Polar routing Tier-1** — region-based lookup, −9% FLOPs | N=1024 | ✅ |

### P2 — Lower priority / gated

| Step | Description | Depends on | Script |
|------|-------------|------------|--------|
| step126 | µP initialization for N-scaling | step72 | ✅ |
| step104 | Compound wave+polar | step102+103 (both weak) | ✅ |
| step122 | DiffPool hierarchical readout | step118 showed readout changes catastrophic | ✅ |

### P3 — Deprioritized

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

| step72 ✓ | N-scaling: 256=57.76%@2.4M, 512=72.03%@4.8M, 1024=87.85%@9.7M, 2048=93.38%@19.4M, 4096=96.56%@38.8M | Monotone scaling confirmed. Efficiency bottleneck: N=512 (4.8M FLOPs) only 72% — too low for ≤5% FLOPs target at 95%+. D=16 track needed. |

### Killed / Null
| Step | Result | Finding |
|------|--------|---------|
| step152 ✗ | All null/negative. B=−0.05pp, C=−3.31pp, D=−3.34pp, A=−15.11pp, E=−13.02pp, F=−13.81pp | Constraint discovery KILLED. Regularization constraints don't help; bottleneck neutral. Network finds its own constraints. |
| step153 ✗ | All null/negative. A=−0.62pp, E=−12.13pp, D=−12.46pp, C=−9.97pp, B=−14.76pp | Progressive capacity reduction KILLED. Pruning during training destroys learned structure. |
| step129 ✗ | −50 to −71pp | Markov routing catastrophic. F.normalize + static AH load-bearing |
| step102 ✗ | −12 to −15pp | Phase polarizer catastrophic |
| step118 ✗ | −60 to −67pp | Attention readout catastrophic. Mean-pool load-bearing |
| step123 ✗ | −35 to −61pp | Stochastic depth catastrophic. Every K_iter step essential |
| step130 ✗ | −3 to −40pp | Beam broadcast hurts local routing |
| step119 ✗ | −2 to −7pp | Adaptive K_iter ACT hurts |
| step110 ~ | −0.15pp best | Muon null. AdamW optimal |
| step103 ~ | +0.51pp best | Wave interference marginal |
| step100 ~ | +1.68pp best | K_in sweep non-monotone, small deltas |
| step115 ~ | −0.51 to −0.89pp | Z-bias/redistrib negative at N=4096 (20ep) |
| step108 ~ | −4.95pp, −9% FLOPs | Polar routing: efficiency track candidate |
| step113 ✗ | +0.86pp best | Intermediate supervision marginal, not adopted |
| step140 ✗ | Ref=90.50% best | N×K tradeoff: N dominates, more K HURTS (−8 to −44pp). K_hh=4 optimal |
| step133 ✓ | B=97.27%(+0.79pp) | α=1.05 WINNER at N=4096 Tier-1. Both 1.05 and 1.10 beat α=1.0 |
| step141 ✗ | all <15.31% | Split-D + residual KILLED. All 5 configs near random (9-15%) |
| step150 ✗ | Ref=80.69%; A=−60.5pp, B=−42.7pp, C=−62.7pp, D=−52.4pp, E=−50.4pp | Positional max-pool KILLED. Mean-pool essential — unit-sphere dynamics require full population aggregation, not winner selection |
| step143 ✓ | B=88.03%(+6.42pp), A=86.88%(+5.27pp), D=82.90%(+1.29pp), C=81.76%(+0.15pp) | Heterogeneous neurons: twopop_weight WINNER (+6.42pp Tier-1), twopop_theta also wins (+5.27pp). freqband/perneuron null |
| step142 ✓ | C=84.66%(+2.85pp), A=83.39%(+1.58pp), B=82.24%(+0.43pp), D=75.16%(−6.65pp) | Curriculum K_iter: 2→4→8→12 ramp WINNER (+2.85pp Tier-1). Reverse schedule catastrophic |
| step127 ✗ | A=83.57%(−4.28pp K=6), B=79.26%(−8.59pp distill α=0.5), C=75.72%(−12.13pp distill α=0.7), D=82.50%(−5.35pp K=8) | K_iter distillation: distillation HURTS. K=6 no-distill is best at −4.28pp/34% FLOPs savings. Routing dynamics don't transfer via KD |
| step156 ✗ | Ref=96.20%; A=88.25%(−7.95pp LayerNorm), B/C pending | LayerNorm does NOT scale to N=4096. Scale failure: diverges immediately. L2 sphere norm is load-bearing at scale |
| step158 ✗ | A=47.62%(−32.6pp drop=0.1), B=28.31%(−52.4pp drop=0.2), C/D pending | DropMessage KILLED at N=1024. Confirms step96: message dropping incompatible with K_iter routing regardless of sparsity |
| step154 ✓ | A=73.83%(+9.75pp) | Safety valve REMOVED — AH handles diversity. lambda_safety=0.0 default |
| step155 ✓ | A=95.06%(N=4096), B=69.25%(N=1024), D=47.08%(N=256), E=44.92%(no-AH) | Diagnostics baseline DONE. AH increases eff_rank; no-AH collapses. N=4096 W_pos barely updating (∇wpos≈0.07 vs ∇θ≈1.0) |
| step144 ✓ | C=93.50%(+5.55pp), B=93.02%(+5.07pp), A=92.38%(+4.43pp), D=90.09%(+2.14pp) | Efficiency stack N=1024 D=32: C=W_proj+RigL WINNER (+5.55pp Tier-1). B=W_proj+α=1.10 strong. Compounding W_proj+RigL synergistic (unlike N=4096 where compounding hurts) |
| step121 ✓ | B=84.71%(+3.54pp), D=83.18%(+2.01pp), C=73.68%(−7.49pp), A=12.23%(KILLED) | W_pos regularization: B=soft_spec WINNER (+3.54pp Tier-1). D=weight_decay +2.01pp. A=hard spectral_norm catastrophic. C=reortho hurts |
| step163 ✓ | E=87.26%(+7.82pp), B=85.35%(+5.91pp), C=85.25%(+5.81pp), D=84.76%(+5.32pp), A=78.88%(−0.56pp) | Progressive KD: WARM-START IS THE MECHANISM not distillation. B(warm K=6 no-distil)≥C/D(warm+distil). More α=more hurt. E(K=8 warm)=BEST. A vs B: warm-start worth +6.47pp. K=12 over-smooths at N=1024 D=16 — fewer steps + good W_pos init wins |
| step149 ✗ | ALL configs −29 to −52pp | Scatter-sum Fourier encoding LOAD-BEARING. Input de-squashification KILLED. Any feature projection before scatter-sum breaks geometric prior |
| step162 ✗ | SGNNET=0.995 < FC_linear=1.003 flip ratio | Flip robustness hypothesis REJECTED. VGG pool5 features already flip-invariant. K_iter propagation does NOT add robustness |
| step165 ✓ | **B=90.96%(+10.32pp warm+W_proj WINNER)**, D=88.76%, A=85.55%, C=81.20% | EFFICIENCY BREAKTHROUGH: first N=1024 D=16 >90%. Warm+W_proj synergistic. RigL kills W_proj at D=16. ~3.1M FLOPs (2.5% of VGG FC) |
| step120 (partial) | Ref=96.51%, A=95.13%(−1.45pp K=16+zbias), B/C/D running | High K_iter hurts. K_iter=12 confirmed optimal at N=4096 |

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
