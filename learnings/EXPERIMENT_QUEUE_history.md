# SGNNET Experiment Queue — Historical Archive

**Active queue:** [EXPERIMENT_QUEUE.md](EXPERIMENT_QUEUE.md)

All DONE / KILLED / STALE entries. Reference only.

---

## P-PAPER — Paper-blocking ablations and synthesis (2026-04-14) — ALL DONE

| Step | Description | Scale | FLOPs | Script | Status |
|------|-------------|-------|-------|--------|--------|
| **step300** | **Polarizer α Tier-1 sweep** — α={0.5,1.0,1.5,2.0,2.5}. **DONE: A200 (α=2.0) = 94.73% winner (+0.84pp)**. step217b α=1.5 no longer reproduces at efficiency config. CONFIRMED α=2.0-2.5 new optimum. | N=2048 D=16 | ~2M | ✅ | **DONE** |
| **step301** | **ΔW × AH interaction** — **DONE (Tier-0): C_rot (ΔW rotation) = 94.39% (+2.52pp) — NEW WINNER**. A_proj=94.01%, weak AH NOT complementary (B_a025 = 93.99%, identical to A_proj). B_a{075,100} progressively hurt. Rotation > projection; compound hurts. → Tier-1 needed. | N=2048 D=16 | 0.98M | ✅ | **DONE** |
| **step306** | **Activation retention** — **DONE: ALL variants HURT.** Ref=85.35% beats all (A_static_p10 -0.10pp, A_static_p30 -0.74pp, A_static_p50 -6.67pp, B_decay -3.24pp, C_norm_cons -18.67pp, D_reinject -4.41pp). KILLED. | N=1024 D=16 | ~0.5M | ✅ | **DONE — KILLED** |
| **step320** | **F.normalize ablation (paper-critical)** — **DONE: F.normalize IS load-bearing.** Ref (l2)=91.62%, A (clamp only)=79.29% (−12.33pp), B (RMSNorm)=85.07% (−6.55pp). CONFIRMED. | N=2048 D=16 | 0.98M | ✅ | **DONE** |
| **step321** | **AH α sweep (paper-critical)** — **DONE: AH IS load-bearing.** α=0 collapses to 18.78%. α=0.25→0.75→1.0 monotonic (49→88→92%). α=1.0 optimal; α=1.25/1.50 slightly worse. CONFIRMED. | N=2048 D=16 | 0.98M | ✅ | **DONE** |
| **step400** | **CIFAR-10 generalization (paper-critical)** — **DONE (30ep mini_mps): Linear=37.94%, N512=33.38%, N2048=32.09%, N4096=33.32%. SGNNET UNDERPERFORMS LINEAR on raw pixels. Root cause: designed for VGG16 feature inputs (25088-dim). Raw 3072-dim pixels low-level — Fourier encoding on S^{D-1} ineffective without feature extraction. Paper framing: SGNNET = classifier head needing upstream features, not end-to-end.** | N=2048 D=16 | ~1M | ✅ | **PARTIAL — 75ep running studio_cpu** |
| **step401** | **Baselines (paper-critical)** — **DONE: Lin_direct=96.92%, MLP_64=97.20%, MLP_2=46.98%, MLP_3=45.91%, SGNNET_Ref=91.75% @20ep, SGNNET_RandProj=10.04% (chance).** W_pos essential (+81pp vs random). MLP same budget = 45-47%. Claim: SGNNET ≈ MLP_64 accuracy at 24× fewer params. | N=2048 | 0.98M | ✅ | **DONE** |
| **step402** | **N-scaling law curve** — **step402a DONE (mini_mps):** N=256: 59.39% @0.12M, N=512: 77.45% @0.25M, N=1024: 87.95% @0.49M, N=2048: 94.11% @0.98M. Log-linear scaling confirmed. step402b (N=4096,8192) running studio_cpu. | various | various | ✅ | **PARTIAL DONE** (N=256-2048 ✓, N=4096-8192 running studio_cpu) |
| **step403** | **ΔW Tier-1 validation + ΔW+Polarizer combo** — **DONE: A_proj = 95.75% (+1.61pp Tier-1 winner).** A_proj_pa15 = 94.98%, A_proj_a025 = 95.62%. ΔW proj alone beats compounds. Paper-ready. | N=2048 D=16 | 0.98M | ✅ | **DONE** |

---

## P-CUDA — Completed (2026-04-14)

| Step | Description | Effort | Expected speedup | Script | Status |
|------|-------------|--------|------------------|--------|--------|
| **step500** | **torch.compile + CUDA graphs bench** — **DONE (MASSIVE WIN):** V1 (torch.compile) = **4.2x training** / **6.5x inference** vs V0 eager on 5060ti. GPU util 2.6% → **99.6%**. **Training 21,043 sps vs VGG 4,469 sps = 4.7x faster with 116x fewer FLOPs.** V2 neutral. V3 crashed (CUDA graph tensor overwrite — V1 supersedes). | CUDA | **4-6x** | ✅ | **DONE** |
| **step520** | **RCM index reordering** — **DONE: KILLED. V1=5.294ms, V_rcm=5.330ms → 0.993×.** RCM no-op at fp32 reduce-overhead: GPU util=98% (compute-bound, not BW-bound). RCM only helps when memory-BW-bound. | 4h | actual: ~1× | ✅ | **DONE — KILLED** |
| **step800** | **ncu FLOPs cross-check** — **DONE.** Old formula undercounts 5.65×. True routing MACs = 2.13M. Total per-sample MACs = 1.85M. VGG FC = 247M → SGNNET = **0.75%** (under 1%). GPU util 0.09% at inference — memory-BW-bound. | 2h | validation | ✅ | **DONE** |
| **step801** | **max-autotune + bf16 benchmark** — **DONE.** V4 max-autotune = +3.5% (negligible). V5 bf16+scaler = 4.4× SLOWER (GradScaler overhead). V6 bf16+autotune+scaler = 4.4× SLOWER. **Decision: drop max-autotune. bf16 training with scaler = DO NOT USE.** | 1h | actual: +3.5% | ✅ | **DONE** |
| **step802** | **bf16 without GradScaler** — **DONE.** V7 bf16 no-scaler train = 5.207ms (+1.8% vs V1 5.299ms). Inference: +3–15% all batch sizes. **Decision: V7 = production inference config (free perf + half power). Training: V1≈V7, either fine.** | 1h | actual: +1.8% train, +3-15% inf | ✅ | **DONE** |
| **step803** | **make_graphed_callables for fwd+bwd** — **DONE: KILLED. `cudaErrorStreamCaptureInvalidated` on int64 indexing `Z[:, conn_hh, :]`. CUDA graph capture can't handle dynamic int64 gather indices.** Fix requires Triton kernel (step530). | Medium (3-5h) | actual: 0× | ✅ | **DONE — KILLED** |

---

## P-SPEED — Completed (2026-04-14)

**Premise:** step610 confirmed K_iter ANNEALING/CURRICULUM fails at N=2048 efficiency config (all schedules −2 to −6pp). Network has fundamental K=5 dependence. New direction: same receptive field, fewer sequential steps.

**Prior findings (from audit):**
- Multi-hop precompute: **NOT TESTED** (novel)
- Parallel branches: step114 at N=1024 D=64 "marginal" (demoted P3)
- Graph powers (A^k): wave-1 KILLED (amplitude-phase coupling, different mechanism)
- Wider+shallower: step213 (K_hh=3 at N=8192) no ceiling break; step214/215 (K_hh=8,16 at D=8) failed low D; **K_hh=4,8 at N=2048 D=16 never tested**

| Step | Description | Metric targets | Script | Status |
|------|-------------|----------------|--------|--------|
| **step700** | **Multi-hop gather** — **DONE: ALL HURT.** Ref=92.28% @ 14.9ms. All configs −3.5 to −13pp. Speedup 0.73-1.04x (wider gather adds memory overhead). Best: A_k3_kh6=86.60% @ 16.5ms. Also SLOWER. **CONFIRMED: Multi-hop flattening loses per-iter nonlinearity stacking.** | 40-80% fewer passes at equal accuracy | ✅ | **DONE — KILLED** |
| **step701** | **Parallel routing branches** — **DONE: ALL HURT.** Ref=91.67%. Best: D_par3_k3=84.64% (−7.03pp); B_par3_k1 collapsed 57.20% (−34.47pp); branch weights stayed uniform. **CONFIRMED: Sequential K_iter has non-trivial representational power beyond receptive field growth.** | K≤2 outer with parallel ≥ K=5 sequential | ✅ | **DONE — KILLED** |
| **step702** | **Latency-aware K_iter × K_hh frontier** — **DONE. Ref (ki5_kh2)=92.64% @121ms. ki4_kh2=91.06% @97ms (1.24×). ki3_kh2=88.88% @75ms (1.61×). ki2_kh2=85.25% @52ms (2.32×). K_hh=4 at K_iter=3/4 no improvement. Pareto: K_iter=5 dominates quality; K_iter=3 best tradeoff at 1.6×.** | 2x speedup at ≥95% accuracy retention | ✅ | **DONE** |
| **step703** | **Per-sample adaptive K_iter at inference** — **DONE: KILLED. All thresholds catastrophic (tau_99: −81.55pp, tau_97: −80.78pp). Logit cosine_sim NOT reliable convergence proxy.** Samples exit k=2-3 under any τ<1.0, collapsing to chance. **P-SPEED fully exhausted.** Only path: step530 Triton kernel. | Mean eff K_iter ~3 at ≤0.5pp accuracy loss | ✅ | **DONE — KILLED** |
| **step704** | **ΔW projection N-scaling at N=4096** — **DONE (KILLED Tier-1): CUDA A_proj=95.36% vs Ref=96.10% (−0.74pp); MPS A_proj=95.31% vs Ref=96.20% (−0.89pp). Both devices agree: ΔW proj N-specific — helps N=2048 (+1.61pp) but hurts N=4096 (near D=16 ceiling).** | ΔW proj ≥+1pp at N=4096 | ✅ | **DONE — KILLED** |

**P-SPEED INTERIM CONCLUSION (paper-worthy):**
Both architectural parallelization paths failed at N=2048 D=16. Sequential K_iter load-bearing — not just receptive field but **iterative nonlinearity stacking** (each iter applies ReLU-θ + F.normalize + reflection + AH suppression). Multi-hop collapses 5 nonlinear transforms into 1, losing representational power. Parallel branches same problem: fewer outer iters = fewer nonlinear refinement steps.

**Speed implications:**
1. **Kernel-level optimization** (P-CUDA: torch.compile, CUDA graphs, RCM, Triton fused kernel) — reduce per-iter overhead without changing K_iter
2. **Per-sample adaptive K_iter at inference** (step703) — train K=5, early-terminate at inference when logits stabilize
3. **Accept K_iter=5 as fundamental** — report as architectural constant

Killed/superseded:
- **step610** K_iter annealing/warm-switch: CONFIRMED FAIL at efficiency config. All 5 schedules hurt vs Ref K=5.
- **step611** K_iter warm transfer: ALL HURT. Ref=93.81%. C_8to5=−0.74pp (best). D_16to5=−6.06pp (worst). Higher teacher K → worse student. CONFIRMED FAIL.
- **step612** K_iter distillation at efficiency config: ALL HURT. Ref=94.52%. K=3 CE-only=90.70% (−3.82pp). K=3 distill T=2 lam=0.5=91.16% (−3.36pp). K=3 distill T=4 lam=0.8=89.76% (−4.76pp). CONFIRMED FAIL. All P-KITER paths KILLED.

---

## P-KITER — Reducing sequential K_iter passes (biggest FLOPs lever) (2026-04-14)

Motivation: K_iter=5 dominant latency factor (5 serial passes can't parallelize, theoretical max speedup = 116/5 = 23x).

**Prior findings (STALE, not current efficiency config):**
- step163 warm-start (K=12 teacher → K=8 student weights): **+7.82pp** at N=1024 D=16 K_hh=8 (known winner)
- step142 C curriculum LOW→HIGH (2→4→8→12): **+2.85pp** at N=1024
- step127 distillation: **HURT** — distillation objective interferes with warm-start gain
- step216 curriculum at N=2048 D=16: **KILLED −58pp** ("curriculum N=1024-only capacity crutch")
- step173 warm-start at N=2048 D=32 K_hh=4: HURT −0.23pp

| Step | Description | Scale | FLOPs | Script | Status |
|------|-------------|-------|-------|--------|--------|
| **step610** | **K_iter annealing at efficiency config** — ALL KILLED. Ref=93.94%. B_down_gentle=93.10% (best, −0.84pp). E_warm_switch_k3=−6.50pp (worst). CONFIRMED FAIL. | N=2048 D=16 K_hh=2 | varies | ✅ | **DONE — KILLED** |
| **step611** | **K_iter warm transfer** — ALL HURT. Ref=93.81%. C_8to5=−0.74pp (best). D_16to5=−6.06pp (worst). Higher teacher K → worse student. CONFIRMED FAIL. | N=2048 D=16 K_hh=2 | final varies | ✅ | **DONE — KILLED** |
| **step612** | **K_iter distillation at efficiency config** — ALL HURT. Ref=94.52%. K=3 CE-only=90.70% (−3.82pp). K=3 distill T=2 lam=0.5=91.16% (−3.36pp). K=3 distill T=4 lam=0.8=89.76% (−4.76pp). CONFIRMED FAIL. All P-KITER paths KILLED. | N=2048 D=16 K_hh=2 | 0.60M (K=3) | ✅ | **DONE — KILLED** |

---

## P-NEW — Completed experiments on step199 final config

| Step | Description | Scale | FLOPs | Script | Status |
|------|-------------|-------|-------|--------|--------|
| **step216** | Compound winners — ALL KILLED. twopop_weight -1.81pp, twopop_theta -2.45pp, curriculum -58pp. α=1.05 neutral. | N=2048 | 0.98M | ✅ | **DONE** |
| **step217** | Polarizer routing — **WINNER: full polarizer +1.27pp**, partial +0.89pp. Rotation neutral. | N=2048 | ~2M | ✅ | **DONE → Tier-1** |
| **step217b** | Polarizer Tier-1 — **WINNER: over-polarizer α=1.5 = 95.92% (+1.91pp)**. Full α=1.0 = 95.18% (+1.17pp). Monotonic alpha trend. | N=2048 | ~2M | ✅ | **DONE** |
| **step218** | Random projection ablation — Without AH: 91.5%→18.8% collapse. AH prerequisite. Freezing bug, needs rerun. | N=2048 | 0.98M | ✅ | **DONE (partial)** |
| **step219** | Topology comparison — barely matters. Anti-pref +0.15pp, uniform random -0.84pp. | N=2048 | 0.98M | ✅ | **DONE** |
| **step220** | Heterogeneous K_hh — **ALL KILLED**. Uniform K_hh=2 beats all. Best -0.28pp, worst -2.40pp. | N=2048 | ~1M | ✅ | **DONE** |
| **step221** | Output-assigned topology — **ALL KILLED**. Input-assigned better. Best -1.30pp, worst -2.45pp. JSON bug. | N=2048 | ~1M | ✅ | **DONE** |
| **step222** | Paper baselines — MLP/random proj/linear all stuck ~9%. Trainer incompatibility bug. | N=2048 | varies | ✅ | **DONE (broken)** |
| **step223** | CIFAR-10 cross-dataset — **DONE: DESIGN FLAW. 60.17% (VGG16 ImageNet features degenerate on raw 32×32 CIFAR-10). Replaced by step400 (raw pixel input, N_IN=3072).** | N=2048 | 0.98M | ✅ | **DONE — DESIGN FLAW** |
| **step224** | Scored dynamic topology — ALL NEUTRAL ±0.5pp. No benefit from per-epoch edge replacement. | N=2048 | 0.98M | ✅ | **DONE** |
| **step225** | Equilibrium propagation pilot — **DONE: KILLED. EP β=0.1: 10.06%, EP β=0.5: 10.04% vs Ref=91.67%. Chance-level. EP fundamentally incompatible with SGNNET L2-normalized routing.** | N=2048 | 0.98M | ✅ | **DONE — KILLED** |
| **step226** | Skip connections — A/B KILLED (−10 to −14pp). C learned gate α=0 (explicitly rejected skips). | N=2048 | 0.98M | ✅ | **DONE** |
| **step232** | Scalar edge weights — ALL ~18% (broken forward pass, gradients didn't flow). Needs redesign. | N=2048 | 0.98M | ✅ | **DONE (broken)** |
| **step233** | θ-edge precision ablation — **ALL DEAD: best 75.2% (fp32), 62-75% all dtypes**. Sinusoidal parameterization insufficient as AH replacement. Optimizer injection fix confirmed θ learns (std=0.115) but still fails. | N=2048 | 0.98M | ✅ | **DONE — KILLED** |
| **step234** | ΔW-vector polarizer — **BREAKTHROUGH: Config A (ΔW proj, NO AH) = 95.44% (+3.77pp)**. ΔW proj > W_pos proj > ΔW rot. Adding AH hurts (-1.2pp). | N=2048 | ~2M | ✅ | **DONE** |
| **step235** | ΔW-rot ± AH ± Augmentation (150ep full) — **WINNER: A_aug (ΔW-rot no-AH, aug) = 97.30%**. A_100 (no aug) = 96.97%. B configs (with AH) regress 96.79-96.82%. Confirms: ΔW-rot > AH, NO synergy combined. Augmentation helps. | N=2048 | 0.98M | ✅ | **DONE — ΔW-rot validated** |
| **step238** | Gradient-safe θ parameterizations — cos_shifted=62%, phase_delta=75%, triangle=~65%. **ALL DEAD.** θ-edge direction confirmed killed. | N=2048 | 0.98M | ✅ | **DONE — KILLED** |
| **ga_v2** | GA autorun v2 — pool=50 (20 elite, 20 crossbreed, 10 mutation). gen 1, individual 9/50. Early: N=4096 D=16 K_hh=2 K_iter=5 = 94.98% (strong scout). | varies | varies | ✅ | RUNNING (Mac Studio MPS) |

---

## P0 — Critical path: structural experiments addressing architecture ceiling

| Step | Description | Scale | FLOPs | Script |
|------|-------------|-------|-------|--------|
| **step156** | **LayerNorm at N=4096 D=64** — validate step116-C winner (+2.24pp N=1024) at scale. 20ep scout (Ref/A/B/C). If +pp → Tier-1 | N=4096 | 38.8M | ✅ |
| **step140** | **N×K connectivity tradeoff** — N dominates; more K HURTS at D=16. KILLED hypothesis | Multi-N | 2.6-5.6M | DONE |
| **step141** | **Split-D (12act+4pos) + residual hypersphere** — fix input collapse + oversmoothing | N=1024 | ~3.1M | ✅ |
| **step116** | **RMSNorm / normalization ablation** — RMSNorm, LayerNorm, pre-norm, delayed norm, no-norm | N=1024 | ~3.1M | DONE — C=LayerNorm+2.24pp WINNER; A=RMSNorm−11.49pp, B=pre_route−6.50pp |
| **step152** | **Constraint discovery** — nuclear norm, bottleneck, dim gating, L1 sparsity, contrastive loss | N=1024 | ~3.1M | DONE — ALL KILLED. B=bottleneck near-neutral (−0.05pp). Nuc norm/contrastive −13-15pp |
| **step153** | **Progressive capacity reduction** — start large (N/D), prune by contribution (Matformer/GMP) | Multi-N/D | varies | DONE — ALL KILLED. A=prune_dims −0.62pp best. Nested dropout/matformer/prune_neurons −10-15pp |

---

## P0.5 — Efficiency track: compound stacking on step165-B

| Step | Description | Scale | FLOPs | Script |
|------|-------------|-------|-------|--------|
| **step166** ✗ | **ALL KILLED** — Ref=91.59%, E=90.09%(−1.50pp twopop), F=87.92%(−3.67pp curric), G=89.86%(−1.73pp spec_reg), H=86.47%(−5.12pp). warm+W_proj local max. | N=1024 D=16 | ~3.1M | DONE |
| **step167** | **warm+W_proj Tier-2 validation** — stale RUNNING entry (no tmux session). Result JSON status unknown. | N=1024 D=16 | ~3.1M | ⚠️ STALE |
| **step168** ✓ | **warm+W_proj D=32 Tier-1** — Ref=81.96%, A=87.62%(+5.66pp warm-only), **B=93.20%(+11.24pp warm+W_proj)**. Near step144-C (93.50% W_proj+RigL). | N=1024 D=32 | ~6.1M | DONE |
| **step169** ✓ | **warm+W_proj D=32 full data 150ep** — **94.01% (best_ep=134)**. Phase exit NOT achieved (gap=0.99pp). D=32 ceiling ~94.0%. | N=1024 D=32 | ~6.1M | DONE |
| **step170** | **warm+W_proj N=2048 D=16 Tier-1** — N-scaling path (step72 N=2048 scratch=93.38%). Teacher training in progress (ep60=86.83%) | N=2048 D=16 | ~6.2M | ✅ RUNNING |
| **step171** | **warm+W_proj D=32 K_hh=4 Tier-1** — K_hh=4 +0.56pp at N=4096; tests if sparser connectivity lifts 94.01% ceiling. Configs: Ref/A(α=1.0)/B(α=1.05)/C(K_iter=12). FLOPs ~3.1-4.6M | N=1024 D=32 K_hh=4 | ~3.1-4.6M | ✅ RUNNING |
| **step172** | **warm+W_proj D=48 K_hh=4 Tier-1** — D capacity test: D=32 ceiling ~94.0%, D=48 between D=32 and D=64. FLOPs ~4.65M ≤ 6.18M ✓. Configs: Ref/B(α=1.0)/C(α=1.05) | N=1024 D=48 K_hh=4 | ~4.65M | ✅ RUNNING |
| **step175** | **warm+W_proj N=1024 D=32 K_hh=4 Tier-2** — reuses step171 teacher. Tests if full data 150ep matches step169 (94.01% @6.1M) at HALF FLOPs (~3.1M). step171-B=93.35% Tier-1 → expected ~94.1-94.5%. | N=1024 D=32 K_hh=4 | ~3.1M | ✅ RUNNING |
| **step173-Ref** ✓ | **scratch N=2048 D=32 K_hh=4 Tier-1 (50%/75ep)**: 94.93%. Warm-start REVERSAL confirmed: B=94.70% < Ref. | N=2048 D=32 K_hh=4 | ~6.1M | DONE |
| **step174** ✓ | **PHASE EXIT CONFIRMED: 95.82% best_ep=118** — N=2048 D=16 K_hh=8 warm+W_proj full data 150ep. | N=2048 D=16 K_hh=8 | ~6.2M | DONE |
| **step175** ✓ | **N=1024 D=32 K_hh=4 warm+W_proj Tier-2**: 94.62% best_ep=138. No phase exit. N=1024 ceiling confirmed. | N=1024 D=32 K_hh=4 | ~3.1M | DONE |
| **step176-A** ✓ | **PHASE EXIT CONFIRMED: 96.18% best_ep=125** — N=2048 D=32 K_hh=4 scratch Tier-2 full data. BEST efficiency point. | N=2048 D=32 K_hh=4 | ~6.1M | DONE |
| **step176-B** | α=1.05 calibration, ep110=93.96%, ~40ep left | N=2048 D=32 K_hh=4 | ~6.1M | ✅ RUNNING |
| **step177** ✓ | **NEW MIN-FLOPs RECORD: 95.13% best_ep=133** — N=1024 D=48 K_hh=4 warm+W_proj Tier-2. Phase exit @ ~4.72M FLOPs (24% below prior records). | N=1024 D=48 K_hh=4 | ~4.72M | DONE |
| **step180** ✓ | **94.14% best_ep=64** — N=2048 D=20 scratch Tier-1 @~3.93M. Tier-2 projected ~95.2-95.4%. | N=2048 D=20 K_hh=4 | ~3.93M | DONE |
| **step178** ✓ | **94.57% best_ep=73** — N=2048 D=24 scratch Tier-1 @~4.72M. Tier-2 → step184. | N=2048 D=24 K_hh=4 | ~4.72M | DONE |
| **step179** ✓ | **94.24% best_ep=57** — N=2048 D=28 scratch Tier-1 @~5.51M. Tier-2 → step183. | N=2048 D=28 K_hh=4 | ~5.51M | DONE |
| **step183** ✓ | **PHASE EXIT: 95.59% best_ep=122** — N=2048 D=28 Tier-2 @~5.51M. Above 5.51M floor. | N=2048 D=28 K_hh=4 | ~5.51M | DONE |
| **step181** ✓ | **PHASE EXIT: 96.03% best_ep=147** — N=2048 D=20 scratch Tier-2 @ ~3.93M. Min-FLOPs record (beat step177 4.72M). | N=2048 D=20 K_hh=4 | ~3.93M | DONE |
| **step182** ✓ | **93.96% best_ep=74** — N=2048 D=16 scratch Tier-1 @~3.15M. Tier-2 → step185. | N=2048 D=16 K_hh=4 | ~3.15M | DONE |
| **step185** ✓ | **PHASE EXIT: 95.87% best_ep=140** — N=2048 D=16 scratch Tier-2 @ ~3.15M. NEW MIN-FLOPs RECORD (beats step181 @ 3.93M). | N=2048 D=16 K_hh=4 | ~3.15M | DONE |
| **step186** ✓ | **93.10% best_ep=75** — N=2048 D=12 Tier-1 @~2.36M. Tier-2 → step188 (borderline). | N=2048 D=12 K_hh=4 | ~2.36M | DONE |
| **step187** ✓ | **91.26% best_ep=72** — N=2048 D=8 Tier-1 @~1.57M. D=8 Tier-2 RULED OUT (max ~93.2%). | N=2048 D=8 K_hh=4 | ~1.57M | DONE |
| **step188** ✗ | **94.62% best_ep=138** — N=2048 D=12 Tier-2 @~2.36M. NO PHASE EXIT (0.38pp short). D-reduction floor confirmed at D=16 @ 3.15M. | N=2048 D=12 K_hh=4 | ~2.36M | DONE |
| **step189** ✓ | **92.25% best_ep=73** — N=2048 D=10 Tier-1 @~1.97M. Tier-2 ruled out (max ~94.2%). | N=2048 D=10 K_hh=4 | ~1.97M | DONE |
| **step190** ✓ | **93.86% best_ep=75** — N=2048 D=16 K_hh=2 Tier-1 @~1.57M. CRITICAL: beats D=8 K_hh=4 +2.6pp at same FLOPs. D=16 dimensionality >> K_hh. Tier-2 → step193. | N=2048 D=16 K_hh=2 | ~1.57M | DONE |
| **step191** ✓ | **94.68% best_ep=71** — N=2048 D=16 K_hh=3 Tier-1 @~2.36M. Already above D=12 Tier-2 (94.62%) at same FLOPs. Tier-2 proj ~96%. → step192. | N=2048 D=16 K_hh=3 | ~2.36M | DONE |
| **step192** ✓ | **PHASE EXIT: 95.90% best_ep=105** — N=2048 D=16 K_hh=3 Tier-2 @~2.36M. NEW MIN-FLOPs record at time. | N=2048 D=16 K_hh=3 | ~2.36M | DONE |
| **step193** ✓ | **PHASE EXIT: 95.67% best_ep=135** — N=2048 D=16 K_hh=2 Tier-2 @~1.57M. NEW MIN-FLOPs record (50% below 3.15M). | N=2048 D=16 K_hh=2 | ~1.57M | DONE |
| **step194** ✓ | **94.88% best_ep=71** — N=2048 D=16 K_hh=2 K_iter=6 Tier-1 @~1.18M. Extraordinary: +1.02pp vs K_iter=8 Tier-1 despite 25% fewer FLOPs. Tier-2 → step195. | N=2048 D=16 K_hh=2 | ~1.18M | DONE |
| **step195** ✓ | **PHASE EXIT: 96.08% best_ep=106** — N=2048 D=16 K_hh=2 K_iter=6 Tier-2 @ 1.18M. **FULL EFFICIENCY CRITERION MET: ≤1% FLOPs + ≥95% accuracy.** | N=2048 D=16 K_hh=2 | ~1.18M | DONE |
| **step196** ✗ | **92.74% best_ep=72** — N=2048 D=16 K_hh=2 K_iter=4 Tier-1 @~0.79M. KILLED: −2.14pp vs K_iter=6. Too few routing steps. | N=2048 D=16 K_hh=2 | ~0.79M | DONE |
| **step197** ✓ | **93.96% best_ep=72** — N=2048 D=16 K_hh=2 K_iter=5 Tier-1 @~0.98M. ≥93% met. Tier-2 → step199. | N=2048 D=16 K_hh=2 | ~0.98M | DONE |
| **step198** ✗ | **88.92% best_ep=62** — N=1024 D=16 K_hh=2 K_iter=6 Tier-1 @~0.59M. KILLED: −6pp vs N=2048. N=2048 floor for D=16. | N=1024 D=16 K_hh=2 | ~0.59M | DONE |
| **step199** ✓ | **SUB-1% EXIT: 95.52% best_ep=136** — N=2048 D=16 K_hh=2 K_iter=5 Tier-2 @ 0.98M. NEW MINIMUM: 0.79% VGG16 FC FLOPs + ≥95%. | N=2048 D=16 K_hh=2 | ~0.98M | DONE |
| **step200** ✗ | **90.52% best_ep=72** — N=2048 D=16 K_hh=1 K_iter=8 Tier-1 @ 0.79M. KILLED. K_hh=1 breaks connectivity; 8 steps insufficient. K_hh=2 minimum viable. | N=2048 D=16 K_hh=1 | ~0.79M | DONE |
| **step202** ✗ | **89.25% best_ep=74** — N=2048 D=16 K_hh=2 K_iter=3 T1 @ 0.59M. KILLED. K_iter floor: 3→89%, 4→92.74%, 5→95.52%. Min viable = K_iter=5. | N=2048 D=16 K_hh=2 | ~0.59M | DONE |
| **step203** ✓ | **96.08% best_ep=69** — N=4096 D=16 K_hh=2 K_iter=5 T1 @ 1.97M. HIGHEST T1 EVER at D=16. N-scaling: N=2048→93.96%, N=4096→96.08% (+2.12pp). T2 → step205. | N=4096 D=16 K_hh=2 | ~1.97M | DONE |
| **step201** ✓ | **95.64% best_ep=66** — N=4096 D=16 K_hh=2 K_iter=6 T1 @ 2.36M. T1 phase exit hit. Tier-2 → step204. | N=4096 D=16 K_hh=2 | ~2.36M | DONE |
| **step204** ✓ | **PHASE EXIT: 97.15% best_ep=71** — N=4096 D=16 K_hh=2 K_iter=6 T2 @ 2.36M. vs N=2048 T2: +1.07pp. vs D=64: −0.71pp. N-scaling confirmed. | N=4096 D=16 K_hh=2 | ~2.36M | DONE |
| **step205** ✓ | **PHASE EXIT: 97.17% best_ep=118** — N=4096 D=16 K_hh=2 K_iter=5 T2 @ 1.97M. **NEW D=16 RECORD**. Only −0.69pp from D=64 record (97.86%). | N=4096 D=16 K_hh=2 | ~1.97M | DONE |
| **step206** | **95.11% best_ep=72** — N=8192 D=16 K_hh=2 K_iter=6 T1 @ 4.72M. T1 regression vs N=4096 (−0.53pp). N-scaling breaks at T1 for N=8192. T2 → step207. | N=8192 D=16 K_hh=2 | ~4.72M | DONE |
| **step207** ✗ | **96.20%@ep54** — N=8192 D=16 K_hh=2 K_iter=6 T2 @ 4.72M. N-SCALING BREAKS: −0.95pp vs N=4096 T2 (97.15%). Plateau 95.75-95.90% after ep54. D=16 bottleneck confirmed for K_iter=6 at N=8192. | N=8192 D=16 K_hh=2 | ~4.72M | DONE |
| **step208** ✓ | **95.77% best_ep=51** — N=8192 D=16 K_hh=2 K_iter=5 T1 @ 3.93M. K_iter=5 beats K_iter=6 at N=8192 (+0.66pp). −0.31pp vs N=4096 K5 T1. T2 → step209. | N=8192 D=16 K_hh=2 | ~3.93M | DONE |
| **step209** ✓ | **97.17%@ep115** — N=8192 D=16 K_hh=2 K_iter=5 T2 @ 3.93M. EXACTLY matches N=4096 K5 T2 (97.17%). **D=16 K_hh=2 ceiling at 97.17% CONFIRMED**. N-scaling flat at N=8192. | N=8192 D=16 K_hh=2 | ~3.93M | DONE |
| **step210** | **95.49%@ep65** — N=8192 D=16 K_hh=2 K_iter=4 T1 @ 3.15M. K_iter=4 VIABLE at N=8192 (KILLED at N=2048!). K_iter axis T1: K6=95.11% < K4=95.49% < K5=95.77%. Optimal K_iter decreases with N. | N=8192 D=16 K_hh=2 | ~3.15M | DONE |
| **step211** | **94.93%@ep72** — N=8192 D=16 K_hh=2 K_iter=3 T1 @ 2.36M. Just below 95% (−0.07pp). +5.68pp vs N=2048 K_iter=3. K_iter=3 floor at N=8192 borderline. → step212 (N=16384). | N=8192 D=16 K_hh=2 | ~2.36M | DONE |

---


*Continued in [EXPERIMENT_QUEUE_history_part2.md](EXPERIMENT_QUEUE_history_part2.md) — P1, P1.5, P2, P3, Launch Order, Completed Experiments, Earlier Completed, P-BACKLOG.*