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

## Currently Running (5 slots, updated 2026-04-15 evening)

| Machine:Device | Session | Step | Status | Note |
|---------|---------|------|--------|------|
| mini:mps | sgn-indra-mini_mps-train_step401b_sgnnet_cifar10 | step401b | RUNNING ep30 val=77% | SGNNET cross-dataset CIFAR-10 |
| mini:cpu | sgn-indra-mini_cpu-train_step614_cifar100_scaling | step614 MLP | RUNNING | CIFAR-100 MLP hidden-scaling CPU |
| studio:mps | sgn-indra-studio_mps-train_step614_cifar100_scaling | step614 all | RUNNING | CIFAR-100 full scaling (SGNNET+MLP) |
| studio:cpu | sgn-indra-studio_cpu-train_step615_fair_mlp_matched | step615 | RUNNING | fair matched-params MLP (LeakyReLU, He-init, dropout, label-smooth, 150ep) |
| 5060ti:cuda | — | step614 SGNNET large-N | PENDING rsync | waiting for store_cifar100.h5 transfer |

**Auto-launch queue when slots free** (`.controller/auto_launch_queue.txt`):
1. bench_step832 PyG torch_scatter (5060ti) — needs CUDA
2. step522 Muon optimizer (5060ti) — needs CUDA
3. step401b CIFAR-10 SGNNET cross-dataset (mini_mps after extract completes)

---

## Priority Queue — Active / QUEUED

### P-PAPER-2026-04-15 — Scripts added from V3 gap-analysis + design log (2026-04-15)

All scripts smoke-tested with `--help`. Launch via `scripts/queue_submit.sh` (once the queue controller is running) or `scripts/launch_slot.sh`.

| Step | Description | Scale | Script | Status |
|------|-------------|-------|--------|--------|
| **step267** | ΔW rot + aug + K=4 @ N=4096 (V3 Gap 2.1) — **SUPERSEDED.** step266 Ref (K=5)=97.71%, A_k4 (K=4)=97.66% ALREADY COMPLETE. Local JSON was stale sync artifact; authoritative result synced from 5060ti. No new script needed. | N=4096 | (n/a) | **DONE — confirmed from synced step266 log** |
| **step268** | ΔW proj + aug + K=4 combo @ N=2048 Tier-1 (V3 Gap 2.2) — stack K=4 equivalence + aug gain. 4-config ablation: Ref K5 no-aug, A K4 no-aug, B K4 aug (COMBO), C K5 aug. | N=2048 | `scripts/train_step268_dwproj_aug_k4.py` | QUEUED |
| **step403b** | Matched-FLOPs MLP_37 baseline — MLP_37=97.71% @ep36 at 1.86M FLOPs. Paper baseline confirmed. | N_in=25088→h=37→10 | `scripts/train_step403b_matched_flops_mlp.py` | **DONE (studio_mps)** |
| **step404** | GCN / GAT / GIN baselines — RUNNING on studio_mps. | N=2048 | `scripts/train_step404_gnn_baselines.py` | **RUNNING (studio_mps)** |
| **step405** | SST-2 cross-modal SGNNET (V3 Gap 2.8 paper-blocker) — DistilBERT CLS [768-d] → Linear / MLP_64 / SGNNET comparison. 2-phase: `--phase extract` (one-time) then `--phase train`. Requires `pip install transformers datasets h5py`. | N=2048 D=16 | `scripts/train_step405_sgnnet_sst2.py` | QUEUED |
| **step524-S1** | Edge-β scalar on frozen topology. Ref=93.94%, S1_init0=93.91% (−0.03pp), S1_init1=93.07% (−0.87pp). | N=2048 | `scripts/train_step524_s1_edge_beta.py` | **DONE — KILLED. Edge-β adds no value. Dynamic direction CLOSED.** |
| **step526/527** | INT8 QAT sweep. Part A weight-only: −0.20pp (lossless). W+Z: −1.27pp. Part B QAT accum=1: −0.97pp viable; cliff at accum=4 (−16.31pp). fp32 ref=87.90%. | N=2048 | `scripts/train_step526_int8_qat.py` | **DONE (mini_mps)**. Result: `results/train_step527_int8_qat_k4_seed42__mini_mps.json` |
| **bench_step830** | K=4 direct wall-clock measurement (V3 Blocker-7) — currently paper says "20% reduction projected"; this measures it. 6 variants: K5/K4 × eager/reduce-overhead/max-autotune fp32. Prints pass/fail vs the ≤0.85× criterion. | N=2048 bs=32 | `scripts/bench_step830_k4_wallclock.py` | QUEUED (5060ti only) |

### P-CUDA — Deferred

| Step | Description | Script | Status |
|------|-------------|--------|--------|
| **step530** | **Triton fused kernel (gather+mul+sum)** — custom kernel eliminating [B,N,K_hh,D] intermediate tensor. Register-tiled for D=16. | DEFERRED | QUEUED |

---

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
| **step405** | SST-2 cross-modal — MLP_37/64 done (84.63%). SGNNET relaunched (fixed use_trainer+Fourier encoding). ep1=78.44%, healthy. 150ep. | **RUNNING (studio_cpu)** |

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
| **step523** | **Alternating W_pos / edge training cycle** (user directive). 30ep warmup → [20ep W_pos / 20ep edges] × 2 cycles → 10ep final. Edge cap 1% per event (≤10 swaps at N=512 K_hh=2). Tests whether temporal separation of W_pos and topology updates rescues the dynamic connectivity direction that step511-514 failed (6/6 variants all negative). **Script ready, queue for next CPU slot.** | QUEUED |
| **step521** | Multi-forward-backward (deep supervision on K_iter). k_only_forward ∈ {0,1,2,3} compared to Ref (single loss). Implemented as accumulated-loss deep supervision, single backward. **Script ready, queue for CUDA slot.** | QUEUED |
| **step522** | Muon optimizer vs AdamW at N=2048 K=5 ΔW proj. Measures convergence speed (epochs-to-93/94/95) AND final accuracy — user directive: "same accuracy at faster convergence is a win". Requires `pip install muon-optimizer`. **Script ready, queue for CUDA slot (after env check).** | QUEUED |
| **direct K=4 wall-clock bench** | Measure SGNNET K=4 inference latency directly (currently projected 0.224ms based on 20% reduction). Add to bench_step811 variant list. | TODO |
| **step524** | **Edge-SHIFT probes** (post-step523 follow-up). 6 configs at N=1024 T0: Ref / P1 step523+Adam-reset / P2 alt-schedule+0%cap / S1 edge-β scalar / S2 W_pos-passive-rebind / S4 cyclic-shift-null-control. Tests H1-H2-H5 of step523 failure + 2 continuous-parameterization alternatives + 1 null control. ~4h one slot. Design in LEARNINGS_design_2026_04_15.md. | TODO (script) |
| **step526** | **INT8 QAT + inference impact + grad-accum sweep.** Part A: fp32 train → quant eval for 3 modes (saturate/modular/crt) × {W only, W+Z}. Tests hypothesis: L2-norm at D=16 bounds components to ±0.25 × scale=100 → int8 range ±25, so wrap never fires. Part B: QAT from scratch (fake-quant+STE forward, fp32 master), grad_accum ∈ {1,4,16,64}. Telemetry: wrap_rate per forward. Script ready: `train_step526_int8_qat.py`. | TODO (launch) |
| **Param count reconciliation** | Bench reports SGNNET=34,976; training reports 67,744. Diff ≈ 32K. Find missing component (likely K_in=25 seed projection). Resolve before paper. | TODO |
| **bench_step832** | **PyTorch Geometric `torch_scatter` wall-clock probe** on 5060ti. Install `torch-scatter`, re-implement routing loop in edge-list format, benchmark vs V2 max-autotune (0.280ms K=5). Hypothesis: 2-3× if memory-BW-bound. Missing baseline — reviewers will ask. | TODO |
| **bench_step830** | **Direct K=4 wall-clock + fullgraph=True audit** on 5060ti. Cheap — measure projected 0.224ms directly AND eliminate torch.compile graph breaks. | TODO |

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
