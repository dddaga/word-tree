---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: completed
stopped_at: Completed 04-05-PLAN.md (Wave Comparison Report)
last_updated: "2026-03-26T09:00:03.624Z"
progress:
  total_phases: 7
  completed_phases: 4
  total_plans: 17
  completed_plans: 12
---

# Project State: neuro_graph

## Project Reference

See: .planning/PROJECT.md (updated 2026-03-23)

**Core value:** SGNNET matches VGG16 FC accuracy at ≤1% of its parameters
**Current focus:** Phase 04 — sgnnet-wave-architecture-experiments

## Current Phase

**Phase 5 — Scalable Architecture Experiments**
Status: In progress (0/5 plans formally, experiments running in tmux)
Next action: Complete remaining Phase 5 experiments; then Phase 6 (PCA Compression)

## Phase Progress

| Phase | Name | Status |
|-------|------|--------|
| 1 | Data Pipeline | Complete (2/2 plans, UAT passed 7/7) |
| 2 | Dense Baseline Benchmark | Complete (2/2 plans) |
| 3 | SGNNET Core Architecture | Complete (3/3 plans) |
| 4 | SGNNET Wave Architecture & Experiments | Complete (5/5 plans) |
| 5 | Scalable Architecture Experiments | In progress — Plan 02 (ProximityWave) complete, dispatched to Mac Studio (exp3_pw); Exp1 scale sweep, Exp2 SmallWorld also running |
| 6 | PCA Compression | Not started |
| 7 | Comparative Analysis & Report | Not started |

## Decisions

- **>= version constraints in requirements.txt**: Python 3.14 may need latest wheels; pinning exact versions risks incompatibility (Phase 1, Plan 01-01)
- **Root-relative /data/ in .gitignore**: Prevents accidentally ignoring src/data/ module directory (Phase 1, Plan 01-01)
- **PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0**: Removes 50% MPS memory cap for full-batch extraction (Phase 1, Plan 01-02)
- **num_workers=0 for extraction**: macOS MPS + Python 3.14 multiprocessing spawn incompatible (Phase 1, Plan 01-02)
- **pin_memory disabled on MPS**: Auto-detected in get_dataloader (Phase 1, Plan 01-02)
- **thop MACs not FLOPs**: thop.profile returns MACs; documented with flops_note in JSON (Phase 2, Plan 02-01)
- **Reuse VGGExtractor for eval**: No duplicate VGG16 loading; extractor handles frozen/eval/MPS (Phase 2, Plan 02-01)
- **Track results/*.json in git**: Baseline JSON is a key deliverable consumed by Phases 4 and 6 (Phase 2, Plan 02-01)
- **Stored soft labels identical to direct eval**: Accuracy diff=0.0 confirms HDF5 tensor store is perfectly faithful (Phase 2, Plan 02-02)
- [Phase 03]: Encoding returns [channel_norm, h_norm, w_norm] (3 coords); value added per-sample in model forward
- [Phase 03]: Split forward into _seed/_iterate_hidden/_output_readout for readability under 250-line limit
- [Phase 03]: Sparsity test threshold 0.88 for small N_hidden due to guaranteed-connectivity row fix
- [Phase 03]: Active param counting: only mask==1 entries count toward budget (649K active vs 6.5M dense)
- [Phase 03]: KL divergence (not MSE) for distillation task loss in total_loss
- [Phase 03]: C_ho sparsity threshold 0.85 due to guaranteed-connectivity on small 256x10 matrix
- [Phase 04]: Switched to phasor activations Z ∈ ℂ^D because amplitude-only dynamic routing is broken: LayerNorm output lives near 0 while W_pos ∈ [0,1]^D, so cdist(A_hidden, W_pos) ≈ 1.0 >> r*≈0.125, killing all proximity gates. Phasor decouples routing (always-positive Gaussian amplitude) from interference (complex phase rotation).
- [Phase 04]: C matrices made binary immutable (0/1 mask, no learned values) to isolate the contribution of dynamic wave routing in experiments. Learned C weights deferred to later generation.
- [Phase 04]: λ = r*/2 = r_repel — one full oscillation in active zone [r*/2, r*], both boundaries at φ=0, single inhibitory ring at d=3r*/4. All constants derived from N and D via r*; no free wavelength hyperparameter.
- [Phase 04]: W_phase ∈ ℝ^D (Exp 2 only) — per-neuron per-dimension learned phase operator applied after incoming phasor accumulation; separate from W_pos which sets geometric proximity.
- [Phase 04]: FP16 mixed precision via `torch.autocast('mps', dtype=torch.float16)` + `torch.amp.GradScaler('mps')` (requires PyTorch ≥2.3). `torch.cuda.amp.*` is CUDA-only — does not work on MPS. Apple Silicon GPU processes FP16 natively (~1.5–2× throughput for large matmuls). If PyTorch <2.3, use autocast alone (GradScaler had inf-detection bugs on MPS before 2.3).
- [Phase 04, Plan 01]: Binary C masks as buffers (D-05): no learned values, 0/1 only, registered as buffers
- [Phase 04, Plan 01]: Masked normalization: only neurons with |Z_j| > eps participate in mean/var
- [Phase 04, Plan 01]: W_phase is None unless use_wphase=True (Stage C only)
- [Phase 04, Plan 01]: N_in=25088 direct kept for Phase 4 wave model (no adapter)
- [Phase 04, Plan 02]: load_balance_loss uses abs sum of scores as proxy for neuron selection frequency
- [Phase 04, Plan 02]: GradScaler support detected at runtime via PyTorch version check (>= 2.3)
- [Phase 04, Plan 02]: PYTORCH_ENABLE_MPS_FALLBACK=1 needed for cdist backward on MPS
- [Phase 04, Plan 03]: Stage A with 1064 learnable params (W_pos only) achieves ~10% accuracy -- binary C masks alone insufficient for distillation
- [Phase 04, Plan 03]: GA selected K=2, N_hidden=256, lr=0.008, lambda_safety=0.52 as best Stage A config
- [Phase 04, Plan 03]: Loss plateaus at ~3.22 after epoch 8 -- static binary wiring hits expressiveness ceiling
- [Phase 04, Plan 04]: Exp 1 GA selected K=2, N_hidden=256, lr=0.0024 -- lower lr suits phasor routing
- [Phase 04, Plan 04]: Exp 2 GA selected K=2, N_hidden=256, lr=0.01, lr_Wphase=0.01 -- aggressive lr for W_phase
- [Phase 04, Plan 04]: Exp 1 (spatial phase) top1=0.1197 vs Stage A 0.1027 -- proximity routing adds marginal benefit
- [Phase 04, Plan 04]: Exp 2 (spatial + W_phase) top1=0.1052 -- W_phase trained (norm=13.36) but did not improve accuracy
- [Phase 04, Plan 05]: Baseline class name casing mismatch handled via explicit mapping dict (VGG16 "English springer" vs experiments "english_springer")
- [Phase 04, Plan 05]: Phase 3 SGNNET config (649K params with learned C) included as reference row alongside Phase 4 binary-C experiments (1K-2K params)
- [Phase 05, Plan 02]: Inline anti-Hebbian suppression in ProximityWave routing weights (not separate wrapper) due to phasor Z_re/Z_im interface mismatch with SGNNET_Resonant
- [Phase 05, Plan 02]: Fourier encoding mode added to ProximityWave for D=64 compatibility
- [Phase 05, Plan 02]: l2 norm mode in ProximityWave (validated winner from Step 1)

## Open Decisions

None currently.

## Key Files

- Architecture spec: `sparse_geometric_network_report.md`
- Requirements: `requirements.txt`
- Dataset module: `src/data/dataset.py`
- Feature extractor: `src/data/extractor.py`
- Tensor store: `data/store.h5` (13,394 records)
- Manifest: `data/manifest.csv` (13,394 rows)
- Metrics module: `src/utils/metrics.py` (compute_all_metrics, count_params, count_flops)
- Baseline eval script: `scripts/eval_baseline.py`
- Baseline results: `results/baseline_vgg16.json` (top1=0.9954, mAP=0.9997)
- Metrics tests: `tests/test_metrics.py` (4 tests)
- Soft label verification: `scripts/verify_soft_labels.py` (accuracy, entropy, class balance checks)
- Training loop: `src/training/trainer.py` (Trainer class with FP16 AMP, position clamping)
- GA search: `src/training/ga_search.py` (GASearch, SEARCH_SPACE_AB, SEARCH_SPACE_C)
- GA CLI: `scripts/run_ga_search.py` (run GA search by experiment name)
- Wave comparison: `scripts/eval_comparison.py` (assembles all stage results)
- Wave comparison JSON: `results/wave_comparison.json` (side-by-side metrics + deltas)
- Wave comparison MD: `results/wave_comparison.md` (human-readable tables)

## Performance Metrics

| Phase-Plan | Duration | Tasks | Files |
|------------|----------|-------|-------|
| 01-01      | 6min     | 2     | 5     |
| 01-02      | 3min     | 2     | 5     |
| 02-01      | 4min     | 2     | 8     |
| 02-02      | 2min     | 1     | 2     |
| Phase 03 P01 | 2min | 1 tasks | 5 files |
| Phase 03 P02 | 2min | 1 tasks | 2 files |
| Phase 03 P03 | 3min | 2 tasks | 5 files |
| 04-01      | 11min    | 3     | 4     |
| 04-02      | 13min    | 2     | 4     |
| 04-03      | 22min    | 2     | 3     |
| 04-04      | 60min    | 2     | 8     |
| 04-05      | 2min     | 1     | 3     |

## Last Session

- **Stopped at:** Phase 5 Plan 02 (ProximityWave) complete -- dispatched exp3_pw on Mac Studio; N=1024 fwd=595ms/batch
- **Timestamp:** 2026-04-03T12:13:23Z

---
*State initialized: 2026-03-23*
*Last updated: 2026-03-23 after Plan 02-02 completion*
