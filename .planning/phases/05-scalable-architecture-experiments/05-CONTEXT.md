# Phase 5: Mechanism Discovery & Architecture Optimization — Context

**Gathered:** 2026-04-03
**Status:** Ready for planning
**Source:** Synthesized from learnings/EXPERIMENT_QUEUE.md, learnings/LEARNINGS_phase5_p*.md, results/ JSONs, and current Mac Studio state

<domain>
## Phase Boundary

Phase 5 delivers two parallel tracks:

**Track A — Architectural exploration (original Phase 5 scope):**
- Neuron scaling sweep: SGNNET_SmallWorld (best mechanism stack) at N=[512,1024,2048,4096,10000]
- SGNNET_ProximityWave: sparse W_pos k-NN topology + dynamic phasor routing + periodic reconnection
- Signal reflection routing: input-dependent active-path routing via negative-activation bounce-back
- Architecture comparison table: SmallWorld vs ProximityWave vs Reflection at multiple N values

**Track B — Mechanism discovery (ARM framework, evolved during Phase 5 execution):**
- ARM 1: Generational compounding — calibrate all mechanisms on step22b base → Gen4 compound (step32)
- ARM 2: New mechanisms — K_iter scaling, LR schedule, high-D routing, low-rank mixing
- ARM 3: Dynamic connectivity — resolve whether O(N·K) input-dependent topology can work at D=64
- ARM 5: Input architecture — learned spatial projections (step55) vs random K_in=50 vs PCA

Phase 5 does NOT deliver: transformer integration (future milestone), or the final comparative report (Phase 6).

</domain>

<decisions>
## Implementation Decisions

### Confirmed Architecture (LOCKED)
- **Model:** SGNNET_SmallWorld (O(N·K) gather-sum, no O(N²) ops)
  - N=1024, D=64, K_iter=8, K_in=50, K_local=4, K_random=2, n_groups=128
  - `src/sgnnet/model_smallworld.py`
- **Phase routing wrapper:** SGNNET_Resonant (mode=dynamic_z_geo, beam_size=32, K_phase=8)
  - `src/sgnnet/model_resonant.py`
- **Inhibition wrapper:** SGNNET_AntiHebbian (variant=wpos)
  - `src/sgnnet/mechanisms_inhibitory.py`
- **Input architecture (current):** Random sparse gather K_in=50 from N_in=25088
  - Being challenged by step55 (spatial grouped learned projection)
- **Encoding:** Fourier encoding on S^{D-1}, D=64 (D=128 architecturally non-viable)
- **Training:** 150 epochs, batch=128, plateau LR (patience=10 factor=0.5), lambda_safety scaled
  - Runner: Mac Studio (256GB RAM, Apple Silicon MPS) via SSH
  - Scripts: `d_env/bin/python3 -u scripts/train_stepXX_*.py --device mps`
  - Remote dir: `/Users/admin/ml/dhiraj/qwen2_omni/testing/`

### Confirmed Winning Mechanisms (MUST BE IN ALL FUTURE BASES)
- **AntiHebb α=0.7 wpos** → **75.24%** (+18.96pp vs 56.28% ceiling) — step29 Config C, ALL-TIME BEST
- **AntiHebb α=0.5 wpos** → 70.14% (+13.86pp) — still valid as intermediate reference
- **K_iter=8** (not 3, not 16+) → K_iter=3→8 gap = 15pp; higher K_iter to be resolved by step48
- **Fourier encoding, D=64** → confirmed ceiling; D=128 non-viable
- **alpha_reflect=0.5** → +2.75pp at 40ep (step22b calibration winner)

### Dead Mechanisms (DO NOT TEST AGAIN)
- **Signed coupling (Z @ Z^T)** at D=64: cos-sim on S^63 ≈ noise; 5 experiments confirm ≤32%
- **D=128 Fourier encoding**: routing collapses, all configs ~10%
- **Cross-dim W_mix (D×D matrices)**: neutral at D=16, hurts D=64 (-15pp)
- **Phase-queried D×D matrix bank**: same interference pattern (-5pp)
- **MoD adaptive K_iter**: 19-20%, all 8 steps necessary
- **Oja's rule routing update**: -32pp; PCA compression destroys diversity
- **Soft beam routing**: gradients through hard selection not the bottleneck

### Experimental Methodology (STRICT RULES)
- **GA compounding rule:** Every new experiment base = ALL confirmed winners. Ablate only the tested variable.
- **Calibration rule:** When base changes scale/generation, run 40-50ep param sweep BEFORE 150ep runs.
- **Concurrency cap:** Maximum 2 experiments on Mac Studio simultaneously.
  - Launch next ONLY when count ≤ 1 AND RAM ≥ 50 GB free+inactive.
  - RAM = (Pages free + Pages inactive) × 16384 / 1073741824
- **Logging:** Results → `results/train_stepXX_*.json`; analysis → `learnings/LEARNINGS_phase5_p*.md`

### Open Questions Being Resolved (Active Experiments)
1. **ARM 1 Gen4 compound (step29c → step32):** Which mechanism combinations beat 75.24%?
   - step29c running on Mac Studio (mechanisms_calibrated): all mechs on step22b-calibrated base
   - step32 (Gen4 compound stack) blocked until step29c completes
2. **K_iter scaling (step48):** Does K_iter > 8 help? Running, e50/150.
3. **LR schedule (step54):** Does CosineWarmRestarts beat plateau? Running, e140/150 (nearly done).
4. **Dynamic connectivity (step36 final, step49, step50, step51):** Can O(N·K) input-dependent topology match what O(N²) signed coupling achieved at D=16?
   - step36 Config 1: 58.62% (+2.34pp over ceiling) — most promising dynamic result yet
5. **Input architecture (step55):** Does learned per-group spatial projection beat random K_in=50?
   - step55 script + model ready locally, not yet dispatched

### Current Best Accuracy Timeline
| Step | Config | top-1 | vs prev |
|------|--------|-------|---------|
| step22 | D=64 N=1024 K_iter=8 | 56.28% | D baseline |
| step22b | + alpha_reflect=0.5 | ~52.94% (40ep) | calibration |
| step29A | + AntiHebb α=0.5 | 70.14% | +13.86pp |
| step29C | + AntiHebb α=0.7 | **75.24%** | **+5.10pp** |

### Claude's Discretion
- Exact order of plans within a wave (executor decides based on experiment completions)
- SSH sync commands and tmux session naming (follow existing patterns in learnings/)
- When step29c takes too long: proceed with ARM 2/3/4 plans in parallel; come back to ARM 1 when done
- LEARNINGS file naming: continue the `LEARNINGS_phase5_p{N}_*.md` convention

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Experiment State (current queue + ARM status)
- `learnings/EXPERIMENT_QUEUE.md` — Full ARM framework, all steps with status and priority
- `learnings/LEARNINGS_phase5_p7_gen3_arm_status.md` — ARM status summary as of 2026-04-01

### Key Results
- `results/train_step29_antihebb_d64.json` — AntiHebb α sweep; Config C = 75.24% (all-time best)
- `results/train_step36_input_gated.json` — Input-gated adjacency; Config 1 = 58.62%
- `results/train_step31_dynamic_zknn.json` — Dynamic Z-KNN; all configs underperform static
- `results/train_step22b_routing_calib_d64.json` — Base routing calibration; alpha_reflect=0.5 winner

### Architecture Source
- `src/sgnnet/model_smallworld.py` — Core SmallWorld model
- `src/sgnnet/model_resonant.py` — Resonant (beam + dynamic_z_geo) wrapper
- `src/sgnnet/mechanisms_inhibitory.py` — AntiHebbian wrapper
- `src/sgnnet/model_spatial_grouped.py` — Spatial grouped input (step55, not yet dispatched)
- `src/training/experiment_config.py` — topology_kwargs, trainer_kwargs, GA_BEST (needs update to 75.24%)
- `src/training/trainer.py` — Training loop with MPS/FP16/plateau

### Mac Studio Infrastructure
- SSH: `ssh mac-studio` (alias configured, key at `keys/id_ed25519`, port 8021)
- Remote: `/Users/admin/ml/dhiraj/qwen2_omni/testing/`
- Python: `d_env/bin/python3`
- Concurrency rule: target 2 concurrent max; launch when count ≤ 1 AND RAM ≥ 50 GB

</canonical_refs>

<specifics>
## Specific Numbers & Constraints

- **Target accuracy:** Not hard-fixed; maximize from current 75.24% base
- **Hard constraint:** O(N·K) complexity — no O(N²) operations in the final model
- **Parameter budget:** ≤1% of VGG16 FC (~1.24M params). AntiHebb adds no params.
- **Epoch budget per experiment:** 150ep for full runs, 40ep for calibration
- **Mac Studio capacity:** 256GB RAM; typically 175+ GB free during training; MPS (Apple Silicon)

## Experiments Currently Running on Mac Studio (2026-04-03)
- **step29c** (PID 89895): mechanisms_calibrated — e10 of ~80; calibrating AntiHebb/phase_exc/interneurons/fast_W_phase on calibrated base
- **step48** (PID 91447): kiter_sweep_d64 — e50/150; K_iter={8,12,16,24,32} ± AntiHebb
- **step54** (PID 95059): warm_restart_lr — e140/150; LR schedule comparison (nearly done)

## Experiments Ready to Dispatch (local scripts written)
- **step55**: `scripts/train_step55_spatial_grouped_input.py` + `src/sgnnet/model_spatial_grouped.py`
  - Needs rsync of `model_spatial_grouped.py` to Mac Studio before launch

</specifics>

<deferred>
## Deferred to Phase 6

- Full PCA compression sweep (k ∈ {64, 128, 256, 512, 1024, 2048}) — Plan 5.4 covers PCA input only at best k
- Comparative report vs VGG16 dense baseline — Phase 6 deliverable
- Transformer integration / SGNNET as FFN replacement — future milestone beyond v1

</deferred>

---

*Phase: 05-scalable-architecture-experiments*
*Context gathered: 2026-04-03 — synthesized from 7 LEARNINGS files, EXPERIMENT_QUEUE.md, results/ JSONs, and current Mac Studio state*
