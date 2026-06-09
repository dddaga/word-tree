# CNN Compress Line — Queue

| Step | What | Tier | Slot | Status |
|---|---|---|---|---|
| cnnc_step001 | Multi-branch full-context CNN vs single-branch control, Imagenette | T0 | mini_mps (scheduler) | DONE |
| cnnc_step002 | Iso-MAC ablation: B/C scaled to ~224M MACs vs step001 Ref (+Ref_deep depth control) | T0 | mini_mps (scheduler) | QUEUED |

## cnnc_step001 details
Script: `scripts/cnn_compress/cnnc_step001_multibranch_t0.py` (models in `models_step001.py`).
Submitted 2026-06-10 03:54 (`.scheduler/pending/20260610_035409_...`). 20ep, 50% data, seed 42, CE only (no distillation).
**Slot note:** raw imagenette2-320 exists only on mini — 5060ti remote (`/home/indra/sgnnet_bench/data`) has .h5 stores only. Hence mini_mps, not 5060ti_cuda.

Configs (param-matched ~0.5M ±10%, smoke-verified counts):

| Config | Params | MACs (224px) | Design |
|---|---|---|---|
| Ref | 483,050 | 224.0M | single-branch plain downsampling stack |
| A_global | 509,850 | 210.4M | Ref + 16×-down global-context branch, channel-wise FC (Pathak 2016) |
| B_multibranch | 476,122 | 110.9M | 3 branches: full-res / 4×-down / 16×-down global |
| C_crelu | 463,930 | 94.1M | B + CReLU in first 2 convs of every branch (Shang 2016) |

Results → `results/cnn_compress/cnnc_step001_t0_seed42__mini_mps.json`.

## cnnc_step002 details
Script: `scripts/cnn_compress/cnnc_step002_isomac_t0.py` (models in `models_step002.py`,
reuses Branch/GlobalContextBranch/MultiBranchNet from models_step001).
Submitted 2026-06-10 04:19 (`.scheduler/pending/20260610_041951_...`). 20ep, 50% data,
seed 42, identical pipeline to step001. Ref NOT rerun — step001 best=0.7557 reused
(same seed/protocol); dRef reported against it.

Configs (MAC-matched 224M ±10%, smoke-verified; params free dimension):

| Config | Params | MACs (224px) | Design |
|---|---|---|---|
| Ref (step001) | 483,050 | 224.0M | reference, reused |
| B_isomac | 963,810 | 236.2M | B_multibranch widths ×1.44 |
| C_isomac | 1,385,302 | 233.3M | B_isomac + CReLU first 2 convs + 1 extra conv on local & mid |
| Ref_deep | 526,794 | 220.4M | depth control: Ref budget as 9-conv single branch |

Results → `results/cnn_compress/cnnc_step002_t0_seed42__mini_mps.json`.

## Results

### cnnc_step001 T0 (20ep, 50% data, mini_mps) — DONE 2026-06-10
| Config | Params | MACs(M) | Acc | dRef | s/ep |
|---|---|---|---|---|---|
| Ref | 483,050 | 224.0 | 0.7557 | — | 5 |
| A_global | 509,850 | 210.4 | 0.7269 | −2.88pp | 5 |
| B_multibranch | 476,122 | 110.9 | 0.7363 | −1.94pp | 4 |
| C_crelu | 463,930 | 94.1 | 0.7353 | −2.04pp | 4 |

Read (dual-axis): at iso-params, B/C lose ~2pp but at 2.0–2.4× fewer MACs —
Pareto-interesting, NOT a kill. A_global weak (−2.88pp, no MAC win) — reject
global-context branch as designed. Next: cnnc_step002 iso-MAC test (scale B/C
to ~224M MACs; do they beat Ref at matched compute?).
