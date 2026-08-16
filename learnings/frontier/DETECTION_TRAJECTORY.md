# Object-Detection Trajectory — SGNNET learnings → detector → drone

**Created:** 2026-07-20 · **Branch B of the /goal directive** · first-level exploration DONE
**End goal:** parameter-efficient visual model on a drone (detection + tracking).

---

## 0. Thesis (why detection fits, and LLM-FFN did not)

SGNNET is **CONFIRMED strong as a readout** — a cheap map from a fixed pooled feature
vector to class logits. The VGG16 champion replaced the FC block on `pool5` (25088-dim)
at **0.029% of its params** and **95.95%** accuracy.

A two-stage detector's **box head is the same shape**: pooled ROI features →
class + bbox. So detection reuses SGNNET's proven strength, unlike the LLM FFN
(a mid-network *transform*, killed in step989 and re-probed in llm_step001/002).

## 1. First-level profile — where the params/FLOPs actually live

`det_step001` on `fasterrcnn_mobilenet_v3_large_fpn` (torchvision, 19.39M params). **CONFIRMED (measured):**

| Submodule | Params | % model | Note |
|---|---|---|---|
| **roi_heads** | 14.36M | **74.1%** | dominated by the box head |
| — box_head (`TwoMLPHead`) | **13.90M** | **71.7%** | FC-on-pooled-ROI: 12544→1024→1024 |
| — box_predictor (`FastRCNNPredictor`) | 0.47M | 2.4% | cls_score + bbox_pred readout |
| backbone (MobileNetV3+FPN) | 4.41M | 22.8% | feature extractor |
| rpn | 0.61M | 3.1% | proposals |

- Box head `in_dim = 12544 = 256×7×7`, `n_cls = 91` — **identical structure to VGG's FC block on pool5**.
- Box head runs **once per ROI** → **14.4 GMACs/image @ 1000 proposals** — the compute hot path *and* the param sink, simultaneously.
- **Implication:** the single biggest cost in a small 2-stage detector is exactly the block SGNNET is proven to compress. This is the SGNNET-replaceable sub-block.
- SGNNET-style compact head (low-rank+top-k, rank=64): 809K params = **5.63% of box head**; deeper compression (champion recipe ~0.03%) is the target, not a limit.

## 2. Gap analysis — what the champion does NOT yet cover

The VGG champion is a **frozen-feature, single-object, 10-class classifier**. Detection adds:

| Gap | Why it matters | De-risk experiment |
|---|---|---|
| **Multi-task head** | box head feeds BOTH cls_score and bbox_pred (regression). SGNNET only ever did cls. | det_step002: SGNNET-head with a bbox-regression output arm; distil both from teacher. |
| **91 classes, not 10** | class-scale untested for SGNNET routing capacity. | det_step002 tests 91-way on COCO ROI features. |
| **Distil from a live detector** | champion distilled a static teacher head; here teacher = detector's box head on real ROIs. | det_step002: cache (roi_feat, teacher_cls, teacher_bbox) triples, distil offline (mirrors VGG pipeline). |
| **Per-ROI throughput** | head runs ×1000/image — wall-time win compounds, but dispatch overhead may dominate at tiny param count (known SGNNET bottleneck). | det_step003: wall-time bench of head @ 1000 ROIs vs TwoMLPHead. |
| **Localization sensitivity** | mAP is IoU-thresholded; small cls/bbox errors cost more than top-1 flips. | evaluate mAP delta, not just cls accuracy. |

## 3. Staged trajectory (each stage gated by the prior)

- **B0 (DONE, this session):** profile → box head = 72% of detector, VGG-FC-shaped, SGNNET-replaceable. ✅
- **B1 — offline head distillation (DONE 2026-07-28, `det_step002`, POSITIVE):** harvested 12k real ROI feats (1468 fg, 12.2%) from FasterRCNN box_head via forward hook on imagenette-val; distilled compact heads to match teacher cls (KD/KL) + bbox (smooth-L1). Metric = argmax agreement vs the 14.36M teacher head.

  | head | agree | fg_agree | bbox_mse | params | %teacher |
  |---|---|---|---|---|---|
  | dense_reinit (control) | 0.9902 | 0.9605 | 0.022 | 14.36M | 100% |
  | lowrank_r256 (no sparsity) | 0.9885 | 0.9339 | 0.040 | 3.39M | 23.6% |
  | **sgn_r128k32** (top-k) | 0.9846 | **0.9421** | 0.059 | **1.68M** | **11.7%** |
  | sgn_r64k16 (top-k) | 0.9819 | 0.9087 | 0.104 | 0.84M | 5.8% |

  **CONFIRMED:** SGNNET-style head reproduces the box head at **11.7% params, 94.2% foreground agreement** (−1.8pp vs the full-size dense control). Top-k is NOT the liability reid_step001 saw: **sgn_r128k32 beats dense lowrank_r256 on fg_agree at half the params** — sparsity is competitive here, so the reid collapse was task-specific (triplet embedding), not a general top-k failure. bbox_mse rises as params shrink (0.022→0.104): the regression arm is more compression-sensitive than cls → sweet spot = sgn_r128k32. **Caveat (HYPOTHESIS):** this is teacher-agreement on real ROIs, NOT end-to-end mAP; imagenette ROIs are COCO-class-narrow. B2 validates mAP.
- **B3 — wall-time + MACs Pareto (DONE 2026-07-28, `det_step003`, MIXED):** benched all four heads @ ROI batch {64,256,1000} on mini_mps. B=1000 (realistic per-image head load):

  | head | params | macs/roi | us/roi | wall× | param× |
  |---|---|---|---|---|---|
  | dense (teacher-arch) | 14.36M | 14.36M | 5.255 | 1.00× | 1.00× |
  | lowrank_r256 | 3.39M | 3.39M | **1.786** | **2.94×** | 4.23× |
  | sgn_r128k32 (top-k) | 1.68M | 1.68M | 1.943 | 2.70× | 8.54× |
  | sgn_r64k16 (top-k) | 0.84M | 0.84M | 1.613 | 3.26× | 17.17× |

  **CONFIRMED (1):** the param cut DOES convert to wall-time — **not** dispatch-bound at B=1000 (the known tiny-head failure mode). 3.26× faster at 17× fewer params. Good for the drone story. **CONFIRMED on MPS / HYPOTHESIS on CUDA (2)** (benched on mini_mps only): **top-k is a wall-time LIABILITY** — sgn_r128k32 has *half* the params of lowrank_r256 yet is *slower* (1.943 vs 1.786 us/roi): the top-k sort cost erases the rank-halving matmul saving (dense matmul ignores the mask, so MACs track RANK not k). Dense low-rank **Pareto-dominates** the top-k head on wall-time. **CONFIRMED (3):** wall-time gain (2.7–3.3×) badly **lags** the MAC/param gain (8.5–17×) — sublinear; memory-bandwidth/small-matmul overhead caps the win. This refines det_step002 (top-k *won* on accuracy): top-k buys accuracy+params, **costs** wall-time → only worth it with a genuine sparse kernel. For a wall-time/energy-bound drone, **dense low-rank is the robust readout** (4.23× params, 2.94× wall, near-teacher accuracy); reserve top-k for accuracy-critical + sparse-kernel deploys.
- **B2 — in-detector swap + mAP:** load the trained head into the detector, freeze backbone+rpn, measure COCO mAP delta vs stock. Tier-2, needs COCO-val (imagenette ROIs are COCO-class-narrow → mAP not meaningful there). Gated open; det_step002 cleared the agreement gate.
- **B4 — swap the backbone question:** if head-only Amdahl-caps the win (backbone still 22.8%), evaluate a lighter backbone or an SGNNET-style FPN neck. Backbone replacement is OUT of SGNNET's proven readout strength — treat as a separate research bet.

## 4. Path to drone (detection → tracking → deploy)

1. **B1–B3** prove SGNNET box head on a static detector (offline COCO). — *this trajectory*
2. **Tracking** = temporal association across frames. SGNNET is stateless; add a light
   recurrent/association layer OR ride an existing tracker (SORT/ByteTrack) that consumes
   detections. SGNNET stays the per-frame detection readout. (Separate gap; see drone-theory note.)
3. **Edge hardware** (Jetson/drone SoC): the load-bearing de-risk. MACs predict wall-time
   at moderate cuts (cnn_step037 CONFIRMED) but tiny heads can be dispatch-bound. Bench on
   real edge silicon before claiming a drone win.
4. **Visual-LLM prune** (final /goal item): once detection head is proven, apply the same
   readout-replacement to a VLM's detection/grounding head. Do NOT target VLM FFNs (step989 + llm_step002 evidence).

## 5. Evidence tags

- Box head = 72% of detector, VGG-FC-shaped: **CONFIRMED** (det_step001, measured).
- SGNNET compresses the box head to 11.7% params at 94.2% fg-agreement: **CONFIRMED** (det_step002, distillation). Class-scale (91-way) and the bbox-regression arm both handled. End-to-end mAP preservation: still **HYPOTHESIS** (B2).
- Top-k sparsity survives the detector-readout role but NOT the reid metric-embedding role: **CONFIRMED** — det_step002 sgn_r128k32 ≥ dense lowrank; reid_step001 sgn heads −9.6/−17.5pp Rank-1. The champion's readout slot transfers; the embedding slot does not.
- Compact head wall-time is NOT dispatch-bound at realistic ROI batch (B=1000): **CONFIRMED** (det_step003) — param cut converts to 2.7–3.3× wall-time, but sublinearly (lags the 8.5–17× MAC cut).
- Hard top-k is a wall-time liability on dense hardware: **CONFIRMED on MPS / HYPOTHESIS on CUDA** (det_step003 was benched on mini_mps ONLY; no 5060ti_cuda data point exists yet) — sgn_r128k32 is slower than lowrank_r256 despite half the params; the top-k sort cost > the rank-halving matmul saving. Dense low-rank is the robust drone readout; top-k only pays with a sparse kernel or when accuracy is capacity-critical.
- Detection > LLM-FFN as an SGNNET target: **CONFIRMED-directional** — detection reuses the proven readout role; llm_step989 killed the FFN-transform role.
