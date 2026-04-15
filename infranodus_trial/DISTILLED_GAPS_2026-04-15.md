# DISTILLED GAPS — SGNNET Research
**Date:** 2026-04-15  
**State:** step266 record attempt (97.71% UNCONFIRMED — see Gap H6); step265 N=4096 K=4 T2 running; CIFAR-10 extraction running; all slots FREE

---

## Cross-Method Validation Table

**V1 corpus:** 62 learnings files + 169 result JSONs (full current corpus, fresh run 2026-04-15)  
**V2 corpus:** 26 files (prior run — pre-2026-04-14 sessions only)  
**V3:** LLM synthesis on full current context

| Gap | V1 (co-occ, fresh 62 files) | V2 (TF-IDF, older 26 files) | V3 (LLM) | Priority |
|-----|-------------|-------------|----------|----------|
| GNN/message-passing baselines absent | **STRONG SIGNAL: cluster [10] "message, passing, style, residual, diffusion" fully isolated from ALL other clusters (density 0.005-0.013 to every other cluster)** | [8] sgnnet↔[3] iter/verdict weak | Explicit blocker | **HIGH** |
| Cross-dataset / cross-modal missing | [5] sgnnet/training/paper↔[10] message-passing weak | [4] ffn/model↔[3] iter/verdict weak | Explicit blocker | **HIGH** |
| Matched-FLOPs MLP missing | [7] accuracy/baseline↔[3] iter/FLOPs weak | [3] iter↔[4] model/sparsity weak | Explicit blocker | **HIGH** |
| Param count inconsistency | [4] proj/warm↔[9] scaling/arch weak | [8] sgnnet/arch↔[3] verdict weak | Explicit (34K vs 67K) | **HIGH** |
| ΔW rot × K=4 combo untested | not captured (recent steps) | [3] iter↔[10] alpha/variant | V3 Gap 2.1 | **HIGH** |
| 97.71% provenance unconfirmed | not captured | not captured | V3 tension T2 | **HIGH** |
| Wall-clock K=4 not measured | not captured | not captured | V3 Blocker-7 | **HIGH** |
| INT8 QAT not started | not captured | not captured | V3 Gap 2.3 | **MEDIUM** |
| Edge-β scalar untested | [0] topology↔[2] sparse/attention gap | [10] dynamic↔[1] routing | V3 Gap 2.5 | **MEDIUM** |
| PyG/torch_scatter baseline | **Cluster [10] message/passing/residual isolated** | [13] encoding/gather weak | V3 Gap 2.4 | **MEDIUM** |
| Muon at current config | not captured | not captured | V3 Gap 2.6 | **MEDIUM** |
| Deep supervision post-mortem | [8] loss/norm/safety isolated | [11]↔[5] rebuild/bug | V3 section 4 T3 | **LOW** |
| AH static framing in manuscript | not captured | not captured | V3 tension T4 | **MEDIUM** |

**Cross-method agreement:** V1 and V3 strongly agree on GNN/message-passing gap (V1 finds it as the most isolated cluster in the full corpus; V3 identifies it as a paper blocker). V1 also picks up that "message passing" concepts are disconnected from the paper/training/evaluation discourse — reviewers will notice the same structural hole. V2 (older corpus) misses this because the ΔW proj experiments that cemented the efficiency story weren't in scope yet.

---

## Priority Tiers

---

### HIGH (Paper-Blocking — must be done before submission)

---

#### H1 — Cross-dataset: CIFAR-10 VGG16 features (step401b)
**What's missing:** SGNNET training on CIFAR-10 VGG16 features (50K samples, 512-dim pool5).  
**Why blocking:** Paper's core claim is "universal FC replacement." One dataset is insufficient.  
**Dependency:** store_cifar10.h5 extraction (currently running on studio_cpu).  
**Experiment:**
```
Step: step401b
Config: N=2048, D=16, K_hh=2, K_iter=5, ΔW proj (efficiency config)
Data: CIFAR-10 VGG16 features (store_cifar10.h5)
Baselines: Linear probe + MLP_64 at same params
Tier: T2 (150ep, 100% data)
Slot: 5060ti_cuda (first free after store_cifar10.h5 done)
Script: copy scripts/train_step235_dwrot_aug_imagenette.py → train_step401b_cifar10.py, swap dataset loader
Expected: ≥90% (CIFAR-10 is easier than Imagenette with VGG features)
Duration: ~2 hours (T2)
```

---

#### H2 — Cross-modal: DistilBERT text classification (step405)
**What's missing:** ANY non-vision modality result.  
**Why blocking:** Without cross-modal, paper cannot claim "universal FC replacement."  
**Experiment:**
```
Step: step405
Config: N=2048, D=16, K_hh=2, K_iter=5, ΔW proj
Data: DistilBERT CLS embeddings on SST-2 (768-dim input) OR AG-News (768-dim, 4 classes)
Baselines: Linear probe (1-layer) + MLP_64 at matched params
Tier: T2 (150ep) — SST-2 has 67K train samples, fast
Slot: studio_cpu or 5060ti_cuda
Script: new train_step405_distilbert_sst2.py using pre-extracted features
Duration: ~4 hours including feature extraction
Expected: competitive with linear probe (if ΔW proj works at 768-dim, paper claim stands)
Note: N_IN must change from 25088 → 768; verify seed gather/projection scales correctly
```

---

#### H3 — Param count reconciliation (bench vs training)
**What's missing:** Explanation of why bench reports 34,976 params vs training's 67,744.  
**Why blocking:** Two conflicting numbers in the paper will trigger reviewer rejection.  
**Resolution (30-minute task, no training needed):**
```
Step: (no step number — diagnostic task)
Action: python3 -c "
from src.model import SGNNET_DeltaAH  # or whichever class bench uses
m = SGNNET_DeltaAH(n=2048, d=16, k_hh=2, k_iter=5, k_in=25)  # bench config
print('bench config params:', sum(p.numel() for p in m.parameters()))
m2 = SGNNET_DeltaAH(n=2048, d=16, k_hh=2, k_iter=5, k_in=50)  # training config
print('training config params:', sum(p.numel() for p in m2.parameters()))
"
Root cause: bench likely uses K_in=25 (not 50). 25*25088 seed proj = 627200 dim... 
Actually: diff=32,768 = 2048*16 = N*D. Likely conn_in buffer or W_pos not counting properly.
Log result in LEARNINGS_ops.md
```

---

#### H4 — Matched-FLOPs MLP baseline (step403b)
**What's missing:** MLP sized to exactly 1.85M FLOPs (SGNNET true FLOPs, ncu-validated).  
**Why blocking:** Reviewer will ask "why not just use a bigger MLP at the same FLOPs?"  
**Calculation:** MLP(25088→h→10). FLOPs = 2*(25088*h + h*10) ≈ 50176h = 1.85M → h ≈ 37. MLP_37 is the fair comparison.  
**Experiment:**
```
Step: step403b
Config: MLP(25088→37→10), same training protocol as step401
Tier: T2 (150ep)
Slot: any CPU slot, ~1 hour
Script: modify scripts/train_step401_paper_baselines.py → add MLP_37 variant
Expected: ~50-60% (MLP_3/2 already collapsed at 75K params; 37-node hidden is tiny)
Paper claim when done: "SGNNET at 1.85M FLOPs achieves 97.30% vs MLP at same FLOPs: ~55%"
```

---

#### H5 — GNN baselines: GCN + GAT at matched params (step404)
**What's missing:** Standard GNN head comparison on same pipeline.  
**Why blocking:** SGNNET is a GNN. Reviewers WILL ask "vs GCN/GAT?"  
**Experiment:**
```
Step: step404
Configs:
  A: GCN head (67K params, 2-layer) on VGG16 features — how to construct graph from 25088-dim? 
     Use: N=2048 neurons as nodes, features = scatter-sum of input (same as SGNNET seed gather), 
     then standard GCN message passing with 2 layers
  B: GAT head (67K params, multi-head) same construction
  C: GIN (67K params) for comparison
Tier: T2
Slot: 5060ti_cuda
Duration: ~3 hours
Note: Graph construction is the key challenge — use same conn_hh as SGNNET for fairness
```

---

#### H6 — Confirm 97.71% provenance (step266 fix + rerun)
**What's missing:** step266 JSON shows a crash at ep1 (top1=23.2%). The 97.71% number from the session headline has no confirmed JSON backing.  
**Why blocking:** Paper headline number must have a result JSON. Cannot submit with a number you can't reproduce.  
**Resolution:**
```
Step: step266b (or identify the actual run)
Action 1: Check session logs for where 97.71% was observed. Search: grep -r "97.71" results/ scripts/ learnings/
Action 2: Debug step266 crash — likely K=4 conn_hh mismatch at N=4096 (K_hh defaults differ)
Action 3: Re-run corrected config
Config: N=4096, K=4, ΔW rot, aug, Tier-2 — requires step265 to finish first (K=4 validated at T2)
Tier: T2
Slot: 5060ti_cuda
Duration: ~4 hours
```

---

#### H7 — Wall-clock K=4 direct measurement (bench_step830)
**What's missing:** K=4 latency is currently "projected at 0.224ms = 0.280ms × 0.8" — not measured.  
**Why blocking:** Paper table with a projected number is unacceptable.  
**Experiment:**
```
Step: bench_step830
Action: Add K_iter=4 variant to bench_step811 script, measure on 5060ti
Expected: ~0.224ms (20% reduction from K=5's 0.280ms)
Slot: 5060ti_cuda, ~30 minutes
Script: modify scripts/bench_step811_compile_modes.py → add K=4 config
```

---

### MEDIUM (Strong paper-impact if explored)

---

#### M1 — ΔW rotation + K=4 at N=4096 (step267)
**What's missing:** The "all winners" combo at N=4096 — rot + aug + K=4. Step266 crashed; needs a fixed rerun.  
**Why it matters:** K=4 wins +0.97pp at N=4096 (step263 seed=42). Rot wins +0.64pp (step729). Combined could approach or exceed 97.86% D=64 record — at D=16 scale with 19× fewer FLOPs.  
**Experiment:**
```
Step: step267
Config: N=4096, D=16, K_hh=4 (scaling rule for N=4096), K_iter=4, ΔW rot, aug=True
Base on: step265 (when done) × step729 results
Tier: T1 first (20ep scout to verify no crash), then T2
Slot: 5060ti_cuda
Duration: T0 scout ~1h, T2 ~4h
Script: base on train_step265_kiter4_n4096_t2.py + add rot + aug
Expected: 97.5-98.0% (if both mechanisms fully compound)
Gating condition: step265 must finish first (need confirmed K=4 T2 base)
```

---

#### M2 — ΔW proj + aug at N=2048 K=4 (step268)
**What's missing:** Augmentation + K=4 combo at the primary efficiency config.  
**Why it matters:** Aug adds +0.33pp (step235); K=4 saves 20% compute. Both benefits at once.  
**Experiment:**
```
Step: step268
Config: N=2048, D=16, K_hh=2, K_iter=4, ΔW proj, aug=True
Base on: step262 (K=4 T2 confirmed tie), step235 (aug confirmed +0.33pp)
Tier: T1 (75ep 50% data first)
Slot: any slot
Duration: T1 ~2h
Script: modify train_step262_kiter4_dwproj_tier2.py → add aug=True
Expected: ~97.0-97.3% (rot+aug was 97.30%; proj is slightly below rot at N=2048)
```

---

#### M3 — Edge-β scalar on frozen topology (step524-S1)
**What's missing:** Learnable per-edge scalar weights — the only untested non-discrete dynamic element.  
**Why it matters:** All discrete/stochastic topology edits killed. If S1 (β scalar, NO topology change) works, it opens "learned edge attention" as a viable mechanism and closes the dynamic element direction definitively either way.  
**Experiment:**
```
Step: step524-S1
Config: N=2048 (or 512), K_hh=2, K_iter=5, ΔW proj, + β[N,K_hh] scalar initialized to 1.0
Mechanism: Z_nb *= (1 + β[h,k].tanh())   # soft edge gate, always ≥0 at β=0
Tier: T0 (20ep scout, N=512 for speed)
Slot: CPU or MPS
Duration: ~30min
Script: new train_step524_edge_shift.py per LEARNINGS_design_2026_04_15.md S1 design
Expected: Neutral or small +pp; if positive, advances to T1
```

---

#### M4 — INT8 QAT inference impact (step526)
**What's missing:** Quantization-aware training test — script already written.  
**Why it matters:** INT8 would give 4× memory reduction + potential 2× CUDA throughput on consumer GPUs, making SGNNET deployable on edge devices. Paper claim: "SGNNET can be quantized to INT8 with ≤0.5pp accuracy loss due to L2-normalized D=16 bounded range."  
**Experiment:**
```
Step: step526
Config: N=2048, D=16, K_hh=2, K_iter=5, ΔW proj
Part A: fp32 train → quant eval (saturate/modular/crt × {W only, W+Z})
Part B: QAT from scratch (fake-quant+STE, grad_accum ∈ {1,4,16,64})
Tier: T0 scout
Slot: 5060ti_cuda (script ready: train_step526_int8_qat.py)
Duration: ~2h
```

---

#### M5 — PyG torch_scatter baseline (bench_step832)
**What's missing:** PyTorch Geometric edge-list routing benchmark.  
**Why it matters:** GNN reviewers will ask "why not use PyG scatter_add?" Missing comparison = weak section 4.  
**Experiment:**
```
Step: bench_step832
Action: pip install torch-scatter on 5060ti; implement SGNNET routing as edge-list message passing;
        benchmark vs V2 max-autotune 0.280ms
Slot: 5060ti_cuda
Duration: ~3h (install + implementation + bench)
Expected: 2-3× faster if BW-bound; neutral if compute-bound at K=5
```

---

#### M6 — Muon optimizer convergence speed (step522)
**What's missing:** Muon benchmark at current config (script ready).  
**Why it matters:** "Same accuracy at 30% faster convergence" is a paper-worthy training efficiency claim. If Muon wins on speed, it changes the paper's "training cost" narrative.  
**Experiment:**
```
Step: step522
Config: N=2048, K=5, ΔW proj, Muon vs AdamW
Metric: BOTH final accuracy AND epochs-to-95%
Tier: T1 (75ep)
Slot: 5060ti_cuda (requires: pip install muon-optimizer)
Duration: ~2h
Script: train_step522_muon_optimizer.py (queued, ready)
```

---

#### M7 — AH static framing correction in manuscript
**What's missing:** The manuscript (MANUSCRIPT_DRAFT.md) likely still describes AH as providing "input-dependent routing." The 2026-04-15 correction establishes: AH = W_pos-only (static post-training); ΔW proj = activation-dependent (true per-input gate).  
**Action (no experiment needed):**
```
Step: (manuscript edit)
Edit: MANUSCRIPT_DRAFT.md + PAPER_OUTLINE.md + claims.md
Change: Everywhere "AH provides input-dependent" → "AH provides static learned suppression"
        Everywhere "input-conditional routing" → attribute to ΔW proj, not AH
Duration: ~30 minutes
```

---

### LOW (Interesting but not critical for paper 1)

---

#### L1 — Step521 deep supervision post-mortem write-up
**What's missing:** step521 killed all deep supervision (−5.5pp), but the failure mechanism is interesting: K_iter intermediate states are NOT independently meaningful until the final iteration. This is a strong negative result worth documenting as a paper claim.  
**Action:** Update LEARNINGS_design_2026_04_15.md with the interpretation from Tension T3.

---

#### L2 — DropMessage at efficiency config
**What's missing:** step158 killed DropMessage at N=1024. Never tested at N=2048 efficiency config.  
**Why low priority:** 2/2 experiments killed this. Pattern is stable.  
**Action:** Skip unless reviewer asks specifically.

---

#### L3 — DistilBERT + AG-News (4-class) for diversity
If step405 (SST-2 binary) works, add AG-News as a second text dataset for diversity. Post-paper-1 scope.

---

#### L4 — K_hh=4 × ΔW proj at N=2048 (step730 follow-up)
**What exists:** step730 done — ΔW proj generalizes to K_hh=4 at N=2048 (+1.83pp, but K_hh=2+proj is still higher at 95.40% mean). K_hh=4 with rotation NOT tested.  
**Action:** step731 — K_hh=4 + ΔW rot at N=2048. Tier-0 scout only.

---

## Queue-Ready Entries

Paste these directly into EXPERIMENT_QUEUE.md under "User-proposed 2026-04-15":

```markdown
| **step267** | **ΔW rot + aug + K=4 at N=4096 (step266 crash fix)** — Debug conn_hh mismatch,
  rerun as T0 scout first. Config: N=4096 K_hh=4 K_iter=4 ΔW-rot aug=True.
  Gated on: step265 T2 completion + step266 crash root cause.
  Slot: 5060ti_cuda. Expected: 97.5-98.0%. | TODO (gated) |

| **step268** | **ΔW proj + aug at N=2048 K=4** — aug+0.33pp (step235) × K=4 tie (step262)
  combo. Config: N=2048 K_hh=2 K_iter=4 ΔW-proj aug=True Tier-1.
  Slot: any. Expected: ~97.0-97.3%. | TODO |

| **step401b** | **CIFAR-10 cross-dataset SGNNET (PAPER-BLOCKER)** — N=2048 K=5 ΔW-proj
  on store_cifar10.h5 VGG16 features. Gated on extraction completing.
  Add Linear + MLP_64 baselines. Tier-2.
  Slot: 5060ti_cuda. | TODO (gated on extraction) |

| **step404** | **GNN baselines: GCN + GAT + GIN at 67K params (PAPER-BLOCKER)** —
  same VGG16 features → graph construction (N=2048 nodes, conn_hh edges) → GNN head.
  Tier-2. Slot: 5060ti_cuda. | TODO |

| **step405** | **Cross-modal: DistilBERT + SST-2 (PAPER-BLOCKER)** — extract
  DistilBERT CLS embeddings (768-dim), train SGNNET N=2048 K=5 ΔW-proj.
  Add Linear + MLP_64 baselines. Tier-2.
  Slot: studio_mps or 5060ti_cuda. | TODO |

| **step403b** | **Matched-FLOPs MLP baseline (PAPER-BLOCKER)** — MLP(25088→37→10) at
  1.85M FLOPs = SGNNET true FLOPs. Add to step401 result table.
  Slot: any CPU. ~1h. | TODO |

| **bench_step830** | **K=4 direct wall-clock measurement (PAPER-BLOCKER)** —
  add K_iter=4 variant to bench_step811, measure 0.2xxms directly.
  Slot: 5060ti_cuda. ~30min. | TODO |

| **param_reconcile** | **Param count 34,976 vs 67,744 root cause (PAPER-BLOCKER)** —
  diagnostic only, compare bench model construction vs training model construction.
  Expected: K_in=25 vs K_in=50 or conn_in buffer not counted.
  No slot needed, ~30 min local. | TODO |
```
