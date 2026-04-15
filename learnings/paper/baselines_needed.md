# Baselines and Experiments Needed for Publication

**Last updated:** 2026-04-14
**Priority shift:** Paper 1 soft-conclusion. Only blockers proceed.

## BLOCKING (paper 1 cannot submit without)

### 1. Cross-modal validation (paper's core claim)
Paper positions SGNNET as a **universal classification head** replacing the FC layer across pre-trained feature extractors. Requires validation across modalities:

| Modality | Feature extractor | Feature dim | Task options | Status |
|---|---|---|---|---|
| Vision (CNN) | VGG16 conv | 25088 | Imagenette 10-class | ✅ 95.52% @ 0.98M routing MACs |
| Text | DistilBERT (66M) | 768 CLS | SST-2, AG News, IMDB | ⏳ PENDING |
| Audio | Whisper-tiny (39M) | 384 | Speech Commands, ESC-50, UrbanSound8K | ⏳ PENDING |
| LLM | Qwen2.5-0.5B (494M, h=896) | 896 last-token | text cls (SST-2 via LLM) | ⏳ PENDING |

For each modality: feature extraction → SGNNET classifier → report {accuracy, params, FLOPs} vs {linear probe, MLP at same param budget, standard FC head}.

### 2. Matched-FLOPs baselines (existing data mostly analytical)
- [x] Same-params baselines done (step401: Lin=96.92%, MLP_64=97.20%, MLP_2=46.98%, MLP_3=45.91%, RandProj=10.04%)
- [ ] **Matched-FLOPs MLP**: MLP sized to equal 0.98M routing MACs (or 1.85M true MACs). Can compute analytically.
- [ ] **Matched-FLOPs pruned VGG FC**: prune FC to reach ≤1M MACs, report retained accuracy

## STRONG-TO-HAVE (makes paper more rigorous)

### 3. Random features + linear classifier baseline
Isolates the contribution of iterative routing. Feed same random W_pos projections into a plain linear classifier (no K_iter loop). If linear ≈ SGNNET → routing contributes nothing and the paper loses its main mechanism claim. If SGNNET >> linear → confirms routing is the value-add. **Note:** step401 SGNNET_RandProj = 10.04% covers part of this (random fixed W_pos in SGNNET architecture), but not "random + vanilla linear classifier" baseline.

### 4. Standard GNN baselines (GCN / GAT / GIN)
SGNNET is a graph-neural-network. Should compare to established GNNs at matched params/FLOPs on the same pipeline (VGG16 features → GNN head → classification). Reviewers will ask.

## DONE (no longer needed)

- [x] N-scaling curve — step72, step402, step402a (log-linear confirmed N=256→8192)
- [x] Ablation: F.normalize (step129/320 CONFIRMED load-bearing −11pp to −71pp)
- [x] Ablation: AH (step321 CONFIRMED load-bearing, α=0 collapses to 18.78%)
- [x] Ablation: W_pos learned vs random (step401 CONFIRMED +81pp delta)
- [x] Ablation: K_iter sequential vs parallel (step700/701 CONFIRMED load-bearing)
- [x] Ablation: ΔW relational axis (step708 CONFIRMED; random direction = chance)
- [x] Pareto frontier: multiple N/D/K configs (step192/193/195/199/205 all on frontier)
- [x] CIFAR-10 on raw pixels (step400 CONFIRMED design flaw — needs feature extractor; addressed by cross-modal plan)
- [x] CUDA throughput: 4.7× faster than VGG FC with torch.compile (step500)
- [x] ncu-validated true FLOPs: 1.85M = 0.75% of VGG16 FC (step800)

## DEFERRED to future work (not paper 1)

- Theoretical analysis (random feature connection, graph signal processing)
- t-SNE/UMAP of neuron activations
- Connectivity pattern visualization
- Triton fused kernel (step530) — 4.7× from compile alone already exceeds paper claim
- ΔW proj × K_hh=4 compound (step730 drafted, not launched)
- Additional scale N=16384+ extensions
