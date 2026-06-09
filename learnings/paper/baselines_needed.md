# Baselines and Experiments Needed for Publication

**Last updated:** 2026-04-22
**Priority shift:** Paper 1 soft-conclusion. Only blockers proceed.

## BLOCKING (paper 1 cannot submit without)

### 1. Cross-modal validation (paper's core claim)
Paper positions SGNNET as a **universal classification head** replacing the FC layer across pre-trained feature extractors. Requires validation across modalities:

| Modality | Feature extractor | Feature dim | Task options | Status |
|---|---|---|---|---|
| Vision (CNN) | VGG16 conv | 25088 | Imagenette 10-class | ✅ 95.52% @ 0.98M routing MACs |
| Vision (CNN) | VGG16 conv | 512 | CIFAR-10 10-class | ⚠️ −5.55pp vs linear at N=2048 (step882); −2.66pp at N=8192 (step914) |
| Text | DistilBERT (66M) | 768 CLS | SST-2, AG News, IMDB | ❌ NEGATIVE (step407/410 — trails linear probe, scope = vision-only) |
| Audio | Whisper-tiny (39M) | 384 | Speech Commands, ESC-50, UrbanSound8K | ❌ NEGATIVE structural (step926–961 — K_in/N tuning insufficient; step962 VGG isolation pending) |
| LLM | Qwen2.5-0.5B (494M, h=896) | 896 last-token | text cls (SST-2 via LLM) | ❌ DEFERRED (text gap confirmed negative) |

For each modality: feature extraction → SGNNET classifier → report {accuracy, params, FLOPs} vs {linear probe, MLP at same param budget, standard FC head}.

### 2. Matched-FLOPs baselines (existing data mostly analytical)
- [x] Same-params baselines done (step401: Lin=96.92%, MLP_64=97.20%, MLP_2=46.98%, MLP_3=45.91%, RandProj=10.04%)
- [x] **Matched-FLOPs MLP**: step403b h=37 → 98.09% @ 1.857M FLOPs, 928K params (26× more params than SGNNET step605). MLP wins +2.14pp but loses params by 26×.
- [ ] **Matched-FLOPs pruned VGG FC**: prune FC to reach ≤1M MACs, report retained accuracy

## STRONG-TO-HAVE (makes paper more rigorous)

### 3. Random features + linear classifier baseline
Isolates the contribution of iterative routing. Feed same random W_pos projections into a plain linear classifier (no K_iter loop). If linear ≈ SGNNET → routing contributes nothing and the paper loses its main mechanism claim. If SGNNET >> linear → confirms routing is the value-add. **Note:** step401 SGNNET_RandProj = 10.04% covers part of this (random fixed W_pos in SGNNET architecture), but not "random + vanilla linear classifier" baseline.

### 4. Standard GNN baselines (GCN / GAT / GIN)
**DONE — step404.** GCN=48.9%, GAT=48.7%, GIN=15.3% at ~35K params on Imagenette (VGG16 features).
SGNNET step605 (34,976 params) = 95.95% → +47pp over GCN/GAT at matched params. Paper-grade result.

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

## Related work gaps (paper writing, no experiment needed)

### Neural gas / SOM citation (qwen attack 5, 2026-04-22)
Anti-Hebbian routing is related to neural gas (Martinetz & Schulten 1991) and Kohonen SOMs (1990) — repulsion-based neighbor update in feature space. Related work section must cite these and distinguish: SGNNET uses W_pos *geometric* repulsion on a graph topology, not feature-space winner-takes-all competition without message-passing.
**Required in paper:** 2-sentence differentiation in related work.

### JL framing softer (qwen attack 2, 2026-04-22)
K_in=25 from 25088 inputs is not a per-neuron JL embedding (JL guarantees d ≥ O(log n/ε²) for n data points). Claim should cite Achlioptas (2003) sparse JL and clarify the N=2048 ensemble provides the full random projection — not each neuron individually.
**Already fixed in claims.md (2026-04-22).**

## DEFERRED to future work (not paper 1)

- Theoretical analysis (random feature connection, graph signal processing)
- t-SNE/UMAP of neuron activations
- Connectivity pattern visualization
- Triton fused kernel (step530) — 4.7× from compile alone already exceeds paper claim
- ΔW proj × K_hh=4 compound (step730 drafted, not launched)
- Additional scale N=16384+ extensions
