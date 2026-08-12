# SGNNET Knowledge Index

**Last updated:** 2026-04-16
**Project best:** 97.86% (step89-A, N=4096, D=64, 150ep)
**Efficiency best:** 95.52% @ 0.98M routing MACs (step199); 96.69% K=5 teacher (step604); 96.36% K=1 KD student @ 5× routing reduction (step605)
**True FLOPs (ncu-validated):** 1.85M/sample = **0.75% of VGG16 FC** (step800). Paper must label "routing MACs" vs "total MACs".
**D=16 ceiling:** 97.17% (step205/209); **97.30% with K_in=15+aug compound** at N=4096/8192 (steps 279/282)
**Aug N-scaling:** scale-invariant +0.4–0.8pp at N=1024–8192 T2 (steps 269/273/276/280)
**K_in reduction:** 26.7× seed FLOP reduction (spatial precomp 16× × K_in 25→15 1.67×); universal at N>=4096 (step293)
**CUDA throughput:** 4.7× vs VGG FC training (step500); 5.26× seed speedup (bench_step831)
**Cross-modal:** Vision ✓, Text ✓ (SST-2 -1pp), Audio KILLED (ESC-50 -14pp), AG News running
**Current direction (2026-04-16):** Paper 1 packaging. Next: K=1+K_in=15 compound (step606 P0 for CUDA), aug T2 validation at N=16384 (step287), cross-modal AG News (step407).

**Updated findings log:** [paper/findings_log_part3.md](paper/findings_log_part3.md) — 2026-04-16 session results (steps 282–294, 604–605).

## Concept Pages

| Concept | File | Status | Key Finding |
|---|---|---|---|
| [[antihebbian]] | [concepts/antihebbian.md](concepts/antihebbian.md) | Confirmed winner | alpha=1.0 alone is Gen4 base; AH = anti-collapse (eff_rank confirmed step155) |
| [[gate_death]] | [concepts/gate_death.md](concepts/gate_death.md) | Theorem established | Multiplicative gates g^K_iter → 0; 8+ experiments confirm |
| [[n_scaling]] | [concepts/n_scaling.md](concepts/n_scaling.md) | Active research | N=4096 peak (97.86%); N=10000 regresses; non-monotonic |
| [[k_iter]] | [concepts/k_iter.md](concepts/k_iter.md) | N-dependent optimal | K_iter=16 at N=1024, K_iter=12 at N=4096 |
| [[phase_routing]] | [concepts/phase_routing.md](concepts/phase_routing.md) (INDEX) | Wave-1 killed; revival via redistribution | All gating variants dead; softmax redistribution promising |
| [[softmax_routing]] | [concepts/softmax_routing.md](concepts/softmax_routing.md) | First dynamic routing win | step73 D=86.34% (+1.78pp); step75 D=87.24% (+3.98pp) |
| [[group_topology]] | [concepts/group_topology.md](concepts/group_topology.md) | +3.01pp N=1024, NULL N=4096 | Bypasses N^2 routing problem via G^2 decisions; doesn't scale |
| [[normalization]] | [concepts/normalization.md](concepts/normalization.md) | LayerNorm winner pending N=4096 | LayerNorm +2.24pp vs L2 sphere (step116); RMSNorm −11.49pp KILLED |
| [[readout]] | [concepts/readout.md](concepts/readout.md) | Hard architectural constraint | C_ho required for unit-sphere activations; global mean-pool → 12% (step149) |
| [[architecture_dead_ends]] | [concepts/architecture_dead_ends.md](concepts/architecture_dead_ends.md) | Reference | All confirmed dead ends; external constraints + pruning KILLED (step152, step153) |
| [[delta_w]] | [concepts/delta_w.md](concepts/delta_w.md) | Confirmed winner (N≤2048) | ΔW projection +1.56pp at efficiency config; non-monotone peak +24pp at N=128; proj 3× cheaper than rot |
| [[sparse_bfs_routing]] | [concepts/sparse_bfs_routing.md](concepts/sparse_bfs_routing.md) | HYPOTHESIS | Beam-gated BFS frontier ≤M·K_hh^(K_iter-1)=176; preserves static conn_hh; reduces per-step cost |
| [[soft_routing_hnsw]] | [concepts/soft_routing_hnsw.md](concepts/soft_routing_hnsw.md) | HYPOTHESIS | Dense training (softmax) → HNSW inference (β-annealed); replaces conn_hh entirely; step855 scripted |
| [[activation_retention]] | [concepts/activation_retention.md](concepts/activation_retention.md) | HYPOTHESIS | Z_t = α·Z_{t-1} + route(...); recurrent SGNNET; step306 killed static/decay/norm_cons — new context needed |
| [[streaming_input]] | [concepts/streaming_input.md](concepts/streaming_input.md) | HYPOTHESIS | Cascading progressive seed; partial features at inference; latency-accuracy tradeoff |

## New Infrastructure (2026-04-10)

| Component | File | Purpose |
|---|---|---|
| Training diagnostics | `src/training/diagnostics.py` | Effective rank, neuron util, W_pos diversity, separability, grad norms |
| Checkpoint system | `src/training/checkpoint.py` | Save/resume full training state; best-model tracking |
| Paper materials | `learnings/paper/` | Novel findings, claims, baseline gaps, figure plans |

## Sequential Logs (Audit Trail)

| File | Covers | Key Steps |
|---|---|---|
| `LEARNINGS_phase5_p1_ladder.md` | D-scaling ladder | D=16→64 calibration |
| `LEARNINGS_phase5_p2_diagnostics.md` | Diagnostic tooling | Activation analysis |
| `LEARNINGS_phase5_p3_breakthrough.md` | AntiHebb breakthrough | step29, alpha calibration |
| `LEARNINGS_phase5_p4_d16.md` | D=16 experiments | Small-scale validation |
| `LEARNINGS_phase5_p5_phase_mechanisms.md` | Phase mechanisms | Phase excitatory, W_phase |
| `LEARNINGS_phase5_p6_gen2_experiments.md` | Gen2 experiments | Compound testing |
| `LEARNINGS_phase5_p6_gen2_gen3.md` | Gen2→Gen3 transition | Generation advancement |
| `LEARNINGS_phase5_p7_gen3_arm_status.md` | Gen3 ARM status | ARM track results |
| `LEARNINGS_phase5_p8_arm1_arm2.md` | ARM 1+2 results | Specific ARM outcomes |
| `LEARNINGS_phase5_p9_arm3_arm5.md` | ARM 3+5 results | Specific ARM outcomes |
| `LEARNINGS_phase5_p11_n_scaling.md` | N-scaling study | step56 N=512→10000 |
| `LEARNINGS_phase5_p12_proxwave.md` | Proximity wave | Distance-based mechanisms |
| `LEARNINGS_phase5_p13_reflection.md` | Reflection routing | alpha_reflect calibration |
| `LEARNINGS_phase5_p14_wave1_verdict.md` | Wave-1 verdict | All wave-1 mechanisms killed |
| `LEARNINGS_phase5_p15_post_wave1.md` | Post-wave-1 INDEX | TOC for p15a–p15e split files |
| `LEARNINGS_phase5_p15a_wave1_closure.md` | Wave-1 closure | steps 60-68 killed experiments |
| `LEARNINGS_phase5_p15b_arch_patch.md` | Arch patch | steps 66, 69, 70 — +9.83pp patch story |
| `LEARNINGS_phase5_p15c_kiter_flops.md` | K_iter + routing | steps 71, 73, 75-77, 79-83 |
| `LEARNINGS_phase5_p15d_flops_track.md` | FLOPs track | steps 86, 88 — Pareto + K_hh defaults |
| `LEARNINGS_phase5_p15e_dynamic_routing_postmortem.md` | Dynamic routing post-mortem | Full failure analysis + next directions |
| `LEARNINGS_phase5_p15f_warmstart_efficiency.md` | Warmstart/efficiency INDEX | TOC for part1/part2 split |
| `LEARNINGS_phase5_p15f_warmstart_efficiency_part1.md` | Warmstart/efficiency pt1 | steps 149–169: encoding killed, distillation, warm+W_proj, D=16/32 ceilings |
| `LEARNINGS_phase5_p15f_warmstart_efficiency_part2.md` | Warmstart/efficiency pt2 | Efficiency Track Status, steps 170–174, PHASE EXIT |
| `CLAUDE_MD_REVIEW_2026-04-14.md` | CLAUDE.md review INDEX | TOC for part1/part2 split |
| `CLAUDE_MD_REVIEW_2026-04-14_part1.md` | CLAUDE.md review pt1 | Sections A–C: Karpathy differences, Current Strengths, Proposed Additions |
| `CLAUDE_MD_REVIEW_2026-04-14_part2.md` | CLAUDE.md review pt2 | Sections D–F: Redundancies, Style Drift, Open Questions |
| `LEARNINGS_design.md` | Design discussions INDEX | TOC for date-split design files |
| `LEARNINGS_design_2026_04_04.md` | Design 04-04 | Phase routing, resonance, ablation protocol |
| `LEARNINGS_design_2026_04_05_06.md` | Design 04-05/06 | Phase routing details, wave-1 failure analysis |
| `LEARNINGS_design_2026_04_07_08.md` | Design 04-07 | Group topology design, group routing, gate-death synthesis |
| `LEARNINGS_design_2026_04_08.md` | Design 04-08 | step83 post-mortem, step87 proximity architecture design |
| `LEARNINGS_design_2026_04_09.md` | Design 04-09 INDEX | TOC for date-split design files (Gemma4/PolarQuant, FLOPs path) |
| `LEARNINGS_design_2026_04_09_part1.md` | Design 04-09 pt1 | Phase-Polarized Neurons, AH Compatibility, Open Design Questions |
| `LEARNINGS_design_2026_04_09_part2.md` | Design 04-09 pt2 | Gemma4/PolarQuant steps 106–109, FLOPs path, gap analysis |
| `LEARNINGS_design_2026_04_10.md` | Design 04-10 | Constraint discovery, progressive capacity, diagnostics, safety removal |
| `LEARNINGS_phase5_p16.md` | 2026-04-10 | steps 116, 149, 152, 153, 155 — normalization, diagnostics, structural constraints |
| `LEARNINGS_phase5_p15i_topology_analysis.md` | 2026-04-11 | K_hh=2 topology graph properties; 12.5% dead-end neurons; out-degree distribution |
| `LEARNINGS_phase5_p15j_belief_update.md` | 2026-04-11 | Belief framework; steps 216-221 unlearnings; polarizer +1.27pp; topology design KILLED; AH prerequisite confirmed |
| `PENDING_DISCUSSIONS.md` | Pending discussions INDEX | TOC for part1/part2 split |
| `PENDING_DISCUSSIONS_part1.md` | Pending pt1 | SCRIPTED entries, steps 87/G4/72/85/60-Redux, Delayed Routing, research-review SCRIPTED/PENDING, step106 |
| `PENDING_DISCUSSIONS_part2.md` | Pending pt2 | Gemma4/PolarQuant (step107–109), gap analysis (step99–101), step163 KD, Resolved |
| `LEARNINGS_arch.md` | Architecture notes | Structural decisions |
| `LEARNINGS_research.md` | Research directions | Literature, external ideas |
| `LEARNINGS_sparse_attention_*.md` | Sparse attention research | External references |
| `HISTORICAL_SPECS.md` | Phase 1-4 archival specs | Data pipeline UAT, Phase 2 baselines, Phase 3 rejected approaches, Phase 4 wave/phasor decisions |
| `LEARNINGS_ops.md` | Operations reference | MPS gotchas, training stability, perf benchmarks, known script bugs, smoke-test protocol |

## Other Index Files

| File | Purpose |
|---|---|
| `EXPERIMENT_QUEUE.md` | Live prioritized queue — RUNNING + QUEUED only |
| `EXPERIMENT_QUEUE_history.md` | Historical DONE/KILLED entries P0–P1 |
| `EXPERIMENT_QUEUE_history_part2.md` | Historical DONE/KILLED entries P1–P3, Completed Experiments |
| `EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md` | All confirmed findings and dead ends (extracted) |
| `EXPERIMENT_REPORT.md` | Comprehensive experiment report |
| `PENDING_DISCUSSIONS.md` | Design ideas not yet scripted (INDEX) |
| `RESEARCH_routing_mechanisms.md` | Literature survey on routing mechanisms |
| `paper/README.md` | Paper materials index |
| `paper/MANUSCRIPT_DRAFT.md` | Manuscript INDEX | TOC for sec1–sec4 split |
| `paper/MANUSCRIPT_DRAFT_sec1_abstract_intro_related.md` | Manuscript sec1 | Abstract, Introduction, Related Work (lines 1–214) |
| `paper/MANUSCRIPT_DRAFT_sec2_arch_experiments_findings.md` | Manuscript sec2 | Architecture, Experimental Setup, Key Findings (lines 215–415) |
| `paper/MANUSCRIPT_DRAFT_sec3_efficiency_discussion.md` | Manuscript sec3 | Efficiency Frontier, Discussion, Baselines, Future Work (lines 416–524) |
| `paper/MANUSCRIPT_DRAFT_sec4_appendices.md` | Manuscript sec4 | Appendices: Dead Ends, Hyperparams, N-Scaling Detail (lines 525–645) |
| `paper/findings_log.md` | Findings log INDEX | TOC for part1/part2 split |
| `paper/findings_log_part1.md` | Findings log pt1 | 2026-04-04 through 2026-04-14 (bench_step811) |
| `paper/findings_log_part2.md` | Findings log pt2 | 2026-04-14 (step260+) through 2026-04-15 |

## Confirmed Laws

| Law | Evidence | Concept Page |
|---|---|---|
| Gate-death: g^K → 0 for K_iter >= 4 | 8+ experiments (steps 58-66) | [[gate_death]] |
| AH compound failure: adding to AH alpha=1.0 kills gain | steps 29c, 32, 58-63 | [[antihebbian]] |
| K_iter optimal is N-dependent | step68 (N=1024→16), step71 (N=4096→12) | [[k_iter]] |
| N-scaling non-monotonic above N=4096 | step56 (N=10000 regresses) | [[n_scaling]] |
| Turing contribution is N-dependent | +1.68pp at N=1024, -0.12pp at N=4096 | [[n_scaling]] |
| Redistribution routing preserves gradient | step73/75 vs wave-1 gating | [[softmax_routing]] |
| Scale transfer failure pattern | Group topo, W_phase, turing all lose gains at N=4096 | [[antihebbian]] |
| Per-step Z-bias = largest single gain at N=1024 | +7.42pp (step106-A, 768 params, additive) | — |
| Phase+AH synergistic (not antagonistic) | +4.21pp combined vs phase alone useless (step105) | [[phase_routing]] |
| N dominates over K (connectivity) | step140: more K HURTS accuracy (−8 to −44pp) | [[n_scaling]] |
| Safety valve redundant with AH active | step154: +9.75pp when REMOVED (AH handles diversity) | [[antihebbian]] |
| Three load-bearing walls | F.normalize, static AH, mean-pool readout — all confirmed | — |
| AH is anti-collapse mechanism | step155: AH=0 → eff_rank collapse; AH=1.0 → eff_rank rises | [[antihebbian]] |
| W_pos barely learns at N=4096 | grad_theta/grad_wpos = 14:1; W_pos static after early epochs | [[antihebbian]] |
| LayerNorm > L2 sphere norm at N=1024 | +2.24pp (step116, pending N=4096) | [[normalization]] |
| C_ho readout required for unit-sphere activations | global mean-pool → 12% (step149) | [[readout]] |
| AH is PREREQUISITE not regularizer | Without AH: 91.5% → 18.8% collapse (step218) — 73pp drop | [[antihebbian]] |
| N=1024 crutches don't transfer | twopop/curriculum/hetero all fail at N=2048 (step216) | [[n_scaling]] |
| Static topology design barely matters | 3 experiments: anti-pref +0.15pp, hetero neg, output-assigned neg | — |
| Polarizer routing is first dynamic win | +1.91pp over-polarizer α=1.5, monotonic alpha trend (step217b) | — |
| Skip connections actively rejected | Network learns skip gate=0.0 (step226) | — |
| D=16 ceiling confirmed at 97.17% | N=4096 and N=8192 both converge to 97.17% (step205/209) | [[n_scaling]] |
| ΔW projection mechanism is specifically load-bearing | random direction = chance (step708); receiver-only = weak; ΔW=W[recv]−W[send] = +2.11pp | [[delta_w]] |
| ΔW proj headroom curve non-monotone, peak at N=128 | N=64(+17pp) < N=128(+24pp) > N=256(+20pp) > N=512(+12) > N=1024(+4.7) > N=2048(+1.6); HURTS at N=4096 | [[delta_w]] |
| ΔW proj vs rot crossover at N=1024→2048 | proj dominates ≤1024; tie at 2048; rot marginal at 4096. Proj 3× cheaper (~2D vs ~6D MACs/edge) | [[delta_w]] |
| Sequential K_iter is load-bearing (not unrollable) | multi-hop precompute (step700) and parallel branches (step701) both killed (−3 to −34pp) | [[k_iter]] |
| W_pos learned geometry is the primary accuracy source | random W_pos (step401) = 10.04% chance; learned = 95.52%. Delta = 81pp | [[antihebbian]] |
| K_iter annealing/distillation/warm-transfer all KILLED at efficiency config | steps 610/611/612 — K=5 sequential passes are architecturally necessary | [[k_iter]] |
| SGNNET is at 0.75% of VGG16 FC true MACs (ncu-validated) | step800 — paper must use "message-passing MACs" (0.98M) vs "total MACs" (1.85M); VGG FC = 247M | — |
| torch.compile gives 4.7× training speedup over VGG FC on 5060ti | step500 — V0 eager 2.6% GPU util → V1 compiled 99.6% (compute-bound). bf16+scaler = 4.4× SLOWER (step801) | — |
| Spatial precomputation gives 16× seed FLOP reduction (mathematical identity) | bench_step831 — 5.26× CUDA speedup, 14× memory reduction, accuracy bit-exact | — |
| K_in=15 helps at N>=4096, costs at N=2048 | step293 (N=4096: +0.33pp, N=8192: +0.38pp T1), step288 (N=16384: +1.27pp T1), step631 (N=2048: -0.36pp); crossover between N=2048-4096 | — |
| Augmentation is scale-invariant (+0.4-0.8pp at all N) | step269/273/276/280 T2 curve; aug delta does not compress at larger N | — |
| Soft-label KD enables K=1 routing with -0.36pp cost | step604/605 — K=5 teacher 96.69% → K=1 student 96.33%; 5× routing reduction; trajectory loss adds only +0.03pp (not load-bearing) | — |
| Deep supervision on K_iter routing KILLED (-5 to -6.5pp) | step521 — all 4 DS variants destroy representation-building dynamics; single final-loss is correct training signal | — |
| Standard GNNs fail as FC replacements (~48% vs SGNNET 95.52%) | step404 — GCN/GAT/GIN assume sparse structural graphs, not dense feature similarity spaces | — |
| SGNNET competitive on text (SST-2: -1pp vs Linear) | step405 — 83.60% CPU vs Linear 84.63%; MPS fails at N_in=768 (numerical issue) | — |
| Audio cross-modal fails (ESC-50: -14pp vs Linear) | step406 — Whisper-tiny 384-dim features don't have enough structure for graph routing | — |

## Dead Ends (Do Not Re-Propose)

| Mechanism | Why Dead | Last Step |
|---|---|---|
| Signed coupling at D=64 | cos-sim on S^63 = noise; 5 experiments | step49 |
| D=128 encoding | All LR/schedule combos stuck ~10% | phase5_p1 |
| MoD adaptive depth | Early exit destroys iterative refinement | step34 |
| Oja's rule routing | PCA compression destroys diversity | step41 |
| Dynamic Z-KNN | Unstable on S^63 per step | step31 |
| Phase-excitatory (static W_phase) | -13pp vs AH alone | step29c |
| Hub interneurons (K_hh=6) | Fan-in too sparse | step61 |
| All wave-1 multiplicative gates | Gate-death theorem | steps 58-66 |
| Group MoE sparsification | Routing capacity collapse | step83, step107 |
| RNN sequential injection from zero | Zero-state bootstrapping failure | step112 |
| DropMessage at K_hh=4 | Too sparse for any dropout | step96 |
| Markov routing | −50 to −71pp; F.normalize + static AH load-bearing | step129 |
| Attention readout | −60 to −67pp; mean-pool load-bearing | step118 |
| Stochastic depth | −35 to −61pp; every K_iter step essential | step123 |
| Beam broadcast | −3 to −40pp; hurts local routing | step130 |
| Adaptive K_iter (ACT) | −2 to −7pp | step119 |
| N×K tradeoff (more K for less N) | N dominates; more K hurts | step140 |
| Low-rank W_proj at N=4096 | Scale transfer compression (+0.06pp NULL) | step132 |
| Safety valve loss | Redundant with AH; REMOVED from defaults | step154 |
| Load balance loss | Only +0.21pp; below noise; REMOVED | step79 |
| Nuclear norm / L1 / contrastive / dim_gate | Network self-organizes; external constraints KILLED (−3 to −15pp) | step152 |
| Pruning during training (all forms) | Stable connectivity required; 100% neuron util = no dead neurons | step153 |
| RMSNorm / pre_route normalize | −11.49pp and −6.50pp vs L2 sphere; LayerNorm wins instead | step116 |
| Global mean-pool readout | −70pp on unit-sphere activations; C_ho required | step149 |
| Heterogeneous K_hh | All 4 variants worse than uniform (best −0.28pp) | step220 |
| Output-assigned topology | All 3 variants worse than input-assigned (best −1.30pp) | step221 |
| Scored dynamic topology | All neutral ±0.5pp, no benefit from per-epoch edge replacement | step224 |
| Skip connections across K_iter | −10 to −14pp; network learns gate=0.0 | step226 |
| Compound N=1024 winners at N=2048 | twopop −1.81pp, curriculum −58pp, α=1.05 neutral | step216 |
| D=8 regardless of K_hh | D=8 K_hh=8 = 90.24%, D=8 K_hh=16 = 89.22%; D wall | step214/215 |
| ConnGA evolutionary topology | ConnGA_rand=68.36% (−1.81pp), deg=70.32% (neutral); static small-world already optimal | step709 |
| Multi-hop gather (K_iter parallelization) | All variants −3.5 to −13pp AND slower; flattening loses nonlinearity stacking | step700 |
| Parallel routing branches (K_iter parallelization) | All −7 to −34pp; learned branch weights stayed uniform, no differentiation | step701 |
| Per-sample adaptive K_iter at inference | All thresholds catastrophic (−80pp); logit cosine_sim is NOT a convergence proxy | step703 |
| ΔW projection at N=4096 (ceiling) | −0.74pp CUDA, −0.89pp MPS — N-specific, breaks down near D=16 ceiling | step704 |
| Activation retention (static/decay/norm_cons/reinject) | All variants hurt (−0.10 to −18.67pp); Ref wins | step306 |
| Equilibrium propagation | β=0.1/0.5 collapse to chance 10%; EP incompatible with L2-normalized routing | step225 |
| Gradient-safe θ parameterizations | cos_shifted/phase_delta/triangle all 62–75%; θ-edge direction fully killed | step238 |
| K_iter annealing at efficiency config | All 5 schedules hurt −0.84 to −6.50pp; K=5 is load-bearing at N=2048 | step610 |
| K_iter warm-transfer (teacher→student) | All hurt; higher teacher K → worse student (C_8to5=−0.74pp, D_16to5=−6.06pp) | step611 |
| K_iter distillation at efficiency config | All hurt −3.36 to −4.76pp at K=3 vs K=5 Ref | step612 |
| bf16 training with GradScaler | 4.4× SLOWER than fp32 on 5060ti; GradScaler per-step overhead dominates | step801 |
| CIFAR-10 on raw pixels (end-to-end SGNNET) | Linear=37.94% > N2048=32.09% > N4096=33.32%; SGNNET needs feature-extracted inputs | step400 |
| RCM index reordering on CUDA | 0.993× speedup = no-op; only helps when memory-BW-bound (we're compute-bound post-compile) | step520 |
| CUDA graph make_graphed_callables (fwd+bwd) | `cudaErrorStreamCaptureInvalidated` on int64 gather indices; needs Triton kernel (step530) | step803 |

## Active Research Directions (2026-04-14)

**Priority shift 2026-04-14:** User directed paper 1 soft-conclusion to take priority. Exploration paused; paper-blocking experiments preferred.

1. **PAPER 1 PACKAGING (top priority)**: manuscript soft-conclusion. Paper framing: "SGNNET as a general-purpose classification head replacing fully-connected layers for pre-trained feature extractors, validated across vision AND language domains." See `paper/MANUSCRIPT_DRAFT.md`, `paper/claims.md`.
2. **Cross-domain validation (paper-blocking)**: language experiment needed. SGNNET on transformer CLS embeddings. Model choice pending (DistilBERT 768-dim vs BERT-tiny 128-dim). Task choice pending (SST-2 vs AG News).
3. **Baselines at matched FLOPs (paper-nice-to-have)**: step401 has matched-params baselines (MLP_64=97.20% at 24× more params). Matched-FLOPs comparison computable analytically.
4. **ΔW mechanism (complete)**: headroom curve N=64–4096 done; proj vs rot crossover complete. See [[delta_w]].
5. **CUDA optimization (on-hold)**: step530 Triton fused kernel deferred — 4.7× from torch.compile alone already exceeds paper efficiency claim.
6. **K_iter reduction (CLOSED)**: all paths killed (annealing, warm-transfer, distillation, multi-hop, parallel branches, adaptive inference). K=5 is architecturally necessary at efficiency config.
7. **Running now**: step729 N=4096 T1 rot vs proj on 5060ti (ceiling verdict — last exploration experiment before paper packaging).
