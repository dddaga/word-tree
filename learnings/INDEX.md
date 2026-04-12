# SGNNET Knowledge Index

**Last updated:** 2026-04-13
**Project best:** 97.86% (step89-A, N=4096, D=64, 150ep)
**Efficiency best:** 95.52% @ 0.98M FLOPs (step199, N=2048, D=16, K_hh=2, K_iter=5)
**D=16 record:** 97.17% (step205/209, N=4096/8192)
**Current direction:** Mechanistic understanding (AH ablations, θ-edge precision), polarizer routing (+1.91pp), architecture simplification

## Concept Pages

| Concept | File | Status | Key Finding |
|---|---|---|---|
| [[antihebbian]] | [concepts/antihebbian.md](concepts/antihebbian.md) | Confirmed winner | alpha=1.0 alone is Gen4 base; AH = anti-collapse (eff_rank confirmed step155) |
| [[gate_death]] | [concepts/gate_death.md](concepts/gate_death.md) | Theorem established | Multiplicative gates g^K_iter → 0; 8+ experiments confirm |
| [[n_scaling]] | [concepts/n_scaling.md](concepts/n_scaling.md) | Active research | N=4096 peak (97.86%); N=10000 regresses; non-monotonic |
| [[k_iter]] | [concepts/k_iter.md](concepts/k_iter.md) | N-dependent optimal | K_iter=16 at N=1024, K_iter=12 at N=4096 |
| [[phase_routing]] | [concepts/phase_routing.md](concepts/phase_routing.md) | Wave-1 killed; revival via redistribution | All gating variants dead; softmax redistribution promising |
| [[softmax_routing]] | [concepts/softmax_routing.md](concepts/softmax_routing.md) | First dynamic routing win | step73 D=86.34% (+1.78pp); step75 D=87.24% (+3.98pp) |
| [[group_topology]] | [concepts/group_topology.md](concepts/group_topology.md) | +3.01pp N=1024, NULL N=4096 | Bypasses N^2 routing problem via G^2 decisions; doesn't scale |
| [[normalization]] | [concepts/normalization.md](concepts/normalization.md) | LayerNorm winner pending N=4096 | LayerNorm +2.24pp vs L2 sphere (step116); RMSNorm −11.49pp KILLED |
| [[readout]] | [concepts/readout.md](concepts/readout.md) | Hard architectural constraint | C_ho required for unit-sphere activations; global mean-pool → 12% (step149) |
| [[architecture_dead_ends]] | [concepts/architecture_dead_ends.md](concepts/architecture_dead_ends.md) | Reference | All confirmed dead ends; external constraints + pruning KILLED (step152, step153) |

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
| `LEARNINGS_design.md` | Design discussions INDEX | TOC for date-split design files |
| `LEARNINGS_design_2026_04_04.md` | Design 04-04 | Phase routing, resonance, ablation protocol |
| `LEARNINGS_design_2026_04_05_06.md` | Design 04-05/06 | Phase routing details, wave-1 failure analysis |
| `LEARNINGS_design_2026_04_07_08.md` | Design 04-07 | Group topology design, group routing, gate-death synthesis |
| `LEARNINGS_design_2026_04_08.md` | Design 04-08 | step83 post-mortem, step87 proximity architecture design |
| `LEARNINGS_design_2026_04_09.md` | Design 04-09 | Gemma4/PolarQuant designs, FLOPs path, 50-experiment gap analysis |
| `LEARNINGS_design_2026_04_10.md` | Design 04-10 | Constraint discovery, progressive capacity, diagnostics, safety removal |
| `LEARNINGS_phase5_p16.md` | 2026-04-10 | steps 116, 149, 152, 153, 155 — normalization, diagnostics, structural constraints |
| `LEARNINGS_phase5_p15i_topology_analysis.md` | 2026-04-11 | K_hh=2 topology graph properties; 12.5% dead-end neurons; out-degree distribution |
| `LEARNINGS_phase5_p15j_belief_update.md` | 2026-04-11 | Belief framework; steps 216-221 unlearnings; polarizer +1.27pp; topology design KILLED; AH prerequisite confirmed |
| `PENDING_DISCUSSIONS.md` | Pending discussions | Ideas discussed but not yet scripted |
| `LEARNINGS_arch.md` | Architecture notes | Structural decisions |
| `LEARNINGS_research.md` | Research directions | Literature, external ideas |
| `LEARNINGS_sparse_attention_*.md` | Sparse attention research | External references |

## Other Index Files

| File | Purpose |
|---|---|
| `EXPERIMENT_QUEUE.md` | Live prioritized queue with status, configs, launch order |
| `EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md` | All confirmed findings and dead ends (extracted) |
| `EXPERIMENT_REPORT.md` | Comprehensive experiment report |
| `PENDING_DISCUSSIONS.md` | Design ideas not yet scripted |
| `RESEARCH_routing_mechanisms.md` | Literature survey on routing mechanisms |
| `paper/README.md` | Paper materials index |

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

## Active Research Directions (2026-04-13)

1. **Mechanistic understanding**: What does hidden W_pos actually encode? θ-edge precision ablation (step233 running) tests whether AH is just learned edge weights. Step231 diagnostics pending.
2. **Polarizer routing**: Over-polarizer α=1.5 = +1.91pp (step217b confirmed Tier-1). Monotonic alpha trend — test α=2.0+. First successful input-dependent routing after 9 failures.
3. **Architecture simplification**: If θ-edge matches AH, replace 32K hidden W_pos + cosine computation with 4K scalar angles. 8× param reduction in hidden routing.
4. **N-scaling continuation**: D=16 ceiling at 97.17%. N=16384 @ K_iter=3 viable (94.78%). K_hh=3 @ N=8192 at 95.44% (needs Tier-2).
5. **Paper track**: 7 core claims, 11+ novel findings logged. MLP baselines broken (step222 trainer bug). CIFAR-10 cross-dataset pending (step223).
