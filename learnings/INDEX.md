# SGNNET Knowledge Index

## Concept Pages

| Concept | File | Status | Key Finding |
|---|---|---|---|
| [[antihebbian]] | [concepts/antihebbian.md](concepts/antihebbian.md) | Confirmed winner | alpha=1.0 alone is Gen4 base; every compound kills gain |
| [[gate_death]] | [concepts/gate_death.md](concepts/gate_death.md) | Theorem established | Multiplicative gates g^K_iter → 0; 8+ experiments confirm |
| [[n_scaling]] | [concepts/n_scaling.md](concepts/n_scaling.md) | Active research | N=4096 peak (97.32%); N=10000 regresses; non-monotonic |
| [[k_iter]] | [concepts/k_iter.md](concepts/k_iter.md) | N-dependent optimal | K_iter=16 at N=1024, K_iter=12 at N=4096 |
| [[phase_routing]] | [concepts/phase_routing.md](concepts/phase_routing.md) | Wave-1 killed; revival via redistribution | All gating variants dead; softmax redistribution promising |
| [[softmax_routing]] | [concepts/softmax_routing.md](concepts/softmax_routing.md) | First dynamic routing win | step73 D=86.34% (+1.78pp); step75 D=87.24% (+3.98pp) |
| [[group_topology]] | [concepts/group_topology.md](concepts/group_topology.md) | +3.01pp N=1024, NULL N=4096 | Bypasses N^2 routing problem via G^2 decisions; doesn't scale |

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
| `LEARNINGS_design_2026_04_09.md` | Design 04-09+ | Open design questions, FLOPs path |
| `PENDING_DISCUSSIONS.md` | Pending discussions | Ideas discussed but not yet scripted |
| `LEARNINGS_arch.md` | Architecture notes | Structural decisions |
| `LEARNINGS_research.md` | Research directions | Literature, external ideas |
| `LEARNINGS_sparse_attention_*.md` | Sparse attention research | External references |

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
| Per-step Z-bias = largest single gain at N=1024 | +7.42pp (step106-A, 768 params, additive) | [[antihebbian]] |
| Phase+AH synergistic (not antagonistic) | +4.21pp combined vs phase alone useless (step105) | [[phase_routing]] |

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
