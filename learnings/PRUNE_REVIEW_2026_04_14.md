# Prune Review — 50 Scripts Proposed for Deletion

Date: 2026-04-14
Review policy: user confirms each row before `rm`.

## STALE (44 scripts — invalidated by later findings)

| Step | Script | Invalidated by | Keep? |
|---|---|---|---|
| 13 | train_step13_beam_iter_scaling.py | Beam routing killed (wave-1 dead ends); beam_size no longer a config axis | [ ] |
| 28 | train_step28_gen3_compound.py | Gen3 on signed-coupling base; signed coupling dead (step49); arch replaced by Gen4+ | [ ] |
| 32 | train_step32_gen4_compound.py | Gen4 compound on buggy pre-patch arch; all mechanisms re-tested post-step69 | [ ] |
| 33 | train_step33_d128.py | D=128 confirmed dead (all LR/schedule combos stuck ~10%, Dead Ends table) | [ ] |
| 33b | train_step33b_d128_calib.py | D=128 calibration dead; root cause documented; direction closed | [ ] |
| 34 | train_step34_mod_adaptive_kiter.py | MoD adaptive depth killed (Dead Ends: "Early exit destroys iterative refinement") | [ ] |
| 37 | train_step37_phase_matrix_bank.py | D×D cross-dim mixing killed (steps 30, 37, fast_W_phase); violates FLOPs budget | [ ] |
| 39 | train_step39_mechanism_loss.py | Mechanism-aware aux losses killed: contrastive −13pp, structural reg entire direction killed (steps 79, 152) | [ ] |
| 41 | train_step41_ojas_rule.py | Oja's rule dead: "PCA compression destroys diversity" (Dead Ends, step41) | [ ] |
| 42 | train_step42_signed_d64_calib.py | Signed coupling at D=64 dead: cos-sim on S^63 = noise, 5 experiments (Dead Ends, step49) | [ ] |
| 44 | train_step44_beam_signed.py | Beam routing killed + signed coupling at D=64 killed; both premises invalid | [ ] |
| 46 | train_step46_wphase_reconnect.py | W_phase barely-learning confirmed at N=4096 (grad_theta/grad_wpos=14:1); phase routing killed | [ ] |
| 47 | train_step47_interneurons_d64.py | Hub interneurons killed (step61); group MoE killed (steps 83, 107); interneuron direction closed | [ ] |
| 50 | train_step50_spatial_dynamic_conn.py | Dynamic Z-KNN dead (step31, Dead Ends); spatial dynamic connectivity same premise | [ ] |
| 51 | train_step51_spatial_phase_gating.py | Phase gating = wave-1 multiplicative gate; gate-death theorem kills all multiplicative variants | [ ] |
| 52 | train_step52_high_d_routing.py | Exploits D=64; project shifted to D=16 efficiency config; mechanisms assume S^63 geometry | [ ] |
| 53 | train_step53_lowrank_mixing.py | Low-rank cross-dim at D=64; D=64 track closed; low-rank W_proj killed (step132, +0.06pp NULL) | [ ] |
| 56 | train_step56_n_scaling.py | N-scaling at D=64 superseded by exhaustive D=16 scaling (steps 198–212); old arch | [ ] |
| 63 | train_step63_act_gated_routing.py | Activation-gated = soft-attention multiplicative weight; gate-death theorem; all multiplicative routing killed | [ ] |
| 65 | train_step65_dist_phase_routing.py | Distance-phase routing at D=64; wave-1 direction; W_phase barely learns; phase routing killed | [ ] |
| 67 | train_step67_safety_ablation.py | Safety valve ablation CONFIRMED redundant at efficiency config (step154: +9.75pp when removed) | [ ] |
| 68 | train_step68_kiter_gen4.py | K_iter sweep at Gen4 D=64 N=1024; superseded by full K_iter sweep at efficiency config | [ ] |
| 69 | train_step69_corrected_baseline.py | Arch-patch baseline on old D=64 config; baseline superseded by efficiency config | [ ] |
| 70 | train_step70_fullscale_gen4plus.py | Full-scale Gen4+ at D=64; 97.86% already achieved; efficiency track dominates | [ ] |
| 75 | train_step75_temp_routing.py | Input-modulated temperature routing; per-neuron dynamic routing killed family | [ ] |
| 79 | train_step79_aux_loss_patched.py | All structural constraints killed (step152); load balance +0.21pp below noise | [ ] |
| 80 | train_step80_nscale_patched.py | N-scaling at D=64 fully superseded by D=16 N-scaling law (steps 190–212) | [ ] |
| 81 | train_step81_hebbian_rewiring.py | Hebbian topology rewiring killed; ConnGA confirmed static small-world optimal (step709); RigL killed (step229) | [ ] |
| 88 | train_step88_alpha_sweep_n4096.py | AH alpha sweep at D=64 superseded by step133 (T1) and step321 (paper-critical at efficiency config) | [ ] |
| 92 | train_step92_relu_group_routing.py | ReLU group routing to fix step83 collapse; group MoE killed at step107 same root cause | [ ] |
| 100 | train_step100_kin_sweep.py | K_in=25 set and validated at efficiency config; K_in reduction not on active agenda | [ ] |
| 102 | train_step102_phase_polarizer.py | Phase polarizer (Malus's law) = multiplicative signal filtering; gate-death theorem; phase routing killed | [ ] |
| 103 | train_step103_wave_interference.py | Wave interference routing; all wave-1 mechanisms confirmed dead | [ ] |
| 104 | train_step104_compound_wave_polar.py | Compounds step102 (Malus's law) + step103 (wave interference); both premises killed | [ ] |
| 107 | train_step107_group_moe.py | Group MoE dead (step107 result in Dead Ends: "Routing capacity collapse") | [ ] |
| 108 | train_step108_polar_routing.py | Hierarchical polar routing at D=64; D=64 closed; coarse-to-fine topology superseded | [ ] |
| 110 | train_step110_muon_optimizer.py | Muon for W_pos; W_pos barely learns at N=4096 (law: grad_theta/grad_wpos=14:1); no leverage | [ ] |
| 112 | train_step112_rnn_input_injection.py | RNN sequential injection dead: "Zero-state bootstrapping failure" (Dead Ends, step112) | [ ] |
| 113 | train_step113_intermediate_supervision.py | Intermediate supervision = K_iter distillation variant; K_iter distillation killed (step612, −3.4 to −4.8pp) | [ ] |
| 114 | train_step114_parallel_chunks.py | Parallel input chunks killed (step701, −7 to −34pp); parallelizing over K_iter non-viable | [ ] |
| 118 | train_step118_attention_readout.py | Attention readout dead: −60 to −67pp (Dead Ends, step118) | [ ] |
| 119 | train_step119_adaptive_kiter.py | Adaptive K_iter (ACT) dead: −2 to −7pp (Dead Ends, step119); per-sample variant killed (step703, −81pp) | [ ] |
| 123 | train_step123_stochastic_depth.py | Stochastic depth dead: −35 to −61pp (Dead Ends, step123) | [ ] |
| 130 | train_step130_beam_broadcast.py | Beam broadcast dead: −3 to −40pp (Dead Ends, step130) | [ ] |
| 140 | train_step140_nk_tradeoff.py | N×K tradeoff dead: "N dominates; more K HURTS" (Dead Ends, step140) | [ ] |

## DUPLICATE (6 scripts — same config as already-executed step)

| Step | Script | Duplicate of | Keep? |
|---|---|---|---|
| 106 | train_step106_perstep_embeddings.py | step115 covers N=4096; step106 is older N=1024 D=64 version of same mechanism | [ ] |
| 115 | train_step115_zbias_redistribution_n4096.py | Both mechanisms individually DONE; combined test superseded by ΔW proj track | [ ] |
| 120 | train_step120_high_kiter_zbias.py | K_iter ceiling fully mapped (steps 68, 71, 194–211); Z-bias superseded by ΔW proj | [ ] |
| 131 | train_step131_tier1_winners.py | Results in queue as DONE (step131-A +3.97pp T1, step131-B +5.48pp T1) | [ ] |
| 217 | train_step217_polarizer_routing.py | step217b Tier-1 DONE (95.92% +1.91pp); this Tier-0 superseded | [ ] |
| 223 | train_step223_cifar10_cross_dataset.py | DESIGN FLAW documented; replaced by step400 which is DONE | [ ] |
