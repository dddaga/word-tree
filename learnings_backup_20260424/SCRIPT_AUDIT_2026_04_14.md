# Script Audit — 2026-04-14

Audited: 91 scripts in `scripts/train_step*.py` with no corresponding JSON in `results/`.

---

## Summary

| Category | Count |
|----------|-------|
| STALE    | 44    |
| DUPLICATE | 6    |
| RELEVANT  | 34   |
| UNKNOWN   | 7    |
| **Total** | **91** |

---

## Full Classification

| Step | Script | Category | Reason |
|------|--------|----------|--------|
| 13 | train_step13_beam_iter_scaling | STALE | Beam routing killed (wave-1 dead ends); beam_size no longer a config axis |
| 28 | train_step28_gen3_compound | STALE | Gen3 compound on signed-coupling base; signed coupling confirmed dead (step49); base arch replaced by Gen4+ |
| 32 | train_step32_gen4_compound | STALE | Gen4 compound design on buggy arch (pre-patch); all mechanisms re-tested on patched arch post-step69 |
| 33 | train_step33_d128 | STALE | D=128 confirmed dead (all LR/schedule combos stuck ~10%, INDEX Dead Ends table) |
| 33b | train_step33b_d128_calib | STALE | D=128 calibration dead — root cause of step33 failure documented; direction closed |
| 34 | train_step34_mod_adaptive_kiter | STALE | MoD adaptive depth killed (step34 listed in Dead Ends: "Early exit destroys iterative refinement") |
| 37 | train_step37_phase_matrix_bank | STALE | D×D cross-dim mixing killed multiple times (steps 30, 37, fast_W_phase); O(N×D²) also violates FLOPs budget |
| 39 | train_step39_mechanism_loss | STALE | Mechanism-aware aux losses killed: load balance +0.21pp, contrastive −13pp, structural regularization entire direction killed (steps 79, 152) |
| 41 | train_step41_ojas_rule | STALE | Oja's rule confirmed dead: "PCA compression destroys diversity" (Dead Ends table, step41) |
| 42 | train_step42_signed_d64_calib | STALE | Signed coupling at D=64 confirmed dead: cos-sim on S^63 = noise, 5 experiments (Dead Ends table, step49) |
| 44 | train_step44_beam_signed | STALE | Beam routing killed + signed coupling at D=64 killed; both premises invalid |
| 46 | train_step46_wphase_reconnect | STALE | W_phase is confirmed barely-learning at N=4096 (grad_theta/grad_wpos = 14:1); phase mechanisms in routing killed (wave-1 verdict) |
| 47 | train_step47_interneurons_d64 | STALE | Interneurons at D=64 — hub interneurons K_hh=6 killed (step61); group MoE killed (step83, step107); whole interneuron direction closed |
| 50 | train_step50_spatial_dynamic_conn | STALE | Dynamic Z-KNN confirmed dead (step31, Dead Ends); spatial dynamic connectivity rests on same premise |
| 51 | train_step51_spatial_phase_gating | STALE | Phase gating = wave-1 multiplicative gate; gate-death theorem kills all multiplicative variants |
| 52 | train_step52_high_d_routing | STALE | High-D routing exploits D=64; project focus shifted to D=16 efficiency config; mechanisms assumed to leverage S^63 geometry |
| 53 | train_step53_lowrank_mixing | STALE | Low-rank cross-dim mixing at D=64; D=64 track closed; low-rank W_proj at N=4096 killed (step132, +0.06pp NULL) |
| 56 | train_step56_n_scaling | STALE | N-scaling at D=64 — superseded by exhaustive D=16 N-scaling (steps 198–212); old arch |
| 63 | train_step63_act_gated_routing | STALE | Activation-gated routing = soft-attention multiplicative weight; gate-death theorem applies; all multiplicative routing killed |
| 65 | train_step65_dist_phase_routing | STALE | Distance-phase routing at D=64; wave-1 direction; W_phase barely learns; all phase routing mechanisms killed |
| 67 | train_step67_safety_ablation | STALE | Safety valve ablation — executed at N=1024 D=64 level, but CONFIRMED at N=2048 efficiency config (step154: +9.75pp when removed); CONFIRMED redundant |
| 68 | train_step68_kiter_gen4 | STALE | K_iter sweep at Gen4 D=64 N=1024 — superseded by full K_iter sweep at efficiency config; K_iter optimal is N-dependent (confirmed law) |
| 69 | train_step69_corrected_baseline | STALE | Arch-patch baseline on old D=64 config; baseline long superseded by efficiency config; all downstream steps from this era completed |
| 70 | train_step70_fullscale_gen4plus | STALE | Full-scale Gen4+ at D=64; project best 97.86% already achieved at D=64; efficiency track dominates |
| 75 | train_step75_temp_routing | STALE | Input-modulated temperature routing — softmax redistribution routing direction superseded; per-neuron temperature is input-dependent dynamic routing killed family |
| 79 | train_step79_aux_loss_patched | STALE | Aux losses re-test on patched arch — all structural constraints killed (step152); load balance +0.21pp below noise; direction closed |
| 80 | train_step80_nscale_patched | STALE | N-scaling survey on patched D=64 arch — fully superseded by D=16 N-scaling law (steps 190–212) |
| 81 | train_step81_hebbian_rewiring | STALE | Hebbian topology rewiring = periodic edge replacement; ConnGA (step709) confirmed static small-world already optimal; scored dynamic topology null (step224); RigL killed (step229) |
| 88 | train_step88_alpha_sweep_n4096 | STALE | AH alpha sweep at N=4096 D=64 — fully superseded by step133 (Tier-1 α=1.05 winner at N=4096) and step321 (paper-critical AH α sweep at efficiency config) |
| 92 | train_step92_relu_group_routing | STALE | ReLU group routing to fix step83 collapse; group MoE killed at step107 with same ReMoE motivation |
| 100 | train_step100_kin_sweep | STALE | K_in sweep — K_in=25 set and validated at efficiency config; K_in reduction not on active research agenda |
| 102 | train_step102_phase_polarizer | STALE | Phase polarizer (Malus's law) — multiplicative signal filtering; gate-death theorem; phase routing killed |
| 103 | train_step103_wave_interference | STALE | Wave interference routing — complex wave mechanism; all wave-1 mechanisms confirmed dead |
| 104 | train_step104_compound_wave_polar | STALE | Compounds step102 (Malus's law) + step103 (wave interference); both premises killed |
| 107 | train_step107_group_moe | STALE | Group MoE confirmed dead at step107 result already in Dead Ends: "Routing capacity collapse" |
| 108 | train_step108_polar_routing | STALE | Hierarchical polar routing at D=64; D=64 track closed; coarse-to-fine topology from geometry superseded by simpler static topology |
| 110 | train_step110_muon_optimizer | STALE | Muon optimizer for W_pos; W_pos barely learns at N=4096 (confirmed law: grad_theta/grad_wpos = 14:1); optimizer change has no leverage |
| 112 | train_step112_rnn_input_injection | STALE | RNN sequential injection confirmed dead: "Zero-state bootstrapping failure" (Dead Ends table, step112) |
| 113 | train_step113_intermediate_supervision | STALE | Intermediate supervision = K_iter distillation variant; K_iter distillation killed at efficiency config (step612, −3.4 to −4.8pp) |
| 114 | train_step114_parallel_chunks | STALE | Parallel input chunks + superimpose; parallel routing branches killed (step701, −7 to −34pp); parallelizing over K_iter confirmed non-viable |
| 118 | train_step118_attention_readout | STALE | Attention readout confirmed dead: −60 to −67pp (Dead Ends table, step118) |
| 119 | train_step119_adaptive_kiter | STALE | Adaptive K_iter (ACT) confirmed dead: −2 to −7pp (Dead Ends table, step119); per-sample adaptive K_iter also killed (step703, −81pp) |
| 123 | train_step123_stochastic_depth | STALE | Stochastic depth confirmed dead: −35 to −61pp (Dead Ends table, step123) |
| 130 | train_step130_beam_broadcast | STALE | Beam broadcast confirmed dead: −3 to −40pp (Dead Ends table, step130) |
| 140 | train_step140_nk_tradeoff | STALE | N×K tradeoff confirmed dead: "N dominates; more K HURTS" (Dead Ends table, step140) |
| 13 | *(beam_iter_scaling JSON matched above)* | — | — |
| 106 | train_step106_perstep_embeddings | DUPLICATE | Per-step Z-bias: step115 tested this at N=4096 (result in queue as DONE); step106 at N=1024 D=64 is an older version |
| 115 | train_step115_zbias_redistribution_n4096 | DUPLICATE | Z-bias + redistribution at N=4096 — both mechanisms individually DONE (step75 redistribution, step106 Z-bias); combined test at N=4096 superseded by ΔW proj track |
| 120 | train_step120_high_kiter_zbias | DUPLICATE | Higher K_iter + Z-bias — K_iter ceiling already mapped (steps 68, 71, 194–211); Z-bias mechanism superseded by ΔW proj |
| 131 | train_step131_tier1_winners | DUPLICATE | Tier-1 validation of step128-A + step117-A winners — both mechanisms referenced in queue as DONE (step131-A weighted_neg +3.97pp Tier-1, step131-B W_proj +5.48pp Tier-1) |
| 217 | train_step217_polarizer_routing | DUPLICATE | Polarizer routing Tier-0 — step217b (Tier-1) already DONE (95.92% +1.91pp); step217 Tier-0 superseded |
| 223 | train_step223_cifar10_cross_dataset | DUPLICATE | CIFAR-10 cross-dataset — DONE in queue: "DESIGN FLAW. Replaced by step400"; step400 is DONE |
| 204 | train_step204_n4096_d16_khh2_kiter6_tier2 | DUPLICATE | N=4096 K_iter=6 T2 — queue shows step204 DONE: "97.15% best_ep=71" |
| 206 | train_step206_n8192_d16_khh2_kiter6_tier1 | DUPLICATE | N=8192 K_iter=6 T1 — queue shows step206 DONE: "95.11% best_ep=72"; result JSON may be missing but experiment is recorded |
| 124 | train_step124_rigl_topology | RELEVANT | RigL at D=16 efficiency config (N=1024, step124-B = +6.82pp largest single gain ever); step229 tests at efficiency config but needs to check if step229 itself executed | 1 |
| 229 | train_step229_rigl_topology | RELEVANT | RigL at efficiency config (N=2048 D=16 K_hh=2) — step229 DONE per queue (ConnGA step709 killed static topology but RigL is gradient-informed pruning/regrowth, not evolutionary) — NEEDS human check: queue says "KILLED" citing step229 but result JSON absent | 2 |
| 760 | train_step760_seed_variance | RELEVANT | Seed-variance replication for paper: measures std for step199 and step706 configs; needed to ground "+0.5pp threshold" claim in paper | 3 |
| 750 | train_step750_kiter3_n4096_khh4 | RELEVANT | K_iter=3 floor at N=4096 K_hh=4 — latency Pareto track; K_iter=3 at N=2048 killed (89%) but N=4096 K_hh=4 never tested; existing K_iter=3 data at N=8192 K_hh=2=94.93% (borderline) | 4 |
| 751 | train_step751_kiter3_n8192_khh8 | RELEVANT | K_iter=3 at N=8192 K_hh=8 — extends K_iter floor search; paper-quality latency Pareto data; includes inline benchmark | 5 |
| 730 | train_step730_proj_khh4_compound | RELEVANT | ΔW proj × K_hh=4 compound at N=2048 T1 — orthogonal mechanism axes (topology vs signal routing); potential +2pp compound; paper efficiency claim strengthener | 6 |
| 231 | train_step231_mechanism_diagnostics | RELEVANT | Mechanistic probes (H1–H3): class-specific W_pos, AH neighbor separation, progressive K_iter refinement; paper explanatory content; no training risk | 7 |
| 218 | train_step218_random_projection_ablation | RELEVANT | AH prerequisite ablation — queue says DONE but notes "Freezing bug, needs rerun"; confirms W_pos is essential (+81pp vs random); paper-critical claim | 8 |
| 230 | train_step230_gumbel_topology | RELEVANT | Gumbel-softmax differentiable topology learning — novel; not killed; ConnGA (evolutionary) killed but end-to-end differentiable is distinct mechanism | 9 |
| 125 | train_step125_alpha_fine_sweep | RELEVANT | AH alpha fine-sweep (1.0–1.3 range never tested at N=4096 D=64) — step321 DONE at N=2048 D=16 efficiency config; D=64 at N=4096 still needs fine sweep for paper completeness | 10 |
| 216 | train_step216_compound_winners | RELEVANT | Compound N=1024 winners on step199 — queue says DONE (twopop killed, curriculum killed, α=1.05 neutral); but script not counted as executed by JSON presence; JSON may be missing | 11 |
| 117 | train_step117_learned_input_proj | RELEVANT | Learned W_proj after scatter-sum; step131-B confirmed +5.48pp T1 at N=1024; scale transfer to efficiency config never tested with this exact version | 12 |
| 128 | train_step128_concat_relu | RELEVANT | ConcatReLU / weighted_neg activation; step131-A confirmed +3.97pp T1 at N=1024; may compound with ΔW proj at efficiency config | 13 |
| 150 | train_step150_positional_maxpool | RELEVANT | Positional max-pool readout; mean-pool confirmed load-bearing but this is an alternative (no attention); P1 priority in queue; attention readout killed but max-pool is different | 14 |
| 163 | train_step163_progressive_kd | RELEVANT | Progressive K_iter distillation with warm-start + intermediate state matching; fixes step127 bugs; K_iter distillation at efficiency config killed but this is at N=1024 D=16 with K=12→8 which showed +7.82pp | 15 |
| 165 | train_step165_compound_warmstart | RELEVANT | Compound warm-start + W_proj + RigL at N=1024 D=16; P0.5 priority; step165-B (warm+W_proj) is DONE but additional compounding (het-neurons) not yet tested | 16 |
| 169 | train_step169_warmproj_d32_fulldata | RELEVANT | warm+W_proj D=32 T2 validation; queue shows step169 DONE (94.01%); JSON missing — verify or re-check | 17 |
| 220 | train_step220_heterogeneous_khh | RELEVANT | Heterogeneous K_hh — DONE per queue ("ALL KILLED"); JSON missing — confirms kill result; low value but verify JSON | 18 |
| 221 | train_step221_output_assigned_topology | RELEVANT | Output-assigned topology — DONE per queue ("ALL KILLED, JSON bug"); JSON missing due to bug; low paper value | 19 |
| 224 | train_step224_scored_dynamic_topology | RELEVANT | Scored dynamic topology — DONE per queue ("ALL NEUTRAL"); JSON missing; low value | 20 |
| 225 | train_step225_equilibrium_propagation | RELEVANT | EP pilot — DONE per queue ("KILLED"); JSON missing; confirms EP incompatible | 21 |
| 226 | train_step226_skip_connections | RELEVANT | Skip connections — DONE per queue ("A/B KILLED, C gate=0"); JSON missing | 22 |
| 238 | train_step238_gradient_safe_theta | RELEVANT | Gradient-safe θ — DONE per queue ("ALL DEAD"); JSON missing | 23 |
| 306 | train_step306_activation_retention | RELEVANT | Activation retention — DONE per queue ("ALL HURT"); JSON missing | 24 |
| 401 | train_step401_baselines | RELEVANT | Paper baselines MLP vs SGNNET — DONE per queue; JSON missing; paper-critical | 25 |
| 520 | train_step520_index_reorder | RELEVANT | RCM reordering — DONE per queue ("KILLED"); JSON missing | 26 |
| 611 | train_step611_kiter_warm_transfer | RELEVANT | K_iter warm transfer — DONE per queue ("ALL HURT"); JSON missing | 27 |
| 703 | train_step703_adaptive_kiter | RELEVANT | Per-sample adaptive K_iter at inference — DONE per queue ("KILLED"); JSON missing | 28 |
| 704 | train_step704_delta_n4096 | RELEVANT | ΔW proj at N=4096 — DONE per queue ("KILLED"); JSON missing | 29 |
| 707 | train_step707_delta_n1024 | RELEVANT | ΔW proj at N=1024 T1 — DONE per queue (93.48%, +4.87pp); JSON missing | 30 |
| 707_tier2 | train_step707_tier2_delta_n1024 | RELEVANT | ΔW proj N=1024 T2 — DONE per queue (95.18%, +4.68pp); JSON missing | 31 |
| 708 | train_step708_delta_direction_ablation | RELEVANT | ΔW direction ablation — DONE per queue (B_random=18.62%, mechanism confirmed); JSON missing | 32 |
| 712 | train_step712_delta_n256 | RELEVANT | ΔW proj N=256 T1 — DONE per queue (+19.80pp); JSON missing | 33 |
| 713 | train_step713_delta_n256_tier2 | RELEVANT | ΔW proj N=256 T2 — DONE per queue (+20.02pp); JSON missing | 34 |
| 715 | train_step715_delta_n128 | RELEVANT | ΔW proj N=128 T1 — DONE per queue (+22pp); JSON missing | 35 |
| 720 | train_step720_delta_n128_d32 | RELEVANT | ΔW proj N=128 D=32 T1 — DONE per queue (+23.90pp); JSON missing | 36 |
| 723 | train_step723_rotation_n1024 | RELEVANT | rot vs proj crossover at N=1024 — DONE per queue (proj beats rot by 1.91pp); JSON missing | 37 |
| 724 | train_step724_delta_n64_khh4 | RELEVANT | K_hh=4 graph diversity test at N=64 — DONE per queue (+1.42pp delta); JSON missing | 38 |
| 729 | train_step729_rotation_n4096_t1 | RELEVANT | rot vs proj at N=4096 T1 — DONE per queue (rot +0.64pp, clears threshold); JSON missing | 39 |
| 740 | train_step740_connga_v2 | RELEVANT | ConnGA v2 stabilised — v1 (step709) used 10ep/child = noisy; v2 uses 30ep; distinct from v1 methodology; moderate paper value | 40 |
| 109 | train_step109_* | UNKNOWN | Script not found in glob (gap in numbering) | — |
| 320 | *(handled in queue DONE)* | — | — | — |

**UNKNOWN scripts (require human review):**

| Step | Script | Issue |
|------|--------|-------|
| 132 | train_step132_wproj_n4096 | W_proj at N=4096 — queue says "Scale transfer failure: +0.06pp NULL" in Dead Ends; but result JSON absent; may be DONE or STALE |
| 143 | train_step143_heterogeneous_neurons | Het neurons at N=1024; step216 killed twopop at N=2048 but that's scale transfer; step143 at N=1024 shows +5–6pp — still relevant at N=1024 level |
| 153 | train_step153_progressive_capacity | All pruning killed (Dead Ends); JSON absent — likely STALE but need confirmation |
| 163_prog | train_step163_progressive_kd | (See RELEVANT #15 above) |
| 166 | train_step166_further_stack | Queue says ALL KILLED; JSON absent — likely STALE |
| 204 (json) | train_step204_n4096_d16_khh2_kiter6_tier2 | Queue says DONE (97.15%); JSON absent — missing artifact, experiment complete |
| 216 (json) | train_step216_compound_winners | Queue says DONE (killed); JSON absent — missing artifact |

---

## Prune-Safe-to-Delete (STALE + DUPLICATE)

**STALE (44 scripts) — design assumption confirmed invalid:**

```
train_step13_beam_iter_scaling.py
train_step28_gen3_compound.py
train_step32_gen4_compound.py
train_step33_d128.py
train_step33b_d128_calib.py
train_step34_mod_adaptive_kiter.py
train_step37_phase_matrix_bank.py
train_step39_mechanism_loss.py
train_step41_ojas_rule.py
train_step42_signed_d64_calib.py
train_step44_beam_signed.py
train_step46_wphase_reconnect.py
train_step47_interneurons_d64.py
train_step50_spatial_dynamic_conn.py
train_step51_spatial_phase_gating.py
train_step52_high_d_routing.py
train_step53_lowrank_mixing.py
train_step56_n_scaling.py
train_step63_act_gated_routing.py
train_step65_dist_phase_routing.py
train_step67_safety_ablation.py
train_step68_kiter_gen4.py
train_step69_corrected_baseline.py
train_step70_fullscale_gen4plus.py
train_step75_temp_routing.py
train_step79_aux_loss_patched.py
train_step80_nscale_patched.py
train_step81_hebbian_rewiring.py
train_step88_alpha_sweep_n4096.py
train_step92_relu_group_routing.py
train_step100_kin_sweep.py
train_step102_phase_polarizer.py
train_step103_wave_interference.py
train_step104_compound_wave_polar.py
train_step107_group_moe.py
train_step108_polar_routing.py
train_step110_muon_optimizer.py
train_step112_rnn_input_injection.py
train_step113_intermediate_supervision.py
train_step114_parallel_chunks.py
train_step118_attention_readout.py
train_step119_adaptive_kiter.py
train_step123_stochastic_depth.py
train_step130_beam_broadcast.py
train_step140_nk_tradeoff.py
```

**DUPLICATE (6 scripts) — superseded by executed version:**

```
train_step106_perstep_embeddings.py    (step115 covers N=4096; step106 old N=1024 D=64 version)
train_step115_zbias_redistribution_n4096.py  (both mechanisms superseded by ΔW proj track)
train_step120_high_kiter_zbias.py      (K_iter range fully mapped; Z-bias superseded)
train_step131_tier1_winners.py         (step131 results in queue as DONE)
train_step217_polarizer_routing.py     (step217b Tier-1 DONE; Tier-0 superseded)
train_step223_cifar10_cross_dataset.py (DESIGN FLAW documented; replaced by step400 DONE)
```

---

## Still Worth Running (Top 10 RELEVANT)

Ranked by expected paper value. Note: many scripts in ranks 18–39 have experiments
that are DONE per queue but lack result JSONs — those need artifact recovery, not re-runs.

| Rank | Step | Script | Rationale |
|------|------|--------|-----------|
| 1 | 760 | train_step760_seed_variance | **Paper-critical:** grounds every ±pp claim. Seeds for step199 + step706 (efficiency records). Reviewer will ask: "is this within noise?" |
| 2 | 750 | train_step750_kiter3_n4096_khh4 | Latency Pareto track: K_iter=3 at N=4096 K_hh=4; fills gap in K_iter floor curve; completes K_iter reduction story for paper |
| 3 | 730 | train_step730_proj_khh4_compound | ΔW proj × K_hh=4 compound — orthogonal axes; potential +2pp at N=2048; strengthens efficiency claim |
| 4 | 751 | train_step751_kiter3_n8192_khh8 | Extends K_iter=3 floor curve to N=8192 K_hh=8; inline latency benchmark; paper Pareto figure data |
| 5 | 231 | train_step231_mechanism_diagnostics | Mechanistic probes for paper: class specificity, AH separation, iterative refinement; zero training risk; explanatory content only |
| 6 | 740 | train_step740_connga_v2 | ConnGA v2 with 30ep/child (v1 had noisy 10ep scoring); topology search question genuinely open |
| 7 | 218 | train_step218_random_projection_ablation | AH prerequisite claim needs clean JSON artifact; queue notes "Freezing bug, needs rerun"; paper claim: +81pp from learned W_pos |
| 8 | 229 | train_step229_rigl_topology | RigL at efficiency config — step124-B at N=1024 gave +6.82pp (largest single gain); efficiency config never tested; step709 ConnGA killed ≠ RigL killed |
| 9 | 230 | train_step230_gumbel_topology | Differentiable topology via Gumbel-Top-K; conceptually distinct from evolutionary search (step709); potential topology learning win |
| 10 | 125 | train_step125_alpha_fine_sweep | AH alpha 1.0–1.3 range untested at N=4096; step321 covers efficiency config; D=64 track completeness for paper Table 2 |

**Note on ranks 18–39:** Steps 204, 206, 216, 218, 220–226, 238, 306, 401, 520, 611, 703–708, 712–713, 715, 720, 723–724, 729, 707, 707_tier2 are all recorded as DONE in the experiment queue but lack result JSON files. These are **missing artifact recovery** tasks, not new runs.

---

*Audit methodology: Classified by cross-referencing script docstrings with Dead Ends table in `learnings/INDEX.md`, Confirmed Laws table, and `learnings/EXPERIMENT_QUEUE.md` status entries. Scripts without docstrings or with ambiguous design assumptions flagged UNKNOWN.*
