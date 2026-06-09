# Architecture Dead Ends

Confirmed dead ends for SGNNET. Do not re-propose without new theoretical motivation.

## Structural Regularization (step152) [CONFIRMED, N=1024 D=16, 75ep]

**Finding:** External structural constraints universally destroy performance. Network self-organizes optimal structure.

| Mechanism | Delta | Why It Fails |
|-----------|-------|--------------|
| Nuclear norm regularization | −15.11pp | Forces low-rank Z; SGNNET uses full rank for routing diversity |
| Contrastive routing | −13.02pp | Explicit competition disrupts emergent specialization |
| Nuclear+bottleneck+dim_gate compound | −13.81pp | Compound of failures |
| L1 sparsity | −3.34pp | Sparsifies weights SGNNET routing depends on |
| Dim gate | −3.31pp | Gating on active dims kills signal flow |
| Bottleneck D→8→D | −0.05pp | Neutral — least harmful, not beneficial |

**Root cause:** SGNNET emergent routing already near information-theoretic optimum for given N/D/K. External constraints fight learned organization. Closes structural regularization direction.

## Pruning During Training (step153) [CONFIRMED, N=1024 D=16, 75ep]

**Finding:** All pruning-during-training methods fail. SGNNET needs stable connectivity throughout training — routing paths co-adapt, removing them mid-training destroys learned structure.

| Mechanism | Delta | Why It Fails |
|-----------|-------|--------------|
| Nested dropout | −14.76pp | Randomly removes paths during co-adaptation phase |
| Prune neurons (GMP) | −12.46pp | Removes load-bearing neurons (100% utilization — no dead neurons) |
| Prune both dims+neurons | −12.13pp | Compound failure |
| Matformer | −9.97pp | Progressive capacity reduction incompatible with iterative refinement |
| Prune dims (GMP) | −0.62pp | Least harmful — still negative |

**Supporting evidence:** step155 shows 100% neuron utilization all configs — no dead neurons to safely prune. Every neuron actively used.

## Other Confirmed Dead Ends

| Mechanism | Key Step | Why Dead |
|-----------|----------|----------|
| Multiplicative gating (all forms) | steps 58-66 | Gate-death: g^K_iter → 0 (see [[gate_death]]) |
| Group MoE sparsification | step83, step107 | Routing capacity collapse |
| Markov routing | step129 | −50 to −71pp; `F.normalize` + static AH are load-bearing |
| Attention readout | step118 | −60 to −67pp; mean-pool readout load-bearing (see [[readout]]) |
| Stochastic depth | step123 | −35 to −61pp; every K_iter step essential |
| Beam broadcast | step130 | −3 to −40pp; hurts local routing |
| Adaptive K_iter (ACT) | step119 | −2 to −7pp |
| N×K tradeoff (more K for less N) | step140 | N dominates; more K hurts |
| RNN sequential injection | step112 | Zero-state bootstrapping failure |
| DropMessage at K_hh=4 | step96 | Too sparse for any dropout |
| Safety valve loss | step154 | Redundant with AH; +9.75pp when removed |
| Load balance loss | step79 | Only +0.21pp; below noise |
| Signed coupling at D=64 | step49 | cos-sim on S^63 = noise |
| Dynamic Z-KNN | step31 | Unstable on S^63 per step |
| Gumbel-Softmax differentiable topology | step230 | All 3 configs KILLED at 20ep T0. Best 48.74% vs ref 91.8% (Δ≈−43pp). Topology entropy stays near uniform (98% of max) — straight-through estimator fails to learn discrete edge structure during training. Decoupled ST-GS (τ_fwd=0.1, τ_bwd=1.0) didn't help. Confirms: simultaneous training + topology learning fails (co-adaptation bug). |

## Dynamic Connectivity at N=512 [CONFIRMED DEAD, steps 511/512/513/514, all 6 variants]

4 experiments, 6 mechanism variants, all fail. Comprehensive closure at efficiency scale.

| Step | Rule | Protocol | Δ vs Ref_static |
|------|------|----------|-----------------|
| 511 | co_act_low destructive | K_hh=2, every 5ep, all-replace | **−3.21pp** |
| 512-A | co_act_low guarded | K_hh=2, every 15ep, ≤1 swap, δ=0.02 | **−2.34pp** |
| 512-B | co_act_hi guarded | K_hh=2, every 15ep, ≤1 swap, δ=0.02 | **−2.24pp** |
| 513-A | co_act_low K_hh=4 | K_hh=4, guarded | **−4.94pp** |
| 513-B | co_act_hi K_hh=4 | K_hh=4, guarded | **−2.42pp** |
| 514 | additive K_hh expansion | K_hh 2→3→4→5 non-destructive | **−0.56pp** |

**Root cause:** At N=512 on Imagenette: (a) pairwise activation correlations at noise level (50% data, 9.5K train), (b) any topology deviation from warmup-learned state disrupts co-adapted W_pos/θ/C_ho, (c) even non-destructive edge addition hurts because readout C_ho tuned to original K_hh=2 routing. Static small-world near-optimal at this scale.

**Design rule:** Do not attempt dynamic connectivity at N≤2048 on Imagenette without jointly-trained routing policy. Future revisit should (i) use richer data for correlation signal, or (ii) learn rewiring rule end-to-end instead of hand-coding correlation thresholds.

## Connectivity Genetic Algorithm (ConnGA v2) [CONFIRMED DEAD, steps 740/741, N=512]

**Hypothesis:** Evolutionary search over hidden-layer connectivity (conn_hh) can find better topologies than random small-world init.

**Protocol:** Population=8, 4 generations, 30–40 epochs/child per eval. Three scoring modes tested. Elite carried forward. Weights retrained from scratch each eval.

| Variant | Scoring | best_seen | Ref | Δ | Steps |
|---------|---------|-----------|-----|---|-------|
| step740 | softmax (τ=0.5) | 72.18% | 79.11% | **−6.93pp** | step740 |
| step741 | rank | 76.05% | 77.81% | **−1.76pp** | step741 |
| step742 | top_k_avg (k=2) | 74.65% | 77.40% | **−2.75pp** | step742 |

**Gen progression step741 (rank):** 74.37 → 76.05 → 74.29 → 74.22. Peaked Gen2, then degraded — no sustained improvement from evolution.

**Root cause (HYPOTHESIS):** Random small-world topology already captures key property (local structure + random long-range shortcuts). GA mutation/crossover on integer edge indices explores space where most perturbations neutral or harmful. Fitness landscape flat near small-world optimum — random init lands in wide basin, so evolution provides no gradient signal.

**Supporting evidence from seed variance:** AH-only at N=512 has σ≈0.5pp (see step760). ConnGA best_seen variance across children within one gen ~2–3pp — much larger than σ. 30ep evaluation noisy relative to true fitness, making selection unreliable.

**All three scoring variants failed (CONFIRMED):** softmax −6.93pp, rank −1.76pp, top_k_avg −2.75pp. No variant found topology better than random small-world. ConnGA track closed.

**Design rule:** Do not attempt topology search via GA on SGNNET. Random small-world optimal; no evolutionary topology search has beaten it.

## See Also

- [[gate_death]] — unified theory why multiplicative mechanisms fail in iterative routing
- [[antihebbian]] — why AH alone optimal; compounding anything onto it fails
- [[readout]] — global mean-pool failure; C_ho requirement
- [[normalization]] — LayerNorm winner (+2.24pp); RMSNorm and pre_route normalize dead