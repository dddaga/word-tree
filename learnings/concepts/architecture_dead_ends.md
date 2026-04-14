# Architecture Dead Ends

Confirmed dead ends for SGNNET. Do not re-propose without new theoretical motivation.

## Structural Regularization (step152) [CONFIRMED, N=1024 D=16, 75ep]

**Finding:** External structural constraints universally destroy performance. The network self-organizes its own optimal structure.

| Mechanism | Delta | Why It Fails |
|-----------|-------|--------------|
| Nuclear norm regularization | −15.11pp | Forces low-rank Z; SGNNET uses full rank for routing diversity |
| Contrastive routing | −13.02pp | Explicit competition disrupts emergent specialization |
| Nuclear+bottleneck+dim_gate compound | −13.81pp | Compound of failures |
| L1 sparsity | −3.34pp | Sparsifies weights SGNNET routing depends on |
| Dim gate | −3.31pp | Gating on active dims kills signal flow |
| Bottleneck D→8→D | −0.05pp | Neutral — least harmful, but not beneficial |

**Root cause:** SGNNET's emergent routing already operates near information-theoretic optimum for the given N/D/K. External constraints fight learned organization. Closes structural regularization direction.

## Pruning During Training (step153) [CONFIRMED, N=1024 D=16, 75ep]

**Finding:** All pruning-during-training methods fail. SGNNET needs stable connectivity throughout training — routing paths co-adapt and removing them mid-training destroys the learned structure.

| Mechanism | Delta | Why It Fails |
|-----------|-------|--------------|
| Nested dropout | −14.76pp | Randomly removes paths during co-adaptation phase |
| Prune neurons (GMP) | −12.46pp | Removes load-bearing neurons (100% utilization — no dead neurons) |
| Prune both dims+neurons | −12.13pp | Compound failure |
| Matformer | −9.97pp | Progressive capacity reduction incompatible with iterative refinement |
| Prune dims (GMP) | −0.62pp | Least harmful — but still negative |

**Supporting evidence:** step155 shows 100% neuron utilization in all configs — there are no dead neurons to safely prune. Every neuron is actively used.

## Other Confirmed Dead Ends

| Mechanism | Key Step | Why Dead |
|-----------|----------|----------|
| Multiplicative gating (all forms) | steps 58-66 | Gate-death: g^K_iter → 0 (see [[gate_death]]) |
| Group MoE sparsification | step83, step107 | Routing capacity collapse |
| Markov routing | step129 | −50 to −71pp; F.normalize + static AH are load-bearing |
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

## See Also

- [[gate_death]] — unified theory of why multiplicative mechanisms fail in iterative routing
- [[antihebbian]] — why AH alone is optimal; compounding anything onto it fails
- [[readout]] — global mean-pool failure; C_ho requirement
- [[normalization]] — LayerNorm winner (+2.24pp); RMSNorm and pre_route normalize dead
