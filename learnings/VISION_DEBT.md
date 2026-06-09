# Vision Debt Register — sparse_geometric_network_report.md promises

Reviewed at every meditation. Any NEVER-TESTED item older than 2 meditations gets a T0
or explicit user-approved WONTFIX. Full audit: [VISION_REVIEW_2026-06-10.md](VISION_REVIEW_2026-06-10.md).

| Brief promise | Status | Evidence | Action |
|---|---|---|---|
| Sparsity O(N·K) | KEPT/EXCEEDED | 34,976 params, 0.029% VGG FC | — |
| Recursive K-iteration loop | KEPT | K_iter=5 load-bearing; K=1 via KD (step605) | — |
| Dynamic connectivity (multiplicative/gated forms) | KILLED-CONFIRMED | gate-death (58–66), step852 8/8, PhaseGate 985/987/988 | no re-attack without new variable |
| Dynamic connectivity (additive r*-threshold, brief §3.5 form) | NEVER-TESTED at modern base | kills predate N=2048/D=16/ΔW base → STALE | **step990 T0** |
| Self-projection readout | KEPT | step118 alternatives killed | — |
| Per-step normalization | KEPT | step958 alternatives killed | — |
| Dead-zone Coulomb safety valve | KILLED-CONFIRMED | step154: removal +9.75pp | — |
| Load-balancing loss | KILLED-CONFIRMED (redundant) | step155: AH gives 100% utilization | — |
| Hebbian prune-and-grow topology (brief §9) | NEVER-TESTED at spec | variants killed (511–514, 740–742, 230); exact epoch-boundary \|c_ij\| prune-grow untested | **step991 T0** |
| K-means init (brief §6.1) | NEVER-TESTED | abandoned without experiment | **step992 T0** |
| Adaptive K stopping (brief §3.4 future) | PARTIALLY TESTED | MoD killed (34, 119); inference-time convergence early-exit untested | low priority; optional T0 |
| Transformer FFN replacement (brief §1 problem statement) | NEVER-TESTED | pivot to VGG FC undocumented, zero experiments | **step989 T0** — decides Paper 2 spine |
| Distillation training method | KEPT (target mutated) | VGG soft-label KD production path | — |
| Hypercube confinement | EVOLVED → S^{D-1} | Fourier breakthrough 2026-03-26 | — |
