# Design Discussions — INDEX

This file is a table of contents. All discussions have been split by date into sub-files.

---

## Sub-Files

| File | Date | Key Topics |
|------|------|------------|
| [LEARNINGS_design_2026_04_04.md](LEARNINGS_design_2026_04_04.md) | 2026-04-04 | Fourier phase routing, resonance-gated excitatory, AntiHebb adaptations, ablation protocol |
| [LEARNINGS_design_2026_04_05_06.md](LEARNINGS_design_2026_04_05_06.md) | 2026-04-05 to 2026-04-06 | Phase routing architecture details, decay mechanism, wave-1 failure analysis, redistribution principle |
| [LEARNINGS_design_2026_04_07_08.md](LEARNINGS_design_2026_04_07_08.md) | 2026-04-07 | Group topology design, group routing design (step83), dynamic routing gate-death synthesis |
| [LEARNINGS_design_2026_04_08.md](LEARNINGS_design_2026_04_08.md) | 2026-04-08 | step83 post-mortem failure analysis, step87 proximity architecture design |
| [LEARNINGS_design_2026_04_09.md](LEARNINGS_design_2026_04_09.md) | 2026-04-09 | Gemma4/PolarQuant-inspired designs (steps 106-109), 50-experiment gap analysis, FLOPs path |
| [LEARNINGS_design_2026_04_10.md](LEARNINGS_design_2026_04_10.md) | 2026-04-10 | Constraint discovery (step152), progressive capacity (step153), safety/LB removal, diagnostics system, checkpoint system, paper track |
| [LEARNINGS_design_2026_04_14.md](LEARNINGS_design_2026_04_14.md) | 2026-04-14 | K_hh scaling rule confirmed, latency-Pareto track, seed variance ΔW proj vs AH-only, ConnGA v2 scoring |

---

## Key Design Decisions (Summary)

| Decision | Date | Outcome |
|----------|------|---------|
| Phase routing → redistribution, not gating | 2026-04-06 | Gate-death theorem established; softmax redistribution is the viable path |
| Group topology for hidden neurons | 2026-04-07 | step82 scripted and run; n_groups=8 wins (+3pp) |
| Group state routing (step83) | 2026-04-07 | Scripted and run; KILLED (S_g=mean too coarse, temporal mismatch) |
| Pure proximity architecture (step87) | 2026-04-08 | Scripted as train_step87_proximity_routing.py |
| Redistribution routing wins at N=1024 | 2026-04-08 | step73 D (+1.78pp), step75 D (+3.98pp) — first dynamic routing wins |
| K_hh=4 new default | 2026-04-09 | step86 confirms K_hh=4 → +0.56pp AND −18% FLOPs |
| Safety valve loss REMOVED | 2026-04-10 | step154: +9.75pp without safety; AH handles positional diversity. lambda_safety=0.0 |
| Load balance loss REMOVED | 2026-04-10 | Only +0.21pp evidence; lambda_lb=0.0. Both removed from computation graph |
| Diagnostics-driven methodology | 2026-04-10 | All new experiments must use TrainingDiagnostics; go beyond loss/accuracy |
| Checkpoint system added | 2026-04-10 | Save/resume full state; best-model tracking. Integrated into Trainer.train() |
| N×K tradeoff KILLED | 2026-04-10 | step140: N dominates, more K hurts. K_hh=4 optimal confirmed |
| Constraint discovery track | 2026-04-10 | step152: nuclear norm, bottleneck, dim gating, L1, contrastive — test core hypothesis |
| Progressive capacity reduction | 2026-04-10 | step153: Matformer/GMP-style prune from large N/D during training |

---

## Pending Design Discussions (see PENDING_DISCUSSIONS.md)

Ideas discussed but not yet scripted:
- Redistribution routing at N=4096 (G4)
- Phase alignment as softmax weight (step60 redemption)
- N-scaling on patched arch (step72)
- Stacked SGNNET parallel on patched arch (step85)
- GRAND-style implicit diffusion (step93)
- Hamiltonian message passing (step94)
- Beltrami flow joint pos+feature evolution (step97)
- Dynamic group KNN topology (step98)
- Next-gen dynamic routing informed by diagnostics (persistent goal)
