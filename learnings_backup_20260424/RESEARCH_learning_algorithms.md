# Learning Algorithms for SGNNET — Survey Index

**Date:** 2026-04-11  
**Status:** Research complete. Efficiency milestone (step199: 95.52% @ 0.98M FLOPs) achieved.  
**Question:** Is backpropagation the best learning algorithm for SGNNET? Can we learn conn_hh?

**Detail files:**
- [Part 1: Algorithm Survey](RESEARCH_learning_algorithms_p1.md) — backprop, PCN, FF, EqProp, CHL, PEPITA/DFA, DTP
- [Part 2: Topology Learning + Recommendations](RESEARCH_learning_algorithms_p2.md) — RigL, CH, Gumbel-Softmax, REINFORCE, NRI, STDP, experiment queue

---

## TL;DR (5 key findings)

1. **Backprop is not the bottleneck — fixed topology is.** Step124-B showed RigL topology learning gave +6.82pp at N=1024 (largest single mechanism gain ever). The random conn_hh has never been trained and is likely a major accuracy ceiling.

2. **Three topology learning approaches are ready to test:** (a) RigL with gradient regrowth — already empirically confirmed in SGNNET; (b) Cannistraci-Hebb (CH) gradient-free topology — ICLR 2024, works best at ultra-sparse regime SGNNET operates in; (c) Gumbel-Softmax differentiable edge selection — direct N×K_candidates parameterization, 2024 Decoupled ST-GS improves gradient fidelity.

3. **Oja's Rule is a low-cost backprop replacement for W_pos.** Native to S^{D-1}, local, 2024 paper shows it "overcomes challenges of training under biological constraints." Good hybrid: Oja for W_pos, AdamW for fc_out. Can try in one 20-epoch scout.

4. **Contrastive Hebbian Learning (CHL) is the deepest theoretical match.** SGNNET's routing IS contrastive learning in disguise — free phase (K_iter routing) and clamped phase (routing toward target class). Single-phase CHL (2024) has same compute cost as backprop. Medium-term experiment.

5. **Predictive coding, EqProp, PEPITA, DTP are theoretically interesting but practically premature.** All require either more compute, architecture changes, or have known performance gaps on non-trivial tasks. Not near-term priorities.

---

## Algorithm Priority Table

| Algorithm | Changes | Topology? | Cost vs BP | Priority | Experiment |
|-----------|---------|-----------|-----------|----------|------------|
| RigL topology | Outer loop only | YES gradient-regrowth | Negligible | **IMMEDIATE** | step-next-A |
| CH topology | Outer loop only | YES gradient-free | Negligible | HIGH | step-next-B |
| Gumbel-Softmax K_hh | Add edge_logits param | YES differentiable | +20% | HIGH | step-next-C |
| Oja's Rule (W_pos) | Replace W_pos optimizer | No | Cheaper | MEDIUM | step-next-D |
| Single-phase CHL | New training loop | Edge weights | ~same | MEDIUM | Phase 6 |
| Predictive Coding EO | Major refactor | No | +50-100% | LOW | Phase 7+ |
| EqProp | Major refactor | Edge weights | 2-4× | LOW | Research |
| Forward-Forward | Shared W_pos problem | No | ~same | LOW | — |
| PEPITA / DFA | Performance gap | No | ~same | VERY LOW | — |
| DTP / FTP | Shared W_pos problem | No | 1-2× | LOW | — |
| REINFORCE | High variance | YES | 10-50× | SKIP | — |
| STDP | Subsumed by CH/RigL | Approx | Complex | SKIP | — |

---

## Immediate Action

**The topology learning opportunity is high-value and requires no algorithm change** — keep backprop for W_pos, add topology evolution as an outer loop:

```python
# Every M=5 epochs, after standard backprop:
if epoch % M == 0:
    conn_hh = rigl_topology_update(model, val_loader, K_drop=1)
    # or:
    conn_hh = ch_topology_update(model.W_pos, conn_hh, K_drop=1)
```

This is compatible with ALL current defaults (AH=1.0, K_iter=5, D=16, N=2048).
