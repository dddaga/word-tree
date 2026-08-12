# IndraAstra × Mem0 — Partnership Pitch

## What IndraAstra Does

IndraAstra provides AI research as a service. We embed into product teams as an AI
research function — running experiments, validating hypotheses, shipping research
artifacts, and building the operating infrastructure that keeps AI collaboration
disciplined at scale.

We are not a consultancy that writes reports. We build, test, and deliver results that
are traceable to empirical evidence. Every claim is tagged: CONFIRMED (clean ablation,
one variable changed, control present), HYPOTHESIS (post-hoc or cited-only), or STALE
(true on prior arch, needs retest). No inflation, no hand-waving.

## Why Mem0 Specifically

Mem0 is building memory infrastructure for AI systems. SGNNET — IndraAstra's flagship
research project — is a study in memory efficiency inside neural networks. The
vocabulary overlaps: what does a model need to remember, at what granularity, at what
cost? SGNNET answers that question at the parameter and FLOP level; Mem0 answers it at
the retrieval and storage level. There is a direct research fit, not a stretch.

Concretely: if Mem0 wants to research efficient memory retrieval at scale, or benchmark
memory footprint tradeoffs for different retrieval architectures, IndraAstra already has
the experimental methodology, the multi-machine compute infrastructure, and the
evidence-tagging discipline to run that work cleanly.

## The SGNNET Proof

What we built: a sparse graph neural network (SGNNET) as a drop-in replacement for the
FC head of VGG16. Trained on Imagenette and CIFAR-10.

Results:
- 95.95% accuracy on Imagenette (vs VGG16 FC baseline)
- 0.20M FLOPs — 0.16% of VGG16's FC compute
- 34,976 parameters — 0.029% of VGG16 FC
- 12.7 µs wall-time at B=32 on RTX 5060 Ti — 5.26× faster than baseline
- CIFAR-10 confirmed: 80.57% ±0.12pp over 3 seeds
- Pareto-dominates VGG16 FC on 4 of 5 efficiency dimensions

How we built it: hundreds of experiments across 6 training slots on 3 machines. Every
experiment placed on a three-tier ladder: T0 (20 ep, 50% data, rejection filter), T1
(75 ep, 50%, calibration), T2 (150 ep, 100%, paper validation). T0 predicts T2 winner
with ρ=0.80 — good enough to kill 80% of dead ends before they consume compute.

Infrastructure: tmux-based slot management, lock files, RAM gates, a single launch
wrapper, Graphiti temporal knowledge graph for cross-session memory persistence.

What this proves about IndraAstra: we run AI-assisted research the same way a
well-managed lab runs experiments — with discipline, traceability, and a methodology
designed to avoid false positives.

## Engagement Model Options

### (a) Embedded Sprint
3 engineers for 3 months. Fixed scope defined upfront. Deliverable: research findings,
validated hypotheses, working artifacts. Pricing: [₹X] for the sprint.

### (b) Retainer
Ongoing research capacity. [N engineers] embedded. Monthly cadence. Suitable for teams
that have a steady stream of open research questions and want a team that compounds
context over time rather than re-ramping every quarter. Pricing: [₹X]/month.

### (c) Specific Project Scope
Example scopes Mem0 could hand off:
- "Research efficient memory retrieval at scale — benchmark 4 architectures, report
  Pareto table across latency / memory / accuracy."
- "Design and validate a memory compression scheme for stored AI conversation context."
- "Ablation study: what retrieval granularity optimizes downstream LLM task accuracy?"

Each scoped project comes with a defined experiment plan, a tier protocol, and a final
report with evidence tags on every conclusion.

## Risk Comparison

| | Hiring 1 FTE at ₹1 Cr/yr | IndraAstra partnership |
|---|---|---|
| Team size | 1 | [N engineers] |
| Ramp time | 4-6 months | Near-zero (methodology already built) |
| Single point of failure | Yes | No |
| Termination cost | High (notice, severance, knowledge loss) | Cancel-anytime |
| Track record | Unknown until hired | Visible in attached transcripts |
| Scope flexibility | Fixed headcount | Adjustable per engagement |

Hiring two engineers at ₹2 Cr/yr total is not wrong. But it is a bet on two individuals.
IndraAstra is a bet on a system — methodology, infrastructure, and a team that already
knows how to work together.

## Contact and Next Step

Reply to this email or schedule directly: [calendar link placeholder]

Dhiraj Daga
CTO, IndraAstra
dhiraj.daga@indraastra.in

Attached in the zip: raw Claude Code session transcripts (`.jsonl`), project summary,
best practices, transcript excerpts, and the operating contract (`CLAUDE.md` snippet).
