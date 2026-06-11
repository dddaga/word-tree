# Paper Audit — Gap Status 2026-05-13

Original audit: PAPER_AUDIT_2026-04-20.md (12 gaps, 7 HIGH priority)

## HIGH Priority Gaps

| Gap | Description | Status |
|-----|-------------|--------|
| GAP 3 | ΔW-proj routing entirely absent from manuscript | **CLOSED** |
| GAP 1 | Efficiency champion still step199; step605 not mentioned | **CLOSED** |
| GAP 2 | "67K params" STALE in Appendix B | **CLOSED** |
| GAP 4 | Multi-seed variance not in manuscript | **CLOSED** |
| GAP 5 | Baselines listed as "NOT DONE" | **CLOSED (partial)** |
| GAP 6 | CIFAR-10 in future work, not results | **CLOSED** |
| GAP 12 | Headline accuracy "95.52%" not updated to 96.38% | **CLOSED** |

## Changes Made

### MANUSCRIPT_DRAFT_sec_dw_proj.md (NEW, 103 lines)
- §3.5 ΔW-Projection Routing: direction vectors, projection coefficient, reflection memory, full routing update, precomputed geometry
- step950 CONFIRMED: Z-collapse with gather-sum + W_edge without ΔW-proj
- step951 CONFIRMED: routing_gain always negative — routing = spatial smoothing
- step967 CONFIRMED: dense non-selective router, zero pathway specialisation
- Proximal gradient interpretation tagged **(HYPOTHESIS)**
- §5.8 ΔW-Projection Ablation: component table (T0+T1 deltas), geometry ablation, cross-dataset transfer, canonical multi-seed result

### MANUSCRIPT_DRAFT_sec2_arch_experiments_findings.md (edited, 200 lines)
- §4.4: "213 experiments" → "~943 experiments across Phases 5–6"; "27 routing variants" → "30+"
- §5.2: N=2048 accuracy 95.52% → 96.38% ± 0.18pp (step887)
- §5.5: removed stale "27 killed mechanisms" count
- §5.8 stub added linking to new sec_dw_proj.md with key results inline

### MANUSCRIPT_DRAFT_sec3_efficiency_discussion.md (edited, 148 lines)
- §6.2: Added step887 row (96.38% ± 0.18pp, canonical) and step605 row (95.95%, efficiency champion); step199 demoted to "Pre-ΔW-proj canonical (legacy)"
- §6.3: Updated to lead with step887 headline and step605 efficiency champion numbers
- §6.4 (NEW): Multi-seed variance — step887 ±0.18pp, step980 CIFAR-10 ±0.12pp, halving vs pre-ΔW-proj
- §7.3: Replaced "Baselines absent" with cross-dataset CIFAR-10 results + MLP/GNN baselines table
- §8: Replaced "NOT DONE" baselines table with status table (MLP DONE, GNN DONE, RandProj DONE, CIFAR-10 DONE, Pruned VGG16 NOT YET)
- §9: CIFAR-10 removed from future work (replaced by "partial" note with pointer to §7.3)

### MANUSCRIPT_DRAFT_sec4_appendices.md (edited, 121 lines)
- Appendix B: Total params 67,744 → **34,976** with correct formula ($N \times D + N + 10 \times D$)
- Appendix B header: "Final Config (step199)" → "Canonical Config (step887/step605)"

## Remaining Open Gaps — Updated 2026-06-11

| Gap | Description | Status |
|-----|-------------|--------|
| GAP 7 | Audio negative result | **CLOSED** — MANUSCRIPT_DRAFT_sec5_multimodal.md §8.3 (2026-06-11) |
| GAP 8 | Time-series results | **CLOSED** — MANUSCRIPT_DRAFT_sec7_timeseries.md §10 (2026-06-11); ts_step030 T0 NEGATIVE |
| GAP 9 | Scaling law section | **CLOSED** — MANUSCRIPT_DRAFT_sec6_scaling.md §9 (2026-06-11) |
| GAP 10 | GLNN distillation cross-dataset | **CLOSED** — MANUSCRIPT_DRAFT_sec_glnn.md §7 (2026-06-11); step621: neutral for large, −60pp for tiny |
| GAP 11 | Theoretical section | **CLOSED (minimal)** — MANUSCRIPT_DRAFT_sec_theory.md §3 (2026-06-11); complexity + geometry + ΔW-proj theory |
| — | Pruned VGG16 FC baseline | **CLOSED (moot)** — 25088→10 at 13.9% sparsity = sparse linear layer, not a meaningful comparison. Existing baselines (Lin_direct=97.12%, MLP h=37=97.71%) cover this space. |
| — | MLP FLOPs-matched baseline | **CLOSED** — step403b: MLP h=37 = 97.71% @ 1.857M FLOPs, 928K params. SGNNET: 96.38% @ 0.98M FLOPs, 34,976 params (26× fewer params). |

## Remaining work before final manuscript

| Item | Status |
|------|--------|
| step982 T2 A_aug result | PENDING — converter relaunched 2026-06-11; launch when done |
| Multi-seed step887 (Imagenette canonical) | DONE — 96.38% ±0.18pp (3 seeds) |
| CIFAR-10 paper claim | DONE — 80.57% ±0.12pp (3 seeds, step980) |
| Abstract/intro update | **DONE** — 2026-06-11: step989 KILLED added; audio nuance (routing hurts); TS negative; §8-10 cited in abstract; sec1 split into sec1 (89L) + sec1b_architecture (119L) |

## Verification

- All updated numbers sourced from claims.md or findings_log_part3.md (CONFIRMED results only)
- No numbers invented; speculative framings tagged (HYPOTHESIS)
- All edited files are at or under 200 lines
- sec1 split 2026-06-11: sec1_abstract_intro_related (89L) + sec1b_architecture (119L)
