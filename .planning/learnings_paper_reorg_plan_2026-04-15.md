# learnings/ Reorganization Plan — 2026-04-15

**Auditor:** Claude Sonnet 4.6  
**Status:** PROPOSAL — do not execute until user approves per-section

---

## 1. Inventory

### 1a. learnings/ root (62 files total, excluding .DS_Store and macOS metadata)

**Live operational files (update every session)**

| File | Lines | Description |
|---|---|---|
| `EXPERIMENT_QUEUE.md` | 501 | Primary live queue — active, pending, done experiments with configs |
| `WINNERS.md` | 121 | All configs that crossed param+FLOPs+accuracy thresholds; mechanism detail |
| `PENDING_DISCUSSIONS.md` | 486 | Design ideas discussed but not yet scripted (PENDING/SCRIPTED/KILLED per entry) |
| `INDEX.md` | 182 | Master index: concept pages, confirmed laws, dead ends, active directions |
| `LEARNINGS_ops.md` | 162 | Ops reference: MPS gotchas, stability, perf benchmarks, known script bugs |

**Sequential experiment log (audit trail — append-only, date-ordered)**

| File | Lines | Description |
|---|---|---|
| `LEARNINGS_phase5_p1_ladder.md` | 193 | Steps 1-4, D-scaling ladder, dynamic connectivity research |
| `LEARNINGS_phase5_p2_diagnostics.md` | 127 | Regression diagnostics, bug fix, infra notes |
| `LEARNINGS_phase5_p3_breakthrough.md` | 147 | Steps 6-10b, D=16 breakthrough |
| `LEARNINGS_phase5_p4_d16.md` | 178 | Steps 11-15, new mechanisms |
| `LEARNINGS_phase5_p5_phase_mechanisms.md` | 169 | Phase mechanisms, W_phase experiments |
| `LEARNINGS_phase5_p6_gen2_experiments.md` | 204 | Steps 14-30: Gen2 inhibition, D-scaling, signed coupling |
| `LEARNINGS_phase5_p6_gen2_gen3.md` | 9 | **Pointer-only file** — links to split sub-files |
| `LEARNINGS_phase5_p7_gen3_arm_status.md` | 184 | Steps 33-44: Gen3 closures + ARM status |
| `LEARNINGS_phase5_p8_arm1_arm2.md` | 207 | ARM 1+2 experiment designs and results |
| `LEARNINGS_phase5_p9_arm3_arm5.md` | 157 | ARM 3+5 experiment designs and results |
| `LEARNINGS_phase5_p11_n_scaling.md` | 103 | N-scaling study (step56, steps 190-212) |
| `LEARNINGS_phase5_p12_proxwave.md` | 86 | Proximity wave, distance-based mechanisms |
| `LEARNINGS_phase5_p13_reflection.md` | 134 | Reflection routing, alpha_reflect calibration |
| `LEARNINGS_phase5_p14_wave1_verdict.md` | 123 | Wave-1 verdict: all multiplicative gates killed |
| `LEARNINGS_phase5_p15_post_wave1.md` | 46 | **Index/pointer only** — TOC for p15a–p15e |
| `LEARNINGS_phase5_p15a_wave1_closure.md` | 187 | Steps 60-68 killed experiments |
| `LEARNINGS_phase5_p15b_arch_patch.md` | 202 | Steps 66/69/70 — +9.83pp patch story |
| `LEARNINGS_phase5_p15c_kiter_flops.md` | 193 | Steps 71-83: K_iter sweep, routing experiments |
| `LEARNINGS_phase5_p15d_flops_track.md` | 133 | Steps 86/88: Pareto + K_hh defaults |
| `LEARNINGS_phase5_p15e_dynamic_routing_postmortem.md` | 177 | Dynamic routing post-mortem, gate-death analysis |
| `LEARNINGS_phase5_p15f_warmstart_efficiency.md` | 289 | **VIOLATION (289 lines)** Warm-start + efficiency track breakthrough |
| `LEARNINGS_phase5_p15g_flops_floor.md` | 249 | **BORDERLINE (249 lines)** FLOPs floor search: D-reduction, K_hh-reduction track |
| `LEARNINGS_phase5_p15h_kiter_axis.md` | 141 | K_iter axis: steps 190-199 K_iter sweep at efficiency config |
| `LEARNINGS_phase5_p15i_topology_analysis.md` | 83 | K_hh=2 topology graph properties (dead-end neurons, out-degree) |
| `LEARNINGS_phase5_p15j_belief_update.md` | 130 | Belief framework; steps 216-221; polarizer +1.27pp; AH prerequisite |
| `LEARNINGS_phase5_p16.md` | 111 | Steps 116, 149, 152, 153, 155: normalization, diagnostics, constraints |
| `LEARNINGS_design.md` | 52 | **Index/pointer only** — TOC for date-split design files |
| `LEARNINGS_design_2026_04_04.md` | 97 | Design: phase routing, resonance, ablation protocol |
| `LEARNINGS_design_2026_04_05_06.md` | 123 | Design: phase routing architecture, wave-1 failure analysis |
| `LEARNINGS_design_2026_04_07_08.md` | 161 | Design: group topology, gate-death synthesis |
| `LEARNINGS_design_2026_04_08.md` | 129 | Design: step83 post-mortem, step87 proximity architecture |
| `LEARNINGS_design_2026_04_09.md` | 305 | **VIOLATION (305 lines)** Gemma4/PolarQuant designs, FLOPs path, gap analysis |
| `LEARNINGS_design_2026_04_10.md` | 124 | Constraint discovery, diagnostics, safety removal |
| `LEARNINGS_design_2026_04_14.md` | 92 | K_hh scaling rule, latency-Pareto, ConnGA v2 scoring |
| `LEARNINGS_design_2026_04_15.md` | 232 | K_iter=4 ΔW discovery; dynamic connectivity closure; AH correction |

**Research/external literature**

| File | Lines | Description |
|---|---|---|
| `LEARNINGS_research.md` | 110 | Translation invariance gap, open hypotheses |
| `LEARNINGS_arch.md` | 103 | Structural decisions (W_pos weight_decay, bool masks, lambda_safety) |
| `LEARNINGS_sparse_attention_research.md` | 9 | **Pointer-only** — links to the two-part split below |
| `LEARNINGS_sparse_attention_mechanisms.md` | 114 | Part A: Longformer, BigBird, Reformer, FlashAttention, MoE survey |
| `LEARNINGS_sparse_attention_sgnnet.md` | 102 | Part B: Binding by synchrony + 7 SGNNET design principles |
| `RESEARCH_routing_mechanisms.md` | 135 | Literature survey on routing mechanisms (GCNII, GRAND, Hamiltonian MP) |
| `RESEARCH_learning_algorithms.md` | 58 | **Pointer-only** — links to p1/p2 split |
| `RESEARCH_learning_algorithms_p1.md` | 177 | Hebbian/anti-Hebbian, STDP, Oja's rule literature |
| `RESEARCH_learning_algorithms_p2.md` | 206 | Predictive coding, energy models, contrastive Hebbian |

**One-off review/audit documents**

| File | Lines | Description |
|---|---|---|
| `CLAUDE_MD_REVIEW_2026-04-14.md` | 256 | **VIOLATION (256 lines)** Karpathy-style CLAUDE.md gap analysis |
| `SCRIPT_AUDIT_2026_04_14.md` | 221 | 91 scripts classified as STALE/DUPLICATE/RELEVANT/UNKNOWN |
| `PRUNE_REVIEW_2026_04_14.md` | 65 | 50 scripts proposed for deletion with user confirmation checkboxes |

**Archival / historical**

| File | Lines | Description |
|---|---|---|
| `HISTORICAL_SPECS.md` | 136 | Phase 1-4 archival: data pipeline UAT, baselines, rejected approaches |
| `learningsA.md` | 217 | Early dynamc routing failure analysis (step700/701 series) — poorly named |
| `EXPERIMENT_REPORT.md` | 249 | **BORDERLINE (249 lines)** Comprehensive report generated 2026-04-08 |
| `EXPERIMENT_REPORT_ADDENDUM.md` | 57 | Overflow from EXPERIMENT_REPORT.md (step82/step83 results) |
| `EXPERIMENT_REPORT_VERIFICATION.md` | 203 | Data cross-check verifying EXPERIMENT_REPORT claims vs JSON |
| `EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md` | 89 | Critical findings extracted from the queue (early ARM results) |

### 1b. learnings/paper/

| File | Lines | Description |
|---|---|---|
| `README.md` | 40 | Paper materials index; thesis, modality table, file listing |
| `MANUSCRIPT_DRAFT.md` | 645 | **VIOLATION (645 lines)** Full manuscript draft v0.1 (2026-04-11) |
| `PAPER_OUTLINE.md` | 204 | Section-by-section outline with headline result, figures plan, baselines |
| `claims.md` | 108 | 7 core claims with CONFIRMED/NEEDS EVIDENCE status |
| `findings_log.md` | 313 | **VIOLATION (313 lines)** Chronological log of paper-worthy discoveries |
| `baselines_needed.md` | 53 | Blocking/strong-to-have/done/deferred baseline experiments |
| `figures_planned.md` | 31 | 7 figures + 3 tables with descriptions |

### 1c. learnings/concepts/

| File | Lines | Description |
|---|---|---|
| `antihebbian.md` | 145 | AH mechanism, alpha calibration, AH-as-prerequisite evidence |
| `gate_death.md` | 178 | Gate-death theorem, 8+ confirming experiments, redistribution fix |
| `n_scaling.md` | 147 | N-scaling law, D=16 ceiling, K_iter×N interaction |
| `k_iter.md` | 113 | K_iter optimal N-dependence, annealing/distillation killed |
| `phase_routing.md` | 279 | **VIOLATION (279 lines)** Full architecture docs for phase routing (now killed) |
| `softmax_routing.md` | 127 | Redistribution routing wins (step73/75), temperature routing |
| `group_topology.md` | 108 | Group topology results, n_groups=8 wins at N=1024, null at N=4096 |
| `normalization.md` | 49 | LayerNorm vs L2-sphere, RMSNorm killed, pending N=4096 |
| `readout.md` | 46 | C_ho requirement, mean-pool failure on unit-sphere activations |
| `architecture_dead_ends.md` | 98 | All confirmed dead ends by category |
| `delta_w.md` | 156 | ΔW projection mechanism, headroom curve, proj vs rot crossover |

---

## 2. Redundancy

**Overlapping content pairs:**

| Set | Files | Overlap | Action |
|---|---|---|---|
| R1 | `EXPERIMENT_REPORT.md` + `EXPERIMENT_REPORT_ADDENDUM.md` + `EXPERIMENT_REPORT_VERIFICATION.md` | Same 2026-04-08 report split across 3 files; findings superseded by INDEX.md confirmed laws | Archive all 3 |
| R2 | `EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md` | Dead ends and findings extracted from the queue; INDEX.md dead ends section is the maintained version | Archive |
| R3 | `PENDING_DISCUSSIONS.md` | Most entries from before step216 are SCRIPTED and ran; their outcomes are in INDEX.md/WINNERS.md. The still-PENDING items (G4, step72, step85, step93-98) may be superseded by current paper focus | Prune resolved entries; move truly-closed designs to archive |
| R4 | `PAPER_OUTLINE.md` + `MANUSCRIPT_DRAFT.md` | PAPER_OUTLINE contains its own figures section (§9 Figures Planned) and MANUSCRIPT_DRAFT has the same figures. `figures_planned.md` is a third copy of figure descriptions. Dataset table also duplicated between README and MANUSCRIPT_DRAFT | Merge figures_planned into PAPER_OUTLINE §9; keep README clean |
| R5 | `claims.md` (FashionMNIST dataset references) | `claims.md` references FashionMNIST (Claim 1, Claim 2) — the project uses Imagenette. This was a stale copy-paste | Fix in-place |
| R6 | `LEARNINGS_phase5_p6_gen2_gen3.md` + `LEARNINGS_sparse_attention_research.md` + `RESEARCH_learning_algorithms.md` | Each is a 9-line pointer to split files; adds no value, creates navigation friction | Delete pointer-only files; update INDEX.md to point directly to the content files |

---

## 3. Freshness — Stale Files

| File | Staleness Reason | Verdict |
|---|---|---|
| `LEARNINGS_phase5.md` (36 lines) | Index of early parts (p1-p4 only, step 9 numbers, 2026-03-29 sessions running); superseded by INDEX.md | Archive |
| `LEARNINGS_phase5_p15_post_wave1.md` (46 lines) | Pure TOC index, all content split to p15a-p15e; INDEX.md covers same | Archive |
| `EXPERIMENT_REPORT.md` + `_ADDENDUM.md` + `_VERIFICATION.md` | Generated 2026-04-08 when best was 97.38%; now best is 97.86% (step89-A); efficiency record 95.52% (step199). Project direction has shifted to efficiency. Numbers are stale | Archive |
| `learningsA.md` | Content covers step700/701 (parallelization kills); those findings are in INDEX.md dead ends; filename is opaque and breaks the naming convention | Rename and archive |
| `concepts/phase_routing.md` (279 lines) | Detailed docs on phase routing architecture. Phase routing is confirmed killed (Wave-1, gate-death theorem). 279 lines dedicated to a dead mechanism is dead weight. Status in INDEX.md: "Wave-1 killed". | Trim to ≤80 lines (keep failure analysis; cut architecture docs). Archive full version |
| `LEARNINGS_research.md` (110 lines) | Translation invariance gap is a pre-step69 concern; with VGG features + no spatial seeding at efficiency config, this is resolved by the feature extractor, not SGNNET. Open hypotheses reference arch that no longer exists | Archive or merge into LEARNINGS_arch.md |
| `PENDING_DISCUSSIONS.md` entries before step200 | step87, step91, step92 are SCRIPTED and ran (results in INDEX.md). step106, step107, step108 experiments ran and killed. step163 (progressive KD) ran and failed | Update status entries; consider archiving old section |

---

## 4. Size Violations (CLAUDE.md limit: ≤250 lines)

| File | Lines | Proposed Split |
|---|---|---|
| `EXPERIMENT_QUEUE.md` | 501 | **Keep as-is** — this is the live operational queue, not a learnings doc. The 250-line rule applies to `learnings/` knowledge files; EXPERIMENT_QUEUE is a planning artifact. Add a note to the file header clarifying this exemption. |
| `PENDING_DISCUSSIONS.md` | 486 | Split: `PENDING_DISCUSSIONS_active.md` (PENDING entries only, ~8 items) + archive everything SCRIPTED/KILLED to `archive/PENDING_DISCUSSIONS_archive.md` |
| `MANUSCRIPT_DRAFT.md` | 645 | Split at §4 boundary: `MANUSCRIPT_p1_intro_related.md` (§1-§3, ~200 lines) + `MANUSCRIPT_p2_arch_results.md` (§4-§6, ~250 lines) + `MANUSCRIPT_p3_deadends_discussion.md` (§7-§10, ~195 lines). Update `paper/README.md` to list all three. |
| `findings_log.md` | 313 | Split at chronological midpoint: `findings_log_p1_pre_efficiency.md` (pre-step180, steps establishing architecture fundamentals) + `findings_log_p2_efficiency_paper.md` (step180+, efficiency track, ΔW proj, CUDA). Update `paper/README.md`. |
| `LEARNINGS_design_2026_04_09.md` | 305 | Split: `LEARNINGS_design_2026_04_09a_flops_gap_analysis.md` (~150 lines — FLOPs path, gap analysis, 50-experiment review) + `LEARNINGS_design_2026_04_09b_new_mechanisms.md` (~155 lines — Gemma4/PolarQuant designs steps 106-109). Update `LEARNINGS_design.md` index. |
| `concepts/phase_routing.md` | 279 | Trim to ≤80 lines (keep: KILLED verdict, gate-death reason, redistribution principle. Cut: full architecture docs, phase split mechanics, code blocks for dead mechanism). Full version → `archive/concepts/phase_routing_full.md`. |
| `LEARNINGS_phase5_p15f_warmstart_efficiency.md` | 289 | Split at step163 boundary: `p15f_warmstart.md` (~145 lines, steps 149-163) + `p15f_efficiency_track.md` (~144 lines, steps 165-177). Update `LEARNINGS_phase5_p15_post_wave1.md` TOC and INDEX.md. |
| `CLAUDE_MD_REVIEW_2026-04-14.md` | 256 | This is a one-off review document, not a living learnings file. Archive to `archive/CLAUDE_MD_REVIEW_2026-04-14.md`. (Content incorporated into CLAUDE.md per the smoke-test/design-to-script rules already.) |

---

## 5. Index Completeness

**INDEX.md status:** largely up-to-date as of 2026-04-14. Gaps found:

| Gap | Detail |
|---|---|
| Missing from Sequential Logs table | `LEARNINGS_design_2026_04_15.md` (K_iter=4 ΔW discovery, AH correction, dynamic connectivity closure) — the most recent session log is not in the index |
| Missing from Sequential Logs table | `learningsA.md` — not listed; content covers step700/701 series |
| Stale entry | INDEX.md line 66 lists `LEARNINGS_phase5_p16.md` (steps 116, 149, 152, 153, 155) but notes steps 149/152/153 are from that range; step155 AH=anti-collapse is correct. Index entry is fine but the line should note it covers the safety-removal cluster |
| Active directions stale | Section "Active Research Directions" still lists step729 as "Running now" — this has completed. Update to current queue state |
| Confirmed Laws missing two entries | ΔW proj K_iter=4 discovery (LEARNINGS_design_2026_04_15) and dynamic connectivity closure (all variants negative) are not yet in the Confirmed Laws table |
| Dead Ends missing | Dynamic connectivity at N=512 (steps 511-514, 2026-04-15): 6 variants all negative. Not yet listed. |

**concepts/*.md cross-linking:**

Sparse. Most concept files do NOT use `[[concept_name]]` wikilink syntax in their body text.

| Status | Files |
|---|---|
| Have inbound wikilinks FROM INDEX.md | All 11 concept files (INDEX.md uses `[[concept_name]]` links correctly) |
| Have outgoing `[[]]` links in body | `antihebbian.md` (none found), `gate_death.md` (none found), `delta_w.md` (none found) |
| Should cross-link | `gate_death.md` → `[[antihebbian]]`, `[[phase_routing]]`; `n_scaling.md` → `[[k_iter]]`; `delta_w.md` → `[[antihebbian]]`; `k_iter.md` → `[[gate_death]]` |

---

## 6. Paper Directory Assessment

**Structure is sound.** README, MANUSCRIPT_DRAFT, PAPER_OUTLINE, claims, findings_log, baselines_needed, figures_planned — correct file roles.

**Issues found:**

| Issue | File | Detail |
|---|---|---|
| Stale dataset name | `claims.md` Claims 1 and 2 | Text says "97.86% on FashionMNIST" — project uses Imagenette. VGG16 FC param count listed as 119.6M (correct) but "FashionMNIST" is wrong throughout claims.md Claims 1-2 |
| Stale working title | `PAPER_OUTLINE.md` line 3 | Title: "Sparse Geometric Neural Networks: Matching Dense Classifier Accuracy at <1% Compute…". README.md title (2026-04-14 revised): "SGNNET: A Sparse Graph Classification Head…". Two different working titles in the same directory |
| Stale headline result | `PAPER_OUTLINE.md` line 7 | Lists "95.52% @ 0.98M FLOPs" as headline; `WINNERS.md` and README both note step235 A_aug achieves 97.30% at same 0.98M FLOPs — this is a better headline |
| figures_planned references FashionMNIST | `figures_planned.md` Table 1 | "SGNNET vs all baselines on FashionMNIST" — should be Imagenette |
| Duplicate figures section | `figures_planned.md` (31 lines) and `PAPER_OUTLINE.md` §9 (17 lines) | Same figure list in two files; `PAPER_OUTLINE.md` §9 is the authoritative location |
| `baselines_needed.md` references DistilBERT | line 14 | Lists DistilBERT as the text feature extractor. README.md (2026-04-14) supersedes this with ModernBERT-base. Stale model choice. |
| `MANUSCRIPT_DRAFT.md` is v0.1 from 2026-04-11 | lines 1-4 | 4 days old; all ΔW proj results (step235, step260-265), dynamic connectivity closure, K_iter=4 discovery, and CUDA benchmarks (step800/801/811) post-date the draft. Draft is increasingly stale relative to actual results. |

---

## 7. Proposed Reorganization

### 7a. Create archive directory

```
create learnings/archive/          (new directory for stale/retired content)
create learnings/archive/concepts/ (for archived concept files)
```

### 7b. Archive stale operational files

```
archive EXPERIMENT_REPORT.md        → learnings/archive/EXPERIMENT_REPORT.md
archive EXPERIMENT_REPORT_ADDENDUM.md → learnings/archive/EXPERIMENT_REPORT_ADDENDUM.md
archive EXPERIMENT_REPORT_VERIFICATION.md → learnings/archive/EXPERIMENT_REPORT_VERIFICATION.md
archive EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md → learnings/archive/EXPERIMENT_QUEUE_CRITICAL_FINDINGS.md
archive CLAUDE_MD_REVIEW_2026-04-14.md → learnings/archive/CLAUDE_MD_REVIEW_2026-04-14.md
archive SCRIPT_AUDIT_2026_04_14.md  → learnings/archive/SCRIPT_AUDIT_2026_04_14.md
archive PRUNE_REVIEW_2026_04_14.md  → learnings/archive/PRUNE_REVIEW_2026_04_14.md
```

Reason: one-off review/verification documents; findings absorbed into INDEX.md/CLAUDE.md. 7 files, ~1,354 lines removed from active browsing.

### 7c. Archive redundant index/pointer-only files

```
archive LEARNINGS_phase5.md         → learnings/archive/LEARNINGS_phase5.md
archive LEARNINGS_phase5_p15_post_wave1.md → learnings/archive/LEARNINGS_phase5_p15_post_wave1.md
archive LEARNINGS_phase5_p6_gen2_gen3.md → learnings/archive/LEARNINGS_phase5_p6_gen2_gen3.md
archive LEARNINGS_sparse_attention_research.md → learnings/archive/LEARNINGS_sparse_attention_research.md
archive RESEARCH_learning_algorithms.md → learnings/archive/RESEARCH_learning_algorithms.md
```

Reason: pure pointer-only files (9 lines each or TOC for split content). INDEX.md covers everything they point to.

### 7d. Archive stale research

```
archive LEARNINGS_research.md       → learnings/archive/LEARNINGS_research.md
```

Reason: translation invariance discussion is a pre-efficiency-track concern; resolved by VGG feature extractor approach. Merge 2 still-relevant architecture notes into LEARNINGS_arch.md before archiving.

### 7e. Rename opaque file

```
mv learningsA.md → learnings/archive/LEARNINGS_dynamic_routing_step700_701.md
```

Reason: `learningsA.md` breaks the naming convention and its content (step700/701 parallelization kills) is fully in INDEX.md dead ends.

### 7f. Split size violations

```
split LEARNINGS_phase5_p15f_warmstart_efficiency.md (289 lines) →
  LEARNINGS_phase5_p15f_warmstart.md         (~145 lines, steps 149-163)
  LEARNINGS_phase5_p15f_efficiency_track.md  (~144 lines, steps 165-177)

split LEARNINGS_design_2026_04_09.md (305 lines) →
  LEARNINGS_design_2026_04_09a_flops_gap_analysis.md (~150 lines)
  LEARNINGS_design_2026_04_09b_new_mechanisms.md     (~155 lines)

split paper/MANUSCRIPT_DRAFT.md (645 lines) →
  paper/MANUSCRIPT_p1_intro_related.md    (§1 Introduction + §2 Related Work, ~200 lines)
  paper/MANUSCRIPT_p2_arch_results.md     (§3 Architecture + §4 Key Findings + §5 Efficiency, ~250 lines)
  paper/MANUSCRIPT_p3_deadends_discuss.md (§6 Dead Ends + §7 Discussion, ~195 lines)

split paper/findings_log.md (313 lines) →
  paper/findings_log_p1_foundations.md   (entries pre-step180: architecture fundamentals, gates, AH)
  paper/findings_log_p2_efficiency.md    (entries step180+: efficiency track, ΔW proj, CUDA)
```

### 7g. Trim concepts/phase_routing.md

```
split concepts/phase_routing.md (279 lines) →
  concepts/phase_routing.md              (≤80 lines: KILLED verdict, gate-death cause, redistribution principle)
  learnings/archive/concepts/phase_routing_full.md  (full 279-line original archived)
```

### 7h. Prune PENDING_DISCUSSIONS.md

```
split PENDING_DISCUSSIONS.md (486 lines) →
  PENDING_DISCUSSIONS.md                 (PENDING-only items ~100-120 lines: G4, step72, step85, step93-98, step163 if still open)
  archive/PENDING_DISCUSSIONS_archive.md (all SCRIPTED/KILLED/GATED entries ~366 lines)
```

### 7i. Update INDEX.md

```
update INDEX.md:
  - Add LEARNINGS_design_2026_04_15.md to Sequential Logs table
  - Add learningsA.md → renamed archive entry
  - Mark Active Directions #7 (step729) as COMPLETE
  - Add confirmed law: "K_iter=4 ΔW proj is better than K_iter=5 at N=4096 (+0.97pp)"
  - Add confirmed law: "Dynamic connectivity at N=512 killed (6 variants, steps 511-514)"
  - Add dead end: "Dynamic connectivity at N=512 — co-activation correlation at noise level"
  - Remove archive targets from Sequential Logs and Other Index Files tables
```

### 7j. Fix stale content in paper/

```
update paper/claims.md:
  - Replace "FashionMNIST" with "Imagenette" in Claims 1 and 2 (at least 4 occurrences)

update paper/PAPER_OUTLINE.md:
  - Align working title with README.md revised title (2026-04-14)
  - Update headline result from 95.52% to 97.30% (step235 A_aug at same 0.98M FLOPs)
  - Add §5.8 "ΔW Projection Mechanism" (step234/235 confirmed +5.77pp over AH baseline)
  - Mark §5.1 D>K_hh claim: add step214/215 confirmation line

update paper/figures_planned.md:
  - Replace "FashionMNIST" with "Imagenette" in Table 1 description
  - Note this file is redundant with PAPER_OUTLINE §9 (consider deleting after merge)

update paper/baselines_needed.md:
  - Update text feature extractor from DistilBERT to ModernBERT-base (README 2026-04-14 decision)
  - Update LLM feature extractor from Qwen2.5-0.5B (896-dim) to Qwen3-0.6B (1024-dim)

update paper/README.md:
  - Update Files table after MANUSCRIPT_DRAFT split (3 entries replacing 1)
  - Update findings_log entries after split (2 entries replacing 1)
```

### 7k. Add cross-links to concepts/

```
update concepts/gate_death.md: add [[antihebbian]] and [[phase_routing]] links
update concepts/n_scaling.md:  add [[k_iter]] link
update concepts/delta_w.md:    add [[antihebbian]] link
update concepts/k_iter.md:     add [[gate_death]] link
update concepts/antihebbian.md: add [[gate_death]] and [[delta_w]] links
```

---

## Summary: Top 3 Highest-Impact Changes

**1. Archive 7 one-off review/verification files (ops §7b) + 5 pointer-only files (§7c) + learningsA.md (§7e) = 13 files removed from active browsing.** These 13 files total ~1,600 lines of noise in the root directory. Every new session that scans `learnings/` for context hits these dead files first. Archiving them immediately reduces cognitive load and speeds up session-start orientation. The content is not lost — it moves to `archive/` and INDEX.md already captures every finding of substance.

**2. Split the 4 size violators in `paper/` (MANUSCRIPT_DRAFT at 645 lines, findings_log at 313 lines) and prune PENDING_DISCUSSIONS.md from 486 to ~120 lines.** These three files are the most actively read and edited. MANUSCRIPT_DRAFT at 645 lines is unworkable in a single read — splitting into three thematic parts (intro, arch+results, dead ends) maps exactly to the paper writing workflow. The PENDING_DISCUSSIONS pruning is high-leverage because it currently contains ~370 lines of SCRIPTED/KILLED design history that is already in INDEX.md; reducing it to the 8 live PENDING entries makes it instantly actionable.

**3. Fix stale dataset name "FashionMNIST" in paper/claims.md and paper/figures_planned.md, and update paper/PAPER_OUTLINE.md headline result from 95.52% to 97.30%.** These are low-effort edits (15 minutes) but high-stakes: the paper claims file is the primary reference for manuscript writing. Having "FashionMNIST" appear in the claims doc for a paper about Imagenette results is a credibility problem that could silently propagate into the manuscript. The headline result update (95.52% → 97.30% at same FLOPs) is also a straightforward improvement — step235 A_aug is a confirmed winner and the better number to lead with.
