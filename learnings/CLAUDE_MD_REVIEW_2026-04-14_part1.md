# CLAUDE.md Review — 2026-04-14

Comparing Karpathy's approach against our 3-layer CLAUDE.md stack.
Scope: gaps, stale content, style drift, open questions.
No files modified — decision document for user.

---

## A. What Karpathy's Approach Does Differently

Karpathy's CLAUDE.md (forrestchang/andrej-karpathy-skills) = 4 principles, ~1 page. Ours = 183 lines project layer alone. Key structural differences:

**1. Assumption surfacing before execution**
> "State your assumptions explicitly. If uncertain, ask. Rather than proceeding blindly, the framework recommends surfacing confusion and presenting multiple interpretations when they exist."

No equivalent. Claude routinely starts writing scripts before confirming ambiguous params (step306 bug, step232 broken forward pass). Added "identify all ambiguous parameters" to design-to-script rule, but buried at line 94 project CLAUDE.md — only applies to design discussions, not script writes.

**2. Simplicity first / no speculative features**
> "No features beyond what was asked. No abstractions for single-use code."

No equivalent. step232 (scalar edge weights) shipped broken forward pass — implementation assumed gradient flow that doesn't work. step222 shipped with trainer incompatibility. Neither caught — no "verify before launch" step.

**3. Surgical changes only**
> "Touch only what you must. Clean up only your own mess. Match existing style. Avoid unrelated refactoring."

No equivalent. Claude occasionally refactors shared utilities when writing new experiment scripts, introducing regressions. step66 `resonant_mode=` kwarg bug: keyword argument added to model that existing scripts didn't pass, crashing on first run.

**4. Goal-driven execution: verifiable success criteria**
> "Convert abstract tasks into measurable outcomes. Transform 'make it work' into concrete goals with specific verification steps."

Design-to-script rule requires script but not verification step. Scripts ship without `python -c "import script; print('ok')"` check. Multiple scripts crashed first run (step66, step306 relaunched, step232 broken).

**5. Minimal CLAUDE.md itself**
Their file ~30 lines. Ours = 183 (project) + 19 (user) + 36 (shared) = 238 lines across 3 layers. Long files get ignored — zombie incident (2026-04-09) broke tmux rule despite being in CLAUDE.md. Length = usability problem.

---

## B. Our Current Strengths — Do Not Touch

**1. Evidence tagging (CONFIRMED / HYPOTHESIS / STALE) — lines 102-113, project CLAUDE.md**
Prevents compounding unvalidated kills into blocked research directions. Addresses historical "KILLED (unvalidated)" entries blocking valid experiments. Non-standard, genuinely valuable.

**2. Two-tier experiment protocol with empirical basis — lines 117-127, project CLAUDE.md**
Tier 0 / Tier 1 pyramid calibrated against 46 actual experiments (Spearman ρ=0.80). Rare empirical backing in CLAUDE.md rule. Keep numbers.

**3. Triple-write rule (Graphiti + LEARNINGS + concepts) — lines 78-87, project CLAUDE.md**
Without this, session-crossing context resets lose experiment history. Core of knowledge persistence system. Working.

**4. Training diagnostics mandatory requirement — lines 131-149, project CLAUDE.md**
Requiring `TrainingDiagnostics` in every script catches silent failures (dimensional collapse, dead neurons) before wasting 75ep runs. Proactive not reactive.

**5. Design-to-script rule — lines 93-98, project CLAUDE.md**
"Later = never" confirmed wisdom (2026-04-05). Every discussion ends with runnable script — enforced, works.

---

## C. Gaps — Proposed Additions

### Gap 1: Script smoke-test before launch

**Problem:** step306 crashed first run, required relaunch. step66 crashed first run (kwarg bug). step232 shipped broken gradient flow. step222 shipped trainer incompatibility. Pattern: scripts written and launched without syntax/import checks.

**Proposed text** (add to "Design discussions → scripts" section, project CLAUDE.md, after line 95):
```
**Script smoke-test (MANDATORY before any tmux launch):**
After writing a new script, run before launching:
  python -c "import importlib.util, sys; spec=importlib.util.spec_from_file_location('s','scripts/SCRIPT.py'); m=importlib.util.module_from_spec(spec)"
Or simply: `d_env/bin/python3 -c "exec(open('scripts/SCRIPT.py').read().split('if __name__')[0])"` — catches import errors, bad kwargs, missing deps.
If smoke-test fails: fix before launching. Never launch a script you haven't syntax-checked.
```

**Which file:** Project CLAUDE.md
**Priority:** HIGH

---

### Gap 2: Queue sync check before session end

**Problem:** step320 and step321 listed RUNNING in `EXPERIMENT_QUEUE.md` (studio:mps, studio:cpu) but no logs exist (`ls logs/train_step320* → no matches`). Claude updated queue to "RUNNING" but either (a) never launched scripts, or (b) launched without verifying tmux session persisted. Queue shows 5 RUNNING entries when real state may be 3.

**Proposed text** (add to Session Start section, project CLAUDE.md, after item 3):
```
4. Verify queue vs reality: for every RUNNING entry in the queue table, confirm the corresponding tmux session exists:
   - Mac Mini: `tmux ls`
   - Mac Studio: `ssh mac-studio '/opt/homebrew/bin/tmux ls'`
   - 5060ti: `ssh indradev 'tmux ls'` (or equivalent)
   If a session is missing for a RUNNING entry, mark it UNKNOWN in the queue and investigate before launching a replacement.
```

**Which file:** Project CLAUDE.md
**Priority:** HIGH

---

### Gap 3: Third machine (5060ti/RTX) not in Training Machines table

**Problem:** CLAUDE.md Training Machines table lists only Mac Studio and Mac Mini ("Total max: 4"). Queue uses 5th slot (`5060ti:cuda`, step301 RUNNING), and `feedback_mac_mini_no_training.md` in memory says "5 training slots: mini:mps, mini:cpu, studio:mps, studio:cpu, 5060ti:cuda". Launch commands, RAM check, tmux path, working dir for 5060ti entirely undocumented.

**Proposed text** (add row to Training Machines table, update "Total max: 4"):
```
| RTX 5060Ti (remote, `indradev`) | 1 | 0 | 1 | ≥ 16 GB | <working_dir_TBD> |

Total max: 5. Prefer Mac Studio if both have slots, then 5060ti for CUDA experiments.
**5060ti launch (CUDA):**
  ssh indradev "cd <working_dir> && tmux new-session -d -s NAME 'd_env/bin/python3 -u scripts/SCRIPT.py --device cuda 2>&1 | tee logs/SCRIPT.log'"
```
(Fill in actual working dir and tmux path once confirmed.)

**Which file:** Project CLAUDE.md, Training Machines section (lines 29-63)
**Priority:** HIGH — missing this causes undocumented launches and ghost RUNNING entries

---

### Gap 4: "DONE (broken)" scripts need explicit re-scripting rule

**Problem:** step222 marked "DONE (broken)" for trainer incompatibility. step232 "DONE (broken)" for broken forward pass. Both sit in queue with no follow-up script. Broken experiments have no protocol: neither closed nor escalated. step400/401 (paper-critical) may replay step222 mistake — root cause (trainer incompatibility with non-SGNNET models) not captured as design rule anywhere.

**Proposed text** (add to Knowledge Persistence section, project CLAUDE.md, after line 87):
```
**Broken experiment protocol:** When a step completes with status "DONE (broken)":
1. Add root cause to `learnings/LEARNINGS_ops.md` under "## Known Failure Modes".
2. Create a follow-up step (e.g., step222b) in the queue with the fix.
3. Do NOT reuse the broken script without fixing the root cause first.
4. If a paper-critical experiment is broken, it blocks the paper — treat as P0.
```

**Which file:** Project CLAUDE.md
**Priority:** HIGH

---

### Gap 5: Stale "Accuracy track" entry

**Problem:** Line 14 lists "Accuracy track (N=4096, D=64): maximize accuracy with confirmed defaults (K_hh=4, K_iter=12, AH=1.0, turing=0.0)". Queue shows no D=64 experiments running or queued since step89 (97.86% record, early phase 5). Current work entirely D=16. "Accuracy track" framing implies D=64 work ongoing when it isn't.

**Proposed text** (replace lines 13-15 project CLAUDE.md):
```
**Current focus:** Paper validation track — ablations confirming paper claims (step300-321), N-scaling law curve (step402), cross-dataset generalization (step400-401, CIFAR-10).
**Accuracy record:** 97.86% (D=64, step89). D=16 record: 97.17% (step205/209). Efficiency record: 95.52% @ 0.98M FLOPs (step199).
```

**Which file:** Project CLAUDE.md, lines 13-15
**Priority:** MEDIUM

---

### Gap 6: Assumption surfacing before script write (Karpathy principle)

**Problem:** Scripts ship with wrong parameters — ambiguities resolved implicitly. Design-to-script rule says "identify all ambiguous parameters" but only for design discussions. Standalone script requests ("write me a step X script") have no pre-check.

**Proposed text** (add to project CLAUDE.md, beginning of "Design discussions → scripts"):
```
**Before writing any experiment script, confirm:**
- Reference config (what is the Ref baseline config?)
- Scale (N, D, K_hh, K_iter — or verify from final efficiency config if unspecified)
- Tier (Tier-0 20ep/50% vs Tier-1 75ep/50% vs Tier-2 150ep/100%)
- Ablation axis (what exactly changes vs Ref?)
- Device target (mps / cpu / cuda)
If any of these is ambiguous: ask. Do not infer silently.
```

**Which file:** Project CLAUDE.md
**Priority:** MEDIUM

---

### Gap 7: Session end checklist

**Problem:** Session Start checklist (4 items) exists but no Session End checklist. Sessions end without: verifying queue current, capturing findings to Graphiti, updating STATE.md. STATE.md still says "last updated: 2026-03-23" despite 300+ experiments since.

**Proposed text** (add new section after Session Start, project CLAUDE.md):
```
## Session End
Before closing any session with completed experiments:
1. Update EXPERIMENT_QUEUE.md "Currently Running" table — mark DONE anything that finished.
2. `mcp__graphiti__add_memory` — one episode per completed experiment (step, result, verdict).
3. Update STATE.md "Last Session" block with what completed and what's still running.
4. If any script shipped, verify it smoke-tested clean.
```

**Which file:** Project CLAUDE.md
**Priority:** MEDIUM

---

### Gap 8: Compounding rule needs positive escape hatch

**Problem:** GA rule "every new experiment base = ALL confirmed winners from all prior generations" (`EXPERIMENT_QUEUE.md` line 53) contradicts compound-kills pattern. AH compatibility rule in LEARNINGS (2026-04-09) clarifies when compounding safe (orthogonal axes), but CLAUDE.md has no positive statement about when compounding IS safe. Caused GA v2 to try ΔW+AH combinations already predicted to fail (step235 confirmed: adding AH to ΔW-rot hurts −1.2pp).

**Proposed text** (add to Knowledge Persistence, project CLAUDE.md):
```
**Compounding rule:** Mechanisms operating on the SAME signal path cancel each other (gate-death / co-adaptation). Compounding is ONLY safe when mechanisms are orthogonal:
- Topology changes (K_hh, RigL) + Signal modifications (AH, ΔW): safe
- Two W_pos-dependent mechanisms: unsafe (double-sparsity)
- Two activation-modifying mechanisms: likely unsafe — test with clean ablation first
Before writing a compound config, identify the signal path of each mechanism. If they share one: do not compound without prior ablation evidence.
```

**Which file:** Project CLAUDE.md
**Priority:** MEDIUM

---

### Gap 9: Karpathy-style brevity principle for CLAUDE.md itself

**Problem:** Project CLAUDE.md = 183 lines. Zombie incident (2026-04-09) broke tmux rule despite being present. Long CLAUDE.md files suffer "too important to read" failure mode. Karpathy's ~30 lines, exactly 4 principles.

**Proposed text** (add to "Maintaining This File" section):
```
**Length discipline:** Every addition must remove something of equal or lesser value. If the file exceeds 200 lines: compress or split. Rules not triggered in the past 20 sessions should be moved to a reference file (e.g., CLAUDE_REFERENCE.md) linked from here.
```

**Which file:** Project CLAUDE.md, "Maintaining This File" section
**Priority:** LOW

---


*Continued in [CLAUDE_MD_REVIEW_2026-04-14_part2.md](CLAUDE_MD_REVIEW_2026-04-14_part2.md) — D. Redundancies/Outdated, E. Style Drift, F. Open Questions.*