# Project: SGNNET / neuro_graph — Dhiraj (group_id: "dhiraj")

## Project State
SGNNET: sparse O(N·K) graph neural network, Fourier encoding on S^{D-1}, Imagenette.
Current best: **97.86%** (D=64 + AntiHebb α=1.0 + K_hh=4 + K_iter=12 + turing=0.0 reflect=0.5, step89 Config A, N=4096, patched arch, full data 150ep, best_ep=134). Previous best: 97.58% (step89 Ref). Update this line when a new best is achieved.
Phase 5 active. **EFFICIENCY MILESTONE ACHIEVED (2026-04-11): step199 FINAL = 95.52% best_ep=136 @ 0.98M FLOPs (0.79% of VGG16 FC). 67K params (0.05% of VGG16 FC). Both ≤1% FLOPs AND ≤1% params criteria met simultaneously. N-scaling ceiling: 97.17% @ 1.97M (N=4096/N=8192, D=16, step205/209).** Core docs: `learnings/EXPERIMENT_QUEUE.md` · `learnings/PENDING_DISCUSSIONS.md` · `learnings/LEARNINGS_design.md` (index)

**Final efficiency config (step199):** N=2048, D=16, K_hh=2, K_iter=5, alpha_ahebb=1.0, alpha_reflect=0.5, alpha_turing=0.0. Script: `scripts/train_step199_n2048_d16_khh2_kiter5_tier2.py`. Eval: `scripts/eval_efficiency_config.py`.

**Primary goal (confirmed 2026-04-04):** Find a general-purpose deep learning architecture more parameter-efficient than transformers. Primary target: replacing the feed-forward (FFN) layer in transformer models. SGNNET is the candidate architecture with O(N×K) hard parameter budget. Imagenette is the testbed; the goal is a generalizable, scalable architecture. Key hypothesis: as problem complexity increases, increasing N incorporates higher orders of complexity — establishing N-scaling laws.
Core value: SGNNET matches VGG16 FC accuracy at ≤1% of its parameters AND ≤1% of its FLOPs (near-term proxy for FFN replacement viability). **STATUS: BOTH CRITERIA MET (step199: 0.79% FLOPs, 0.05% params, 95.52% accuracy).**

**Two parallel research tracks:**
1. **Accuracy track** (N=4096, D=64): maximize accuracy with confirmed defaults (K_hh=4, K_iter=12, AH=1.0, turing=0.0)
2. **Efficiency track** (small N/D): **COMPLETE** — ≤1% FLOPs criterion met at step195 (1.18M), sub-1% (0.98M) met at step199. Next: cross-dataset/model generalizability testing.

---

## Session Start
Before responding to the user on every new session or after /clear:
1. `mcp__graphiti__search_memory_facts("recent experiments results running", group_ids=["dhiraj"])`
2. `mcp__graphiti__search_memory_facts("confirmed winners dead ends D=64", group_ids=["dhiraj"])`
3. Read `learnings/EXPERIMENT_QUEUE.md` and `.planning/STATE.md`
4. Create training monitor cron job (every 10 min, recurring) — see Training Monitor section below

---

## Training Machines

| Machine | MPS cap | CPU cap | Total cap | RAM threshold | Working dir |
|---|---|---|---|---|---|
| Mac Studio (remote, `mac-studio`) | 1 | 1 | 2 | ≥ 50 GB | `/Users/admin/ml/dhiraj/qwen2_omni/testing/` |
| Mac Mini (local host) | 1 | 1 | 2 | ≥ 20 GB | `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/` |

Total max: 4. Prefer Mac Studio if both have slots. Launch rule: per-device count < 1 AND RAM ≥ threshold — both required.
RAM: `vm_stat | grep -E 'free|inactive'` → (Pages free + Pages inactive) × 16384 / 1073741824

**STRICT: All training launches MUST use tmux.** No bare `nohup`, no `&` backgrounding, no detached processes. tmux gives: (1) `tmux ls` shows exactly what's running, (2) `tmux capture-pane` reads live output, (3) sessions survive SSH disconnects cleanly, (4) no zombie processes burning compute silently. If `tmux ls` shows nothing, nothing is running — period. Violation of this rule caused 5 zombie processes burning 3 MPS + 2 CPU slots on Mac Studio for hours with no output (2026-04-09 incident).

**Mac Studio launch (MPS):**
```bash
rsync -avz scripts/SCRIPT.py mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/scripts/
ssh mac-studio "cd /Users/admin/ml/dhiraj/qwen2_omni/testing && /opt/homebrew/bin/tmux new-session -d -s NAME 'd_env/bin/python3 -u scripts/SCRIPT.py --device mps 2>&1 | tee logs/SCRIPT.log'"
```

**Mac Studio launch (CPU):**
```bash
rsync -avz scripts/SCRIPT.py mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/scripts/
ssh mac-studio "cd /Users/admin/ml/dhiraj/qwen2_omni/testing && /opt/homebrew/bin/tmux new-session -d -s NAME 'd_env/bin/python3 -u scripts/SCRIPT.py --device cpu 2>&1 | tee logs/SCRIPT.log'"
```

**Mac Studio status check:** `/opt/homebrew/bin/tmux ls` (tmux is at `/opt/homebrew/bin/tmux`, not in default PATH)

**Mac Mini launch (MPS):** (results/logs land in repo directly — no rsync needed)
```bash
cd /Volumes/T9/IndraAstra/dhiraj/neuro_graph && tmux new-session -d -s NAME 'd_env/bin/python3 -u scripts/SCRIPT.py --device mps 2>&1 | tee logs/SCRIPT.log'
```

**Mac Mini launch (CPU):**
```bash
cd /Volumes/T9/IndraAstra/dhiraj/neuro_graph && tmux new-session -d -s NAME 'd_env/bin/python3 -u scripts/SCRIPT.py --device cpu 2>&1 | tee logs/SCRIPT.log'
```
Python env: `/Volumes/T9/IndraAstra/dhiraj/neuro_graph/d_env`

---

## Knowledge Persistence

Four layers — always keep in sync:

| Layer | Purpose | How |
|---|---|---|
| Graphiti | Semantic search across sessions | `mcp__graphiti__add_memory` |
| `learnings/LEARNINGS_*.md` | Sequential human-readable audit trail | Edit `LEARNINGS_phase5_p<N>.md` or `LEARNINGS_design.md` |
| `learnings/concepts/*.md` | Consolidated per-concept wiki pages | Update relevant concept page(s) |
| `EXPERIMENT_QUEUE.md` | Live queue + critical findings | Edit directly |

**Triple-write rule:** every result, decision, or finding goes to Graphiti AND learnings/ AND the relevant concept page(s).

| Event | Graphiti | Learnings | Concepts |
|---|---|---|---|
| Experiment completed | step, config, result, delta, verdict | `LEARNINGS_phase5_p<N>.md` | Update relevant concept page(s) results table |
| Winner confirmed | gain, config, why | `EXPERIMENT_QUEUE.md` + LEARNINGS | Update concept page + `learnings/INDEX.md` |
| Approach killed | why it failed | Mark KILLED in queue + LEARNINGS | Update concept page failure table |
| Calibration finding | param, best value, scale | Current LEARNINGS part | Update concept parameter section |
| Design decision | reasoning behind generation/ARM choice | Current LEARNINGS part | New concept page if novel concept |
| Design discussion | hypothesis, idea, architectural debate | `learnings/LEARNINGS_design.md` (dated) | Cross-link from relevant concept pages |

**Concept pages** (`learnings/concepts/*.md`): LLM-maintained, per-concept consolidated knowledge. Sequential LEARNINGS files are the audit trail (append-only); concept pages are the compounding layer (always current). Use `[[concept_name]]` wiki-link syntax for cross-references between concept pages. When a new concept emerges (e.g., a novel mechanism), create a concept page in the same session.

**Index** (`learnings/INDEX.md`): Master catalog of all concept pages, key findings, and which LEARNINGS parts cover which topics. Regenerate after any concept page is created or significantly updated.

**Design discussions → scripts (STRICT):** Every design discussion is an experiment-generation session. The outcome is always a script + EXPERIMENT_QUEUE.md entry — not notes alone.
- Before writing any code: identify all ambiguous parameters and cross-question until every config is fully specified (mode, hyperparams, ablation axis, reference config).
- Once resolved: write model file + training script in the same session. Never end a design discussion without a script.
- Add the step to EXPERIMENT_QUEUE.md immediately after writing the script.
- "We'll script it later" is not acceptable — later = never (confirmed by Dhiraj 2026-04-05).
- Signal: "I wonder if...", "what if...", "why did we..." → engage fully, then produce script before moving on.

**Before designing any experiment:** `mcp__graphiti__search_memory_facts("<mechanism>", group_ids=["dhiraj"])` — check if already tried.

**Evidence standards for findings (STRICT):**
When logging experiment results, every causal claim MUST be tagged:
- **CONFIRMED**: clean ablation — exactly one variable changed, control present.
- **HYPOTHESIS**: post-hoc explanation for a result, OR confounded experiment (multiple variables changed), OR inferred from indirect evidence.
- **STALE**: tested on a prior architecture/scale that has since changed significantly (e.g., pre-patch findings after +9.83pp arch fix).

Rules:
1. Post-hoc explanations for failures are HYPOTHESES, not facts. Never treat them as confirmed without a controlled experiment.
2. When a hypothesis gates multiple future experiments (e.g., "compounding kills" blocking all compound work), prioritize validation — design a clean ablation to confirm or invalidate.
3. When the base changes significantly (arch patch, new N scale, new defaults), all conclusions from the old base become STALE. Retest the load-bearing ones first.
4. "KILLED" status requires CONFIRMED evidence. If the evidence is HYPOTHESIS or STALE, mark as "KILLED (unvalidated)" and note what control is missing.
5. Before closing an entire research direction, verify the evidence is CONFIRMED and not confounded. One bad experiment with multiple variables changed is not sufficient to kill a direction.

**Learnings propagation rule:** When a running experiment completes with a notable result, immediately check all pending/future scripts for assumptions that result changes. Update scripts in-place. Don't wait for all results before adapting.

**Two-tier experiment protocol (STRICT):**
Every new mechanism/ablation follows this pyramid — never skip Tier 0 to jump straight to Tier 1. Tier 2 (150ep full runs) is deferred until project endgame.

| Tier | Budget | Data | Purpose | When to advance |
|------|--------|------|---------|-----------------|
| **Tier 0 (Scout)** | 20 epochs | 50% | Rejection filter. Kill obviously bad configs (large negative delta). | All configs NOT clearly failing advance to Tier 1 |
| **Tier 1 (Calibration)** | 75 epochs | 50% | Reliable comparison. Confirm winner vs Ref. | Winner with ≥+0.5pp → add to defaults |

**Tier 0 is a rejection filter, not a selection cap.** Kill configs with clearly bad performance (e.g., >5pp below Ref, or catastrophic failure). All configs with positive or neutral delta advance — do NOT artificially cap at "top-2" when multiple configs show promise.

Empirical basis (46 experiments, D=64 arch): 20ep scouts predict the final winner ~80% of the time (Spearman ρ=0.80). 30ep scouts hit 91%. Tier 0 catches dead configs before wasting 75ep × 4 slots on them.

**Autorun mode** (`scripts/autorun_sgnnet.py`): Autoresearch-style tight loop — runs Tier 0 scouts back-to-back with a monotonic ratchet (keep wins, revert losses). Use for overnight exploration when human is away. Journal logged to `results/autorun_journal.tsv`.

**Training diagnostics (STRICT — all new experiments):**
Every experiment script MUST integrate `src/training/diagnostics.py` to log architectural health metrics at epoch boundaries (every 5 epochs, not per-batch). Diagnostics run one forward pass on 32 val samples — negligible overhead.

Key metrics tracked:
- **Effective rank of Z** — how many dims the representation actually uses (low = dimensional collapse)
- **Neuron utilization %** — dead neuron detection (should be >90% for healthy training)
- **W_pos cosine similarity** — positional diversity (high = neurons clustering → AH failing)
- **Separability ratio** — inter-class / intra-class distance (higher = better class separation)
- **Gradient norms** — per param group (θ, W_pos, fc_out) — who's learning?

Usage in custom training loops:
```python
from src.training.diagnostics import TrainingDiagnostics
diag = TrainingDiagnostics(model, device, log_every=5)
# At end of each epoch:
diag.log_epoch(epoch, model, val_loader, optimizer)
```

Why: loss/accuracy are lagging indicators. Diagnostics reveal WHY training works or fails — dimensional collapse, dead neurons, positional clustering, gradient imbalance. This enables data-driven experiment design: if effective rank is low, try nuclear norm regularization; if separability is low, try contrastive loss; if W_pos diversity drops, AH strength may need tuning.

**Paper materials** (`learnings/paper/`): Capture novel findings as they emerge. After every major experiment, check if the result is paper-worthy (not already in literature) and add to `learnings/paper/findings_log.md`. Cross-reference `baselines_needed.md` to track gaps toward publication.

**File size rule:** No file in `learnings/` should exceed 250 lines. Split into sub-files with an index when approaching the limit.

---

## GSD ↔ Graphiti Integration

GSD subagents can't call Graphiti — inject context at orchestrator level. Before spawning: search recent results + dead ends. After phase completes: one Graphiti episode with decisions and rationale.

---

## Training Monitor

**Always use tmux for all training launches** (both Mac Studio and Mac Mini). Sessions auto-clean when the command exits — `tmux ls` gives a live view of what's running.

**Training monitor cron** must be active whenever experiments are running. Create at session start and after `/clear`:
```
CronCreate(cron="*/10 * * * *", prompt="<training monitor prompt>", recurring=True)
```
When the cron fires and all experiments are done, it stays idle (stays silent). Re-create the cron at the start of every new session — it does not persist across sessions.

**CRITICAL — background agent rule:** The cron prompt MUST instruct Claude to launch a background Agent (`run_in_background=true`) that performs all tmux/ssh checks inside the agent. Never run tmux capture-pane or ssh commands inline in the main context — this pollutes the conversation with tool calls and output. The background agent reads panes, compares against known state, and reports only if something notable happened; otherwise it returns "SILENT". The main context only ever sees the final report.

---

## Maintaining This File

This file is the current truth — git tracks history. Do not append logs.
- **Update** existing instructions when context changes (new best accuracy, env path, machine added)
- **Add** only when a recurring pattern isn't covered
- **Remove** when an instruction is obsolete or no longer reflects reality
- Observe workflow friction across sessions; synthesise before proposing. Structural changes need confirmation; wording/query tweaks do not.
