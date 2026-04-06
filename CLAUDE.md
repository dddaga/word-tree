# Project: SGNNET / neuro_graph — Dhiraj (group_id: "dhiraj")

## Project State
SGNNET: sparse O(N·K) graph neural network, Fourier encoding on S^{D-1}, FashionMNIST.
Current best: **84.36%** (D=64 + AntiHebb α=1.0 + Gen4 calibrated base, step56 N=4096, best_ep=146/150, full data 150ep). Previous best: 81.10% (step56 N=2048). Update this line when a new best is achieved.
Phase 5 active — 6 ARM tracks. Core docs: `learnings/EXPERIMENT_QUEUE.md` · `learnings/LEARNINGS_phase5_p*.md` · `.planning/STATE.md`

**Primary goal (confirmed 2026-04-04):** Find a general-purpose deep learning architecture more parameter-efficient than transformers. Primary target: replacing the feed-forward (FFN) layer in transformer models. SGNNET is the candidate architecture with O(N×K) hard parameter budget. FashionMNIST is the testbed; the goal is a generalizable, scalable architecture. Key hypothesis: as problem complexity increases, increasing N incorporates higher orders of complexity — establishing N-scaling laws.
Core value: SGNNET matches VGG16 FC accuracy at ≤1% of its parameters (near-term proxy for FFN replacement viability).

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

**Mac Studio launch (MPS):**
```bash
rsync -avz scripts/SCRIPT.py mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/scripts/
ssh mac-studio "cd /Users/admin/ml/dhiraj/qwen2_omni/testing && tmux new-session -d -s NAME 'd_env/bin/python3 -u scripts/SCRIPT.py --device mps 2>&1 | tee logs/SCRIPT.log'"
```

**Mac Studio launch (CPU):**
```bash
rsync -avz scripts/SCRIPT.py mac-studio:/Users/admin/ml/dhiraj/qwen2_omni/testing/scripts/
ssh mac-studio "cd /Users/admin/ml/dhiraj/qwen2_omni/testing && tmux new-session -d -s NAME 'd_env/bin/python3 -u scripts/SCRIPT.py --device cpu 2>&1 | tee logs/SCRIPT.log'"
```

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

Three layers — always keep in sync:

| Layer | Purpose | How |
|---|---|---|
| Graphiti | Semantic search across sessions | `mcp__graphiti__add_memory` |
| `learnings/` | Sequential human-readable audit trail | Edit `LEARNINGS_phase5_p<N>.md` or `LEARNINGS_design.md` |
| `EXPERIMENT_QUEUE.md` | Live queue + critical findings | Edit directly |

**Dual-write rule:** every result, decision, or finding goes to BOTH Graphiti AND learnings/.

| Event | Graphiti | Learnings |
|---|---|---|
| Experiment completed | step, config, result, delta, verdict | `LEARNINGS_phase5_p<N>.md` |
| Winner confirmed | gain, config, why | `EXPERIMENT_QUEUE.md` critical findings + LEARNINGS |
| Approach killed | why it failed | Mark KILLED in queue + LEARNINGS |
| Calibration finding | param, best value, scale | Current LEARNINGS part |
| Design decision | reasoning behind generation/ARM choice | Current LEARNINGS part |
| Design discussion | hypothesis, idea, architectural debate | `learnings/LEARNINGS_design.md` (dated) |

**Design discussions → scripts (STRICT):** Every design discussion is an experiment-generation session. The outcome is always a script + EXPERIMENT_QUEUE.md entry — not notes alone.
- Before writing any code: identify all ambiguous parameters and cross-question until every config is fully specified (mode, hyperparams, ablation axis, reference config).
- Once resolved: write model file + training script in the same session. Never end a design discussion without a script.
- Add the step to EXPERIMENT_QUEUE.md immediately after writing the script.
- "We'll script it later" is not acceptable — later = never (confirmed by Dhiraj 2026-04-05).
- Signal: "I wonder if...", "what if...", "why did we..." → engage fully, then produce script before moving on.

**Before designing any experiment:** `mcp__graphiti__search_memory_facts("<mechanism>", group_ids=["dhiraj"])` — check if already tried.

```python
mcp__graphiti__add_memory(
    name="step29 AntiHebb result",
    episode_body="AntiHebb α=0.5, D=64, N=1024, K_iter=8: 70.14% val acc. +13.86pp over D=64 ceiling. Must include in all future D=64 experiments.",
    group_id="dhiraj", source="text", source_description="results/train_step29_antihebb_d64.json"
)
```

---

## GSD ↔ Graphiti Integration

GSD subagents can't call Graphiti — inject context at orchestrator level before spawning.

**Before /gsd:discuss-phase or /gsd:plan-phase:**
Search `<topic> results` + `<topic> rejected failed dead` + `confirmed winners`. Inject into agent prompt: what to include, what not to re-propose.

**Before /gsd:execute-phase:**
Search running experiments + calibration findings for the mechanism being implemented. Include in executor prompt.

**After any GSD phase completes:**
One Graphiti episode: goal, key decision, what was confirmed/rejected. No task lists — decisions and rationale only.

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
