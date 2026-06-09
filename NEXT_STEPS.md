# NEXT_STEPS — session handoff (2026-06-10, ~01:30)

Resume point for next agent. Context: vision-debt T0 batch (step989–992) testing
never-tried promises from `sparse_geometric_network_report.md` (original brief).
Full audit: `learnings/VISION_REVIEW_2026-06-10.md` + `learnings/VISION_DEBT.md`.

## 1. IMMEDIATE — launch step990 v2 + step991 v2 (blocked on tmux bug)

Both scripts FIXED and smoke-passed, NOT yet launched.

**Blocker:** `scripts/launch_slot.sh` local tmux launch fails with
`error creating tmux (Operation not supported)` when invoked from a non-TTY
shell (Claude Code Bash tool). The same `tmux new-session -d ...` command works
when run directly in the Bash tool. Last diagnosis state:
- Script has `set -euo pipefail` (line 23), no allexport, TMUX var is shell-local.
- `setsid` fallback added at line ~188 does NOT fix it (setsid not on macOS —
  remove that line; it silently falls through to the bare call which also fails
  inside the script but works outside).
- WORKAROUND that works (use it): write launch cmd to /tmp script, run
  `tmux new-session -d -s <SESSION> bash /tmp/<file>.sh` DIRECTLY in Bash tool.

Launch commands (run directly, not via launch_slot.sh, until bug fixed):
```bash
# step990 v2 on mini_mps
S=sgn-indra-mini_mps-train_step990_additive_dynamic_t0
L=/Volumes/T9/IndraAstra/dhiraj/neuro_graph/logs/train_step990_additive_dynamic_t0__mini_mps.log
echo "cd /Volumes/T9/IndraAstra/dhiraj/neuro_graph && SGN_SLOT=mini_mps d_env/bin/python3 -u scripts/train_step990_additive_dynamic_t0.py --device mps 2>&1 | tee $L" > /tmp/l990.sh
tmux new-session -d -s $S bash /tmp/l990.sh

# step991 v2 on mini_cpu
S=sgn-indra-mini_cpu-train_step991_hebbian_t0
L=/Volumes/T9/IndraAstra/dhiraj/neuro_graph/logs/train_step991_hebbian_t0__mini_cpu.log
echo "cd /Volumes/T9/IndraAstra/dhiraj/neuro_graph && SGN_SLOT=mini_cpu d_env/bin/python3 -u scripts/train_step991_hebbian_t0.py --device cpu 2>&1 | tee $L" > /tmp/l991.sh
tmux new-session -d -s $S bash /tmp/l991.sh
```
Validity check after ~10 min: Ref must reach ≥~60% by ep5-10 (canonical chain).
If Ref ~11% again → chain still broken, debug before trusting deltas.

## 2. step989 — GPT-2 FFN distillation (founding-vision test)

- Extraction RUNNING on 5060ti, tmux `sgn-indra-extract-gpt2`,
  log `/home/indra/sgnnet_bench/logs/extract_gpt2_ffn.log`.
  transformers==4.44.0 installed (5.5.4 broke on torchvision::nms — fixed).
- When `data/gpt2_ffn_layer6.pt` exists on 5060ti: launch
  `scripts/train_step989_ffn_distil_t0.py` there. CUDA audit blocks it
  (custom MSE loop, no Resonant_CUDA import) — legitimate exemption, use:
  `bash scripts/launch_slot.sh 5060ti_cuda scripts/train_step989_ffn_distil_t0.py --unsafe-cuda-launch`
- Advance: val_mse ≤ 2× Ref_mlp AND cos_sim ≥ 0.5.
- Outcome decides Paper 2 spine (FFN replacement vs dynamic routing).

## 3. step982 T2 — CIFAR-10 aug paper claim (INCOMPLETE, relaunch)

Ref DONE (80.79% @ep140, 471s). **A_aug crashed silently at start** — likely OOM
loading 2.3GB `store_cifar10_aug.h5` (file exists on 5060ti). Log ends after Ref.
Relaunch on 5060ti after extraction finishes (script reruns Ref too, ~16 min
total): `bash scripts/launch_slot.sh 5060ti_cuda scripts/train_step982_cifar10_aug_t2.py`
If A_aug OOMs again: add chunked h5 load to script (T1 version had sequential fix).
Paper claim if A_aug ≥ Ref + 0.5pp at T2.

## 4. Results landed this session (record + propagate)

| Step | Result | Verdict |
|---|---|---|
| step985 v1/v2, 987, 988 | all configs ~11-14% | PhaseGate direction KILLED-CONFIRMED (16+ gate kills) |
| step992 K-means init | Ref=85.58, A=−1.83pp, B=−3.11pp | KILLED — random init confirmed; debt retired (VALID run) |
| step990 v1 | Ref=11.6% bare base | INVALID — v2 relaunch (§1) |
| step991 v1 | Ref=14% bare base | INVALID — v2 relaunch (§1) |
| step986 T1 | N=16384 → 82.85% @ep69 (75ep, 50%) | scaling curve continues; consider T2 |
| step982 T2 | Ref=80.79%; A_aug crashed | incomplete (§3) |

TODO bookkeeping:
- EXPERIMENT_QUEUE.md: mark step992 DONE/KILLED, step990/991 v1 INVALID + v2
  RUNNING, step986 DONE, step982 partial. Update VISION_DEBT.md (K-means row →
  KILLED-CONFIRMED).
- Graphiti episodes (group_id="dhiraj") for: 985/987/988 kill, 992 kill,
  986 result, vision review. MCP was down earlier — retry.
- git add + commit + push (token in remote URL works; commit f81bf71 pushed OK).

## 5. Key technical facts (do not relearn)

- **Canonical chain (MANDATORY for any experiment):** `SGNNET_SmallWorld` →
  `SGNNET_Resonant_CUDA(alpha_reflect=0.5, alpha_turing=0, mode="dynamic_z_geo")`
  → `SGNNET_AntiHebbian_CUDA(alpha_ahebb=1.0, variant="wpos")`.
  **Bare SmallWorld does NOT learn** (AH prerequisite, step218: −73pp).
  On CPU/MPS pass `compile=(device=="cuda")`.
- step990 v2 model: `SGNNET_AH_AdditiveDynamic` in
  `src/sgnnet/model_additive_dynamic.py` — eager canonical-chain semantics +
  additive `dynamic_connectivity_hh` term. Ref = alpha_dyn=0. N=512 (O(N²) cdist).
- step991 v2: `HebbianRewirer` in `src/sgnnet/model_hebbian.py` — controller,
  not forward wrapper. Scores edges by ΔW-proj on diagnostic batch (`_DIAG_X`
  module global set in main), rewires base.conn_hh at tick_epoch, then
  `ah._invalidate_supp_w()` (supp_w depends on conn_hh).
- Trainer auto-calls `model.tick_epoch()` (trainer.py:331).
- Slots: mini_mps, mini_cpu, 5060ti_cuda only (Studio removed).
- Auto-compact: globally disabled (`~/.claude/settings.json autoCompactEnabled:
  false`); project window 150k. Manual /compact needed.

## 6. After T0 batch lands — decision tree

- Any of step989/990/991 positive (≥ +0.5pp or step989 criteria) → T1 per tier
  protocol (75ep, 50%).
- All killed → vision-debt CLOSED; write closing entry in VISION_DEBT.md +
  architecture_dead_ends.md; original brief fully adjudicated (every promise
  KEPT/EVOLVED/KILLED-CONFIRMED with evidence). Then back to Paper 1 finish:
  step982 T2 claim + step986 T2 (N=16384, 150ep) if scaling matters for paper.
- mem0_application: pricing placeholders [₹X] + calendar link still unfilled
  (user action).

## 7. Open files / uncommitted state

Modified this session (uncommitted): launch_slot.sh (TERM removal + setsid line
— clean up per §1), model_hebbian.py (v2 rewrite), model_additive_dynamic.py
(appended AH variant), train_step990/991 scripts (v2), EXPERIMENT_QUEUE.md
(earlier edits committed in f81bf71; new edits pending), NEXT_STEPS.md (this).
