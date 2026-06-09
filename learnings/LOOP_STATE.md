# Loop State — single source of truth for /loop session
*Maintained by main agent. Reload THIS file (not transcript) on every wakeup.
Keep ≤120 lines. Prune DONE rows to bottom section weekly.*

Updated: 2026-06-10 ~04:00

## Active loop
`/loop till all taskks in NEXT_STEPS.md are done` — dynamic mode, ScheduleWakeup ~1500s.

## Context-management protocol (Dhiraj directive 2026-06-10)
1. All findings/state → .md files immediately; context holds pointers only.
2. On wakeup: read LOOP_STATE.md first, then ONLY files needed for current task.
3. Heavy work (design, debug, research, multi-file edits) → background sub-agents
   with self-contained prompts; main agent stays thin orchestrator.
4. Maintain TODO below religiously — it replaces transcript memory.

## TODO
| # | Task | Status | Notes |
|---|---|---|---|
| 1 | step982 T2 A_aug on 5060ti | RUNNING | session sgn-indra-5060ti_cuda-train_step982_cifar10_aug_t2. Ref=80.58% banked in JSON (resume logic). Paper claim if A_aug ≥81.08%. Check log via ssh. |
| 2 | cnnc_step001 design sub-agent | RUNNING | bg agent designing multi-branch CNN T0, will submit via scheduler. Notification on completion. |
| 3 | ffn T1 (75ep) in scheduler queue | QUEUED | job 20260610_034743, slots mini_mps/mini_cpu. Scheduler daemon tmux sgn-scheduler. Verify pickup. |
| 4 | step989 GPT-2 extraction | BLOCKED | teammate PID 1098714 holds 3.8 GB on 5060ti; need ~5+ GB free. Then extraction → train_step989_ffn_distil_t0.py --unsafe-cuda-launch. |
| 5 | Graphiti episodes backlog | BLOCKED | MCP unavailable all session. Episodes: step985/987/988 kill, step992 kill, step986, step991 kill, step990/993 kill, step982 result, ffn_step001 T0 result. |
| 6 | ffn_baseline/QUEUE.md T0 results row | TODO | b1_A=92.79, b1_C=92.84, b5_A=92.94, b5_C=92.92, B variants 92.46; sparsity≈0.50/layer. |
| 7 | Commit all new files | TODO | research reports, scheduler, ffn line, cnn_compress line, step982 fixes, this file. |
| 8 | NEXT_STEPS §6 decision tree | WAIT | after step982 + step989 resolve. |
| 9 | step960 quaternion T0 + expert-choice routing T0 | PROPOSED | from paper2 research report — Mac slots via scheduler. Not yet user-approved as queue item; loop scope = NEXT_STEPS.md. |

## Key facts (so transcript not needed)
- ffn_step001 T0: per-channel FFN @0.93% FC budget = 92.8% Imagenette T0. Budget
  saturated (5%→+0.1pp). Sparsity ≈50% as predicted. Results in
  results/ffn_baseline/ffn_step001_perchannel_t0_seed42__mini_mps.json.
- Research reports saved: learnings/research/IMPROVEMENT_PLANS_2026-06-10_*.md
  (paper2_routing, efficiency_frontier, cifar10_gap, audio_gap).
- Scheduler: scripts/scheduler/{scheduler.py,submit.py}; state .scheduler/{pending,running,done,failed}.
  Daemon polls 60s, round-robin across lines, foreign-load guards.
- step993 T1 KILLED (additive dynamic, sign reversal); recorded in
  EXPERIMENT_QUEUE.md + VISION_DEBT.md.
- NON-SGN tmux sessions DO NOT TOUCH: manik-1-0, neuro_g, dashboard, oracle,
  oracle-debug-ui, qwen_embed, summaries, val_benchmark, sf-manik-5060ti_cuda-main.
- Caveman mode full active.
