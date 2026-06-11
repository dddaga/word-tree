# Loop State — single source of truth for /loop session
*Maintained by main agent. Reload THIS file (not transcript) on every wakeup.
Keep ≤120 lines. Prune DONE rows to bottom section weekly.*

Updated: 2026-06-11 ~18:00

## Active loop
`/goal to do all the action items to have a rigourous 1st paper are done` — Stop hook active.
Paper action items being worked: manuscript sections (audio/scaling/TS/CIFAR100/multimodal),
step982 T2 (pending converter), ts_step030 T0 (DONE-NEGATIVE), step989 (DONE-KILLED).

## Context-management protocol (Dhiraj directive 2026-06-10)
1. All findings/state → .md files immediately; context holds pointers only.
2. On wakeup: read LOOP_STATE.md first, then ONLY files needed for current task.
3. Heavy work (design, debug, research, multi-file edits) → background sub-agents
   with self-contained prompts; main agent stays thin orchestrator.
4. Maintain TODO below religiously — it replaces transcript memory.

## TODO
| # | Task | Status | Notes |
|---|---|---|---|
| 1 | step982 T2 A_aug on 5060ti | PENDING-LAUNCH | converter (h5→npy) running on 5060ti. Manik GPU at 99%/4GB. Launch step982 when converter done + GPU free. Ref resume JSON exists. |
| 2 | step989 T0 | DONE-KILLED | cos_sim=0.187 (d16), 0.213 (d32). Both below 0.5 threshold. Founding vision retired. VISION_DEBT complete. |
| 3 | ts_step030 T0 | DONE-NEGATIVE | All models dir_acc≈50%, sharpe<−100. Task near-random. Paper = honest negative. |
| 4 | §8 multimodal section | DONE | MANUSCRIPT_DRAFT_sec5_multimodal.md written. CIFAR-10/100, audio ESC-50, text. |
| 5 | §9 scaling law section | DONE | MANUSCRIPT_DRAFT_sec6_scaling.md written. Imagenette + CIFAR-10 N-scaling. |
| 6 | §10 time series section | DONE | MANUSCRIPT_DRAFT_sec7_timeseries.md written. Negative result. |
| 7 | Graphiti episodes backlog | TODO | Record: step989 KILLED, ts_step030 NEGATIVE, step982 pending. |
| 8 | Commit new files | TODO | sec5_multimodal, sec6_scaling, sec7_timeseries, ts_common, ts_model_sgnnet, ts_step030 (rewrite), QUEUE edits. |
| 9 | Paper audit gaps remaining | TODO | GAP10 (GLNN cross-dataset), GAP11 (theory). Also: step982 T2 result when done. |

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
- launch_slot.sh bug FIXED 2026-06-10: shell var TMUX="tmux" overwrote exported
  $TMUX (sessions run inside tmux) → client socket path "tmux" on exFAT cwd →
  "Operation not supported". Renamed TMUX_BIN. Committed.
- Caveman mode full active.
