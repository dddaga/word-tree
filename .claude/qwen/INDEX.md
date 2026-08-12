# Qwen Workspace Index

## Context (read these before working)

| File | Purpose |
|------|---------|
| `context/project_briefing.md` | 1-page SGNNET project overview — architecture, notation, FLOPs scope, key results |
| `context/do_not_do.md` | Known failure modes and misconceptions to avoid |

## Task Instructions (use with `qwen -p <task>`)

| Task file | Invocation | Purpose |
|-----------|-----------|---------|
| `tasks/red_team_critique.md` | `qwen -p red_team_critique` | Attack paper claims, rank by severity |
| `tasks/scaffold_script.md` | `qwen -p scaffold_script` | Draft training scripts from queue entries |
| `tasks/cuda_checklist_review.md` | `qwen -p cuda_checklist_review` | Validate scripts against 5060ti infra rules |

## Sessions

JSONL session logs: `sessions/YYYY-MM-DD.jsonl`
Each line: `{"ts": ..., "task": ..., "input": ..., "output": ..., "tool_calls": [...]}`

## Outputs

Critique and analysis outputs from completed tasks: `outputs/`

## Python Harness

`scripts/qwen_agent.py` — tool-calling harness (see file for usage)
- Loads `context/project_briefing.md` as system prompt
- Whitelisted tools: `read_file` (project paths only), `run_bash` (read-only: ls/grep/wc/jq/tail)
- Logs every session to `sessions/`
