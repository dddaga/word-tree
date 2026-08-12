"""
Qwen agentic harness — tool-calling loop with project context injection.

Usage:
    python scripts/qwen_agent.py --task "critique the K_iter ablation"
    python scripts/qwen_agent.py --task-file .claude/qwen/tasks/red_team_critique.md --input "see claims.md"
    python scripts/qwen_agent.py --task "..." --context  # inject project briefing (default: on)
    python scripts/qwen_agent.py --task "..." --no-context  # skip briefing injection

Tools available to qwen (read-only):
    read_file(path)  — read any file under PROJECT_ROOT (relative paths only)
    run_bash(cmd)    — restricted shell: ls, grep, wc, jq, tail, head, cat, find

Session logs: .claude/qwen/sessions/YYYY-MM-DD.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import requests

# ── Config ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/Volumes/T9/IndraAstra/dhiraj/neuro_graph")
QWEN_DIR = PROJECT_ROOT / ".claude" / "qwen"
SESSIONS_DIR = QWEN_DIR / "sessions"
CONTEXT_DIR = QWEN_DIR / "context"
TASKS_DIR = QWEN_DIR / "tasks"

API_BASE = os.getenv("LM_STUDIO_URL", "http://localhost:1234/v1")
MODEL = os.getenv("QWEN_MODEL", "qwen/qwen3-coder-next")
MAX_TOOL_ROUNDS = 10
MAX_TOKENS = 4096

# Bash commands allowed (prefix match)
ALLOWED_BASH_PREFIXES = ("ls", "grep", "wc", "jq", "tail", "head", "cat", "find", "echo")


# ── Tool definitions (sent to qwen) ─────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the SGNNET project. "
                "Path must be relative to project root (/Volumes/T9/IndraAstra/dhiraj/neuro_graph). "
                "Examples: 'learnings/paper/claims.md', 'scripts/train_step977_kd_vs_ce_multiseed_t1.py'"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative path from project root"},
                    "lines": {"type": "integer", "description": "Max lines to return (default 200)"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a read-only shell command in the project root. "
                "Allowed: ls, grep, wc, jq, tail, head, cat, find, echo. "
                "No write operations, no python, no training scripts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Shell command to run"},
                },
                "required": ["cmd"],
            },
        },
    },
]


# ── Tool dispatch ────────────────────────────────────────────────────────────
def dispatch_tool(name: str, args: dict) -> str:
    if name == "read_file":
        return _tool_read_file(args["path"], args.get("lines", 200))
    elif name == "run_bash":
        return _tool_run_bash(args["cmd"])
    else:
        return f"ERROR: unknown tool '{name}'"


def _tool_read_file(rel_path: str, max_lines: int) -> str:
    # Security: resolve to absolute, must stay within project root
    abs_path = (PROJECT_ROOT / rel_path).resolve()
    try:
        abs_path.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return f"ERROR: path '{rel_path}' escapes project root"

    if not abs_path.exists():
        return f"ERROR: file not found: {rel_path}"
    if not abs_path.is_file():
        return f"ERROR: not a file: {rel_path}"

    try:
        lines = abs_path.read_text(errors="replace").splitlines()
        if len(lines) > max_lines:
            truncated = len(lines) - max_lines
            lines = lines[:max_lines] + [f"... [{truncated} lines truncated]"]
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR reading file: {e}"


def _tool_run_bash(cmd: str) -> str:
    # Security: only allow whitelisted command prefixes
    stripped = cmd.strip()
    allowed = any(stripped.startswith(p) for p in ALLOWED_BASH_PREFIXES)
    if not allowed:
        return (
            f"ERROR: command not allowed. "
            f"Permitted prefixes: {', '.join(ALLOWED_BASH_PREFIXES)}"
        )

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=15,
        )
        out = result.stdout
        if result.stderr:
            out += f"\n[stderr]: {result.stderr[:500]}"
        return out[:4000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out (15s)"
    except Exception as e:
        return f"ERROR: {e}"


# ── API call ─────────────────────────────────────────────────────────────────
def chat(messages: list, tools: Optional[list] = None) -> dict:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.3,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    resp = requests.post(
        f"{API_BASE}/chat/completions",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ── Agent loop ───────────────────────────────────────────────────────────────
def run_agent(task: str, inject_context: bool = True):
    system_parts = []

    if inject_context:
        briefing_path = CONTEXT_DIR / "project_briefing.md"
        dont_path = CONTEXT_DIR / "do_not_do.md"
        if briefing_path.exists():
            system_parts.append(briefing_path.read_text())
        if dont_path.exists():
            system_parts.append(dont_path.read_text())

    system_parts.append(
        "You are a research assistant for the SGNNET project. "
        "Use tools to read project files when you need evidence. "
        "Be precise, critical, and evidence-based. "
        "Cite specific file paths and line numbers when making claims."
    )

    system_prompt = "\n\n---\n\n".join(system_parts)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task},
    ]

    tool_call_log = []
    final_content = ""

    for round_num in range(MAX_TOOL_ROUNDS):
        response = chat(messages, tools=TOOLS)
        choice = response["choices"][0]
        msg = choice["message"]
        finish_reason = choice["finish_reason"]

        # Accumulate content
        if msg.get("content"):
            final_content = msg["content"]

        # No more tool calls
        if finish_reason != "tool_calls" or not msg.get("tool_calls"):
            break

        # Append assistant message with tool calls
        messages.append(msg)

        # Dispatch each tool call
        tool_results = []
        for tc in msg["tool_calls"]:
            fn_name = tc["function"]["name"]
            try:
                fn_args = json.loads(tc["function"]["arguments"])
            except json.JSONDecodeError:
                fn_args = {}

            result = dispatch_tool(fn_name, fn_args)
            tool_call_log.append({"tool": fn_name, "args": fn_args, "result_len": len(result)})

            print(f"  [tool] {fn_name}({fn_args}) → {len(result)} chars", file=sys.stderr)

            tool_results.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })

        messages.extend(tool_results)

    return final_content, tool_call_log


# ── Session logging ───────────────────────────────────────────────────────────
def log_session(task: str, output: str, tool_calls: list) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = SESSIONS_DIR / f"{datetime.now().strftime('%Y-%m-%d')}.jsonl"

    entry = {
        "ts": datetime.now().isoformat(),
        "task": task[:200],
        "output_len": len(output),
        "tool_rounds": len(tool_calls),
        "tool_calls": tool_calls,
        "output_preview": output[:500],
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen agentic harness")
    parser.add_argument("--task", type=str, help="Task string (inline)")
    parser.add_argument("--task-file", type=str, help="Path to task instruction file")
    parser.add_argument("--input", type=str, default="", help="Additional input appended to task file")
    parser.add_argument("--no-context", action="store_true", help="Skip project briefing injection")
    parser.add_argument("--save", type=str, help="Save output to this path")
    args = parser.parse_args()

    # Build task
    if args.task_file:
        task_path = Path(args.task_file)
        if not task_path.exists():
            # Try relative to tasks dir
            task_path = TASKS_DIR / args.task_file
            if not task_path.exists():
                print(f"ERROR: task file not found: {args.task_file}", file=sys.stderr)
                sys.exit(1)
        task = task_path.read_text()
        if args.input:
            task += f"\n\n---\n# INPUT\n{args.input}"
    elif args.task:
        task = args.task
    else:
        print("ERROR: provide --task or --task-file", file=sys.stderr)
        sys.exit(1)

    inject = not args.no_context
    print(f"[qwen_agent] model={MODEL} context={inject} task_len={len(task)}", file=sys.stderr)

    output, tool_calls = run_agent(task, inject_context=inject)
    log_session(task, output, tool_calls)

    print(output)

    if args.save:
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(output)
        print(f"\n[saved to {save_path}]", file=sys.stderr)


if __name__ == "__main__":
    main()
