#!/usr/bin/env bash
# PreCompact hook: use Haiku to summarize the session, inject summary so it
# survives into the compacted context. The built-in compact still clears
# conversation history; this ensures Haiku's summary is what gets preserved.
#
# Returns JSON with:
#   continue: true          — let built-in compact proceed (clears context)
#   additionalContext       — Haiku summary injected before compact runs

set -euo pipefail

MONITOR_DIR="/Volumes/T9/IndraAstra/dhiraj/neuro_graph/.monitor"
STATE_FILE="$MONITOR_DIR/session_state.md"
QUEUE_FILE="/Volumes/T9/IndraAstra/dhiraj/neuro_graph/learnings/EXPERIMENT_QUEUE.md"
mkdir -p "$MONITOR_DIR"

# Read conversation JSON from stdin (passed by Claude Code)
CONVERSATION=$(cat)

# Build context snippet: running experiments from queue (fast, no API)
RUNNING_CTX=""
if [[ -f "$QUEUE_FILE" ]]; then
  RUNNING_CTX=$(grep -A3 "RUNNING\|step[0-9]" "$QUEUE_FILE" 2>/dev/null | head -40 || true)
fi

PROMPT="You are a research-session state distiller for an ML research project (SGNNET — sparse graph neural network, Imagenette image classification, paper-in-progress).

Produce a COMPACT (≤80 lines) markdown summary of the session. Include ONLY information the main agent CANNOT recover from git/code files — i.e. decisions made, experiment verdicts with delta-pp numbers, what was about to happen next, and any non-obvious constraints discovered.

Structure:
## Session State — $(date -u '+%Y-%m-%d %H:%M UTC')
### Running Experiments
(slot, step, config, what it's testing)
### Key Findings This Session
(bullet per result: step, configs, delta-pp, verdict CONFIRMED/KILLED/T0-artifact)
### Decisions Made
(design choices, defaults changed, directions opened/closed)
### Next Actions
(what was in-flight when compaction fired)

Be terse. No filler. Omit anything derivable from code or git log.

CURRENT QUEUE STATE (for running experiments):
$RUNNING_CTX

SESSION CONVERSATION:
$CONVERSATION"

# Call Haiku via claude CLI
CLAUDE_BIN="${CLAUDE_BIN:-/Users/indra/.local/bin/claude}"
SUMMARY=$(echo "$PROMPT" | "$CLAUDE_BIN" \
  --model claude-haiku-4-5-20251001 \
  --output-format text \
  2>/dev/null || echo "## Haiku summary failed — check claude CLI availability")

# Save to file for cross-session reference
{
  echo "$SUMMARY"
  echo ""
  echo "---"
} > "$STATE_FILE"

# Return JSON: let built-in compact proceed, inject Haiku summary as preserved context
ESCAPED=$(echo "$SUMMARY" | python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))")

cat <<EOF
{
  "continue": true,
  "hookSpecificOutput": {
    "hookEventName": "PreCompact",
    "additionalContext": $ESCAPED
  }
}
EOF
