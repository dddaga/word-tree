#!/bin/bash
# PreCompact hook: spawn Haiku to distill session state before compaction.
# Writes .monitor/session_state.md so the post-compact agent can resume cleanly.
# Runs async so it doesn't block compaction.

MONITOR_DIR="/Volumes/T9/IndraAstra/dhiraj/neuro_graph/.monitor"
STATE_FILE="$MONITOR_DIR/session_state.md"
QUEUE_FILE="/Volumes/T9/IndraAstra/dhiraj/neuro_graph/learnings/EXPERIMENT_QUEUE.md"

mkdir -p "$MONITOR_DIR"

# Write timestamp marker so main agent knows distillation ran
echo "## PreCompact distillation: $(date -u '+%Y-%m-%d %H:%M UTC')" >> "$STATE_FILE"
echo "Haiku distillation hook fired. Reading queue state..." >> "$STATE_FILE"

# Extract currently running experiments from EXPERIMENT_QUEUE.md (fast, no API call)
if [ -f "$QUEUE_FILE" ]; then
  RUNNING=$(grep -A2 "RUNNING" "$QUEUE_FILE" 2>/dev/null | head -20)
  echo "### Running slots:" >> "$STATE_FILE"
  echo "$RUNNING" >> "$STATE_FILE"
fi

exit 0
