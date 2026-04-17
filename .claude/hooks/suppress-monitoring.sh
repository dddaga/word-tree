#!/bin/bash
# Suppress noisy monitoring bash commands from entering context.
# Returns {"suppressOutput": true} for routine status-check commands.
# The command still executes — only the stdout is hidden from the transcript.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

# Patterns that are purely status/monitoring — output is noise, not signal
if echo "$COMMAND" | grep -qE '^(bash scripts/slot_status\.sh|ssh (mac-studio|5060ti) "tmux|tmux ls|tmux list|vm_stat|nvidia-smi|pgrep|ps aux.*grep|tail.*\.log|cat.*\.log)'; then
  echo '{"suppressOutput": true}'
  exit 0
fi

exit 0
