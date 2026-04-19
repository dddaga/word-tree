#!/bin/bash
# Suppress noisy monitoring bash commands from the transcript.
# Returns {"suppressOutput": true} for routine status-check and file-sync commands.
# The command still executes — only stdout is hidden from the transcript view.

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty' 2>/dev/null)

# Slot status checks
if echo "$COMMAND" | grep -qE 'slot_status\.sh'; then
  echo '{"suppressOutput": true}'
  exit 0
fi

# tmux session inspection
if echo "$COMMAND" | grep -qE 'tmux (ls|list-sessions|list-panes|capture-pane)'; then
  echo '{"suppressOutput": true}'
  exit 0
fi

# SSH monitoring commands (tmux, nvidia-smi, ps, tail logs, ls results)
if echo "$COMMAND" | grep -qE 'ssh (mac-studio|indra@5060ti|5060ti).*(tmux|nvidia-smi|ps |tail |cat.*\.log|ls -t|cat.*\.json)'; then
  echo '{"suppressOutput": true}'
  exit 0
fi

# rsync result syncing
if echo "$COMMAND" | grep -qE '^rsync .*(results/|\.json)'; then
  echo '{"suppressOutput": true}'
  exit 0
fi

# cat/tail on log and result JSON files
if echo "$COMMAND" | grep -qE '(cat|tail).*(\.log|results/.*\.json|/tmp/.*\.log)'; then
  echo '{"suppressOutput": true}'
  exit 0
fi

# System resource checks
if echo "$COMMAND" | grep -qE 'vm_stat|nvidia-smi|pgrep|ps aux.*grep|free -h'; then
  echo '{"suppressOutput": true}'
  exit 0
fi

exit 0
