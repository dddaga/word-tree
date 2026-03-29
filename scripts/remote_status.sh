#!/usr/bin/env bash
# remote_status.sh — check what's running on Mac Studio and tail all active logs.
#
# Usage:
#   ./scripts/remote_status.sh

REMOTE_HOST="mac-studio"
REMOTE_DIR="/Users/admin/ml/dhiraj/qwen2_omni/testing"
TMUX="/opt/homebrew/bin/tmux"

echo "=== Mac Studio status ==="
ssh "$REMOTE_HOST" bash << ENDSSH
echo "-- tmux sessions --"
$TMUX ls 2>/dev/null || echo "(no tmux sessions)"

echo ""
echo "-- active python training processes --"
ps aux | grep "python.*train_" | grep -v grep || echo "(none)"

echo ""
echo "-- all logs (last 10 lines each) --"
LOGS=\$(ls -t "$REMOTE_DIR/logs/"*.log 2>/dev/null)
if [ -n "\$LOGS" ]; then
    for LOG in \$LOGS; do
        echo ""
        echo ">>> \$LOG"
        tail -10 "\$LOG"
    done
else
    echo "(no log files yet)"
fi
ENDSSH
