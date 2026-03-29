#!/usr/bin/env bash
# remote_dispatch.sh — sync code+data to Mac Studio and run training in a tmux session.
#
# Usage:
#   ./scripts/remote_dispatch.sh "<training command>"
#   ./scripts/remote_dispatch.sh --session NAME "<training command>"   # named session for parallel jobs
#   ./scripts/remote_dispatch.sh --session NAME --no-sync "<cmd>"     # skip rsync (already synced)
#
# Parallel training example (jobs run simultaneously on Mac Studio):
#   ./scripts/remote_dispatch.sh --session neuro_a "python -u scripts/train_A.py ... 2>&1 | tee logs/A.log"
#   ./scripts/remote_dispatch.sh --session neuro_b --no-sync "python -u scripts/train_B.py ... 2>&1 | tee logs/B.log"

set -euo pipefail

REMOTE_HOST="mac-studio"
REMOTE_DIR="/Users/admin/ml/dhiraj/qwen2_omni/testing"
LOCAL_DIR="/Volumes/T9/IndraAstra/dhiraj/neuro_graph"
TMUX="/opt/homebrew/bin/tmux"
PYTHON="/opt/homebrew/bin/python3.13"
SESSION="neuro"
SKIP_SYNC=0

# Parse optional flags before the command
while [[ $# -gt 0 ]]; do
    case "$1" in
        --session)   SESSION="$2"; shift 2 ;;
        --no-sync)   SKIP_SYNC=1;  shift   ;;
        *)           break ;;
    esac
done

CMD="${1:-}"
if [[ -z "$CMD" ]]; then
    echo "Usage: $0 [--session NAME] [--no-sync] \"<training command>\""
    echo "  Training command is relative to $REMOTE_DIR"
    exit 1
fi

if [[ $SKIP_SYNC -eq 0 ]]; then
    echo "=== [1/4] Syncing code + data to Mac Studio ==="
    rsync -avz --progress \
        --exclude='d_env/' \
        --exclude='learnings/' \
        --exclude='.planning/' \
        --exclude='.claude/' \
        --exclude='.git/' \
        --exclude='.jj/' \
        --exclude='results/' \
        --exclude='logs/' \
        --exclude='checkpoints/' \
        --exclude='data/imagenette2-320/' \
        --exclude='keys/' \
        "$LOCAL_DIR/" \
        "$REMOTE_HOST:$REMOTE_DIR/"
else
    echo "=== [1/4] Skipping sync (--no-sync) ==="
fi

echo ""
echo "=== [2/4] Ensuring remote dirs exist ==="
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR/{logs,results,checkpoints}"

echo ""
echo "=== [3/4] Bootstrapping Python venv (idempotent) ==="
ssh "$REMOTE_HOST" bash << ENDSSH
set -e
cd "$REMOTE_DIR"
if [ ! -f "d_env/bin/activate" ]; then
    echo "Creating venv with $PYTHON ..."
    $PYTHON -m venv d_env
fi
echo "Installing/updating requirements..."
d_env/bin/pip install --quiet --upgrade pip
d_env/bin/pip install --quiet -r requirements.txt
echo "Venv ready: \$(d_env/bin/python --version)"
ENDSSH

echo ""
echo "=== [4/4] Starting tmux session '$SESSION' on Mac Studio ==="
ssh "$REMOTE_HOST" bash << ENDSSH
set -e
cd "$REMOTE_DIR"

# Kill existing session if present
$TMUX kill-session -t "$SESSION" 2>/dev/null && echo "Killed old session '$SESSION'" || true

# Window 0: training job (source venv, then run command)
$TMUX new-session -d -s "$SESSION" -x 220 -y 50 \
    "cd $REMOTE_DIR && source d_env/bin/activate && $CMD; echo '--- DONE --- exit code: \$?'; read"

# Window 1: live log monitor (opens after brief pause so log file exists)
$TMUX new-window -t "$SESSION:1" \
    "sleep 5 && cd $REMOTE_DIR && watch -n 10 'tail -n 15 \$(ls -t logs/*.log 2>/dev/null | head -1 || echo /dev/null)'"

$TMUX select-window -t "$SESSION:0"
echo "Session '$SESSION' started."
$TMUX ls
ENDSSH

echo ""
echo "=== Done ==="
echo "Training dispatched to Mac Studio."
echo ""
echo "Useful commands:"
echo "  Check status  : ./scripts/remote_status.sh"
echo "  Pull results  : ./scripts/sync_results.sh"
echo "  Attach to tmux: ssh mac-studio -t '$TMUX attach -t $SESSION'"
