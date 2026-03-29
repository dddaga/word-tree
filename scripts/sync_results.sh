#!/usr/bin/env bash
# sync_results.sh — pull results, logs, and checkpoints from Mac Studio back to Mac Mini.
#
# Run this after a training job completes (or at any time to get partial results).
# Safe to run repeatedly — rsync only copies changed/new files.
#
# Usage:
#   ./scripts/sync_results.sh

set -euo pipefail

REMOTE_HOST="mac-studio"
REMOTE_DIR="/Users/admin/ml/dhiraj/qwen2_omni/testing"
LOCAL_DIR="/Volumes/T9/IndraAstra/dhiraj/neuro_graph"

echo "=== Syncing results from Mac Studio → Mac Mini ==="

echo ""
echo "--- results/ ---"
rsync -avz --progress \
    "$REMOTE_HOST:$REMOTE_DIR/results/" \
    "$LOCAL_DIR/results/"

echo ""
echo "--- logs/ ---"
rsync -avz --progress \
    "$REMOTE_HOST:$REMOTE_DIR/logs/" \
    "$LOCAL_DIR/logs/"

echo ""
echo "--- checkpoints/ ---"
rsync -avz --progress \
    "$REMOTE_HOST:$REMOTE_DIR/checkpoints/" \
    "$LOCAL_DIR/checkpoints/" 2>/dev/null || echo "(no checkpoints yet)"

echo ""
echo "=== Sync complete ==="
echo "New/updated results:"
ls -lt "$LOCAL_DIR/results/"*.json 2>/dev/null | head -5 || echo "(none)"
