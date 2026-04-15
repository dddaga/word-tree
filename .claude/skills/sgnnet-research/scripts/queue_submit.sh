#!/usr/bin/env bash
# queue_submit.sh — Submit an experiment to the SGNNET controller queue.
#
# Identity resolution (in order):
#   1. --user <name> CLI flag
#   2. SGNNET_USER env var
#   3. .sgnnet_user file in the current working directory
#   4. Interactive prompt (first-run setup); inferred default from parent dir
#      name. Writes .sgnnet_user so subsequent runs are silent.
#
# Usage:
#   queue_submit.sh <script_path> \
#       [--user <name>] \
#       [--step-name <name>] \
#       [--device-pref any|cuda|mps|cpu] \
#       [--slot-pref mini_mps|mini_cpu|studio_mps|studio_cpu|5060ti_cuda] \
#       [--args "<extra args>"] \
#       [--priority <int>]
#
# Environment variables:
#   SGNNET_USER            — override identity (optional; .sgnnet_user preferred)
#   SGNNET_CONTROLLER_URL  — default http://mac-mini.local:7433

set -euo pipefail

CONTROLLER_URL="${SGNNET_CONTROLLER_URL:-http://mac-mini.local:7433}"
USER_FILE="$(pwd)/.sgnnet_user"

# --- parse args -------------------------------------------------------------
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <script_path> [--step-name <name>] [--device-pref any|cuda|mps|cpu] [--slot-pref <slot>] [--args \"...\"] [--priority <n>]" >&2
  exit 1
fi

SCRIPT_PATH="$1"
shift

STEP_NAME=""
DEVICE_PREF="any"
SLOT_PREF=""
EXTRA_ARGS=""
PRIORITY=0
USER_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)         USER_FLAG="$2";   shift 2 ;;
    --step-name)    STEP_NAME="$2";   shift 2 ;;
    --device-pref)  DEVICE_PREF="$2"; shift 2 ;;
    --slot-pref)    SLOT_PREF="$2";   shift 2 ;;
    --args)         EXTRA_ARGS="$2";  shift 2 ;;
    --priority)     PRIORITY="$2";    shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# --- resolve identity ------------------------------------------------------
# Priority: --user flag > SGNNET_USER env var > .sgnnet_user file > prompt
if [[ -n "$USER_FLAG" ]]; then
  SGNNET_USER="$USER_FLAG"
elif [[ -n "${SGNNET_USER:-}" ]]; then
  :  # env var already set
elif [[ -f "$USER_FILE" ]]; then
  SGNNET_USER=$(head -1 "$USER_FILE" | tr -d '[:space:]')
fi

if [[ -z "${SGNNET_USER:-}" ]]; then
  # First-run interactive setup
  if [[ ! -t 0 && ! -r /dev/tty ]]; then
    echo "ERROR: no identity configured and no TTY for interactive setup." >&2
    echo "  Create $USER_FILE with your user id, or pass --user <name>." >&2
    exit 1
  fi
  DEFAULT=$(basename "$(dirname "$(pwd)")")
  echo "" >&2
  echo "── First-time setup: SGNNET identity not configured ──" >&2
  echo "  No $USER_FILE found, no --user flag, no \$SGNNET_USER env var." >&2
  echo "  This identity tags your experiments in the shared queue." >&2
  echo "" >&2
  printf "  Enter your SGNNET user id [%s]: " "$DEFAULT" >&2
  read -r input < /dev/tty
  SGNNET_USER="${input:-$DEFAULT}"
  SGNNET_USER=$(echo "$SGNNET_USER" | tr -d '[:space:]')
  if [[ -z "$SGNNET_USER" ]]; then
    echo "ERROR: empty user id; aborting." >&2
    exit 1
  fi
  echo "$SGNNET_USER" > "$USER_FILE"
  echo "  Saved → $USER_FILE" >&2
  echo "  (Future runs will read this silently. Delete the file to re-configure.)" >&2
  echo "" >&2
fi

# --- derive step name if not given ------------------------------------------
if [[ -z "$STEP_NAME" ]]; then
  STEP_NAME=$(basename "$SCRIPT_PATH" .py)
fi

# --- validate device_pref ---------------------------------------------------
case "$DEVICE_PREF" in
  any|cuda|mps|cpu) ;;
  *) echo "ERROR: --device-pref must be one of: any cuda mps cpu" >&2; exit 1 ;;
esac

# --- validate slot_pref if given --------------------------------------------
if [[ -n "$SLOT_PREF" ]]; then
  case "$SLOT_PREF" in
    mini_mps|mini_cpu|studio_mps|studio_cpu|5060ti_cuda) ;;
    *) echo "ERROR: --slot-pref must be one of: mini_mps mini_cpu studio_mps studio_cpu 5060ti_cuda" >&2; exit 1 ;;
  esac
fi

# --- build JSON payload -----------------------------------------------------
PAYLOAD=$(python3 -c "
import json, sys
d = {
    'user':        sys.argv[1],
    'step_name':   sys.argv[2],
    'script_path': sys.argv[3],
    'args':        sys.argv[4],
    'device_pref': sys.argv[5],
    'priority':    int(sys.argv[6]),
}
slot = sys.argv[7]
if slot:
    d['slot_pref'] = slot
print(json.dumps(d))
" "$SGNNET_USER" "$STEP_NAME" "$SCRIPT_PATH" "$EXTRA_ARGS" "$DEVICE_PREF" "$PRIORITY" "$SLOT_PREF")

# --- submit -----------------------------------------------------------------
RESPONSE=$(curl -sf \
  -X POST \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" \
  "${CONTROLLER_URL}/submit") || {
  echo "ERROR: Could not reach controller at ${CONTROLLER_URL}" >&2
  echo "  Is the controller running?  python3 scripts/experiment_controller.py" >&2
  exit 1
}

# --- print queue id ---------------------------------------------------------
QUEUE_ID=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$RESPONSE" 2>/dev/null || echo "?")
echo "Queued: id=${QUEUE_ID}  user=${SGNNET_USER}  step=${STEP_NAME}  device_pref=${DEVICE_PREF}${SLOT_PREF:+  slot_pref=}${SLOT_PREF}"
