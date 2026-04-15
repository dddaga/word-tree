#!/usr/bin/env bash
# queue_list.sh — Pretty-print the current experiment queue.
#
# Usage:
#   queue_list.sh [--user <username>] [--all-statuses]
#
# Options:
#   --user <name>    Filter to a specific user (default: all users)
#   --all-statuses   Show done/failed/cancelled entries too (default: queued+running only)
#
# Environment:
#   SGNNET_CONTROLLER_URL  — default http://mac-mini.local:7433

set -euo pipefail

CONTROLLER_URL="${SGNNET_CONTROLLER_URL:-http://mac-mini.local:7433}"
USER_FILTER=""
ALL_STATUSES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user)         USER_FILTER="$2"; shift 2 ;;
    --all-statuses) ALL_STATUSES=1;   shift ;;
    -h|--help)
      echo "Usage: $0 [--user <name>] [--all-statuses]"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

# Build URL
URL="${CONTROLLER_URL}/queue"
if [[ -n "$USER_FILTER" ]]; then
  URL="${URL}?user=${USER_FILTER}"
fi

# Fetch
RESPONSE=$(curl -sf "$URL") || {
  echo "ERROR: Could not reach controller at ${CONTROLLER_URL}" >&2
  echo "  Is the controller running?  python3 scripts/experiment_controller.py" >&2
  exit 1
}

# Pretty-print via Python
python3 - "$RESPONSE" "$ALL_STATUSES" <<'PYEOF'
import json
import sys

data = json.loads(sys.argv[1])
all_statuses = sys.argv[2] == "1"

ACTIVE = {"queued", "running"}
STATUS_COLOR = {
    "queued":    "\033[33m",   # yellow
    "running":   "\033[32m",   # green
    "done":      "\033[90m",   # gray
    "failed":    "\033[31m",   # red
    "cancelled": "\033[90m",   # gray
}
RESET = "\033[0m"

entries = [e for e in data if all_statuses or e["status"] in ACTIVE]

if not entries:
    if all_statuses:
        print("Queue is empty.")
    else:
        print("No queued or running experiments.  Use --all-statuses to see history.")
    sys.exit(0)

header = f"{'ID':>5}  {'USER':<12} {'STATUS':<11} {'STEP_NAME':<30} {'DEVICE':>6} {'SLOT_PREF':<14} {'SUBMITTED_AT'}"
print(header)
print("-" * len(header))

for e in entries:
    status = e["status"]
    color = STATUS_COLOR.get(status, "")
    slot_pref = e["slot_pref"] or ""
    submitted = (e["submitted_at"] or "")[:19].replace("T", " ")
    step = e["step_name"][:29]
    print(f"{e['id']:>5}  {e['user']:<12} {color}{status:<11}{RESET} {step:<30} {e['device_pref']:>6} {slot_pref:<14} {submitted}")

print()
print(f"Total: {len(entries)} entries shown")
PYEOF
