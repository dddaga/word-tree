#!/usr/bin/env bash
# Mac Studio asyncRewake watcher
# Polls /tmp/mac_studio_status.json every 20 minutes.
# When the file is newer than last check, reads it and exits 2 with a
# systemMessage JSON so Claude Code wakes automatically.

set -euo pipefail

STATUS_FILE="/tmp/mac_studio_status.json"
INTERVAL=1200  # 20 minutes
MTIME_FILE="/tmp/mac_studio_watcher_mtime"

# Read last-seen mtime (0 if not yet set)
last_mtime=0
[[ -f "$MTIME_FILE" ]] && last_mtime=$(cat "$MTIME_FILE" 2>/dev/null || echo 0)

while true; do
  sleep "$INTERVAL"

  # Get current mtime of status file
  current_mtime=0
  if [[ -f "$STATUS_FILE" ]]; then
    current_mtime=$(stat -f %m "$STATUS_FILE" 2>/dev/null || echo 0)
  fi

  if (( current_mtime > last_mtime )); then
    # Update mtime record
    echo "$current_mtime" > "$MTIME_FILE"
    last_mtime=$current_mtime

    # Read status content
    local_ts=""
    local_count=0
    local_ram=0
    local_synced="[]"
    local_launched=""
    if [[ -f "$STATUS_FILE" ]]; then
      local_ts=$(python3 -c "import json,sys; d=json.load(open('$STATUS_FILE')); print(d.get('timestamp',''))" 2>/dev/null || echo "")
      local_count=$(python3 -c "import json,sys; d=json.load(open('$STATUS_FILE')); print(d.get('running_count',0))" 2>/dev/null || echo 0)
      local_ram=$(python3 -c "import json,sys; d=json.load(open('$STATUS_FILE')); print(d.get('ram_gb_free',0))" 2>/dev/null || echo 0)
      local_synced=$(python3 -c "import json,sys; d=json.load(open('$STATUS_FILE')); print(json.dumps(d.get('synced',[])))" 2>/dev/null || echo "[]")
      local_launched=$(python3 -c "import json,sys; d=json.load(open('$STATUS_FILE')); print(d.get('launched',''))" 2>/dev/null || echo "")
    fi

    # Build human-readable summary for the systemMessage
    summary="Mac Studio update: ${local_count} experiments running, ${local_ram}GB RAM free."
    [[ "$local_synced" != "[]" && "$local_synced" != "" ]] && summary="$summary Synced: $local_synced."
    [[ -n "$local_launched" ]] && summary="$summary Launched: $local_launched."
    summary="$summary Check Mac Studio status and take action if needed (sync results, dispatch next experiment if count ≤ 1 AND RAM ≥ 50GB)."

    # Exit 2 with systemMessage JSON to wake Claude Code
    printf '%s' "{\"systemMessage\": \"${summary}\"}"
    exit 2
  fi
done
