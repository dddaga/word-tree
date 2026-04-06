#!/usr/bin/env bash
# Mac Studio experiment monitor
# Usage: check_mac_studio.sh          — run once and exit
#        check_mac_studio.sh --loop   — loop every 30 minutes

set -euo pipefail

REPO="/Volumes/T9/IndraAstra/dhiraj/neuro_graph"
REMOTE_DIR="/Users/admin/ml/dhiraj/qwen2_omni/testing"
STATUS_FILE="/tmp/mac_studio_status.json"
LOG_FILE="/tmp/mac_studio_monitor.log"
INTERVAL=1800  # 30 minutes

# Ordered dispatch queue (launch in this order when a slot opens)
QUEUE=(
  "step56_nscale:scripts/train_step56_n_scaling.py:step56_nscale"
  "exp3_proxwave:scripts/train_exp3_proxwave.py:exp3_pw:src/sgnnet/model_proximity_wave.py"
  "exp4_reflect:scripts/train_exp4_reflection.py:exp4_reflect:src/sgnnet/model_reflection.py"
  "step55_spatial:scripts/train_step55_spatial_grouped_input.py:step55:src/sgnnet/model_spatial_grouped.py"
  "pca_sweep:scripts/pca_input_sweep.py:pca_sweep"
  "step32_gen4:scripts/train_step32_gen4_compound.py:step32"
  "step53_lowrank:scripts/train_step53_lowrank_mixing.py:step53"
)

log() {
  local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$msg" | tee -a "$LOG_FILE"
}

check_once() {
  local changes=0

  # ── 1. Count running experiments ─────────────────────────────────────────
  local count
  count=$(ssh mac-studio "ps aux | grep -E 'train_step|train_exp' | grep -v grep | grep -v tee | grep -v 'zsh -c' | wc -l" 2>/dev/null | tr -d ' ' || echo "99")

  # ── 2. Check RAM ─────────────────────────────────────────────────────────
  local ram_gb=0
  local vm_out
  vm_out=$(ssh mac-studio "vm_stat" 2>/dev/null || echo "")
  if [[ -n "$vm_out" ]]; then
    local free inactive
    free=$(echo "$vm_out"   | awk '/Pages free:/     {print $3+0}')
    inactive=$(echo "$vm_out" | awk '/Pages inactive:/ {print $3+0}')
    ram_gb=$(( (free + inactive) * 16384 / 1073741824 ))
  fi

  # ── 3. Sync new result JSONs ──────────────────────────────────────────────
  local synced=()
  local remote_results
  remote_results=$(ssh mac-studio "ls -t ${REMOTE_DIR}/results/train_step*.json ${REMOTE_DIR}/results/exp*.json 2>/dev/null" || echo "")

  while IFS= read -r remote_path; do
    [[ -z "$remote_path" ]] && continue
    local fname
    fname=$(basename "$remote_path")
    local local_path="${REPO}/results/${fname}"

    # Check if local copy is missing or older
    local remote_mtime local_mtime
    remote_mtime=$(ssh mac-studio "stat -f %m '${remote_path}' 2>/dev/null" || echo "0")
    local_mtime=0
    [[ -f "$local_path" ]] && local_mtime=$(stat -f %m "$local_path" 2>/dev/null || echo "0")

    if (( remote_mtime > local_mtime )); then
      rsync -q "mac-studio:${remote_path}" "$local_path" 2>/dev/null && {
        synced+=("$fname")
        changes=1
      }
    fi
  done <<< "$remote_results"

  # ── 4. Launch next experiment if slot available ───────────────────────────
  local launched=""
  if (( count <= 1 && ram_gb >= 50 )); then
    for entry in "${QUEUE[@]}"; do
      IFS=':' read -r name script session extra_file <<< "${entry}:::"
      # Check if already running (tmux session exists)
      if ssh mac-studio "tmux has-session -t '${session}' 2>/dev/null"; then
        continue  # already running
      fi
      # Check if result already exists locally
      local result_name="${name//_/-}"
      # Skip if we already have a result JSON for this experiment
      if ls "${REPO}/results/"*"${name}"*.json 2>/dev/null | grep -q .; then
        continue
      fi

      # Rsync script (and optional model file)
      rsync -q "${REPO}/${script}" "mac-studio:${REMOTE_DIR}/${script}" 2>/dev/null || true
      if [[ -n "${extra_file:-}" ]]; then
        rsync -q "${REPO}/${extra_file}" "mac-studio:${REMOTE_DIR}/${extra_file}" 2>/dev/null || true
      fi

      # Launch in tmux
      local script_base
      script_base=$(basename "$script")
      local log_name="${script_base%.py}.log"
      ssh mac-studio "cd ${REMOTE_DIR} && tmux new-session -d -s '${session}' \
        'd_env/bin/python3 -u ${script} --device mps 2>&1 | tee logs/${log_name}'" 2>/dev/null && {
        launched="$script_base"
        changes=1
        break  # only launch one at a time
      }
    done
  fi

  # ── 5. Write status JSON only when something changed ─────────────────────
  if (( changes )); then
    local ts
    ts=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
    cat > "$STATUS_FILE" <<JSON
{
  "timestamp": "${ts}",
  "running_count": ${count},
  "ram_gb_free": ${ram_gb},
  "synced": $(printf '%s\n' "${synced[@]+"${synced[@]}"}" | python3 -c "import sys,json; print(json.dumps(sys.stdin.read().split()))" 2>/dev/null || echo "[]"),
  "launched": "${launched}"
}
JSON
    [[ ${#synced[@]} -gt 0 ]] && log "SYNCED: ${synced[*]}"
    [[ -n "$launched" ]]      && log "LAUNCHED: $launched (slot opened: count=${count}, RAM=${ram_gb}GB)"
  fi

  return 0
}

# ── Main ────────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--loop" ]]; then
  log "Mac Studio monitor started (interval=${INTERVAL}s)"
  while true; do
    check_once || log "ERROR: check_once failed"
    sleep "$INTERVAL"
  done
else
  check_once
fi
