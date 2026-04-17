#!/usr/bin/env bash
# Show current state of all 5 training slots across 3 machines.
#
# No lock files. State is determined entirely from tmux + OS process state:
#   RUNNING  — pane alive AND shell has active child processes (python3 still running)
#   DONE     — pane alive but shell is idle (script finished, safe to reuse)
#   FREE     — no sgn-*-<slot>-* session found
#
# Session naming convention: sgn-<user>-<slot>-<step>
# Sessions without sgn- prefix are teammates' — never touched.
#
# Usage: scripts/slot_status.sh

set -uo pipefail

PREFIX="sgn-"

# remote_cmd <host> <cmd> — runs cmd locally or via ssh
remote_cmd() {
  local host="$1" cmd="$2"
  if [[ "$host" == "local" ]]; then
    TERM=xterm-256color bash -c "$cmd"
  else
    ssh -o ConnectTimeout=4 "$host" "$cmd"
  fi
}

# check_crash <host> <log_dir> <slot>
# Scans the most recent log for slot for Traceback/Error.
# Prints "[CRASHED: <last-error-line>]" if found, else prints nothing.
check_crash() {
  local host="$1" log_dir="$2" slot="$3"
  local last_log crash_line
  last_log=$(remote_cmd "$host" \
    "ls -t ${log_dir}/*__${slot}.log 2>/dev/null | head -1" \
    2>/dev/null || true)
  [[ -z "$last_log" ]] && return
  crash_line=$(remote_cmd "$host" \
    "grep -m1 'Traceback\|RuntimeError\|CUDA error\|AssertionError\|KeyboardInterrupt' '${last_log}' 2>/dev/null | tail -1" \
    2>/dev/null || true)
  [[ -n "$crash_line" ]] && printf "  *** CRASHED: %s ***\n" "$crash_line"
}

# log_dir_for_host <host> — returns the log directory path on that host
log_dir_for_host() {
  case "$1" in
    local)      echo "/Volumes/T9/IndraAstra/dhiraj/neuro_graph/logs" ;;
    mac-studio) echo "/Users/admin/ml/dhiraj/qwen2_omni/testing/logs" ;;
    5060ti)     echo "/home/indra/sgnnet_bench/logs" ;;
    *)          echo "logs" ;;
  esac
}

# check_slot <slot> <host> <tmux_cmd>
check_slot() {
  local slot="$1" host="$2" tmux_cmd="$3"
  printf "%-13s " "$slot"

  # Find sessions matching sgn-<any_user>-<slot>-<step>
  local sessions
  sessions=$(remote_cmd "$host" \
    "$tmux_cmd list-sessions -F '#{session_name}' 2>/dev/null | grep -E '^sgn-[^-]+-${slot}-'" \
    2>/dev/null || true)

  local log_dir
  log_dir=$(log_dir_for_host "$host")

  if [[ -z "$sessions" ]]; then
    printf "[FREE]\n"
    check_crash "$host" "$log_dir" "$slot"
    return
  fi

  local s pane_info pane_dead pane_pid n_children
  while IFS= read -r s; do
    pane_info=$(remote_cmd "$host" \
      "$tmux_cmd list-panes -t '$s' -F '#{pane_dead} #{pane_pid}' 2>/dev/null | head -1" \
      2>/dev/null || echo "1 0")
    pane_dead=$(awk '{print $1}' <<< "$pane_info")
    pane_pid=$(awk  '{print $2}' <<< "$pane_info")

    if [[ "$pane_dead" == "1" ]]; then
      # pane shell exited entirely — treat as FREE
      printf "[FREE]       (dead pane from session=%s)\n" "$s"
      check_crash "$host" "$log_dir" "$slot"
      return
    fi

    # Count active child processes of the pane shell.
    # While the pipeline "python3 ... | tee" runs, bash has ≥2 children.
    # When done, bash has 0 children (idle at prompt).
    n_children=$(remote_cmd "$host" \
      "pgrep -P $pane_pid 2>/dev/null | wc -l | tr -d ' '" \
      2>/dev/null || echo "0")

    if [[ "${n_children:-0}" -gt 0 ]]; then
      printf "[RUNNING]    session=%s\n" "$s"
    else
      printf "[DONE]       session=%s  (script finished, shell idle)\n" "$s"
      check_crash "$host" "$log_dir" "$slot"
    fi
    return
  done <<< "$sessions"

  printf "[FREE]\n"
  check_crash "$host" "$log_dir" "$slot"
}

# list_sessions <host> <tmux_cmd>
list_sessions() {
  local host="$1" tmux_cmd="$2"
  local sessions s pane_dead pane_pid n_children state

  sessions=$(remote_cmd "$host" "$tmux_cmd list-sessions 2>/dev/null" || true)
  if [[ -z "$sessions" ]]; then
    echo "  (no tmux sessions)"
    return
  fi

  local sgn other
  sgn=$(grep  "^${PREFIX}" <<< "$sessions" || true)
  other=$(grep -v "^${PREFIX}" <<< "$sessions" || true)

  if [[ -n "$sgn" ]]; then
    echo "  [SGN training]"
    while IFS= read -r line; do
      s="${line%%:*}"
      pane_info=$(remote_cmd "$host" \
        "$tmux_cmd list-panes -t '$s' -F '#{pane_dead} #{pane_pid}' 2>/dev/null | head -1" \
        2>/dev/null || echo "1 0")
      pane_dead=$(awk '{print $1}' <<< "$pane_info")
      pane_pid=$(awk  '{print $2}' <<< "$pane_info")

      if [[ "$pane_dead" == "1" ]]; then
        state="DEAD"
      else
        n_children=$(remote_cmd "$host" \
          "pgrep -P $pane_pid 2>/dev/null | wc -l | tr -d ' '" \
          2>/dev/null || echo "0")
        if [[ "${n_children:-0}" -gt 0 ]]; then state="RUNNING"; else state="DONE"; fi
      fi
      printf "    [%-7s] %s\n" "$state" "$line"
    done <<< "$sgn"
  fi

  if [[ -n "$other" ]]; then
    echo "  [NON-SGN — teammates' work, DO NOT TOUCH]"
    grep -v "^${PREFIX}" <<< "$sessions" | sed 's/^/    /'
  fi
}

echo "===== SGNNET Slot Status ====="
check_slot "mini_mps"    "local"      "tmux"
check_slot "mini_cpu"    "local"      "tmux"
check_slot "studio_mps"  "mac-studio" "/opt/homebrew/bin/tmux"
check_slot "studio_cpu"  "mac-studio" "/opt/homebrew/bin/tmux"
check_slot "5060ti_cuda" "5060ti"     "/usr/bin/tmux"

echo
echo "===== Tmux sessions by machine ====="
echo "--- Mac Mini (local) ---"
list_sessions "local" "tmux"
echo "--- Mac Studio ---"
list_sessions "mac-studio" "/opt/homebrew/bin/tmux"
echo "--- 5060ti ---"
list_sessions "5060ti" "/usr/bin/tmux"
