#!/usr/bin/env bash
# Multi-user-safe training launcher for SGNNET.
#
# No lock files. Slot occupancy is determined from tmux pane + OS process state:
#   RUNNING  — pane alive AND shell has active child processes
#   DONE/FREE — pane dead, shell idle, or no session → slot is available
#
# Session naming: sgn-<user>-<slot>-<step_name>
# This encodes slot identity in the session name so slot_status.sh can find it.
#
# Usage:
#   scripts/launch_slot.sh <slot> <script_path> [extra_args...]
#
# Slots:
#   mini_mps     — Mac Mini, MPS
#   mini_cpu     — Mac Mini, CPU
#   studio_mps   — Mac Studio, MPS
#   studio_cpu   — Mac Studio, CPU
#   5060ti_cuda  — RTX 5060 Ti, CUDA
#
# Exit codes: 0=launched, 1=usage/error, 2=slot occupied, 3=host unreachable

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <slot> <script_path> [extra_args...]"
  echo "Slots: mini_mps mini_cpu studio_mps studio_cpu 5060ti_cuda"
  exit 1
fi

SLOT="$1"
SCRIPT="$2"
shift 2

# Check for --unsafe-cuda-launch flag anywhere in remaining args
UNSAFE_CUDA_LAUNCH=0
NEW_ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--unsafe-cuda-launch" ]]; then
    UNSAFE_CUDA_LAUNCH=1
  else
    NEW_ARGS+=("$arg")
  fi
done
set -- "${NEW_ARGS[@]+"${NEW_ARGS[@]}"}"

USER_PREFIX="${SGNNET_USER:-$(whoami)}"
REPO_LOCAL="/Volumes/T9/IndraAstra/dhiraj/neuro_graph"

# ----- Layer 1 guard: CUDA checklist audit -----------------------------------
# For 5060ti_cuda launches, enforce known-good markers from
# .claude/skills/sgnnet-research/CUDA_CHECKLIST.md.
# Override with --unsafe-cuda-launch.
audit_cuda_script() {
  local script="$1"
  if [[ ! -f "$script" ]]; then
    echo "ERROR: CUDA audit cannot read script: $script" >&2
    return 1
  fi
  local content
  content=$(cat "$script")
  local fails=()

  # Blanket pass: explicit human-review marker. Use when the script delegates
  # transfer/compile to an audited helper (Trainer, etc.) and you've verified
  # the path is CUDA-optimal.
  if grep -q "# CUDA-5060ti-validated" <<< "$content"; then
    return 0
  fi

  # Required markers (grep literal substring anywhere — imports, calls, or comments)
  if ! grep -qE "SGNNET_Resonant_CUDA|SGNNET_AntiHebbian_CUDA" <<< "$content"; then
    fails+=("  - Missing CUDA variant: import SGNNET_Resonant_CUDA or SGNNET_AntiHebbian_CUDA. Evidence: step500 (2.6%→99.6% util, 4x speedup).")
  fi
  if ! grep -q "pin_memory=True" <<< "$content"; then
    fails+=("  - Missing pin_memory=True on DataLoader. Needed for async H→D overlap.")
  fi
  if ! grep -q "non_blocking=True" <<< "$content"; then
    fails+=("  - Missing non_blocking=True on .to(device). Needed to overlap transfer with compute.")
  fi
  # GradScaler absence is OK; if present, must be disabled (step801 evidence)
  if grep -q "GradScaler" <<< "$content" && \
     ! grep -qE "GradScaler\([^)]*enabled=False|use_amp=False" <<< "$content"; then
    fails+=("  - GradScaler present without enabled=False. On Blackwell fp16+GradScaler is 4.4x SLOWER (step801). Disable it.")
  fi

  if [[ ${#fails[@]} -gt 0 ]]; then
    echo "" >&2
    echo "┌── CUDA LAUNCH BLOCKED: $script" >&2
    echo "│   Failing the 5060ti_cuda checklist. See:" >&2
    echo "│     .claude/skills/sgnnet-research/CUDA_CHECKLIST.md" >&2
    echo "│" >&2
    for f in "${fails[@]}"; do
      echo "│ $f" >&2
    done
    echo "│" >&2
    echo "│   To override (last resort): append --unsafe-cuda-launch to the command." >&2
    echo "└──" >&2
    return 1
  fi
  return 0
}

if [[ "$SLOT" == "5060ti_cuda" ]]; then
  if [[ "$UNSAFE_CUDA_LAUNCH" == "1" ]]; then
    echo "WARNING: --unsafe-cuda-launch set; skipping CUDA checklist audit for $SCRIPT" >&2
  else
    audit_cuda_script "$SCRIPT" || exit 1
  fi
fi

case "$SLOT" in
  mini_mps)    DEVICE=mps;  HOST=local;      REMOTE_DIR="$REPO_LOCAL";                            TMUX="tmux";                    PY="d_env/bin/python3" ;;
  mini_cpu)    DEVICE=cpu;  HOST=local;      REMOTE_DIR="$REPO_LOCAL";                            TMUX="tmux";                    PY="d_env/bin/python3" ;;
  studio_mps)  DEVICE=mps;  HOST=mac-studio; REMOTE_DIR="/Users/admin/ml/dhiraj/qwen2_omni/testing"; TMUX="/opt/homebrew/bin/tmux"; PY="d_env/bin/python3" ;;
  studio_cpu)  DEVICE=cpu;  HOST=mac-studio; REMOTE_DIR="/Users/admin/ml/dhiraj/qwen2_omni/testing"; TMUX="/opt/homebrew/bin/tmux"; PY="d_env/bin/python3" ;;
  5060ti_cuda) DEVICE=cuda; HOST=5060ti;     REMOTE_DIR="/home/indra/sgnnet_bench";               TMUX="/usr/bin/tmux";           PY="venv/bin/python3" ;;
  *) echo "ERROR: unknown slot '$SLOT'"; exit 1 ;;
esac

# Session name encodes slot for cross-user visibility
STEP_NAME=$(basename "$SCRIPT" .py)
SESSION="sgn-${USER_PREFIX}-${SLOT}-${STEP_NAME}"

# ----- helpers ---------------------------------------------------------------
run() {
  if [[ "$HOST" == "local" ]]; then TERM=xterm-256color bash -c "$1"
  else ssh -o ConnectTimeout=5 "$HOST" "$1"
  fi
}

# Returns number of active child processes for a given PID (0 = idle)
count_children() {
  local pid="$1"
  run "pgrep -P $pid 2>/dev/null | wc -l | tr -d ' '" 2>/dev/null || echo "0"
}

# ----- check if slot is occupied by any user ----------------------------------
# Find any sgn-*-<slot>-* session on target machine
EXISTING=$(run "$TMUX list-sessions -F '#{session_name}' 2>/dev/null | grep -E '^sgn-[^-]+-${SLOT}-'" 2>/dev/null || true)

if [[ -n "$EXISTING" ]]; then
  while IFS= read -r s; do
    pane_info=$(run "$TMUX list-panes -t '$s' -F '#{pane_dead} #{pane_pid}' 2>/dev/null | head -1" 2>/dev/null || echo "1 0")
    pane_dead=$(awk '{print $1}' <<< "$pane_info")
    pane_pid=$(awk  '{print $2}' <<< "$pane_info")

    if [[ "$pane_dead" == "1" ]]; then
      echo "Cleaning up dead pane session '$s'"
      run "$TMUX kill-session -t '$s' 2>/dev/null" || true
      continue
    fi

    n_children=$(count_children "$pane_pid")
    if [[ "${n_children:-0}" -gt 0 ]]; then
      echo "SLOT OCCUPIED: $SLOT has running session '$s' (script still running)"
      echo "Wait for it to finish or kill: $TMUX kill-session -t $s (on $HOST)"
      exit 2
    else
      echo "Cleaning up finished session '$s' (script done, shell idle)"
      run "$TMUX kill-session -t '$s' 2>/dev/null" || true
    fi
  done <<< "$EXISTING"
fi

# ----- smoke test ------------------------------------------------------------
# (Caller should run --help before launching; this wrapper does not re-check)

# ----- launch ----------------------------------------------------------------
LOG_DIR="$REMOTE_DIR/logs"
# Slot-suffixed log for cross-machine discoverability after sync
LOG_PATH="$LOG_DIR/${STEP_NAME}__${SLOT}.log"
run "mkdir -p $LOG_DIR"

# Export SGN_SLOT so training scripts can suffix their OUT_PATH:
#   slot = os.environ.get("SGN_SLOT", "local")
#   OUT_PATH = ROOT / "results" / f"train_stepXXX_seed{SEED}__{slot}.json"
EXTRA="$*"
CMD="cd $REMOTE_DIR && SGN_SLOT=$SLOT $PY -u $SCRIPT --device $DEVICE $EXTRA 2>&1 | tee $LOG_PATH"

# Write CMD to a temp script to avoid shell-quoting issues when SSH passes
# the command through a remote shell before tmux receives it.
# Direct "tmux new-session ... \"CMD\"" causes the remote zsh to split on && and |.
# Solution: pipe CMD via stdin (no shell metacharacter expansion), then run the file.
LAUNCH_SCRIPT="/tmp/sgnnet_launch_${SESSION}.sh"
if [[ "$HOST" == "local" ]]; then
  echo "$CMD" > "$LAUNCH_SCRIPT" && chmod +x "$LAUNCH_SCRIPT"
  TERM=xterm-256color $TMUX new-session -d -s "$SESSION" bash "$LAUNCH_SCRIPT"
else
  ssh -o ConnectTimeout=5 "$HOST" "cat > $LAUNCH_SCRIPT && chmod +x $LAUNCH_SCRIPT" <<< "$CMD"
  ssh -o ConnectTimeout=5 "$HOST" "$TMUX new-session -d -s $SESSION bash $LAUNCH_SCRIPT"
fi

# Give tmux a moment to spawn the shell
sleep 1

# Verify session is alive and script started (pane has children)
PANE_INFO=$(run "$TMUX list-panes -t $SESSION -F '#{pane_dead} #{pane_pid}' 2>/dev/null | head -1" 2>/dev/null || echo "1 0")
PANE_DEAD=$(awk '{print $1}' <<< "$PANE_INFO")
PANE_PID=$(awk  '{print $2}' <<< "$PANE_INFO")

if [[ "$PANE_DEAD" == "1" || -z "$PANE_PID" ]]; then
  echo "ERROR: session created but pane died immediately. Check $LOG_PATH on $HOST"
  exit 3
fi

echo "LAUNCHED: $SESSION on $SLOT (host=$HOST)"
echo "  Log: $LOG_PATH"
echo "  Monitor: bash scripts/slot_status.sh"
