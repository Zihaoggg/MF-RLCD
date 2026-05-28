#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

WORK_DIR="${WORK_DIR:-./opt_run/RL_Diffusion/T001}"
HISTORY_ROOT="${HISTORY_ROOT:-./opt_run/RL_Diffusion}"
E_TARGET="${E_TARGET:-110000}"
SIGMA_Y_TARGET="${SIGMA_Y_TARGET:-910}"
KT_TARGET="${KT_TARGET:-1.20}"
NUM_EPISODES="${NUM_EPISODES:-300}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-10}"
NO_RESUME="${NO_RESUME:-0}"

print_title "Run single-target optimization"

cmd=(
  "$PYTHON_BIN" "./run_project.py" "optimize"
  --work_dir "$WORK_DIR"
  --E_target "$E_TARGET"
  --sigma_y_target "$SIGMA_Y_TARGET"
  --Kt_target "$KT_TARGET"
  --num_episodes "$NUM_EPISODES"
  --steps_per_episode "$STEPS_PER_EPISODE"
)

if [[ -n "$HISTORY_ROOT" ]]; then
  cmd+=(--history_root "$HISTORY_ROOT")
fi

if [[ "$NO_RESUME" == "1" ]]; then
  cmd+=(--no_resume)
fi

run_cmd "${cmd[@]}"
