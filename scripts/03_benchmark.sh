#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

WORK_DIR="${WORK_DIR:-./opt_run/BENCH/T001}"
E_TARGET="${E_TARGET:-110000}"
SIGMA_Y_TARGET="${SIGMA_Y_TARGET:-910}"
KT_TARGET="${KT_TARGET:-1.20}"
BUDGET_EVALS="${BUDGET_EVALS:-80}"
NUM_TASKS="${NUM_TASKS:-10}"
TASK_MODE="${TASK_MODE:-jitter}"
RUN_DIFFUSION_DIRECT="${RUN_DIFFUSION_DIRECT:-1}"
RUN_CMAES="${RUN_CMAES:-1}"
RUN_RL_ONLY="${RUN_RL_ONLY:-1}"
MFRLCD_ENABLE_UPDATES="${MFRLCD_ENABLE_UPDATES:-0}"
NO_RESUME="${NO_RESUME:-0}"

print_title "Run standard benchmark"

cmd=(
  "$PYTHON_BIN" "./run_project.py" "benchmark"
  --work_dir "$WORK_DIR"
  --E_target "$E_TARGET"
  --sigma_y_target "$SIGMA_Y_TARGET"
  --Kt_target "$KT_TARGET"
  --budget_evals "$BUDGET_EVALS"
  --num_tasks "$NUM_TASKS"
  --task_mode "$TASK_MODE"
)

if [[ "$RUN_DIFFUSION_DIRECT" == "1" ]]; then
  cmd+=(--run_diffusion_direct)
fi

if [[ "$RUN_CMAES" == "1" ]]; then
  cmd+=(--run_cmaes)
fi

if [[ "$RUN_RL_ONLY" == "1" ]]; then
  cmd+=(--run_rl_only)
fi

if [[ "$MFRLCD_ENABLE_UPDATES" == "1" ]]; then
  cmd+=(--mfrlcd_enable_updates)
fi

if [[ "$NO_RESUME" == "1" ]]; then
  cmd+=(--no_resume)
fi

run_cmd "${cmd[@]}"
