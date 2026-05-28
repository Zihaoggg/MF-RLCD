#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

BASE_OUT="${BASE_OUT:-./opt_run/RL_Diffusion}"
BUDGET_EVALS="${BUDGET_EVALS:-80}"
RUN_DIFFUSION_DIRECT="${RUN_DIFFUSION_DIRECT:-1}"
RUN_CMAES="${RUN_CMAES:-1}"
RUN_RL_ONLY="${RUN_RL_ONLY:-1}"
NO_RESUME="${NO_RESUME:-0}"

print_title "Run default benchmark batch"

cmd=(
  "$PYTHON_BIN" "./run_project.py" "batch"
  --base_out "$BASE_OUT"
  --budget_evals "$BUDGET_EVALS"
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

if [[ "$NO_RESUME" == "1" ]]; then
  cmd+=(--no_resume)
fi

run_cmd "${cmd[@]}"
