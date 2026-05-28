#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

OPT_WORK_DIR="${OPT_WORK_DIR:-./opt_run/RL_Diffusion/FULL_T001}"
BENCH_WORK_DIR="${BENCH_WORK_DIR:-./opt_run/BENCH/FULL_T001}"
ANALYSIS_DIR="${ANALYSIS_DIR:-./opt_run/BENCH/analysis_FULL_T001}"
HISTORY_ROOT="${HISTORY_ROOT:-./opt_run/RL_Diffusion}"

E_TARGET="${E_TARGET:-110000}"
SIGMA_Y_TARGET="${SIGMA_Y_TARGET:-910}"
KT_TARGET="${KT_TARGET:-1.20}"

NUM_EPISODES="${NUM_EPISODES:-100}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-10}"
BUDGET_EVALS="${BUDGET_EVALS:-40}"
NUM_TASKS="${NUM_TASKS:-3}"
TASK_MODE="${TASK_MODE:-jitter}"

RUN_DIFFUSION_DIRECT="${RUN_DIFFUSION_DIRECT:-1}"
RUN_CMAES="${RUN_CMAES:-1}"
RUN_RL_ONLY="${RUN_RL_ONLY:-1}"
MFRLCD_ENABLE_UPDATES="${MFRLCD_ENABLE_UPDATES:-0}"

print_title "Step 1/4: optimize"
cmd_opt=(
  "$PYTHON_BIN" "./run_project.py" "optimize"
  --work_dir "$OPT_WORK_DIR"
  --history_root "$HISTORY_ROOT"
  --E_target "$E_TARGET"
  --sigma_y_target "$SIGMA_Y_TARGET"
  --Kt_target "$KT_TARGET"
  --num_episodes "$NUM_EPISODES"
  --steps_per_episode "$STEPS_PER_EPISODE"
)
run_cmd "${cmd_opt[@]}"

print_title "Step 2/4: benchmark"
cmd_bench=(
  "$PYTHON_BIN" "./run_project.py" "benchmark"
  --work_dir "$BENCH_WORK_DIR"
  --E_target "$E_TARGET"
  --sigma_y_target "$SIGMA_Y_TARGET"
  --Kt_target "$KT_TARGET"
  --budget_evals "$BUDGET_EVALS"
  --num_tasks "$NUM_TASKS"
  --task_mode "$TASK_MODE"
)
if [[ "$RUN_DIFFUSION_DIRECT" == "1" ]]; then
  cmd_bench+=(--run_diffusion_direct)
fi
if [[ "$RUN_CMAES" == "1" ]]; then
  cmd_bench+=(--run_cmaes)
fi
if [[ "$RUN_RL_ONLY" == "1" ]]; then
  cmd_bench+=(--run_rl_only)
fi
if [[ "$MFRLCD_ENABLE_UPDATES" == "1" ]]; then
  cmd_bench+=(--mfrlcd_enable_updates)
fi
run_cmd "${cmd_bench[@]}"

print_title "Step 3/4: analyze"
cmd_analyze=(
  "$PYTHON_BIN" "./run_project.py" "analyze"
  --input_root "$BENCH_WORK_DIR"
  --output_dir "$ANALYSIS_DIR"
)
run_cmd "${cmd_analyze[@]}"

print_title "Step 4/4: verify"
cmd_verify=(
  "$PYTHON_BIN" "./run_project.py" "verify"
  --work_dir "$OPT_WORK_DIR"
)
run_cmd "${cmd_verify[@]}"

echo
echo "Full pipeline finished."
echo "Optimization output: $OPT_WORK_DIR"
echo "Benchmark output: $BENCH_WORK_DIR"
echo "Analysis output: $ANALYSIS_DIR"
