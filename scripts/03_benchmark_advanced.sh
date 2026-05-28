#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

FRAMEWORK_PATH="${FRAMEWORK_PATH:-./rl_diffusion_framework_v7.py}"
WORK_DIR="${WORK_DIR:-./opt_run/BENCH/T001_advanced}"
SEED="${SEED:-42}"
E_TARGET="${E_TARGET:-110000}"
SIGMA_Y_TARGET="${SIGMA_Y_TARGET:-910}"
KT_TARGET="${KT_TARGET:-1.20}"
NUM_TASKS="${NUM_TASKS:-10}"
TASK_MODE="${TASK_MODE:-jitter}"
BUDGET_EVALS="${BUDGET_EVALS:-80}"
RUN_DIFFUSION_DIRECT="${RUN_DIFFUSION_DIRECT:-1}"
RUN_CMAES="${RUN_CMAES:-1}"
RUN_RL_ONLY="${RUN_RL_ONLY:-1}"
NO_RESUME="${NO_RESUME:-0}"

MFRLCD_ENABLE_UPDATES="${MFRLCD_ENABLE_UPDATES:-1}"
MFRLCD_DETERMINISTIC_POLICY="${MFRLCD_DETERMINISTIC_POLICY:-0}"
MFRLCD_PPO_MIN_BATCH="${MFRLCD_PPO_MIN_BATCH:-64}"
MFRLCD_UPDATE_EVERY_STEPS="${MFRLCD_UPDATE_EVERY_STEPS:-1}"
MFRLCD_DIFFUSION_TRAIN_EVERY="${MFRLCD_DIFFUSION_TRAIN_EVERY:-2}"
MFRLCD_DIFFUSION_STEPS="${MFRLCD_DIFFUSION_STEPS:-50}"
MFRLCD_DIFFUSION_TOP_FRAC="${MFRLCD_DIFFUSION_TOP_FRAC:-0.30}"
MFRLCD_COST_PER_EVAL="${MFRLCD_COST_PER_EVAL:-0.08}"
MFRLCD_COST_STEPS_SCALE="${MFRLCD_COST_STEPS_SCALE:-0.04}"
MFRLCD_LOCAL_REFINE_ROUNDS="${MFRLCD_LOCAL_REFINE_ROUNDS:-2}"
MFRLCD_LOCAL_REFINE_SCALE="${MFRLCD_LOCAL_REFINE_SCALE:-0.08}"
MFRLCD_LOCAL_REFINE_DECAY="${MFRLCD_LOCAL_REFINE_DECAY:-0.65}"

print_title "Run advanced benchmark"

cmd=(
  "$PYTHON_BIN" "-m" "benchmark.main"
  --framework_path "$FRAMEWORK_PATH"
  --work_dir "$WORK_DIR"
  --seed "$SEED"
  --E_target "$E_TARGET"
  --sigma_y_target "$SIGMA_Y_TARGET"
  --Kt_target "$KT_TARGET"
  --num_tasks "$NUM_TASKS"
  --task_mode "$TASK_MODE"
  --budget_evals "$BUDGET_EVALS"
  --run_mfrlcd
  --mfrlcd_ppo_min_batch "$MFRLCD_PPO_MIN_BATCH"
  --mfrlcd_update_every_steps "$MFRLCD_UPDATE_EVERY_STEPS"
  --mfrlcd_diffusion_train_every "$MFRLCD_DIFFUSION_TRAIN_EVERY"
  --mfrlcd_diffusion_steps "$MFRLCD_DIFFUSION_STEPS"
  --mfrlcd_diffusion_top_frac "$MFRLCD_DIFFUSION_TOP_FRAC"
  --mfrlcd_cost_per_eval "$MFRLCD_COST_PER_EVAL"
  --mfrlcd_cost_steps_scale "$MFRLCD_COST_STEPS_SCALE"
  --mfrlcd_local_refine_rounds "$MFRLCD_LOCAL_REFINE_ROUNDS"
  --mfrlcd_local_refine_scale "$MFRLCD_LOCAL_REFINE_SCALE"
  --mfrlcd_local_refine_decay "$MFRLCD_LOCAL_REFINE_DECAY"
  --save_json
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

if [[ "$MFRLCD_ENABLE_UPDATES" == "1" ]]; then
  cmd+=(--mfrlcd_enable_updates)
fi

if [[ "$MFRLCD_DETERMINISTIC_POLICY" == "1" ]]; then
  cmd+=(--mfrlcd_deterministic_policy)
fi

run_cmd "${cmd[@]}"
