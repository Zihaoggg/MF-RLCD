#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/common.sh"

TARGET_FILE="${TARGET_FILE:-./configs/mfrlcd_paper_targets.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./opt_run/MFRLCD_PAPER_15}"
FRAMEWORK_PATH="${FRAMEWORK_PATH:-./rl_diffusion_framework_v7.py}"
BASE_SEED="${BASE_SEED:-59}"
BUDGET_EVALS="${BUDGET_EVALS:-200}"
NUM_TASKS="${NUM_TASKS:-1}"

MFRLCD_ENABLE_UPDATES="${MFRLCD_ENABLE_UPDATES:-1}"
MFRLCD_PPO_MIN_BATCH="${MFRLCD_PPO_MIN_BATCH:-8}"
MFRLCD_UPDATE_EVERY_STEPS="${MFRLCD_UPDATE_EVERY_STEPS:-1}"
MFRLCD_DIFFUSION_TRAIN_EVERY="${MFRLCD_DIFFUSION_TRAIN_EVERY:-5}"
MFRLCD_DIFFUSION_STEPS="${MFRLCD_DIFFUSION_STEPS:-50}"
MFRLCD_DIFFUSION_TOP_FRAC="${MFRLCD_DIFFUSION_TOP_FRAC:-0.30}"
MFRLCD_COST_PER_EVAL="${MFRLCD_COST_PER_EVAL:-0.08}"
MFRLCD_COST_STEPS_SCALE="${MFRLCD_COST_STEPS_SCALE:-0.04}"
MFRLCD_LOCAL_REFINE_ROUNDS="${MFRLCD_LOCAL_REFINE_ROUNDS:-1}"
MFRLCD_LOCAL_REFINE_SCALE="${MFRLCD_LOCAL_REFINE_SCALE:-0.08}"
MFRLCD_LOCAL_REFINE_DECAY="${MFRLCD_LOCAL_REFINE_DECAY:-0.65}"
MFRLCD_DETERMINISTIC_POLICY="${MFRLCD_DETERMINISTIC_POLICY:-0}"
NO_RESUME="${NO_RESUME:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

RUNTIME_MANIFEST="${OUTPUT_ROOT}/paper_target_runs.csv"
ANALYSIS_DIR="${OUTPUT_ROOT}/analysis"

print_title "Prepare paper target list"

if [[ -f "$PROJECT_ROOT/${TARGET_FILE#./}" ]]; then
  TARGET_FILE_ABS="$PROJECT_ROOT/${TARGET_FILE#./}"
elif [[ -f "$TARGET_FILE" ]]; then
  TARGET_FILE_ABS="$TARGET_FILE"
else
  echo "[Error] Target list not found: $TARGET_FILE"
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
printf 'difficulty,target_id,target_name,E_target,sigma_y_target,Kt_target,seed,budget_evals,run_dir\n' > "$RUNTIME_MANIFEST"

target_index=0
while IFS=, read -r difficulty target_id target_name E_TARGET SIGMA_Y_TARGET KT_TARGET; do
  if [[ "$difficulty" == "difficulty" ]]; then
    continue
  fi

  target_index=$((target_index + 1))
  seed=$((BASE_SEED + target_index - 1))
  run_dir="${OUTPUT_ROOT}/${difficulty}/${target_name}"

  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$difficulty" "$target_id" "$target_name" "$E_TARGET" "$SIGMA_Y_TARGET" "$KT_TARGET" "$seed" "$BUDGET_EVALS" "$run_dir" \
    >> "$RUNTIME_MANIFEST"

  if [[ "$SKIP_EXISTING" == "1" && -f "$run_dir/benchmark_results.json" ]]; then
    echo "[Skip] $target_name already has benchmark_results.json"
    continue
  fi

  print_title "Run ${target_name} (${difficulty})"

  cmd=(
    "$PYTHON_BIN" "-m" "benchmark.main"
    --framework_path "$FRAMEWORK_PATH"
    --work_dir "$run_dir"
    --seed "$seed"
    --E_target "$E_TARGET"
    --sigma_y_target "$SIGMA_Y_TARGET"
    --Kt_target "$KT_TARGET"
    --budget_evals "$BUDGET_EVALS"
    --num_tasks "$NUM_TASKS"
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

  if [[ "$MFRLCD_ENABLE_UPDATES" == "1" ]]; then
    cmd+=(--mfrlcd_enable_updates)
  fi

  if [[ "$MFRLCD_DETERMINISTIC_POLICY" == "1" ]]; then
    cmd+=(--mfrlcd_deterministic_policy)
  fi

  if [[ "$NO_RESUME" == "1" ]]; then
    cmd+=(--no_resume)
  fi

  run_cmd "${cmd[@]}"
done < "$TARGET_FILE_ABS"

print_title "Aggregate paper results"

analysis_cmd=(
  "$PYTHON_BIN" "-m" "benchmark.paper_analysis"
  --input_root "$OUTPUT_ROOT"
  --manifest "$RUNTIME_MANIFEST"
  --output_dir "$ANALYSIS_DIR"
)
run_cmd "${analysis_cmd[@]}"

echo
echo "Paper batch finished."
echo "Run manifest: $RUNTIME_MANIFEST"
echo "Target summary: $ANALYSIS_DIR/paper_target_summary.csv"
echo "Difficulty summary: $ANALYSIS_DIR/paper_difficulty_summary.csv"
echo "Evaluations: $ANALYSIS_DIR/paper_evaluations.csv"
echo "Diagnostics: $ANALYSIS_DIR/paper_diagnostics.csv"
echo "Updates: $ANALYSIS_DIR/paper_updates.csv"
