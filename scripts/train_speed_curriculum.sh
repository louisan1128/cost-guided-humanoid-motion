#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIMICKIT_ROOT="${1:-/home/an/Desktop/MimicKit-main}"
PYTHON_BIN="${PYTHON_BIN:-/home/an/miniconda3/envs/issaclab/bin/python}"
NUM_ENVS="${NUM_ENVS:-4096}"
ITERATIONS="${ITERATIONS:-500}"
STEPS_PER_ITER=32
MAX_SAMPLES=$((NUM_ENVS * STEPS_PER_ITER * ITERATIONS))
SEED="${SEED:-0}"

BASELINE_MODEL="${BASELINE_MODEL:-${PROJECT_ROOT}/outputs/ppo_g1_baseline_seed0/int_models/model_0000000500.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/ppo_speed_curriculum_seed${SEED}}"
INIT_MODEL="${OUTPUT_ROOT}/baseline_speed_init.pt"
AGENT_CONFIG="${PROJECT_ROOT}/configs/agents/ppo_g1_speed_stage1.yaml"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_speed_expert.py"

if [[ ! -f "${BASELINE_MODEL}" ]]; then
  echo "Baseline checkpoint not found: ${BASELINE_MODEL}" >&2
  exit 1
fi

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Output directory already exists; refusing to mix runs: ${OUTPUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/convert_baseline_checkpoint.py" \
  --input "${BASELINE_MODEL}" \
  --output "${INIT_MODEL}" \
  --command-mean 1.0 \
  --command-std 0.2

run_stage() {
  local stage_name="$1"
  local env_config="$2"
  local input_model="$3"
  local output_dir="${OUTPUT_ROOT}/${stage_name}"

  echo
  echo "Starting ${stage_name}: ${ITERATIONS} iterations, ${NUM_ENVS} environments"
  "${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --mimickit-root "${MIMICKIT_ROOT}" \
    --mode train \
    --num-envs "${NUM_ENVS}" \
    --max-samples "${MAX_SAMPLES}" \
    --device cuda:0 \
    --env-config "${env_config}" \
    --agent-config "${AGENT_CONFIG}" \
    --model-file "${input_model}" \
    --out-dir "${output_dir}" \
    --seed "${SEED}" \
    --logger tb \
    --save-int-models
}

run_stage \
  "stage1_fixed_1p0" \
  "${PROJECT_ROOT}/configs/envs/g1_speed_fixed_1p0.yaml" \
  "${INIT_MODEL}"

run_stage \
  "stage2_range_0p8_1p2" \
  "${PROJECT_ROOT}/configs/envs/g1_speed_range_0p8_1p2.yaml" \
  "${OUTPUT_ROOT}/stage1_fixed_1p0/model.pt"

run_stage \
  "stage3_range_0p6_1p4" \
  "${PROJECT_ROOT}/configs/envs/g1_speed_range_0p6_1p4.yaml" \
  "${OUTPUT_ROOT}/stage2_range_0p8_1p2/model.pt"

echo
echo "Curriculum complete: ${OUTPUT_ROOT}"
