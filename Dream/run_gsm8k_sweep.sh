#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/eval_gsm8k.py"
SUMMARY_SCRIPT="${SCRIPT_DIR}/scripts/summarize_gsm8k_sweep.py"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/models/Dream-v0-Base-7B}"
DATASET_PATH="${DATASET_PATH:-}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/gsm8k_sweeps/$(date +%Y%m%d_%H%M%S)}"

GEN_LENGTHS=(256 512 1024)
BLOCK_LENGTHS=(32 64 128)
N_SHOTS=(5)
MODES=(baseline fast_dllm_dual_cache focus_dual_cache)
FOCUS_LAYER="${FOCUS_LAYER:-3}"
FOCUS_TOPK="${FOCUS_TOPK:-8}"

if [[ -n "${GEN_LENGTHS_OVERRIDE:-}" ]]; then
  read -r -a GEN_LENGTHS <<< "${GEN_LENGTHS_OVERRIDE}"
fi
if [[ -n "${BLOCK_LENGTHS_OVERRIDE:-}" ]]; then
  read -r -a BLOCK_LENGTHS <<< "${BLOCK_LENGTHS_OVERRIDE}"
fi
if [[ -n "${N_SHOTS_OVERRIDE:-}" ]]; then
  read -r -a N_SHOTS <<< "${N_SHOTS_OVERRIDE}"
fi
if [[ -n "${MODES_OVERRIDE:-}" ]]; then
  read -r -a MODES <<< "${MODES_OVERRIDE}"
fi

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
  exit 1
fi

run_case() {
  local mode_name="$1"
  local gen_length="$2"
  local steps="$3"
  local n_shot="$4"
  local block_length="$5"

  local run_name="mode${mode_name}_len${gen_length}_steps${steps}_shot${n_shot}_blk${block_length}"
  if [[ "${mode_name}" == "focus_dual_cache" ]]; then
    run_name="${run_name}_layer${FOCUS_LAYER}_topk${FOCUS_TOPK}"
  fi

  local run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"

  local mode_args=()
  if [[ "${mode_name}" == "fast_dllm_prefix_cache" ]]; then
    mode_args=(--use_cache)
  elif [[ "${mode_name}" == "fast_dllm_dual_cache" ]]; then
    mode_args=(--use_cache --dual_cache)
  elif [[ "${mode_name}" == "focus_dual_cache" ]]; then
    mode_args=(--use_cache --dual_cache --focus_decode --focus_layer "${FOCUS_LAYER}" --focus_topk "${FOCUS_TOPK}")
  fi

  local cmd=(
    "${PYTHON_BIN}" -u "${PYTHON_SCRIPT}"
    --model_path "${MODEL_PATH}"
    --split "${SPLIT}"
    --start "${START_INDEX}"
    --n_shot "${n_shot}"
    --max_new_tokens "${gen_length}"
    --steps "${steps}"
    --block_length "${block_length}"
    --output_path "${run_dir}/gsm8k_results.jsonl"
    --stats_path "${run_dir}/gsm8k_stats.json"
    "${mode_args[@]}"
  )

  if [[ -n "${DATASET_PATH}" ]]; then
    cmd+=(--dataset_path "${DATASET_PATH}")
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    cmd+=(--max_samples "${MAX_SAMPLES}")
  fi
  if [[ -n "${END_INDEX}" ]]; then
    cmd+=(--end "${END_INDEX}")
  fi

  echo "Running ${run_name}"
  "${cmd[@]}" 2>&1 | tee "${run_dir}/stdout.log"
}

mkdir -p "${OUTPUT_ROOT}"

for gen_length in "${GEN_LENGTHS[@]}"; do
  steps="${gen_length}"
  for n_shot in "${N_SHOTS[@]}"; do
    for block_length in "${BLOCK_LENGTHS[@]}"; do
      for mode_name in "${MODES[@]}"; do
        run_case "${mode_name}" "${gen_length}" "${steps}" "${n_shot}" "${block_length}"
      done
    done
  done
done

if [[ -f "${SUMMARY_SCRIPT}" ]]; then
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --input_root "${OUTPUT_ROOT}"
fi

echo "Done. Outputs: ${OUTPUT_ROOT}"
