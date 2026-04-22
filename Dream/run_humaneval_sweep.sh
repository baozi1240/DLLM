#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/eval_humaneval.py"
MODEL_PATH="${SCRIPT_DIR}/models/Dream-v0-Base-7B"
#MODEL_PATH="${SCRIPT_DIR}/models/test" # debug
DATASET_PATH="data/HumanEval.jsonl.gz"
OUTPUT_ROOT="${SCRIPT_DIR}/humaneval_sweeps/$(date +%Y%m%d_%H%M%S)"

TIMEOUT=5.0
GEN_LENGTHS=(256 512)
BLOCK_LENGTHS=(32)
if [[ -n "${BLOCK_LENGTHS_OVERRIDE:-}" ]]; then
  read -r -a BLOCK_LENGTHS <<< "${BLOCK_LENGTHS_OVERRIDE}"
fi
MODES=(baseline fast_dllm_dual_cache focus_dual_cache)
if [[ -n "${MODES_OVERRIDE:-}" ]]; then
  read -r -a MODES <<< "${MODES_OVERRIDE}"
fi
FOCUS_LAYER=3
FOCUS_TOPK=8

run_case() {
  local mode_name="$1"
  local gen_length="$2"
  local steps="$3"
  local block_length="$4"

  local run_name="mode${mode_name}_len${gen_length}_steps${steps}_blk${block_length}"
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

  echo "Running ${run_name}"
  "${PYTHON_BIN}" -u "${PYTHON_SCRIPT}" \
    --model_path "${MODEL_PATH}" \
    --dataset_path "${DATASET_PATH}" \
    --max_new_tokens "${gen_length}" \
    --steps "${steps}" \
    --block_length "${block_length}" \
    --timeout "${TIMEOUT}" \
    --confirm_run_unsafe_code \
    --output_dir "${run_dir}" \
    "${mode_args[@]}" \
    2>&1 | tee "${run_dir}/stdout.log"
}

mkdir -p "${OUTPUT_ROOT}"

for gen_length in "${GEN_LENGTHS[@]}"; do
  steps="${gen_length}"
  for block_length in "${BLOCK_LENGTHS[@]}"; do
    for mode_name in "${MODES[@]}"; do
      run_case "${mode_name}" "${gen_length}" "${steps}" "${block_length}"
    done
  done
done

echo "Done. Outputs: ${OUTPUT_ROOT}"
