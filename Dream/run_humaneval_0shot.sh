#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/eval_humaneval.py"
FOCUS_DECODE_MODEL_PATH="${FOCUS_DECODE_MODEL_PATH:-${SCRIPT_DIR}/models/Dream-v0-Base-7B-Softmax}"
FASTDLLM_MODEL_PATH="${SCRIPT_DIR}/models/Dream-v0-Base-7B-Fastdllm"
BASELINE_MODEL_PATH="${SCRIPT_DIR}/models/Dream-v0-Base-7B"
DATASET_PATH="data/HumanEval.jsonl.gz"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${SCRIPT_DIR}/humaneval_sweeps/${OUTPUT_RUN_NAME}"

TIMEOUT=5.0
ALGS=(confidence_threshold)
if [[ -n "${ALGS_OVERRIDE:-}" ]]; then
  read -r -a ALGS <<< "${ALGS_OVERRIDE}"
elif [[ -n "${ALG:-}" ]]; then
  ALGS=("${ALG}")
fi
THRESHOLDS=(0.9)
if [[ -n "${THRESHOLDS_OVERRIDE:-}" ]]; then
  read -r -a THRESHOLDS <<< "${THRESHOLDS_OVERRIDE}"
elif [[ -n "${THRESHOLD:-}" ]]; then
  THRESHOLDS=("${THRESHOLD}")
fi
GAMMAS=(0.1)
if [[ -n "${GAMMAS_OVERRIDE:-}" ]]; then
  read -r -a GAMMAS <<< "${GAMMAS_OVERRIDE}"
fi
GEN_LENGTHS=(256 512)
if [[ -n "${GEN_LENGTHS_OVERRIDE:-}" ]]; then
  read -r -a GEN_LENGTHS <<< "${GEN_LENGTHS_OVERRIDE}"
fi
BLOCK_LENGTHS=(32)
if [[ -n "${BLOCK_LENGTHS_OVERRIDE:-}" ]]; then
  read -r -a BLOCK_LENGTHS <<< "${BLOCK_LENGTHS_OVERRIDE}"
fi
#MODES=(fast_dllm_dual_cache focus_dual_cache)
MODES=(focus_dual_cache fast_dllm_dual_cache)
if [[ -n "${MODES_OVERRIDE:-}" ]]; then
  read -r -a MODES <<< "${MODES_OVERRIDE}"
fi
FOCUS_LAYER=3
FOCUS_TOPK=16

run_case() {
  local mode_name="$1"
  local gen_length="$2"
  local steps="$3"
  local block_length="$4"
  local model_path="$5"
  local alg="$6"
  local threshold="$7"
  local gamma="$8"

  local alg_tag="${alg//[^[:alnum:]]/_}"
  local threshold_tag="na"
  if [[ "${alg}" == "confidence_threshold" ]]; then
    threshold_tag="${threshold//./p}"
  fi
  local gamma_tag="na"
  if [[ "${mode_name}" == "focus_dual_cache" && "${alg}" == "confidence_threshold" ]]; then
    gamma_tag="${gamma//./p}"
  fi
  local run_name="mode${mode_name}_alg${alg_tag}_th${threshold_tag}_gamma${gamma_tag}_len${gen_length}_steps${steps}_blk${block_length}"
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

  echo "Running ${run_name} with model ${model_path}"
  "${PYTHON_BIN}" -u "${PYTHON_SCRIPT}" \
    --model_path "${model_path}" \
    --dataset_path "${DATASET_PATH}" \
    --max_new_tokens "${gen_length}" \
    --steps "${steps}" \
    --block_length "${block_length}" \
    --alg "${alg}" \
    --threshold "${threshold}" \
    --gamma "${gamma}" \
    --timeout "${TIMEOUT}" \
    --confirm_run_unsafe_code \
    --output_dir "${run_dir}" \
    --add_bos_token \
    "${mode_args[@]}" \
    2>&1 | tee "${run_dir}/stdout.log"
}

mkdir -p "${OUTPUT_ROOT}"

for gen_length in "${GEN_LENGTHS[@]}"; do
  steps="${gen_length}"
  for block_length in "${BLOCK_LENGTHS[@]}"; do
    for mode_name in "${MODES[@]}"; do
      for alg in "${ALGS[@]}"; do
        if [[ "${mode_name}" == "baseline" && "${alg}" == "confidence_threshold" ]]; then
          continue
        fi

        if [[ "${alg}" == "confidence_threshold" ]]; then
          thresholds_to_run=("${THRESHOLDS[@]}")
        else
          thresholds_to_run=("${THRESHOLDS[0]}")
        fi
        # threshold 扫描 (Fastdllm, Focus)
        for threshold in "${thresholds_to_run[@]}"; do
          if [[ "${mode_name}" == "fast_dllm_dual_cache" ]]; then
            model_path="${FASTDLLM_MODEL_PATH}"
            run_case "${mode_name}" "${gen_length}" "${steps}" "${block_length}" "${model_path}" "${alg}" "${threshold}" "${GAMMAS[0]}"
          elif [[ "${mode_name}" == "baseline" ]]; then
            model_path="${BASELINE_MODEL_PATH}"
            run_case "${mode_name}" "${gen_length}" "${steps}" "${block_length}" "${model_path}" "${alg}" "${threshold}" "${GAMMAS[0]}"
          elif [[ "${mode_name}" == "focus_dual_cache" ]]; then
            model_path="${FOCUS_DECODE_MODEL_PATH}"
            if [[ "${alg}" == "confidence_threshold" ]]; then
              for gamma in "${GAMMAS[@]}"; do
                run_case "${mode_name}" "${gen_length}" "${steps}" "${block_length}" "${model_path}" "${alg}" "${threshold}" "${gamma}"
              done
            else
              run_case "${mode_name}" "${gen_length}" "${steps}" "${block_length}" "${model_path}" "${alg}" "${threshold}" "${GAMMAS[0]}"
            fi
          fi
        done
      done
    done
  done
done

echo "Done. Outputs: ${OUTPUT_ROOT}"
