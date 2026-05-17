#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/eval_gsm8k.py"
SUMMARY_SCRIPT="${SCRIPT_DIR}/scripts/summarize_gsm8k_sweep.py"
BASELINE_MODEL_PATH="${BASELINE_MODEL_PATH:-${MODEL_PATH:-${SCRIPT_DIR}/models/Dream-v0-Base-7B}}"
FASTDLLM_MODEL_PATH="${FASTDLLM_MODEL_PATH:-${SCRIPT_DIR}/models/Dream-v0-Base-7B-Fastdllm}"
FOCUS_DECODE_MODEL_PATH="${FOCUS_DECODE_MODEL_PATH:-${SCRIPT_DIR}/models/Dream-v0-Base-7B-Sliding}"
DATASET_PATH="${DATASET_PATH:-}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"
OUTPUT_RUN_NAME="${OUTPUT_RUN_NAME:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/gsm8k_sweeps/${OUTPUT_RUN_NAME}}"

ALGS=(confidence_threshold)
THRESHOLDS=(0.9)
GAMMAS=(0.1 0.05 0)
GEN_LENGTHS=(256)
BLOCK_LENGTHS=(32)
N_SHOTS=(5)
MODES=(focus_dual_cache fast_dllm_dual_cache)
FOCUS_LAYERS=("${FOCUS_LAYER:-3}")
FOCUS_TOPKS=("${FOCUS_TOPK:-16}")

if [[ -n "${ALGS_OVERRIDE:-}" ]]; then
  read -r -a ALGS <<< "${ALGS_OVERRIDE}"
elif [[ -n "${ALG:-}" ]]; then
  ALGS=("${ALG}")
fi
if [[ -n "${THRESHOLDS_OVERRIDE:-}" ]]; then
  read -r -a THRESHOLDS <<< "${THRESHOLDS_OVERRIDE}"
elif [[ -n "${THRESHOLD:-}" ]]; then
  THRESHOLDS=("${THRESHOLD}")
fi
if [[ -n "${GAMMAS_OVERRIDE:-}" ]]; then
  read -r -a GAMMAS <<< "${GAMMAS_OVERRIDE}"
fi
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
if [[ -n "${FOCUS_LAYERS_OVERRIDE:-}" ]]; then
  read -r -a FOCUS_LAYERS <<< "${FOCUS_LAYERS_OVERRIDE}"
fi
if [[ -n "${FOCUS_TOPKS_OVERRIDE:-}" ]]; then
  read -r -a FOCUS_TOPKS <<< "${FOCUS_TOPKS_OVERRIDE}"
fi
if [[ -n "${FOCUS_LAYERS_OVERRIDE:-}" ]]; then
  read -r -a FOCUS_LAYERS <<< "${FOCUS_LAYERS_OVERRIDE}"
fi
if [[ -n "${FOCUS_TOPKS_OVERRIDE:-}" ]]; then
  read -r -a FOCUS_TOPKS <<< "${FOCUS_TOPKS_OVERRIDE}"
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
  local model_path="$6"
  local alg="$7"
  local threshold="$8"
  local gamma="$9"
  local focus_layer="${10:-}"
  local focus_topk="${11:-}"

  local alg_tag="${alg//[^[:alnum:]]/_}"
  local threshold_tag="na"
  if [[ "${alg}" == "confidence_threshold" ]]; then
    threshold_tag="${threshold//./p}"
  fi
  local gamma_tag="na"
  if [[ "${mode_name}" == "focus_dual_cache" && "${alg}" == "confidence_threshold" ]]; then
    gamma_tag="${gamma//./p}"
  fi
  local run_name="mode${mode_name}_alg${alg_tag}_th${threshold_tag}_gamma${gamma_tag}_len${gen_length}_steps${steps}_shot${n_shot}_blk${block_length}"
  if [[ "${mode_name}" == "focus_dual_cache" ]]; then
    run_name="${run_name}_layer${focus_layer}_topk${focus_topk}"
  fi

  local run_dir="${OUTPUT_ROOT}/${run_name}"
  mkdir -p "${run_dir}"

  local mode_args=()
  if [[ "${mode_name}" == "fast_dllm_prefix_cache" ]]; then
    mode_args=(--use_cache)
  elif [[ "${mode_name}" == "fast_dllm_dual_cache" ]]; then
    mode_args=(--use_cache --dual_cache)
  elif [[ "${mode_name}" == "focus_dual_cache" ]]; then
    mode_args=(--use_cache --dual_cache --focus_decode --focus_layer "${focus_layer}" --focus_topk "${focus_topk}")
  elif [[ "${mode_name}" != "baseline" ]]; then
    echo "ERROR: Unknown mode: ${mode_name}"
    exit 1
  fi

  local cmd=(
    "${PYTHON_BIN}" -u "${PYTHON_SCRIPT}"
    --model_path "${model_path}"
    --split "${SPLIT}"
    --start "${START_INDEX}"
    --n_shot "${n_shot}"
    --max_new_tokens "${gen_length}"
    --steps "${steps}"
    --alg "${alg}"
    --threshold "${threshold}"
    --gamma "${gamma}"
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

  echo "Running ${run_name} with model ${model_path}"
  "${cmd[@]}" 2>&1 | tee "${run_dir}/stdout.log"
}

mkdir -p "${OUTPUT_ROOT}"

for gen_length in "${GEN_LENGTHS[@]}"; do
  steps="${gen_length}"
  for n_shot in "${N_SHOTS[@]}"; do
    for block_length in "${BLOCK_LENGTHS[@]}"; do
      for mode_name in "${MODES[@]}"; do
        for alg in "${ALGS[@]}"; do
          # baseline 不启用 threshold
          if [[ "${mode_name}" == "baseline" && "${alg}" == "confidence_threshold" ]]; then
            continue
          fi

          if [[ "${alg}" == "confidence_threshold" ]]; then
            thresholds_to_run=("${THRESHOLDS[@]}")
          else
            thresholds_to_run=("${THRESHOLDS[0]}")
          fi

          for threshold in "${thresholds_to_run[@]}"; do
            if [[ "${mode_name}" == "fast_dllm_prefix_cache" || "${mode_name}" == "fast_dllm_dual_cache" ]]; then
              model_path="${FASTDLLM_MODEL_PATH}"
              run_case "${mode_name}" "${gen_length}" "${steps}" "${n_shot}" "${block_length}" "${model_path}" "${alg}" "${threshold}" "${GAMMAS[0]}"
            elif [[ "${mode_name}" == "baseline" ]]; then
              model_path="${BASELINE_MODEL_PATH}"
              run_case "${mode_name}" "${gen_length}" "${steps}" "${n_shot}" "${block_length}" "${model_path}" "${alg}" "${threshold}" "${GAMMAS[0]}"
            elif [[ "${mode_name}" == "focus_dual_cache" ]]; then
              model_path="${FOCUS_DECODE_MODEL_PATH}"
              for focus_layer in "${FOCUS_LAYERS[@]}"; do
                for focus_topk in "${FOCUS_TOPKS[@]}"; do
                  if [[ "${alg}" == "confidence_threshold" ]]; then
                    # gamma 扫描只对 focus_decode + confidence_threshold 生效
                    for gamma in "${GAMMAS[@]}"; do
                      run_case "${mode_name}" "${gen_length}" "${steps}" "${n_shot}" "${block_length}" "${model_path}" "${alg}" "${threshold}" "${gamma}" "${focus_layer}" "${focus_topk}"
                    done
                  else
                    run_case "${mode_name}" "${gen_length}" "${steps}" "${n_shot}" "${block_length}" "${model_path}" "${alg}" "${threshold}" "${GAMMAS[0]}" "${focus_layer}" "${focus_topk}"
                  fi
                done
              done
            else
              echo "ERROR: Unknown mode: ${mode_name}"
              exit 1
            fi
          done
        done
      done
    done
  done
done

if [[ -f "${SUMMARY_SCRIPT}" ]]; then
  "${PYTHON_BIN}" "${SUMMARY_SCRIPT}" --input_root "${OUTPUT_ROOT}"
fi

echo "Done. Outputs: ${OUTPUT_ROOT}"
