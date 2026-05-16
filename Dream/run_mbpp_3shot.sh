#!/usr/bin/env bash
set -euo pipefail

export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export HF_DATASETS_TRUST_REMOTE_CODE="${HF_DATASETS_TRUST_REMOTE_CODE:-true}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT_DEFAULT="${SCRIPT_DIR}/../../Fast-dLLM/dream/eval.py"
EVAL_SCRIPT="${EVAL_SCRIPT:-$EVAL_SCRIPT_DEFAULT}"
LAUNCHER="${LAUNCHER:-accelerate launch}"

TASK="${TASK:-mbpp}"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/models/Dream-v0-Base-7B}"
NUM_FEWSHOT="${NUM_FEWSHOT:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
STEPS="${STEPS:-256}"
BLOCK_LENGTHS=(32)
FOCUS_TOPKS=(4 6 8)
FOCUS_LAYERS=(0 1 3 5)
EXTRA_MODEL_ARGS="${EXTRA_MODEL_ARGS:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT_DEFAULT="${SCRIPT_DIR}/mbpp_sweeps/${timestamp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"

usage() {
  cat <<'EOF'
Usage:
  bash run_mbpp_3shot.sh [options]

Options:
  --output_root PATH
  --model_path PATH
  --task NAME
  --num_fewshot N
  --batch_size N
  --block_lengths CSV     e.g. 32,64
  --focus_layers CSV      e.g. 0,1,3,5
  --focus_topks CSV       e.g. 4,6,8
  --extra_model_args "..."
  --extra_eval_args "..."
EOF
}

csv_to_array() {
  local csv="$1"
  local -n out_arr=$2
  IFS=',' read -r -a out_arr <<< "$csv"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --num_fewshot)
      NUM_FEWSHOT="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --block_lengths)
      csv_to_array "$2" BLOCK_LENGTHS
      shift 2
      ;;
    --focus_layers)
      csv_to_array "$2" FOCUS_LAYERS
      shift 2
      ;;
    --focus_topks)
      csv_to_array "$2" FOCUS_TOPKS
      shift 2
      ;;
    --extra_model_args)
      EXTRA_MODEL_ARGS="$2"
      shift 2
      ;;
    --extra_eval_args)
      EXTRA_EVAL_ARGS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "ERROR: eval script not found: ${EVAL_SCRIPT}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
SUMMARY_LOG="${OUTPUT_ROOT}/summary.log"

{
  echo "Sweep started at $(date)"
  echo "Eval script: ${EVAL_SCRIPT}"
  echo "Task: ${TASK}"
  echo "Model path: ${MODEL_PATH}"
  echo "Output root: ${OUTPUT_ROOT}"
  echo "num_fewshot: ${NUM_FEWSHOT}"
  echo "batch_size: ${BATCH_SIZE}"
  echo "max_new_tokens: ${MAX_NEW_TOKENS}"
  echo "steps: ${STEPS}"
  echo "block_lengths: ${BLOCK_LENGTHS[*]}"
  echo "focus_layers: ${FOCUS_LAYERS[*]}"
  echo "focus_topks: ${FOCUS_TOPKS[*]}"
  echo
} | tee "${SUMMARY_LOG}"

total_runs=0
for block_length in "${BLOCK_LENGTHS[@]}"; do
  total_runs=$((total_runs + 1))
  total_runs=$((total_runs + ${#FOCUS_LAYERS[@]} * ${#FOCUS_TOPKS[@]}))
done

run_id=0
for block_length in "${BLOCK_LENGTHS[@]}"; do
  for mode_name in fast_dllm_dual_cache; do
    run_id=$((run_id + 1))
    run_name="mode${mode_name}_layer0_topk0_len${MAX_NEW_TOKENS}_steps${STEPS}_shot${NUM_FEWSHOT}_blk${block_length}"
    run_dir="${OUTPUT_ROOT}/${run_name}"
    stdout_log="${run_dir}/stdout.log"
    output_path="${run_dir}/${TASK}"

    mkdir -p "${run_dir}"

    model_args="pretrained=${MODEL_PATH},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${STEPS},add_bos_token=true,alg=entropy,escape_until=true,block_length=${block_length},use_cache=true,dual_cache=true"
    if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
      model_args="${model_args},${EXTRA_MODEL_ARGS}"
    fi

    cmd=(
      ${LAUNCHER}
      "${EVAL_SCRIPT}"
      --model dream
      --model_args "${model_args}"
      --tasks "${TASK}"
      --num_fewshot "${NUM_FEWSHOT}"
      --batch_size "${BATCH_SIZE}"
      --output_path "${output_path}"
      --log_samples
      --confirm_run_unsafe_code
    )

    echo "[${run_id}/${total_runs}] Running ${run_name}" | tee -a "${SUMMARY_LOG}"
    echo "Command: ${cmd[*]} ${EXTRA_EVAL_ARGS}" | tee -a "${SUMMARY_LOG}"

    if [[ -n "${EXTRA_EVAL_ARGS}" ]]; then
      # shellcheck disable=SC2086
      if ${cmd[@]} ${EXTRA_EVAL_ARGS} 2>&1 | tee "${stdout_log}"; then
        echo "[${run_id}/${total_runs}] Finished ${run_name}" | tee -a "${SUMMARY_LOG}"
      else
        echo "[${run_id}/${total_runs}] FAILED ${run_name}" | tee -a "${SUMMARY_LOG}"
        echo "See log: ${stdout_log}" | tee -a "${SUMMARY_LOG}"
      fi
    else
      if "${cmd[@]}" 2>&1 | tee "${stdout_log}"; then
        echo "[${run_id}/${total_runs}] Finished ${run_name}" | tee -a "${SUMMARY_LOG}"
      else
        echo "[${run_id}/${total_runs}] FAILED ${run_name}" | tee -a "${SUMMARY_LOG}"
        echo "See log: ${stdout_log}" | tee -a "${SUMMARY_LOG}"
      fi
    fi

    echo | tee -a "${SUMMARY_LOG}"
  done

  for focus_layer in "${FOCUS_LAYERS[@]}"; do
    for focus_topk in "${FOCUS_TOPKS[@]}"; do
      run_id=$((run_id + 1))
      run_name="modefocus_dual_cache_layer${focus_layer}_topk${focus_topk}_len${MAX_NEW_TOKENS}_steps${STEPS}_shot${NUM_FEWSHOT}_blk${block_length}"
      run_dir="${OUTPUT_ROOT}/${run_name}"
      stdout_log="${run_dir}/stdout.log"
      output_path="${run_dir}/${TASK}"

      mkdir -p "${run_dir}"

      model_args="pretrained=${MODEL_PATH},max_new_tokens=${MAX_NEW_TOKENS},diffusion_steps=${STEPS},add_bos_token=true,alg=entropy,escape_until=true,block_length=${block_length},use_cache=true,dual_cache=true,focus_decode=true,focus_layer=${focus_layer},focus_topk=${focus_topk}"
      if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
        model_args="${model_args},${EXTRA_MODEL_ARGS}"
      fi

      cmd=(
        ${LAUNCHER}
        "${EVAL_SCRIPT}"
        --model dream
        --model_args "${model_args}"
        --tasks "${TASK}"
        --num_fewshot "${NUM_FEWSHOT}"
        --batch_size "${BATCH_SIZE}"
        --output_path "${output_path}"
        --log_samples
        --confirm_run_unsafe_code
      )

      echo "[${run_id}/${total_runs}] Running ${run_name}" | tee -a "${SUMMARY_LOG}"
      echo "Command: ${cmd[*]} ${EXTRA_EVAL_ARGS}" | tee -a "${SUMMARY_LOG}"

      if [[ -n "${EXTRA_EVAL_ARGS}" ]]; then
        # shellcheck disable=SC2086
        if ${cmd[@]} ${EXTRA_EVAL_ARGS} 2>&1 | tee "${stdout_log}"; then
          echo "[${run_id}/${total_runs}] Finished ${run_name}" | tee -a "${SUMMARY_LOG}"
        else
          echo "[${run_id}/${total_runs}] FAILED ${run_name}" | tee -a "${SUMMARY_LOG}"
          echo "See log: ${stdout_log}" | tee -a "${SUMMARY_LOG}"
        fi
      else
        if "${cmd[@]}" 2>&1 | tee "${stdout_log}"; then
          echo "[${run_id}/${total_runs}] Finished ${run_name}" | tee -a "${SUMMARY_LOG}"
        else
          echo "[${run_id}/${total_runs}] FAILED ${run_name}" | tee -a "${SUMMARY_LOG}"
          echo "See log: ${stdout_log}" | tee -a "${SUMMARY_LOG}"
        fi
      fi

      echo | tee -a "${SUMMARY_LOG}"
    done
  done
done

echo "Sweep finished at $(date)" | tee -a "${SUMMARY_LOG}"
echo "All outputs are under: ${OUTPUT_ROOT}" | tee -a "${SUMMARY_LOG}"
