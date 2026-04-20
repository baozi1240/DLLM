#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/eval_gsm8k.py"
MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/models/Dream-v0-Base-7B}"
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-50}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT_DEFAULT="${SCRIPT_DIR}/gsm8k_sweeps/${timestamp}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"

# Default sweep values. You can override them with CLI args.
USE_CACHE_VALUES=(0 1)
DUAL_CACHE_VALUES=(0 1)
FOCUS_DECODE_VALUES=(0 1)
FOCUS_LAYERS=(1 3 5)
FOCUS_TOPKS=(8)
GEN_LENGTHS=(512 2048 4096)
STEPS_LIST=(512 2048 4096)
N_SHOTS=(5)
BLOCK_LENGTHS=(32 64 128)

usage() {
  cat <<'EOF'
Usage:
  bash run_gsm8k_sweep.sh [options]

Options:
  --output_root PATH
  --model_path PATH
  --split NAME
  --max_samples N
  --start N
  --end N
  --use_cache_values CSV      e.g. 0,1
  --dual_cache_values CSV     e.g. 0,1
  --focus_decode_values CSV   e.g. 0,1
  --focus_layers CSV          e.g. 3,4
  --focus_topks CSV           e.g. 8,16
  --gen_lengths CSV           e.g. 256,512
  --steps_list CSV            e.g. 256,512
  --n_shots CSV               e.g. 0,5
  --block_lengths CSV         e.g. 32,64
  --extra_args "..."

Examples:
  bash run_gsm8k_sweep.sh
  bash run_gsm8k_sweep.sh --max_samples 50 --gen_lengths 512 --steps_list 512 --n_shots 0,8
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
    --split)
      SPLIT="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --start)
      START_INDEX="$2"
      shift 2
      ;;
    --end)
      END_INDEX="$2"
      shift 2
      ;;
    --use_cache_values)
      csv_to_array "$2" USE_CACHE_VALUES
      shift 2
      ;;
    --dual_cache_values)
      csv_to_array "$2" DUAL_CACHE_VALUES
      shift 2
      ;;
    --focus_decode_values)
      csv_to_array "$2" FOCUS_DECODE_VALUES
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
    --gen_lengths)
      csv_to_array "$2" GEN_LENGTHS
      shift 2
      ;;
    --steps_list)
      csv_to_array "$2" STEPS_LIST
      shift 2
      ;;
    --n_shots)
      csv_to_array "$2" N_SHOTS
      shift 2
      ;;
    --block_lengths)
      csv_to_array "$2" BLOCK_LENGTHS
      shift 2
      ;;
    --extra_args)
      EXTRA_ARGS="$2"
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

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"
SUMMARY_LOG="${OUTPUT_ROOT}/summary.log"

{
  echo "Sweep started at $(date)"
  echo "Python script: ${PYTHON_SCRIPT}"
  echo "Model path: ${MODEL_PATH}"
  echo "Output root: ${OUTPUT_ROOT}"
  echo "Split: ${SPLIT}"
  echo "use_cache values: ${USE_CACHE_VALUES[*]}"
  echo "dual_cache values: ${DUAL_CACHE_VALUES[*]}"
  echo "focus_decode values: ${FOCUS_DECODE_VALUES[*]}"
  echo "focus_layers: ${FOCUS_LAYERS[*]}"
  echo "focus_topks: ${FOCUS_TOPKS[*]}"
  echo "gen_lengths: ${GEN_LENGTHS[*]}"
  echo "steps_list: ${STEPS_LIST[*]}"
  echo "n_shots: ${N_SHOTS[*]}"
  echo "block_lengths: ${BLOCK_LENGTHS[*]}"
  echo
} | tee "${SUMMARY_LOG}"

if [[ "${#GEN_LENGTHS[@]}" -ne "${#STEPS_LIST[@]}" ]]; then
  echo "ERROR: gen_lengths and steps_list must have the same number of entries because they are paired by index." | tee -a "${SUMMARY_LOG}"
  exit 1
fi

total_runs=0
for use_cache in "${USE_CACHE_VALUES[@]}"; do
  for dual_cache in "${DUAL_CACHE_VALUES[@]}"; do
    if [[ "${dual_cache}" == "1" && "${use_cache}" != "1" ]]; then
      continue
    fi
    for focus_decode in "${FOCUS_DECODE_VALUES[@]}"; do
      if [[ "${focus_decode}" == "1" && ( "${dual_cache}" != "1" || "${use_cache}" != "1" ) ]]; then
        continue
      fi
      for idx in "${!GEN_LENGTHS[@]}"; do
        gen_length="${GEN_LENGTHS[$idx]}"
        steps="${STEPS_LIST[$idx]}"
        if [[ "${gen_length}" != "${steps}" ]]; then
          echo "ERROR: gen_length and steps must match at index ${idx}; got gen_length=${gen_length}, steps=${steps}" | tee -a "${SUMMARY_LOG}"
          exit 1
        fi
        for n_shot in "${N_SHOTS[@]}"; do
          for block_length in "${BLOCK_LENGTHS[@]}"; do
            if [[ "${focus_decode}" == "1" ]]; then
              for focus_layer in "${FOCUS_LAYERS[@]}"; do
                for focus_topk in "${FOCUS_TOPKS[@]}"; do
                  total_runs=$((total_runs + 1))
                done
              done
            else
              total_runs=$((total_runs + 1))
            fi
          done
        done
      done
    done
  done
done

run_id=0
for use_cache in "${USE_CACHE_VALUES[@]}"; do
  for dual_cache in "${DUAL_CACHE_VALUES[@]}"; do
    if [[ "${dual_cache}" == "1" && "${use_cache}" != "1" ]]; then
      continue
    fi

    for focus_decode in "${FOCUS_DECODE_VALUES[@]}"; do
      if [[ "${focus_decode}" == "1" && ( "${dual_cache}" != "1" || "${use_cache}" != "1" ) ]]; then
        continue
      fi

      for idx in "${!GEN_LENGTHS[@]}"; do
        gen_length="${GEN_LENGTHS[$idx]}"
        steps="${STEPS_LIST[$idx]}"
        if [[ "${gen_length}" != "${steps}" ]]; then
          echo "ERROR: gen_length and steps must match at index ${idx}; got gen_length=${gen_length}, steps=${steps}" | tee -a "${SUMMARY_LOG}"
          exit 1
        fi
        for n_shot in "${N_SHOTS[@]}"; do
          for block_length in "${BLOCK_LENGTHS[@]}"; do
            focus_layers_to_run=(0)
            focus_topks_to_run=(0)
            if [[ "${focus_decode}" == "1" ]]; then
              focus_layers_to_run=("${FOCUS_LAYERS[@]}")
              focus_topks_to_run=("${FOCUS_TOPKS[@]}")
            fi

            for focus_layer in "${focus_layers_to_run[@]}"; do
              for focus_topk in "${focus_topks_to_run[@]}"; do
                run_id=$((run_id + 1))

                  run_name="uc${use_cache}_dc${dual_cache}_fd${focus_decode}_fl${focus_layer}_ftk${focus_topk}_len${gen_length}_steps${steps}_shot${n_shot}_blk${block_length}"
                  run_dir="${OUTPUT_ROOT}/${run_name}"
                  stdout_log="${run_dir}/stdout.log"
                  results_jsonl="${run_dir}/gsm8k_results.jsonl"
                  stats_json="${run_dir}/gsm8k_stats.json"

                  mkdir -p "${run_dir}"

                  cmd=(
                    "${PYTHON_BIN}" -u "${PYTHON_SCRIPT}"
                    --model_path "${MODEL_PATH}"
                    --split "${SPLIT}"
                    --start "${START_INDEX}"
                    --max_new_tokens "${gen_length}"
                    --steps "${steps}"
                    --n_shot "${n_shot}"
                    --block_length "${block_length}"
                    --output_path "${results_jsonl}"
                    --stats_path "${stats_json}"
                  )

                  if [[ -n "${END_INDEX}" ]]; then
                    cmd+=(--end "${END_INDEX}")
                  fi
                  if [[ -n "${MAX_SAMPLES}" ]]; then
                    cmd+=(--max_samples "${MAX_SAMPLES}")
                  fi
                  if [[ "${use_cache}" == "1" ]]; then
                    cmd+=(--use_cache)
                  fi
                  if [[ "${dual_cache}" == "1" ]]; then
                    cmd+=(--dual_cache)
                  fi
                  if [[ "${focus_decode}" == "1" ]]; then
                    cmd+=(--focus_decode --focus_layer "${focus_layer}" --focus_topk "${focus_topk}")
                  fi

                  echo "[${run_id}/${total_runs}] Running ${run_name}" | tee -a "${SUMMARY_LOG}"
                  echo "Command: ${cmd[*]} ${EXTRA_ARGS}" | tee -a "${SUMMARY_LOG}"

                  if [[ -n "${EXTRA_ARGS}" ]]; then
                    if "${cmd[@]}" ${EXTRA_ARGS} 2>&1 | tee "${stdout_log}"; then
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
        done
      done
    done
  done
done

echo "Sweep finished at $(date)" | tee -a "${SUMMARY_LOG}"
echo "All outputs are under: ${OUTPUT_ROOT}" | tee -a "${SUMMARY_LOG}"
