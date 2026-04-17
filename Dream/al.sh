#!/usr/bin/env bash
set -euo pipefail

PYTHON_SCRIPT="analyse.py"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="/home/xuefeng/DLLM/DLLM/Dream/models/Dream-v0-Base-7B"
PROMPTS_FILE="../original/prompts_50.txt"

GENERATED_TXT="all_generated_texts.txt"
STATS_TXT="all_stats.txt"
ALL_RECORDS_TXT="all_records_jsonl.txt"

TMP_DIR="batch_run_outputs"
mkdir -p "${TMP_DIR}"

#GEN_LENGTHS=(256 512 1024 2048 4096)
GEN_LENGTHS=(256)
BLOCK_LENGTHS=(32)
TOPKS=(8)
LAYERS_FROM_END=(1 2 3 5 10)
ALGS=(entropy)
CACHE_MODES=(dual_cache use_cache no_cache)

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
  echo "ERROR: Python script not found: ${PYTHON_SCRIPT}"
  exit 1
fi

if [[ ! -f "${PROMPTS_FILE}" ]]; then
  echo "ERROR: Prompts file not found: ${PROMPTS_FILE}"
  exit 1
fi

mapfile -t PROMPTS < "${PROMPTS_FILE}"

if [[ "${#PROMPTS[@]}" -ne 50 ]]; then
  echo "ERROR: ${PROMPTS_FILE} must contain exactly 50 lines, but got ${#PROMPTS[@]}"
  exit 1
fi

: > "${GENERATED_TXT}"
: > "${STATS_TXT}"
: > "${ALL_RECORDS_TXT}"

echo "Batch run started at $(date)" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Python script: ${PYTHON_SCRIPT}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Model path: ${MODEL_PATH}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Prompts file: ${PROMPTS_FILE}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "============================================================" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"

run_id=0
total_runs=$(( ${#GEN_LENGTHS[@]} * ${#BLOCK_LENGTHS[@]} * ${#TOPKS[@]} * ${#LAYERS_FROM_END[@]} * ${#ALGS[@]} * ${#CACHE_MODES[@]} * 50 ))

for gen_length in "${GEN_LENGTHS[@]}"; do
  steps="${gen_length}"

  for block_length in "${BLOCK_LENGTHS[@]}"; do
    for topk in "${TOPKS[@]}"; do
      for layer in "${LAYERS_FROM_END[@]}"; do
        for alg in "${ALGS[@]}"; do
          for idx in "${!PROMPTS[@]}"; do
            for cache_mode in "${CACHE_MODES[@]}"; do
              prompt="${PROMPTS[$idx]}"
              prompt_id=$((idx + 1))
              run_id=$((run_id + 1))

              run_name="mode${cache_mode}_len${gen_length}_blk${block_length}_topk${topk}_layer${layer}_alg${alg}_prompt${prompt_id}"
              output_prefix="${TMP_DIR}/${run_name}"
              stdout_log="${TMP_DIR}/${run_name}.stdout.log"
              summary_json="${output_prefix}_summary.json"
              records_jsonl="${output_prefix}_stats.jsonl"

              echo "[${run_id}/${total_runs}] Running ${run_name} ..." | tee -a "${STATS_TXT}"

              if CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} "${PYTHON_BIN}" "${PYTHON_SCRIPT}" \
                --model_path "${MODEL_PATH}" \
                --prompt_text "${prompt}" \
                --gen_length "${gen_length}" \
                --steps "${steps}" \
                --block_length "${block_length}" \
                --topk "${topk}" \
                --analyze_layer_from_end "${layer}" \
                --alg "${alg}" \
                --cache_mode "${cache_mode}" \
                --output_prefix "${output_prefix}" \
                > "${stdout_log}" 2>&1
              then
                echo "[${run_id}/${total_runs}] Finished ${run_name}" | tee -a "${STATS_TXT}"
              else
                echo "[${run_id}/${total_runs}] FAILED ${run_name}" | tee -a "${STATS_TXT}"
                echo "See log: ${stdout_log}" | tee -a "${STATS_TXT}"
                continue
              fi

              {
                echo "------------------------------------------------------------"
                echo "RUN_NAME: ${run_name}"
                echo "PROMPT_ID: ${prompt_id}"
                echo "PROMPT: ${prompt}"
                echo "CACHE_MODE: ${cache_mode}"
                echo "GEN_LENGTH: ${gen_length}"
                echo "STEPS: ${steps}"
                echo "BLOCK_LENGTH: ${block_length}"
                echo "TOPK: ${topk}"
                echo "ANALYZE_LAYER_FROM_END: ${layer}"
                echo "ALG: ${alg}"
                echo
                echo "[GENERATED_TEXT]"
                awk '
                  /^Generated text:$/ {flag=1; next}
                  /^================================================================================$/ && flag==1 {flag=0}
                  flag {print}
                ' "${stdout_log}"
                echo
              } >> "${GENERATED_TXT}"

              {
                echo "------------------------------------------------------------"
                echo "RUN_NAME: ${run_name}"
                echo "PROMPT_ID: ${prompt_id}"
                echo "PROMPT: ${prompt}"
                echo "CACHE_MODE: ${cache_mode}"
                echo "GEN_LENGTH: ${gen_length}"
                echo "STEPS: ${steps}"
                echo "BLOCK_LENGTH: ${block_length}"
                echo "TOPK: ${topk}"
                echo "ANALYZE_LAYER_FROM_END: ${layer}"
                echo "ALG: ${alg}"
                echo

                if [[ -f "${summary_json}" ]]; then
                  echo "[SUMMARY_JSON]"
                  cat "${summary_json}"
                  echo
                else
                  echo "[WARNING] summary json not found: ${summary_json}"
                  echo
                fi
              } >> "${STATS_TXT}"

              if [[ -f "${records_jsonl}" ]]; then
                {
                  echo "------------------------------------------------------------"
                  echo "RUN_NAME: ${run_name}"
                  echo "CACHE_MODE: ${cache_mode}"
                  cat "${records_jsonl}"
                  echo
                } >> "${ALL_RECORDS_TXT}"
              fi
          done
          done
        done
      done
    done
  done
done

echo "============================================================" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Batch run finished at $(date)" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Generated texts saved to: ${GENERATED_TXT}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Stats saved to: ${STATS_TXT}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "All records saved to: ${ALL_RECORDS_TXT}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
