#!/usr/bin/env bash
set -euo pipefail

# =========================
# Config
# =========================
PYTHON_SCRIPT="analyse.py"
MODEL_PATH="/DISK1/home/yx_zhao31/LLaDA/weigh/LLaDA-8B-Base"
PROMPTS_FILE="prompts_50.txt"

# 分开输出
GENERATED_TXT="all_generated_texts.txt"
STATS_TXT="all_stats.txt"

# 临时目录
TMP_DIR="batch_run_outputs"
mkdir -p "${TMP_DIR}"

# 参数组合
GEN_LENGTHS=(256 512 1024 2048 4096)
BLOCK_LENGTHS=(32 64)
TOPKS=(4 6 8)
LAYERS_FROM_END=(1 2 3 5 10)

# =========================
# Check inputs
# =========================
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

# 清空旧结果
: > "${GENERATED_TXT}"
: > "${STATS_TXT}"

echo "Batch run started at $(date)" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Python script: ${PYTHON_SCRIPT}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Model path: ${MODEL_PATH}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Prompts file: ${PROMPTS_FILE}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "============================================================" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"

run_id=0
total_runs=$(( ${#GEN_LENGTHS[@]} * ${#BLOCK_LENGTHS[@]} * ${#TOPKS[@]} * ${#LAYERS_FROM_END[@]} * 50 ))

for gen_length in "${GEN_LENGTHS[@]}"; do
  steps="${gen_length}"

  for block_length in "${BLOCK_LENGTHS[@]}"; do
    for topk in "${TOPKS[@]}"; do
      for layer in "${LAYERS_FROM_END[@]}"; do
        for idx in "${!PROMPTS[@]}"; do
          prompt="${PROMPTS[$idx]}"
          prompt_id=$((idx + 1))
          run_id=$((run_id + 1))

          run_name="len${gen_length}_blk${block_length}_topk${topk}_layer${layer}_prompt${prompt_id}"
          output_prefix="${TMP_DIR}/${run_name}"
          stdout_log="${TMP_DIR}/${run_name}.stdout.log"
          summary_json="${output_prefix}_summary.json"

          echo "[${run_id}/${total_runs}] Running ${run_name} ..." | tee -a "${STATS_TXT}"

          CUDA_VISIBLE_DEVICES=0 python "${PYTHON_SCRIPT}" \
            --model_path "${MODEL_PATH}" \
            --prompt_text "${prompt}" \
            --gen_length "${gen_length}" \
            --steps "${steps}" \
            --block_length "${block_length}" \
            --topk "${topk}" \
            --analyze_layer_from_end "${layer}" \
            --output_prefix "${output_prefix}" \
            > "${stdout_log}" 2>&1

          # =========================
          # 1) 生成文本 / 回答 输出到 GENERATED_TXT
          # =========================
          {
            echo "------------------------------------------------------------"
            echo "RUN_NAME: ${run_name}"
            echo "PROMPT_ID: ${prompt_id}"
            echo "PROMPT: ${prompt}"
            echo "GEN_LENGTH: ${gen_length}"
            echo "STEPS: ${steps}"
            echo "BLOCK_LENGTH: ${block_length}"
            echo "TOPK: ${topk}"
            echo "ANALYZE_LAYER_FROM_END: ${layer}"
            echo
            echo "[GENERATED_TEXT]"
            awk '
              /^Generated text:$/ {flag=1; next}
              /^================================================================================$/ && flag==1 {flag=0}
              flag {print}
            ' "${stdout_log}"
            echo
          } >> "${GENERATED_TXT}"

          # =========================
          # 2) 统计数值 输出到 STATS_TXT
          # =========================
          {
            echo "------------------------------------------------------------"
            echo "RUN_NAME: ${run_name}"
            echo "PROMPT_ID: ${prompt_id}"
            echo "PROMPT: ${prompt}"
            echo "GEN_LENGTH: ${gen_length}"
            echo "STEPS: ${steps}"
            echo "BLOCK_LENGTH: ${block_length}"
            echo "TOPK: ${topk}"
            echo "ANALYZE_LAYER_FROM_END: ${layer}"
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

        done
      done
    done
  done
done

echo "============================================================" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Batch run finished at $(date)" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Generated texts saved to: ${GENERATED_TXT}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
echo "Stats saved to: ${STATS_TXT}" | tee -a "${GENERATED_TXT}" "${STATS_TXT}"
