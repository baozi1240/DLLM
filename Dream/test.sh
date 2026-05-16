#!/bin/bash
set -euo pipefail

model_path="/home/xuefeng/DLLM/DLLM/Dream/models/Dream-v0-Base-7B-Softmax"
fastdllm_model_path="/home/xuefeng/DLLM/DLLM/Dream/models/Dream-v0-Base-7B-Fastdllm"
gen_len=512
blk_len=32
topk=16
# ops="--trace_decode --use_chat_template"
ops="--trace_decode"

# 可通过环境变量覆盖，消除 warmup 影响：
#   WARMUP_ROUNDS=2 MEASURE_ROUNDS=5 bash test.sh
WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"
MEASURE_ROUNDS="${MEASURE_ROUNDS:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

declare -a CASE_NAMES=(
  "fastdllm_reference"
  "focus_gamma_0"
  "focus_gamma_0p05"
  "focus_gamma_0p5"
)

declare -a CASE_ARGS=(
  "--show_time --use_cache --block_length ${blk_len} --alg confidence_threshold --model_path ${fastdllm_model_path} --max_new_tokens ${gen_len} --dual_cache ${ops}"
  "--show_time --use_cache --block_length ${blk_len} --alg confidence_threshold --model_path ${model_path} --max_new_tokens ${gen_len} --dual_cache --focus_decode --focus_topk ${topk} ${ops} --gamma 0.0"
  "--show_time --use_cache --block_length ${blk_len} --alg confidence_threshold --model_path ${model_path} --max_new_tokens ${gen_len} --dual_cache --focus_decode --focus_topk ${topk} ${ops} --gamma 0.05"
  "--show_time --use_cache --block_length ${blk_len} --alg confidence_threshold --model_path ${model_path} --max_new_tokens ${gen_len} --dual_cache --focus_decode --focus_topk ${topk} ${ops} --gamma 0.5"
)

run_case() {
  local case_name="$1"
  local case_args="$2"
  local round_label="$3"

  local output
  output="$(eval "python demo_completion.py ${case_args}" 2>&1)"
  local infer_time
  infer_time="$(printf '%s\n' "${output}" | awk -F'Inference time: |s' '/Inference time:/{print $2}' | tail -n 1)"
  if [[ -z "${infer_time}" ]]; then
    echo "[${case_name}] ${round_label} 未解析到 Inference time，原始输出如下："
    printf '%s\n' "${output}"
    return 1
  fi
  printf '%s' "${infer_time}"
}

echo "Warmup rounds: ${WARMUP_ROUNDS}"
echo "Measure rounds: ${MEASURE_ROUNDS}"
echo
printf "%-22s %10s %10s %10s\n" "case" "avg_s" "min_s" "max_s"
printf "%-22s %10s %10s %10s\n" "----------------------" "--------" "--------" "--------"

for i in "${!CASE_NAMES[@]}"; do
  case_name="${CASE_NAMES[$i]}"
  case_args="${CASE_ARGS[$i]}"

  echo
  echo "==> ${case_name}"
  for ((w = 1; w <= WARMUP_ROUNDS; w++)); do
    t="$(run_case "${case_name}" "${case_args}" "warmup-${w}")"
    echo "  warmup ${w}/${WARMUP_ROUNDS}: ${t}s"
  done

  times=()
  for ((m = 1; m <= MEASURE_ROUNDS; m++)); do
    t="$(run_case "${case_name}" "${case_args}" "measure-${m}")"
    times+=("${t}")
    echo "  measure ${m}/${MEASURE_ROUNDS}: ${t}s"
  done

  stats="$(
    printf '%s\n' "${times[@]}" | awk '
      BEGIN {min=1e30; max=0; sum=0; n=0}
      {
        x=$1+0
        if (x<min) min=x
        if (x>max) max=x
        sum+=x
        n++
      }
      END {
        if (n==0) {
          printf "0 0 0"
        } else {
          printf "%.4f %.4f %.4f", sum/n, min, max
        }
      }
    '
  )"
  avg_s="$(echo "${stats}" | awk '{print $1}')"
  min_s="$(echo "${stats}" | awk '{print $2}')"
  max_s="$(echo "${stats}" | awk '{print $3}')"
  printf "%-22s %10s %10s %10s\n" "${case_name}" "${avg_s}" "${min_s}" "${max_s}"
done
