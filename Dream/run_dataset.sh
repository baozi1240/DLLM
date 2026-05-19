#!/bin/bash

OUTPUT_DIR_NAME=OptBenchmark

echo "================================================"
echo "=== Running Fast-dllm and Focus ==="
echo "================================================"
echo "   Running HumanEval-0shot..."
OUTPUT_RUN_NAME=$OUTPUT_DIR_NAME FOCUS_DECODE_MODEL_PATH=models/Dream-v0-Base-7B-OptSliding/ ./run_humaneval_0shot.sh
echo "   Running GSM8K-5shot..."
OUTPUT_RUN_NAME=$OUTPUT_DIR_NAME FOCUS_DECODE_MODEL_PATH=models/Dream-v0-Base-7B-OptSliding/ ./run_gsm8k_5shot.sh
echo "   Running MATH-4shot..."
OUTPUT_RUN_NAME=$OUTPUT_DIR_NAME FOCUS_DECODE_MODEL_PATH=models/Dream-v0-Base-7B-OptSliding/ ./run_math_4shot.sh
echo "   Running MBPP-3shot..."
OUTPUT_RUN_NAME=$OUTPUT_DIR_NAME FOCUS_DECODE_MODEL_PATH=models/Dream-v0-Base-7B-OptSliding/ ./run_mbpp_3shot.sh

echo "================================================"
echo "=== Running Baseline ==="
echo "================================================"
echo "   Running HumanEval-0shot..."
OUTPUT_RUN_NAME=$OUTPUT_DIR_NAME MODES_OVERRIDE=baseline ALGS_OVERRIDE=entropy ./run_humaneval_0shot.sh
echo "   Running GSM8K-5shot..."
OUTPUT_RUN_NAME=$OUTPUT_DIR_NAME MODES_OVERRIDE=baseline ALGS_OVERRIDE=entropy ./run_gsm8k_5shot.sh
echo "   Running MATH-4shot..."
OUTPUT_RUN_NAME=$OUTPUT_DIR_NAME MODES_OVERRIDE=baseline ALGS_OVERRIDE=entropy ./run_math_4shot.sh
echo "   Running MBPP-3shot..."
OUTPUT_RUN_NAME=$OUTPUT_DIR_NAME MODES_OVERRIDE=baseline ALGS_OVERRIDE=entropy ./run_mbpp_3shot.sh
