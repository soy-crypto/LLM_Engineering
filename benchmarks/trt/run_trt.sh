#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"
RESULTS_DIR="$PROJECT/results"

ENGINE_DIR="$WORKSPACE/trt_engine/qwen2p5_7b_bf16_b16_s2560"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8,16"
MAX_NEW_TOKENS=512

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running TensorRT-LLM benchmark"
echo "================================="

########################################
# Validate inputs
########################################

if [ ! -d "$ENGINE_DIR" ]; then
    echo "ERROR: Engine directory not found at $ENGINE_DIR"
    exit 1
fi

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: Prompts file not found at $PROMPTS"
    exit 1
fi

########################################
# Ensure running inside TRT container
########################################

if ! command -v trtllm-build >/dev/null; then
    echo "ERROR: Must run inside NVIDIA TensorRT-LLM container."
    exit 1
fi

if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
    echo "ERROR: tensorrt_llm Python module not available."
    exit 1
fi

########################################
# Run benchmark
########################################

python "$PROJECT/benchmarks/trt/bm_trtllm.py" \
  --engine_dir "$ENGINE_DIR" \
  --model_id "Qwen/Qwen2.5-7B-Instruct" \
  --prompts "$PROMPTS" \
  --batch_size "$BATCH_SIZES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --out_csv "$RESULTS_DIR/trt_results.csv" \
  --backend "TensorRT-LLM"

echo ""
echo "TensorRT benchmark complete."
echo "Results saved to:"
echo "$RESULTS_DIR/trt_results.csv"