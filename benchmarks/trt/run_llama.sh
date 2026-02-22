#!/bin/bash
set -e

PROJECT="/workspace/LLM_Engineering"
RESULTS_DIR="$PROJECT/results"

ENGINE_DIR="$PROJECT/trt_engine/llama3_1_8b_bf16_b16_s4096"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8,16"
MAX_NEW_TOKENS=512

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running TensorRT-LLM benchmark (Llama-3.1-8B)"
echo "================================="

# Ensure running inside TRT container
if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
    echo "ERROR: Must run inside NVIDIA TensorRT-LLM container."
    exit 1
fi

python "$PROJECT/benchmarks/trt/bm_trtllm.py" \
  --engine_dir "$ENGINE_DIR" \
  --model_id "meta-llama/Llama-3.1-8B" \
  --prompts "$PROMPTS" \
  --batch_size "$BATCH_SIZES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --out_csv "$RESULTS_DIR/trt_llama3_1_results.csv" \
  --backend "TensorRT-LLM"

echo "TensorRT benchmark complete."
