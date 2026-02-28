#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"
RESULTS_DIR="$PROJECT/results"

ENGINE_DIR="$WORKSPACE/trt_engine/mistral_7b_bf16_b16_s4096"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8,16"
MAX_NEW_TOKENS=512

mkdir -p "$RESULTS_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================="
echo "Running TensorRT-LLM benchmark (Mistral-7B)"
echo "================================="

if [ ! -f "$ENGINE_DIR/config.json" ] || ! ls "$ENGINE_DIR"/*.engine >/dev/null 2>&1; then
    echo "ERROR: Engine incomplete at $ENGINE_DIR"
    exit 1
fi

python3 "$PROJECT/benchmarks/trt/bm_trtllm.py" \
  --engine_dir "$ENGINE_DIR" \
  --model_id "mistralai/Mistral-7B-Instruct-v0.3" \
  --prompts "$PROMPTS" \
  --batch_size "$BATCH_SIZES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --out_csv "$RESULTS_DIR/trt_mistral_7b_results.csv" \
  --backend "TensorRT-LLM"

echo "Done."