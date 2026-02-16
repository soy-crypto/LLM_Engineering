#!/bin/bash
set -e

PROJECT="/workspace/LLM_Engineering"
RESULTS_DIR="$PROJECT/results"

MODEL_PATH="$PROJECT/hf_models/qwen2p5_7b"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8,16"
MAX_NEW_TOKENS=512

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running vLLM benchmark"
echo "================================="

# Activate vLLM venv
source "$PROJECT/.venv_vllm/bin/activate"

python "$PROJECT/benchmarks/vllm/bm_vllm.py" \
  --model "$MODEL_PATH" \
  --prompts "$PROMPTS" \
  --batch_size "$BATCH_SIZES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --out_csv "$RESULTS_DIR/vllm_results.csv" \
  --backend vLLM

deactivate

echo "vLLM benchmark complete."
