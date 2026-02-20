#!/bin/bash
set -e

PROJECT="/workspace/LLM_Engineering"
RESULTS_DIR="$PROJECT/results"

MODEL_PATH="$PROJECT/hf_models/llama3_1_8b"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8"
MAX_NEW_TOKENS=512
DTYPE="bfloat16"

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running vLLM benchmark (Llama-3.1-8B)"
echo "================================="

# Activate vLLM environment
source "$PROJECT/.venv_vllm/bin/activate"

# Ensure required deps (vLLM optional extras)
pip install -q pyairports

python "$PROJECT/benchmarks/vllm/bm_vllm.py" \
  --model "$MODEL_PATH" \
  --prompts "$PROMPTS" \
  --batch_size "$BATCH_SIZES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --out_csv "$RESULTS_DIR/vllm_llama3_1_results.csv" \
  --backend vLLM

deactivate

echo "vLLM benchmark complete."
