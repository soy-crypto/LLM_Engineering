#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"
RESULTS_DIR="$PROJECT/results"

MODEL_PATH="$WORKSPACE/hf_models/qwen2p5_7b"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8,16"
MAX_NEW_TOKENS=512

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running HuggingFace benchmark"
echo "================================="

########################################
# Validate inputs
########################################

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: Prompts file not found at $PROMPTS"
    exit 1
fi

########################################
# Activate HF venv (FIXED PATH)
########################################

source "$WORKSPACE/.venv_hf/bin/activate"

########################################
# Run benchmark
########################################

python "$PROJECT/benchmarks/hf/bm_hf.py" \
  --model "$MODEL_PATH" \
  --prompts "$PROMPTS" \
  --batch_size "$BATCH_SIZES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --out_csv "$RESULTS_DIR/hf_results.csv" \
  --backend HF

deactivate

echo ""
echo "HF benchmark complete."
echo "Results saved to:"
echo "$RESULTS_DIR/hf_results.csv"