#!/bin/bash
set -euo pipefail

########################################
# Paths
########################################

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"
HF_VENV="$WORKSPACE/.venv_hf"
RESULTS_DIR="$PROJECT/results"

MODEL_PATH="$WORKSPACE/hf_models/qwen2_5_7b"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8,16"
MAX_NEW_TOKENS=512

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running HuggingFace benchmark (Qwen 2.5 7B)"
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
# Create HF venv if missing
########################################

if [ ! -d "$HF_VENV" ]; then
    echo "Creating HF virtual environment..."
    python3 -m venv "$HF_VENV"
fi

########################################
# Install dependencies if missing
########################################

if ! "$HF_VENV/bin/python" -c "import transformers" >/dev/null 2>&1; then
    echo "Installing torch + transformers..."
    "$HF_VENV/bin/pip" install --upgrade pip
    "$HF_VENV/bin/pip" install torch transformers
fi

########################################
# Debug info
########################################

echo "Using Python:"
"$HF_VENV/bin/python" -c "import sys; print(sys.executable)"
"$HF_VENV/bin/python" -c "import transformers; print('Transformers:', transformers.__version__)"

########################################
# Run benchmark
########################################

"$HF_VENV/bin/python" \
  "$PROJECT/benchmarks/hf/bm_hf.py" \
  --model "$MODEL_PATH" \
  --prompts "$PROMPTS" \
  --batch_size "$BATCH_SIZES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --out_csv "$RESULTS_DIR/hf_qwen_results.csv" \
  --backend HF

echo ""
echo "================================="
echo "HF benchmark complete."
echo "Results saved to:"
echo "$RESULTS_DIR/hf_qwen_results.csv"
echo "================================="