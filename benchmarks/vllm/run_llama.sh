#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"

VLLM_VENV="$WORKSPACE/.venv_vllm"
RESULTS_DIR="$PROJECT/results"

MODEL_PATH="$WORKSPACE/hf_models/llama3_1_8b"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8"
MAX_NEW_TOKENS=512
DTYPE="bfloat16"

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running vLLM benchmark (Llama-3.1-8B)"
echo "================================="

#######################################
# Validate inputs
#######################################

if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: Prompts file not found at $PROMPTS"
    exit 1
fi

#######################################
# Create vLLM venv (if missing)
#######################################

if [ ! -d "$VLLM_VENV" ]; then
    echo "Creating vLLM virtual environment..."

    python3 -m venv "$VLLM_VENV"

    source "$VLLM_VENV/bin/activate"
    pip install --upgrade pip

    pip install \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    pip install \
        vllm==0.15.1 \
        transformers==4.43.3

    deactivate
else
    echo "vLLM venv already exists."
fi

#######################################
# Activate and run benchmark
#######################################

source "$VLLM_VENV/bin/activate"

python "$PROJECT/benchmarks/vllm/bm_vllm.py" \
  --model "$MODEL_PATH" \
  --prompts "$PROMPTS" \
  --batch_size "$BATCH_SIZES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --dtype "$DTYPE" \
  --out_csv "$RESULTS_DIR/vllm_llama3_1_results.csv" \
  --backend vLLM

deactivate

echo ""
echo "vLLM benchmark complete."
echo "Results saved to:"
echo "$RESULTS_DIR/vllm_llama3_1_results.csv"