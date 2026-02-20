#!/bin/bash
set -e

PROJECT="/workspace/LLM_Engineering"
VLLM_VENV="$PROJECT/.venv_vllm"
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

#######################################
# Create vLLM venv (Python 3.10)
#######################################
if [ ! -d "$VLLM_VENV" ]; then
    echo "Creating vLLM virtual environment (Python 3.10)..."

    python3 -m venv "$VLLM_VENV"

    source "$VLLM_VENV/bin/activate"
    pip install --upgrade pip

    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    pip install vllm[all]==0.5.5
    pip install transformers==4.43.3

    deactivate
else
    echo "vLLM venv already exists."
fi

# Activate vLLM environment
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

echo "vLLM benchmark complete."
