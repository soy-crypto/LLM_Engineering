#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"

VLLM_VENV="$WORKSPACE/.venv_vllm"
RESULTS_DIR="$PROJECT/results"

MODEL_PATH="$WORKSPACE/hf_models/llama3_1_8b"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES=(1 2 4 8)
MAX_NEW_TOKENS=512
DTYPE="bfloat16"

OUTPUT="$RESULTS_DIR/vllm_llama3_1_scaling.csv"

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running vLLM Scaling Study (Llama-3.1-8B)"
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
# Ensure venv exists
#######################################

if [ ! -d "$VLLM_VENV" ]; then
    echo "ERROR: vLLM venv not found at $VLLM_VENV"
    exit 1
fi

#######################################
# Activate
#######################################

source "$VLLM_VENV/bin/activate"

echo "backend,batch_size,max_new_tokens,total_latency_ms,tokens_per_sec,gpu_mem_mb" > "$OUTPUT"

for B in "${BATCH_SIZES[@]}"
do
    echo ""
    echo "Running batch size $B..."

    python "$PROJECT/benchmarks/vllm/bm_vllm.py" \
      --model "$MODEL_PATH" \
      --prompts "$PROMPTS" \
      --batch_size "$B" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --dtype "$DTYPE" \
      --append_csv "$OUTPUT" \
      --backend vLLM

done

deactivate

echo ""
echo "Scaling study complete."
echo "Results saved to:"
echo "$OUTPUT"