#!/bin/bash
set -euo pipefail

########################################
# Paths
########################################

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"

VLLM_VENV="$WORKSPACE/.venv_vllm"

CONFIG_FILE="$PROJECT/scripts/config/models.conf"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

MODEL_ROOT="$WORKSPACE/hf_models"
RESULTS_DIR="$PROJECT/results"

BATCH_SIZES=(1 2 4 8)
MAX_NEW_TOKENS=512
DTYPE="bfloat16"

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running vLLM Scaling Study"
echo "================================="

########################################
# Validate
########################################

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file not found"
    exit 1
fi

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: prompts file not found"
    exit 1
fi

if [ ! -d "$VLLM_VENV" ]; then
    echo "ERROR: vLLM venv not found"
    exit 1
fi

########################################
# Activate environment
########################################

source "$VLLM_VENV/bin/activate"

########################################
# Benchmark function
########################################

run_model () {

    NAME="$1"

    MODEL_PATH="$MODEL_ROOT/$NAME"
    OUTPUT="$RESULTS_DIR/vllm_${NAME}_scaling.csv"

    echo ""
    echo "================================="
    echo "Running vLLM: $NAME"
    echo "================================="

    if [ ! -d "$MODEL_PATH" ]; then
        echo "WARNING: Model not found locally: $MODEL_PATH"
        echo "Skipping..."
        return
    fi

    echo "backend,batch_size,max_new_tokens,latency_ms,tokens_per_sec,gpu_mem_mb" > "$OUTPUT"

    for B in "${BATCH_SIZES[@]}"
    do
        echo "Batch size $B"

        python "$PROJECT/benchmarks/vllm/bm_vllm.py" \
            --model "$MODEL_PATH" \
            --prompts "$PROMPTS" \
            --batch_size "$B" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --dtype "$DTYPE" \
            --append_csv "$OUTPUT" \
            --backend vLLM
    done

    echo "Saved: $OUTPUT"

    # Clean GPU memory between models
    sleep 3
    python - <<EOF
import torch
torch.cuda.empty_cache()
EOF
}

########################################
# Loop through config
########################################

while IFS="|" read -r NAME MODEL_ID
do
    [[ -z "$NAME" ]] && continue
    [[ "$NAME" =~ ^# ]] && continue

    run_model "$NAME"

done < "$CONFIG_FILE"

########################################

deactivate

echo ""
echo "================================="
echo "All vLLM scaling studies complete"
echo "Results saved to:"
echo "$RESULTS_DIR"
echo "================================="