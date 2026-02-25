#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"

VLLM_VENV="$WORKSPACE/.venv_vllm"
RESULTS_DIR="$PROJECT/results"
CONFIG_FILE="$PROJECT/scripts/config/models.conf"

PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES=(1 2 4 8)
MAX_NEW_TOKENS=512
DTYPE="bfloat16"

MODEL_ROOT="$WORKSPACE/hf_models"

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running vLLM Scaling Study (CONFIG MODE)"
echo "================================="

#######################################
# Validate prompts
#######################################

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: Prompts file not found at $PROMPTS"
    exit 1
fi

#######################################
# Validate config file
#######################################

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

#######################################
# Validate venv
#######################################

if [ ! -d "$VLLM_VENV" ]; then
    echo "ERROR: vLLM venv not found at $VLLM_VENV"
    exit 1
fi

#######################################
# Activate environment
#######################################

source "$VLLM_VENV/bin/activate"

#######################################
# Benchmark function
#######################################

run_model () {

    NAME="$1"
    MODEL_PATH="$MODEL_ROOT/$NAME"
    OUTPUT="$RESULTS_DIR/vllm_${NAME}_scaling.csv"

    echo ""
    echo "================================="
    echo "Running vLLM Scaling Study ($NAME)"
    echo "================================="

    if [ ! -d "$MODEL_PATH" ]; then
        echo "WARNING: Model not found. Skipping $NAME"
        return
    fi

    echo "backend,batch_size,max_new_tokens,total_latency_ms,tokens_per_sec,gpu_mem_mb" > "$OUTPUT"

    for B in "${BATCH_SIZES[@]}"
    do
        echo "Batch size $B..."

        python "$PROJECT/benchmarks/vllm/bm_vllm.py" \
            --model "$MODEL_PATH" \
            --prompts "$PROMPTS" \
            --batch_size "$B" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --dtype "$DTYPE" \
            --append_csv "$OUTPUT" \
            --backend vLLM
    done

    echo "Done: $NAME"
    echo "Saved: $OUTPUT"
}

#######################################
# Loop through config file
#######################################

while IFS="|" read -r NAME MODEL_ID
do
    run_model "$NAME"
done < "$CONFIG_FILE"

#######################################

deactivate

echo ""
echo "================================="
echo "All vLLM scaling studies complete."
echo "Results saved in:"
echo "$RESULTS_DIR"
echo "================================="