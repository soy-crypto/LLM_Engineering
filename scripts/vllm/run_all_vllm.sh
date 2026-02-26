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

source "$VLLM_VENV/bin/activate"

echo "================================="
echo "Running vLLM Scaling Study"
echo "================================="

run_model () {

    NAME="$1"
    MODEL_PATH="$MODEL_ROOT/$NAME"
    OUTPUT="$RESULTS_DIR/vllm_${NAME}_scaling.csv"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "Skipping missing model: $NAME"
        return
    fi

    echo ""
    echo "Running: $NAME"

    python "$PROJECT/benchmarks/vllm/bm_vllm.py" \
        --model "$MODEL_PATH" \
        --prompts "$PROMPTS" \
        --batch_size "1,2,4,8" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --dtype "$DTYPE" \
        --out_csv "$OUTPUT" \
        --backend vLLM
}

while IFS="|" read -r NAME MODEL_ID
do
    [[ -z "$NAME" ]] && continue
    [[ "$NAME" =~ ^# ]] && continue

    run_model "$NAME"

done < "$CONFIG_FILE"

deactivate

echo ""
echo "Done."