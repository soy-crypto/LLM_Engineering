#!/bin/bash
set -euo pipefail

########################################
# Paths
########################################

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"

HF_VENV="$WORKSPACE/.venv_hf"
PYTHON="$HF_VENV/bin/python"

CONFIG_FILE="$PROJECT/scripts/config/models.conf"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

RESULTS_DIR="$PROJECT/results"

BATCH_SIZES="1,2,4,8"
MAX_NEW_TOKENS=512
DTYPE="bfloat16"

mkdir -p "$RESULTS_DIR"

########################################
# Validate environment
########################################

if [ ! -x "$PYTHON" ]; then
    echo "ERROR: HuggingFace venv not found."
    echo "Run bootstrap first:"
    echo "./scripts/bootstrap/bootstrap_hf.sh"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: models.conf not found at $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: prompts file not found at $PROMPTS"
    exit 1
fi

########################################
# Run benchmark function
########################################

run_model () {

    local NAME="$1"
    local MODEL_PATH="$WORKSPACE/hf_models/$NAME"
    local OUT_CSV="$RESULTS_DIR/hf_${NAME}_results.csv"

    echo "========================================"
    echo "Running HuggingFace benchmark: $NAME"
    echo "========================================"

    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "ERROR: Model not found at $MODEL_PATH"
        exit 1
    fi

    $PYTHON "$PROJECT/benchmarks/hf/bm_hf.py" \
        --model "$MODEL_PATH" \
        --prompts "$PROMPTS" \
        --batch_size "$BATCH_SIZES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --dtype "$DTYPE" \
        --out_csv "$OUT_CSV" \
        --backend HF

    echo "Completed: $NAME"
}

########################################
# Loop through models.conf
########################################

while IFS="|" read -r NAME MODEL_ID
do
    [ -z "$NAME" ] && continue
    [[ "$NAME" =~ ^# ]] && continue

    run_model "$NAME"

done < "$CONFIG_FILE"

########################################
# Done
########################################

echo "========================================"
echo "All HF benchmarks completed"
echo "Results saved to: $RESULTS_DIR"
echo "========================================"