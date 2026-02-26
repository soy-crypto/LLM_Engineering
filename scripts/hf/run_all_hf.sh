#!/bin/bash
set -euo pipefail

########################################
# Paths
########################################

CONFIG="/workspace/LLM_Engineering/scripts/hf/models.conf"
MODEL_DIR="/workspace/hf_models"
PROMPTS="/workspace/LLM_Engineering/prompts/prompts_mid.txt"
OUT="/workspace/LLM_Engineering/results/results_hf.csv"
PYTHON="/workspace/.venv_hf/bin/python"
BENCH="/workspace/LLM_Engineering/benchmarks/hf/bm_hf.py"

########################################
# Validate config
########################################

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

########################################
# Read config line-by-line
########################################

while IFS= read -r model || [[ -n "$model" ]]; do

    # skip empty lines
    [[ -z "$model" ]] && continue

    # skip comments
    [[ "$model" =~ ^# ]] && continue

    MODEL_PATH="$MODEL_DIR/$model"

    echo "========================================"
    echo "Running HuggingFace benchmark: $model"
    echo "========================================"

    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: Model not found at $MODEL_PATH"
        continue
    fi

    "$PYTHON" "$BENCH" \
        --model "$MODEL_PATH" \
        --prompts "$PROMPTS" \
        --out_csv "$OUT" \
        --backend HF

    echo "Completed: $model"
    echo ""

    sleep 2

done < "$CONFIG"

echo "========================================"
echo "All benchmarks complete"
echo "========================================"