#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"
RESULTS_DIR="$PROJECT/results"
CONFIG_FILE="$PROJECT/scripts/config/models.conf"

PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8,16"
MAX_NEW_TOKENS=512

ENGINE_ROOT="$WORKSPACE/trt_engine"

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running TensorRT-LLM benchmarks (CONFIG MODE)"
echo "================================="

########################################
# Validate prompts
########################################

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: Prompts file not found at $PROMPTS"
    exit 1
fi

########################################
# Validate config file
########################################

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

########################################
# Validate TensorRT environment
########################################

if ! command -v trtllm-build >/dev/null; then
    echo "ERROR: Must run inside TensorRT-LLM container."
    exit 1
fi

if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
    echo "ERROR: tensorrt_llm module not available."
    exit 1
fi

########################################
# Benchmark function
########################################

run_model () {

    local NAME="$1"
    local MODEL_ID="$2"

    ENGINE_DIR="$ENGINE_ROOT/${NAME}_bf16_b16_s4096"
    OUT_CSV="$RESULTS_DIR/trt_${NAME}_results.csv"

    echo "---------------------------------"
    echo "Running TensorRT benchmark: $NAME"
    echo "Engine: $ENGINE_DIR"
    echo "---------------------------------"

    if [ ! -d "$ENGINE_DIR" ]; then
        echo "WARNING: Engine not found. Skipping $NAME"
        return
    fi

    python "$PROJECT/benchmarks/trt/bm_trtllm.py" \
        --engine_dir "$ENGINE_DIR" \
        --model_id "$MODEL_ID" \
        --prompts "$PROMPTS" \
        --batch_size "$BATCH_SIZES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --out_csv "$OUT_CSV" \
        --backend "TensorRT-LLM"

    echo "Done: $NAME"
    echo ""
}

########################################
# Loop through config file
########################################

while IFS="|" read -r NAME MODEL_ID
do

    run_model "$NAME" "$MODEL_ID"

done < "$CONFIG_FILE"

########################################

echo "================================="
echo "All TensorRT-LLM benchmarks complete."
echo "Results saved in:"
echo "$RESULTS_DIR"
echo "================================="