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

export TRANSFORMERS_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================="
echo "Running TensorRT-LLM benchmarks"
echo "================================="

########################################
# Validate environment
########################################

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: prompts file missing: $PROMPTS"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file missing: $CONFIG_FILE"
    exit 1
fi

if ! command -v trtllm-build >/dev/null; then
    echo "ERROR: Must run inside TensorRT-LLM container"
    exit 1
fi

if ! python3 -c "import tensorrt_llm" &>/dev/null; then
    echo "ERROR: tensorrt_llm not installed"
    exit 1
fi

########################################
# Benchmark function
########################################

run_model () {

    local NAME="$1"
    local MODEL_ID="$2"

    local ENGINE_DIR="$ENGINE_ROOT/$NAME"
    local OUT_CSV="$RESULTS_DIR/trt_${NAME}.csv"

    echo ""
    echo "================================="
    echo "Running TensorRT benchmark: $NAME"
    echo "Model ID: $MODEL_ID"
    echo "Engine: $ENGINE_DIR"
    echo "================================="

    ####################################
    # FIX 1: validate engine properly
    ####################################
    if [ ! -f "$ENGINE_DIR/config.json" ]; then
        echo "WARNING: Engine not found or incomplete: $ENGINE_DIR"
        return
    fi

    ####################################
    # FIX 2: ensure output file header once
    ####################################
    if [ ! -f "$OUT_CSV" ]; then
        echo "backend,model,batch_size,avg_ttft,avg_latency,tokps_new" > "$OUT_CSV"
    fi

    ####################################
    # FIX 3: use absolute python path
    ####################################
    python3 "$PROJECT/benchmarks/trt/bm_trtllm.py" \
        --engine_dir "$ENGINE_DIR" \
        --model_id "$MODEL_ID" \
        --prompts "$PROMPTS" \
        --batch_size "$BATCH_SIZES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --out_csv "$OUT_CSV" \
        --backend TensorRT-LLM

    echo "Finished: $NAME"
}

########################################
# Loop config safely
########################################

while IFS="|" read -r NAME MODEL_ID || [[ -n "$NAME" ]]
do

    [[ -z "$NAME" ]] && continue
    [[ "$NAME" =~ ^# ]] && continue

    run_model "$NAME" "$MODEL_ID"

done < "$CONFIG_FILE"

echo ""
echo "================================="
echo "All TensorRT benchmarks done"
echo "Results:"
echo "$RESULTS_DIR"
echo "================================="