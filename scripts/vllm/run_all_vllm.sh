#!/bin/bash
set -euo pipefail

########################################
# Paths
########################################

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"

TRT_VENV="$WORKSPACE/.venv_trt"

CONFIG_FILE="$PROJECT/scripts/config/models.conf"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

MODEL_ROOT="$WORKSPACE/hf_models"
ENGINE_ROOT="$WORKSPACE/trt_engines"

RESULTS_DIR="$PROJECT/results"

BATCH_SIZES=(1 2 4 8)
MAX_NEW_TOKENS=512
DTYPE="bfloat16"

mkdir -p "$RESULTS_DIR"
mkdir -p "$ENGINE_ROOT"

echo "================================="
echo "Running TensorRT-LLM Scaling Study"
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

if [ ! -d "$TRT_VENV" ]; then
    echo "ERROR: TRT venv not found"
    exit 1
fi

########################################
# Activate TRT environment
########################################

source "$TRT_VENV/bin/activate"

########################################
# Build TRT engine
########################################

build_engine () {

    NAME="$1"

    HF_MODEL="$MODEL_ROOT/$NAME"
    ENGINE_DIR="$ENGINE_ROOT/$NAME"

    if [ -d "$ENGINE_DIR" ]; then
        echo "Engine already exists: $NAME"
        return
    fi

    echo ""
    echo "================================="
    echo "Building TensorRT engine: $NAME"
    echo "================================="

    python "$PROJECT/benchmarks/trt/build_trt_engine.py" \
        --hf_model_dir "$HF_MODEL" \
        --engine_dir "$ENGINE_DIR" \
        --dtype "$DTYPE"

}

########################################
# Benchmark TRT engine
########################################

run_model () {

    NAME="$1"

    ENGINE_DIR="$ENGINE_ROOT/$NAME"
    OUTPUT="$RESULTS_DIR/trt_${NAME}_scaling.csv"

    echo ""
    echo "================================="
    echo "Running TRT benchmark: $NAME"
    echo "================================="

    if [ ! -d "$ENGINE_DIR" ]; then
        echo "Engine missing, building..."
        build_engine "$NAME"
    fi

    echo "backend,batch_size,max_new_tokens,latency_ms,tokens_per_sec,gpu_mem_mb" > "$OUTPUT"

    for B in "${BATCH_SIZES[@]}"
    do

        echo "Batch size $B"

        python "$PROJECT/benchmarks/trt/bm_trt.py" \
            --engine_dir "$ENGINE_DIR" \
            --prompts "$PROMPTS" \
            --batch_size "$B" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --append_csv "$OUTPUT" \
            --backend TRT

    done

    echo "Saved: $OUTPUT"

}

########################################
# Loop config
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
echo "All TensorRT benchmarks complete"
echo "Results in:"
echo "$RESULTS_DIR"
echo "================================="