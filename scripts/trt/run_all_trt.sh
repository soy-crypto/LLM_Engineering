```bash
#!/bin/bash
set -euo pipefail

########################################
# TensorRT-LLM Benchmark Runner
########################################

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"

RESULTS_DIR="$PROJECT/results"
CONFIG_FILE="$PROJECT/scripts/config/models.conf"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

ENGINE_ROOT="$WORKSPACE/trt_engine"

BATCH_SIZES="1,2,4,8,16"
MAX_NEW_TOKENS=512

mkdir -p "$RESULTS_DIR"
mkdir -p "$ENGINE_ROOT"

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
# Validate engine completeness
########################################

engine_valid () {

    local ENGINE_DIR="$1"

    if [ ! -d "$ENGINE_DIR" ]; then
        return 1
    fi

    if [ ! -f "$ENGINE_DIR/config.json" ]; then
        return 1
    fi

    if ! ls "$ENGINE_DIR"/*.engine >/dev/null 2>&1; then
        return 1
    fi

    return 0
}

########################################
# Run benchmark for one model
########################################

run_model () {

    local NAME="$1"
    local MODEL_ID="$2"

    local ENGINE_DIR="$ENGINE_ROOT/$NAME"
    local OUT_CSV="$RESULTS_DIR/trt_${NAME}.csv"

    echo ""
    echo "================================="
    echo "Model: $NAME"
    echo "HF ID: $MODEL_ID"
    echo "Engine: $ENGINE_DIR"
    echo "================================="

    ####################################
    # Validate engine
    ####################################

    if ! engine_valid "$ENGINE_DIR"; then
        echo "WARNING: Engine missing or incomplete: $ENGINE_DIR"
        echo "Skipping benchmark."
        return
    fi

    ####################################
    # Create CSV header if needed
    ####################################

    if [ ! -f "$OUT_CSV" ]; then
        echo "backend,model,batch_size,avg_ttft,avg_latency,tokps_new" > "$OUT_CSV"
    fi

    ####################################
    # Run benchmark
    ####################################

    python3 "$PROJECT/benchmarks/trt/bm_trtllm.py" \
        --engine_dir "$ENGINE_DIR" \
        --model_id "$MODEL_ID" \
        --prompts "$PROMPTS" \
        --batch_size "$BATCH_SIZES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --out_csv "$OUT_CSV" \
        --backend TensorRT-LLM \
        || echo "ERROR: Benchmark failed for $NAME"

    echo "Finished: $NAME"
}

########################################
# Main loop
########################################

while IFS="|" read -r NAME MODEL_ID || [[ -n "$NAME" ]]
do

    [[ -z "$NAME" ]] && continue
    [[ "$NAME" =~ ^# ]] && continue

    run_model "$NAME" "$MODEL_ID"

done < "$CONFIG_FILE"

echo ""
echo "================================="
echo "All TensorRT benchmarks complete"
echo "Results directory:"
echo "$RESULTS_DIR"
echo "================================="
```
