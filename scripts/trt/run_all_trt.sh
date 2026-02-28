```bash
#!/bin/bash
set -euo pipefail

# Enable debug output (REMOVE later if you want quieter logs)
set -x

########################################
# TensorRT-LLM Full Pipeline Runner
# HF → TRT checkpoint → TRT engine → Benchmark
########################################

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"

RESULTS_DIR="$PROJECT/results"
CONFIG_FILE="$PROJECT/scripts/config/models.conf"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

HF_ROOT="$WORKSPACE/hf_models"
CKPT_ROOT="$WORKSPACE/trt_ckpt"
ENGINE_ROOT="$WORKSPACE/trt_engine"

BATCH_SIZES="1,2,4,8,16"
MAX_NEW_TOKENS=512

mkdir -p "$RESULTS_DIR"
mkdir -p "$CKPT_ROOT"
mkdir -p "$ENGINE_ROOT"

export TRANSFORMERS_TRUST_REMOTE_CODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "================================="
echo "TensorRT-LLM Full Pipeline Runner"
echo "================================="

echo "WORKSPACE: $WORKSPACE"
echo "HF_ROOT: $HF_ROOT"
echo "CKPT_ROOT: $CKPT_ROOT"
echo "ENGINE_ROOT: $ENGINE_ROOT"
echo "CONFIG_FILE: $CONFIG_FILE"
echo "PROMPTS: $PROMPTS"

########################################
# Validate environment
########################################

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file missing: $CONFIG_FILE"
    exit 1
fi

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: prompts file missing: $PROMPTS"
    exit 1
fi

if ! command -v trtllm-build >/dev/null; then
    echo "ERROR: trtllm-build not found (not in TensorRT-LLM container)"
    exit 1
fi

if ! python3 -c "import tensorrt_llm" &>/dev/null; then
    echo "ERROR: tensorrt_llm not installed"
    exit 1
fi

echo "Environment validation OK"

########################################
# Check checkpoint exists
########################################

ckpt_valid() {

    local DIR="$1"

    if [ ! -d "$DIR" ]; then
        return 1
    fi

    if [ ! -f "$DIR/config.json" ]; then
        return 1
    fi

    if ! ls "$DIR"/*.safetensors >/dev/null 2>&1; then
        return 1
    fi

    return 0
}

########################################
# Check engine exists
########################################

engine_valid() {

    local DIR="$1"

    if [ ! -d "$DIR" ]; then
        return 1
    fi

    if [ ! -f "$DIR/config.json" ]; then
        return 1
    fi

    if ! ls "$DIR"/*.engine >/dev/null 2>&1; then
        return 1
    fi

    return 0
}

########################################
# Convert HF → TRT checkpoint
########################################

convert_checkpoint() {

    local NAME="$1"

    local HF_DIR="$HF_ROOT/$NAME"
    local CKPT_DIR="$CKPT_ROOT/$NAME"

    if ckpt_valid "$CKPT_DIR"; then
        echo "Checkpoint already exists: $CKPT_DIR"
        return
    fi

    if [ ! -d "$HF_DIR" ]; then
        echo "ERROR: HF model directory missing: $HF_DIR"
        exit 1
    fi

    echo "================================="
    echo "Converting checkpoint: $NAME"
    echo "================================="

    python3 -u /usr/local/lib/python3.12/dist-packages/tensorrt_llm/examples/llama/convert_checkpoint.py \
        --model_dir "$HF_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16

    echo "Checkpoint conversion complete: $NAME"
}

########################################
# Build TRT engine
########################################

build_engine() {

    local NAME="$1"

    local CKPT_DIR="$CKPT_ROOT/$NAME"
    local ENGINE_DIR="$ENGINE_ROOT/$NAME"

    if engine_valid "$ENGINE_DIR"; then
        echo "Engine already exists: $ENGINE_DIR"
        return
    fi

    echo "================================="
    echo "Building TensorRT engine: $NAME"
    echo "================================="

    trtllm-build \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$ENGINE_DIR" \
        --max_batch_size 8 \
        --max_seq_len 4096 \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --context_fmha enable \
        --remove_input_padding enable \
        --kv_cache_type paged

    echo "Engine build complete: $NAME"
}

########################################
# Benchmark engine
########################################

run_benchmark() {

    local NAME="$1"
    local MODEL_ID="$2"

    local ENGINE_DIR="$ENGINE_ROOT/$NAME"
    local OUT_CSV="$RESULTS_DIR/trt_${NAME}.csv"

    echo "================================="
    echo "Benchmarking model: $NAME"
    echo "================================="

    if [ ! -f "$OUT_CSV" ]; then
        echo "backend,model,batch_size,avg_ttft,avg_latency,tokps_new" > "$OUT_CSV"
    fi

    python3 "$PROJECT/benchmarks/trt/bm_trtllm.py" \
        --engine_dir "$ENGINE_DIR" \
        --model_id "$MODEL_ID" \
        --prompts "$PROMPTS" \
        --batch_size "$BATCH_SIZES" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --out_csv "$OUT_CSV" \
        --backend TensorRT-LLM

    echo "Benchmark complete: $NAME"
}

########################################
# Main loop (FIXED — no hanging)
########################################

echo "================================="
echo "Reading model config"
echo "================================="

cat "$CONFIG_FILE"

while IFS="|" read -r NAME MODEL_ID
do

    NAME="${NAME:-}"
    MODEL_ID="${MODEL_ID:-}"

    # skip empty lines
    if [[ -z "$NAME" ]]; then
        continue
    fi

    # skip comments
    if [[ "$NAME" =~ ^# ]]; then
        continue
    fi

    echo ""
    echo "================================="
    echo "Processing model: $NAME"
    echo "================================="

    convert_checkpoint "$NAME"
    build_engine "$NAME"
    run_benchmark "$NAME" "$MODEL_ID"

done < "$CONFIG_FILE"

echo ""
echo "================================="
echo "Pipeline complete"
echo "Results directory:"
echo "$RESULTS_DIR"
echo "================================="
```
