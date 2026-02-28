```bash
#!/bin/bash
set -euo pipefail

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
echo "TensorRT-LLM Full Pipeline"
echo "================================="

########################################
# Validation
########################################

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: prompts file missing: $PROMPTS"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config file missing: $CONFIG_FILE"
    exit 1
fi

########################################
# Check checkpoint exists
########################################

ckpt_valid () {

    local DIR="$1"

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

engine_valid () {

    local DIR="$1"

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

convert_checkpoint () {

    local NAME="$1"

    local HF_DIR="$HF_ROOT/$NAME"
    local CKPT_DIR="$CKPT_ROOT/$NAME"

    if ckpt_valid "$CKPT_DIR"; then
        echo "Checkpoint exists: $CKPT_DIR"
        return
    fi

    echo "Converting checkpoint: $NAME"

    python3 /usr/local/lib/python3.12/dist-packages/tensorrt_llm/examples/llama/convert_checkpoint.py \
        --model_dir "$HF_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16

}

########################################
# Build TRT engine
########################################

build_engine () {

    local NAME="$1"

    local CKPT_DIR="$CKPT_ROOT/$NAME"
    local ENGINE_DIR="$ENGINE_ROOT/$NAME"

    if engine_valid "$ENGINE_DIR"; then
        echo "Engine exists: $ENGINE_DIR"
        return
    fi

    echo "Building engine: $NAME"

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

}

########################################
# Benchmark engine
########################################

run_benchmark () {

    local NAME="$1"
    local MODEL_ID="$2"

    local ENGINE_DIR="$ENGINE_ROOT/$NAME"
    local OUT_CSV="$RESULTS_DIR/trt_${NAME}.csv"

    echo "Benchmarking: $NAME"

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

}

########################################
# Main loop
########################################

while IFS="|" read -r NAME MODEL_ID || [[ -n "$NAME" ]]
do

    [[ -z "$NAME" ]] && continue
    [[ "$NAME" =~ ^# ]] && continue

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
echo "Results:"
echo "$RESULTS_DIR"
echo "================================="
```
