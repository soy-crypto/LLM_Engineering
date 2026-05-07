#!/bin/bash
set -euo pipefail

echo "========================================"
echo "TensorRT-LLM Benchmark Runner"
echo "GPU: L40S"
echo "========================================"

PROJECT="/workspace/LLM_Engineering"

RESULTS_DIR="$PROJECT/results"
PROMPTS="$PROJECT/prompts/prompts_mid.txt"

ENGINE_ROOT="/workspace/trt_engine"

mkdir -p "$RESULTS_DIR"

########################################
# Validate environment
########################################

if ! command -v trtllm-build >/dev/null; then
    echo "ERROR: Must run inside TensorRT-LLM container"
    exit 1
fi

if ! python3 -c "import tensorrt_llm" &>/dev/null; then
    echo "ERROR: tensorrt_llm not available"
    exit 1
fi

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: Missing prompts file"
    exit 1
fi

########################################
# Benchmark function
########################################

run_benchmark () {

    local NAME="$1"
    local MODEL_ID="$2"
    local ENGINE_DIR="$ENGINE_ROOT/$3"

    local OUT_CSV="$RESULTS_DIR/trt_${NAME}.csv"

    echo ""
    echo "========================================"
    echo "Benchmarking: $NAME"
    echo "Engine: $ENGINE_DIR"
    echo "========================================"

    if [ ! -d "$ENGINE_DIR" ] || ! ls "$ENGINE_DIR"/*.engine >/dev/null 2>&1; then
        echo "WARNING: Engine not found, skipping"
        return
    fi

    /usr/bin/python3 -u "$PROJECT/v2/benchmarks/trt/benchmark.py" \
        --engine_dir "$ENGINE_DIR" \
        --model_id "$MODEL_ID" \
        --prompts "$PROMPTS" \
        --batch_size "1,2,4,8,16" \
        --max_new_tokens 512 \
        --out_csv "$OUT_CSV" \
        --backend TensorRT-LLM

    echo "Completed: $NAME"
}

########################################
# Run benchmarks
########################################

run_benchmark \
    llama3_1_8b \
    meta-llama/Llama-3.1-8B \
    llama3_1_8b_bf16_b16_s4096

run_benchmark \
    mistral_7b \
    mistralai/Mistral-7B-Instruct-v0.3 \
    mistral_7b_bf16_b16_s4096

run_benchmark \
    gemma_7b \
    google/gemma-7b-it \
    gemma_7b_bf16_b16_s4096

run_benchmark \
    qwen2_5_7b \
    Qwen/Qwen2.5-7B-Instruct \
    qwen2_5_7b_bf16_b16_s4096

run_benchmark \
    phi_3_mini \
    microsoft/phi-3-mini-4k-instruct \
    phi_3_mini_bf16_b16_s4096

run_benchmark \
    deepseek_moe_16b \
    deepseek-ai/deepseek-moe-16b-base \
    deepseek_moe_16b_bf16_b16_s4096

echo ""
echo "========================================"
echo "All benchmarks complete"
echo ""
echo "Results saved to:"
echo "$RESULTS_DIR"
echo "========================================"