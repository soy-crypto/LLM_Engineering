#!/bin/bash
set -euo pipefail

echo "========================================"
echo "Building TensorRT Engine: LLaMA 3.1 8B"
echo "GPU: L40S"
echo "========================================"

########################################
# Paths
########################################

CKPT_DIR="/workspace/trt_ckpt/llama3_1_8b_bf16_1gpu"
ENGINE_DIR="/workspace/trt_engine/llama3_1_8b_bf16_b16_s4096"

########################################
# Validate container
########################################

if ! command -v trtllm-build >/dev/null; then
    echo "ERROR: Must run inside TensorRT-LLM container"
    exit 1
fi

python3 -c "import tensorrt_llm" || {
    echo "ERROR: tensorrt_llm not installed"
    exit 1
}

########################################
# Validate checkpoint
########################################

if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "ERROR: TRT checkpoint not found:"
    echo "$CKPT_DIR"
    exit 1
fi

########################################
# Skip if engine exists
########################################

if ls "$ENGINE_DIR"/*.engine >/dev/null 2>&1; then

    echo ""
    echo "Engine already exists:"
    echo "$ENGINE_DIR"

    exit 0
fi

########################################
# Build engine
########################################

echo ""
echo "Building engine..."

mkdir -p "$ENGINE_DIR"

trtllm-build \
    --checkpoint_dir "$CKPT_DIR" \
    --output_dir "$ENGINE_DIR" \
    --max_batch_size 16 \
    --max_seq_len 4096 \
    --kv_cache_type paged \
    --gemm_plugin bfloat16 \
    --gpt_attention_plugin bfloat16 \
    --context_fmha enable \
    --use_paged_context_fmha enable \
    --remove_input_padding enable

echo ""
echo "========================================"
echo "Engine build complete"
echo "$ENGINE_DIR"
echo "========================================"