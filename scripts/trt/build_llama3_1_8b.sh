#!/bin/bash
set -euo pipefail

echo "========================================"
echo "TensorRT-LLM Build Script: LLaMA 3.1 8B"
echo "GPU: L40S"
echo "========================================"

########################################
# Paths
########################################

MODEL_ID="meta-llama/Llama-3.1-8B"

HF_DIR="/workspace/hf_models/llama3_1_8b"
CKPT_DIR="/workspace/trt_ckpt/llama3_1_8b_bf16_1gpu"
ENGINE_DIR="/workspace/trt_engine/llama3_1_8b_bf16_b16_s4096"

CONVERT_SCRIPT="/app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py"

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
# Download HF model
########################################

if [ ! -f "$HF_DIR/config.json" ]; then

    echo ""
    echo "Downloading HuggingFace model..."

    mkdir -p "$HF_DIR"

    huggingface-cli download "$MODEL_ID" \
        --local-dir "$HF_DIR" \
        --local-dir-use-symlinks False
fi

echo "HF model ready: $HF_DIR"

########################################
# Convert checkpoint
########################################

if [ ! -f "$CKPT_DIR/config.json" ]; then

    echo ""
    echo "Converting HF → TensorRT checkpoint..."

    mkdir -p "$CKPT_DIR"

    python3 -u "$CONVERT_SCRIPT" \
        --model_dir "$HF_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16
fi

echo "Checkpoint ready: $CKPT_DIR"

########################################
# Build TensorRT engine
########################################

if ! ls "$ENGINE_DIR"/*.engine >/dev/null 2>&1; then

    echo ""
    echo "Building TensorRT engine..."

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
fi

echo ""
echo "========================================"
echo "TensorRT engine ready"
echo "Location:"
echo "$ENGINE_DIR"
echo "========================================"