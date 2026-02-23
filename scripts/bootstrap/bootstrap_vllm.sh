#!/bin/bash
set -euo pipefail

MODEL_DIR="/workspace/hf_models/llama3_1_8b"

CKPT_DIR="/workspace/trt_ckpt/llama3_1_8b_bf16_1gpu"
ENGINE_DIR="/workspace/trt_engine/llama3_1_8b_bf16_b16_s4096"

echo "Bootstrapping TensorRT-LLM backend"

########################################
# Validate container
########################################
command -v trtllm-build >/dev/null || {
    echo "Run inside TensorRT-LLM container"
    exit 1
}

########################################
# Convert checkpoint
########################################
if [ ! -f "$CKPT_DIR/config.json" ]; then

    mkdir -p "$CKPT_DIR"

    python3 /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
        --model_dir "$MODEL_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16
fi

########################################
# Build engine
########################################
if [ ! -f "$ENGINE_DIR/rank0.engine" ]; then

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

echo "TensorRT-LLM backend ready."