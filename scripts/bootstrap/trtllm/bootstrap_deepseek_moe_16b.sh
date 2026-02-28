#!/bin/bash
set -euo pipefail

echo "========================================"
echo "Bootstrap DeepSeek MoE 16B → TensorRT-LLM"
echo "========================================"

MODEL_ID="deepseek-ai/deepseek-moe-16b-base"
MODEL_DIR="/workspace/hf_models/deepseek_moe_16b"

CKPT_DIR="/workspace/trt_ckpt/deepseek_moe_16b_bf16_1gpu"
ENGINE_DIR="/workspace/trt_engine/deepseek_moe_16b_bf16_b16_s4096"

########################################
# Download HF model
########################################

if [ ! -f "$MODEL_DIR/config.json" ]; then

    echo "Downloading model..."

    mkdir -p "$MODEL_DIR"

    huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
fi

echo "HF model ready"

########################################
# Convert checkpoint (CORRECT PATH)
########################################

if [ ! -f "$CKPT_DIR/config.json" ]; then

    echo "Converting checkpoint..."

    mkdir -p "$CKPT_DIR"

    python3 /app/tensorrt_llm/examples/models/core/deepseek/convert_checkpoint.py \
        --model_dir "$MODEL_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16
fi

echo "Checkpoint ready"

########################################
# Build engine
########################################

if [ ! -f "$ENGINE_DIR/rank0.engine" ]; then

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
fi

echo "DONE"