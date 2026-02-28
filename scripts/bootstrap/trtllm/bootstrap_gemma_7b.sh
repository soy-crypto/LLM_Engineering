#!/bin/bash
set -euo pipefail

echo "Bootstrap Gemma 7B → TensorRT-LLM"

MODEL_ID="google/gemma-7b-it"

MODEL_DIR="/workspace/hf_models/gemma_7b"
CKPT_DIR="/workspace/trt_ckpt/gemma_7b_bf16_1gpu"
ENGINE_DIR="/workspace/trt_engine/gemma_7b_bf16_b16_s4096"

HF_VENV="/workspace/.venv_hf"
CONVERT_SCRIPT="/app/tensorrt_llm/examples/models/core/gemma/convert_checkpoint.py"

command -v trtllm-build >/dev/null || exit 1

source "$HF_VENV/bin/activate"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    mkdir -p "$MODEL_DIR"
    huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
fi

deactivate

if [ ! -f "$CKPT_DIR/config.json" ]; then

    mkdir -p "$CKPT_DIR"

    python3 -u "$CONVERT_SCRIPT" \
        --model_dir "$MODEL_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16
fi

if ! ls "$ENGINE_DIR"/*.engine >/dev/null 2>&1; then

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
        --remove_input_padding enable
fi

echo "Done: Gemma 7B"