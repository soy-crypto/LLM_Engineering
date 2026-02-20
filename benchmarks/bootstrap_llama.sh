#!/bin/bash
set -euo pipefail

PROJECT="/workspace/LLM_Engineering"

MODEL_ID="meta-llama/Llama-3.1-8B"
MODEL_DIR="$PROJECT/hf_models/llama3_1_8b"

HF_VENV="$PROJECT/.venv_hf"
VLLM_VENV="$PROJECT/.venv_vllm"

CKPT_DIR="$PROJECT/trt_ckpt/llama3_1_8b_bf16_1gpu"
ENGINE_DIR="$PROJECT/trt_engine/llama3_1_8b_bf16_b16_s4096"

echo "Bootstrapping LLaMA 3.1 8B"

########################################
# HF venv
########################################
if [ ! -d "$HF_VENV" ]; then
    python3 -m venv "$HF_VENV"
    source "$HF_VENV/bin/activate"

    pip install --upgrade pip
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.43.3 huggingface_hub==0.23.4 protobuf

    deactivate
fi

########################################
# vLLM venv
########################################
if [ ! -d "$VLLM_VENV" ]; then
    python3 -m venv "$VLLM_VENV"
    source "$VLLM_VENV/bin/activate"

    pip install --upgrade pip
    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121
    pip install vllm==0.15.1 transformers==4.43.3
    
    pip install --upgrade transformers

    deactivate
fi

########################################
# Download model
########################################
if [ ! -f "$MODEL_DIR/config.json" ]; then
    mkdir -p "$MODEL_DIR"
    source "$HF_VENV/bin/activate"

    huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False

    deactivate
fi

########################################
# Require TRT container
########################################
python3 -c "import tensorrt_llm" &>/dev/null || {
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
        --gpt_attention_plugin bfloat16
fi

echo "Done."