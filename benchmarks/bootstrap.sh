#!/bin/bash
set -e

echo "================================="
echo "Setting up benchmark environments"
echo "================================="

PROJECT="/workspace/LLM_Engineering"
PYTHON_BIN="python3"

#######################################
# HF venv
#######################################
if [ ! -d "$PROJECT/.venv_hf" ]; then
    echo "Creating HF venv..."
    $PYTHON_BIN -m venv $PROJECT/.venv_hf
    source $PROJECT/.venv_hf/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers huggingface_hub
    deactivate
fi

#######################################
# vLLM venv
#######################################
if [ ! -d "$PROJECT/.venv_vllm" ]; then
    echo "Creating vLLM venv..."
    $PYTHON_BIN -m venv $PROJECT/.venv_vllm
    source $PROJECT/.venv_vllm/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install vllm transformers
    deactivate
fi

#######################################
# Model Download (HF format)
#######################################

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
HF_MODEL_DIR="/workspace/hf_models/qwen2p5_7b"

if [ ! -d "$HF_MODEL_DIR" ]; then
    echo "Downloading model: $MODEL_ID"
    mkdir -p /workspace/hf_models

    $PROJECT/.venv_hf/bin/python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$HF_MODEL_DIR",
    local_dir_use_symlinks=False
)
EOF

    echo "Model download complete."
else
    echo "Model already exists."
fi

########################################
# TensorRT-LLM Bootstrap (bf16)
########################################

# Must be inside TRT container
if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
    echo "ERROR: TensorRT-LLM not found."
    echo "Run this inside NVIDIA NGC tensorrt-llm container."
    exit 1
fi

CKPT_DIR="/workspace/trt_ckpt/qwen2p5_7b_bf16_1gpu"
ENGINE_DIR="/workspace/trt_engine/qwen2p5_7b_bf16_b16_s2560"

# Convert (only if not exists)
if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "Converting HF → TRT checkpoint (bf16)..."
    mkdir -p $CKPT_DIR

    python3 /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
      --model_dir $HF_MODEL_DIR \
      --output_dir $CKPT_DIR \
      --dtype bfloat16
fi

# Build engine (only if not exists)
if [ ! -f "$ENGINE_DIR/config.json" ]; then
    echo "Building TensorRT engine (bf16)..."
    mkdir -p $ENGINE_DIR

    trtllm-build \
      --checkpoint_dir $CKPT_DIR \
      --output_dir $ENGINE_DIR \
      --max_batch_size 16 \
      --max_seq_len 2560 \
      --kv_cache_type paged \
      --gemm_plugin bfloat16 \
      --gpt_attention_plugin bfloat16
fi

echo "TensorRT bootstrap complete."

echo "================================="
echo "Environment setup complete."
echo "================================="