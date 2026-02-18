#!/bin/bash
set -e

PROJECT="/workspace/LLM_Engineering"
PYTHON_BIN="python3"

MODEL_ID="meta-llama/Meta-Llama-Guard-2-8B"
HF_MODEL_DIR="$PROJECT/hf_models/llama_guard_2_8b"

CKPT_DIR="$PROJECT/trt_ckpt/llama_guard_2_8b_bf16_1gpu"
ENGINE_DIR="$PROJECT/trt_engine/llama_guard_2_8b_bf16_b16_s2560"

echo "================================="
echo "Setting up benchmark environments"
echo "================================="

#######################################
# HF venv
#######################################
if [ ! -d "$PROJECT/.venv_hf" ]; then
    echo "Creating HF venv..."
    $PYTHON_BIN -m venv "$PROJECT/.venv_hf"
    source "$PROJECT/.venv_hf/bin/activate"
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
    $PYTHON_BIN -m venv "$PROJECT/.venv_vllm"
    source "$PROJECT/.venv_vllm/bin/activate"
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install vllm transformers
    deactivate
fi

#######################################
# HuggingFace Login Check
#######################################
echo "Ensure you have logged in:"
echo "Run: huggingface-cli login"
echo "Llama models require access approval."

#######################################
# Model Download
#######################################
if [ ! -d "$HF_MODEL_DIR" ]; then
    echo "Downloading model: $MODEL_ID"
    mkdir -p "$PROJECT/hf_models"

    source "$PROJECT/.venv_hf/bin/activate"
    python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$HF_MODEL_DIR",
    local_dir_use_symlinks=False
)
EOF
    deactivate

    echo "Model download complete."
else
    echo "Model already exists."
fi

########################################
# TensorRT-LLM Bootstrap (bf16)
########################################

# Ensure running inside TRT container
if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
    echo "ERROR: Must run inside NVIDIA TensorRT-LLM container."
    exit 1
fi

########################################
# Convert HF → TRT Checkpoint (Llama)
########################################

if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "Converting HF → TRT checkpoint (bf16)..."
    mkdir -p "$CKPT_DIR"

    python3 /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
        --model_dir "$HF_MODEL_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16
fi

########################################
# Build TensorRT Engine
########################################

if [ ! -f "$ENGINE_DIR/rank0.engine" ]; then
    echo "Building TensorRT engine (bf16)..."
    mkdir -p "$ENGINE_DIR"

    trtllm-build \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$ENGINE_DIR" \
        --max_batch_size 16 \
        --max_seq_len 2560 \
        --kv_cache_type paged \
        --gemm_plugin bfloat16 \
        --gpt_attention_plugin bfloat16
fi

echo "================================="
echo "Bootstrap complete."
echo "================================="
