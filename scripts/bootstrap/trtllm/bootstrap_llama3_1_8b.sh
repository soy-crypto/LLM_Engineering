#!/bin/bash
set -euo pipefail

echo "========================================"
echo "Bootstrap LLaMA 3.1 8B → TensorRT-LLM"
echo "GPU: L40S"
echo "========================================"

########################################
# Paths
########################################

PROJECT="/workspace/LLM_Engineering"

MODEL_ID="meta-llama/Llama-3.1-8B"
MODEL_DIR="/workspace/hf_models/llama3_1_8b"

HF_VENV="/workspace/.venv_hf"

CKPT_DIR="/workspace/trt_ckpt/llama3_1_8b_bf16_1gpu"
ENGINE_DIR="/workspace/trt_engine/llama3_1_8b_bf16_b16_s4096"

CONVERT_SCRIPT="/app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py"

########################################
# Validate TensorRT container
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
# Create HF venv if needed
########################################

if [ ! -d "$HF_VENV" ]; then

    echo ""
    echo "Creating HuggingFace venv..."

    python3 -m venv "$HF_VENV"
    source "$HF_VENV/bin/activate"

    pip install --upgrade pip

    pip install \
        torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    pip install \
        transformers==4.43.3 \
        huggingface_hub \
        protobuf

    deactivate
fi

########################################
# Activate HF venv
########################################

source "$HF_VENV/bin/activate"

echo "Using existing HuggingFace authentication (no check required)"

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Downloading model..."
    huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
fi

deactivate

########################################
# Download model
########################################

if [ ! -f "$MODEL_DIR/config.json" ]; then

    echo ""
    echo "Downloading model from HuggingFace..."

    mkdir -p "$MODEL_DIR"

    huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
fi

echo "HF model ready: $MODEL_DIR"

deactivate

########################################
# Convert checkpoint
########################################

if [ ! -f "$CKPT_DIR/config.json" ]; then

    echo ""
    echo "Converting HF → TensorRT checkpoint..."

    mkdir -p "$CKPT_DIR"

    python3 -u "$CONVERT_SCRIPT" \
        --model_dir "$MODEL_DIR" \
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
echo "Bootstrap complete"
echo ""
echo "HF model:"
echo "$MODEL_DIR"
echo ""
echo "TRT checkpoint:"
echo "$CKPT_DIR"
echo ""
echo "TRT engine:"
echo "$ENGINE_DIR"
echo "========================================"