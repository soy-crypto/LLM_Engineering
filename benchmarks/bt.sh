#!/bin/bash
set -euo pipefail

########################################
# Config
########################################
PROJECT="/workspace/LLM_Engineering"
PYTHON_BIN="python3"

MODEL_ID="meta-llama/Llama-3.1-8B"
MODEL_DIR="$PROJECT/hf_models/llama3_1_8b"

HF_VENV="$PROJECT/.venv_hf"
VLLM_VENV="$PROJECT/.venv_vllm"
HF_PYTHON="$HF_VENV/bin/python"

echo "================================="
echo "Bootstrapping LLaMA 3.1 8B"
echo "================================="

#######################################
# 0️⃣ Check token
#######################################
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "❌ HUGGINGFACE_HUB_TOKEN not set."
    echo ""
    echo "Run:"
    echo "export HUGGINGFACE"
    echo ""
    exit 1
fi

#######################################
# 1️⃣ Create HF venv (if missing)
#######################################
if [ ! -d "$HF_VENV" ]; then
    echo "Creating HF virtual environment..."
    $PYTHON_BIN -m venv "$HF_VENV"

    source "$HF_VENV/bin/activate"
    pip install --upgrade pip

    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    pip install transformers==4.43.3
    pip install huggingface_hub==0.23.4
    pip install protobuf

    deactivate
else
    echo "HF venv already exists."
fi

#######################################
# 2️⃣ Create vLLM venv (if missing)
#######################################
if [ ! -d "$VLLM_VENV" ]; then
    echo "Creating vLLM virtual environment..."
    $PYTHON_BIN -m venv "$VLLM_VENV"

    source "$VLLM_VENV/bin/activate"
    pip install --upgrade pip

    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
    pip install vllm==0.5.5
    pip install transformers==4.43.3

    deactivate
else
    echo "vLLM venv already exists."
fi

#######################################
# 3️⃣ Download model (non-interactive)
#######################################
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Downloading $MODEL_ID ..."
    mkdir -p "$MODEL_DIR"

    "$HF_PYTHON" -m huggingface_hub download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False \
        --token "$HUGGINGFACE_HUB_TOKEN"

    echo "Download complete."
else
    echo "Model already exists."
fi

#######################################
# 4️⃣ Verify
#######################################
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ ERROR: config.json missing."
    echo "Possible reasons:"
    echo " - Token invalid"
    echo " - Access not approved for gated model"
    echo " - Network issue"
    exit 1
fi


#######################################
# 6️⃣ TensorRT-LLM Engine Build
#######################################

TRT_ENGINE_DIR="$PROJECT/trt_engine/llama3_1_8b_bf16"

if [ ! -d "$TRT_ENGINE_DIR" ]; then
    echo "================================="
    echo "Building TensorRT-LLM Engine"
    echo "================================="

    mkdir -p "$TRT_ENGINE_DIR"

    # Convert HF checkpoint
    python /opt/tensorrt_llm/examples/llama/convert_checkpoint.py \
        --model_dir "$MODEL_DIR" \
        --output_dir "$TRT_ENGINE_DIR" \
        --dtype bfloat16

    # Build engine
    trtllm-build \
        --checkpoint_dir "$TRT_ENGINE_DIR" \
        --output_dir "$TRT_ENGINE_DIR" \
        --max_batch_size 8 \
        --max_input_len 2048 \
        --max_seq_len 4096 \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16

    echo "Engine build complete."
else
    echo "TensorRT engine already exists."
fi

echo "================================="
echo "✅ Bootstrap complete"
echo "Model location:"
echo "$MODEL_DIR"
echo "================================="
