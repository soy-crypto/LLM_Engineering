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

CKPT_DIR="$PROJECT/trt_ckpt/llama3_1_8b_bf16_1gpu"
ENGINE_DIR="$PROJECT/trt_engine/llama3_1_8b_bf16_b16_s4096"

echo "================================="
echo "Bootstrapping LLaMA 3.1 8B"
echo "================================="

########################################
# Load .env if exists
########################################
if [ -f "$PROJECT/.env" ]; then
    echo "Loading environment from .env"
    export $(grep -v '^#' "$PROJECT/.env" | xargs)
fi

#######################################
# Check HuggingFace token
#######################################
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "❌ HUGGINGFACE_HUB_TOKEN not set."
    echo "Add it to .env file:"
    echo "HUGGINGFACE_HUB_TOKEN=hf_xxxxx"
    exit 1
fi

#######################################
# 1️⃣ Create HF venv
#######################################
if [ ! -d "$HF_VENV" ]; then
    echo "Creating HF virtual environment..."
    $PYTHON_BIN -m venv "$HF_VENV"

    source "$HF_VENV/bin/activate"
    pip install --upgrade pip

    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    pip install transformers==4.43.3
    pip install huggingface_hub==0.23.4
    pip install protobuf

    pip check
    deactivate
else
    echo "HF venv already exists."
fi

#######################################
# 2️⃣ Create vLLM venv
#######################################
if [ ! -d "$VLLM_VENV" ]; then
    echo "Creating vLLM virtual environment..."
    $PYTHON_BIN -m venv "$VLLM_VENV"

    source "$VLLM_VENV/bin/activate"
    pip install --upgrade pip

    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    pip install vllm==0.15.1
    pip install transformers==4.43.3

    pip check
    deactivate
else
    echo "vLLM venv already exists."
fi

#######################################
# 3️⃣ Download model
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
# 4️⃣ Verify model
#######################################
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ ERROR: config.json missing."
    echo "Possible reasons:"
    echo " - Token invalid"
    echo " - Access not approved for gated model"
    echo " - Network issue"
    exit 1
fi

########################################
# 5️⃣ TensorRT-LLM Bootstrap
########################################

# Ensure running inside TRT container
if ! python3 -c "import tensorrt_llm" &>/dev/null; then
    echo "❌ ERROR: Must run inside NVIDIA TensorRT-LLM container."
    echo ""
    echo "Run with:"
    echo "docker run --gpus all -it \\"
    echo "  -v \$PWD:/workspace \\"
    echo "  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3"
    exit 1
fi

########################################
# Convert HF → TRT checkpoint (once)
########################################
if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "Converting HF → TRT checkpoint (bf16)..."
    mkdir -p "$CKPT_DIR"

    python3 /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
        --model_dir "$MODEL_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16
fi

########################################
# Build TRT Engine (once)
########################################
if [ ! -f "$ENGINE_DIR/rank0.engine" ]; then
    echo "Building TensorRT engine (bf16)..."
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

echo "================================="
echo "✅ Bootstrap complete"
echo "Model location:  $MODEL_DIR"
echo "Engine location: $ENGINE_DIR"
echo "================================="

ChatGPT said:
Perfect — I’ll give you the corrected full script with the HuggingFace download fixed properly.

The only broken part was this line:

"$HF_PYTHON" -m huggingface_hub download ...
We will replace it with the correct CLI usage.

✅ FINAL CORRECT VERSION (Fully Fixed)
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

CKPT_DIR="$PROJECT/trt_ckpt/llama3_1_8b_bf16_1gpu"
ENGINE_DIR="$PROJECT/trt_engine/llama3_1_8b_bf16_b16_s4096"

echo "================================="
echo "Bootstrapping LLaMA 3.1 8B"
echo "================================="

########################################
# Load .env if exists
########################################
if [ -f "$PROJECT/.env" ]; then
    echo "Loading environment from .env"
    export $(grep -v '^#' "$PROJECT/.env" | xargs)
fi

#######################################
# Check HuggingFace token
#######################################
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "❌ HUGGINGFACE_HUB_TOKEN not set."
    echo "Add it to .env file:"
    echo "HUGGINGFACE_HUB_TOKEN=hf_xxxxx"
    exit 1
fi

#######################################
# 1️⃣ Create HF venv
#######################################
if [ ! -d "$HF_VENV" ]; then
    echo "Creating HF virtual environment..."
    $PYTHON_BIN -m venv "$HF_VENV"

    source "$HF_VENV/bin/activate"
    pip install --upgrade pip

    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    pip install transformers==4.43.3
    pip install huggingface_hub==0.23.4
    pip install protobuf

    pip check
    deactivate
else
    echo "HF venv already exists."
fi

#######################################
# 2️⃣ Create vLLM venv
#######################################
if [ ! -d "$VLLM_VENV" ]; then
    echo "Creating vLLM virtual environment..."
    $PYTHON_BIN -m venv "$VLLM_VENV"

    source "$VLLM_VENV/bin/activate"
    pip install --upgrade pip

    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    pip install vllm==0.15.1
    pip install transformers==4.43.3

    pip check
    deactivate
else
    echo "vLLM venv already exists."
fi

#######################################
# 3️⃣ Download model (FIXED)
#######################################
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Downloading $MODEL_ID ..."
    mkdir -p "$MODEL_DIR"

    source "$HF_VENV/bin/activate"

    huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False \
        --token "$HUGGINGFACE_HUB_TOKEN"

    deactivate

    echo "Download complete."
else
    echo "Model already exists."
fi

#######################################
# 4️⃣ Verify model
#######################################
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ ERROR: config.json missing."
    echo "Possible reasons:"
    echo " - Token invalid"
    echo " - Access not approved for gated model"
    echo " - Network issue"
    exit 1
fi

########################################
# 5️⃣ TensorRT-LLM Bootstrap
########################################

if ! python3 -c "import tensorrt_llm" &>/dev/null; then
    echo "❌ ERROR: Must run inside NVIDIA TensorRT-LLM container."
    echo ""
    echo "Run with:"
    echo "docker run --gpus all -it \\"
    echo "  -v \$PWD:/workspace \\"
    echo "  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3"
    exit 1
fi

########################################
# Convert HF → TRT checkpoint
########################################
if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "Converting HF → TRT checkpoint (bf16)..."
    mkdir -p "$CKPT_DIR"

    python3 /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
        --model_dir "$MODEL_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16
fi

########################################
# Build TRT Engine
########################################
if [ ! -f "$ENGINE_DIR/rank0.engine" ]; then
    echo "Building TensorRT engine (bf16)..."
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

echo "================================="
echo "✅ Bootstrap complete"
echo "Model location:  $MODEL_DIR"
echo "Engine location: $ENGINE_DIR"
echo "================================="