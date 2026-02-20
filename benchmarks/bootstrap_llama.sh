#!/bin/bash
set -e

########################################
# Config
########################################
PROJECT="/workspace/LLM_Engineering"
PYTHON_BIN="python3"

MODEL_ID="meta-llama/Llama-3.1-8B"
MODEL_DIR="$PROJECT/hf_models/llama3_1_8b"

HF_VENV="$PROJECT/.venv_hf"
VLLM_VENV="$PROJECT/.venv_vllm"

echo "================================="
echo "Bootstrapping LLaMA 3.1 8B"
echo "================================="

#######################################
# 1️⃣ Create HF venv (if missing)
#######################################
if [ ! -d "$HF_VENV" ]; then
    echo "Creating HF virtual environment..."
    $PYTHON_BIN -m venv "$HF_VENV"

    source "$HF_VENV/bin/activate"
    pip install --upgrade pip

    # Pin versions for reproducibility
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
    echo "Creating vLLM virtual environment (Python 3.10)..."

    # Force Python 3.10 (required for vLLM stability)
    python3 -m venv "$VLLM_VENV"

    source "$VLLM_VENV/bin/activate"
    pip install --upgrade pip

    # Install torch first
    pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121

    # Install vLLM with full optional deps
    pip install vllm[all]==0.5.5

    # Match transformers version
    pip install transformers==4.43.3

    deactivate
else
    echo "vLLM venv already exists."
fi

#######################################
# 3️⃣ Check HuggingFace login (inside HF venv)
#######################################
if ! "$HF_VENV/bin/huggingface-cli" whoami &> /dev/null; then
    echo ""
    echo "❌ Not logged into HuggingFace."
    echo "Run:"
    echo "source $HF_VENV/bin/activate"
    echo "huggingface-cli login"
    echo ""
    exit 1
fi

#######################################
# 4️⃣ Download model (if missing)
#######################################
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Downloading $MODEL_ID ..."
    mkdir -p "$MODEL_DIR"

    "$HF_VENV/bin/huggingface-cli" download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False

    echo "Download complete."
else
    echo "Model already exists."
fi

#######################################
# 5️⃣ Verify
#######################################
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ ERROR: config.json missing."
    echo "Possible reasons:"
    echo " - Access not approved on HuggingFace"
    echo " - Login failed"
    exit 1
fi

echo "================================="
echo "✅ Bootstrap complete"
echo "Model location:"
echo "$MODEL_DIR"
echo "================================="
