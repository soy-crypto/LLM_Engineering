#!/bin/bash
set -euo pipefail

HF_VENV="/workspace/.venv_hf"
PYTHON_BIN="python3"

echo "========================================"
echo "Bootstrapping HuggingFace inference stack"
echo "========================================"

########################################
# Remove broken venv automatically
########################################

if [ -d "$HF_VENV" ] && [ ! -f "$HF_VENV/bin/python" ]; then
    echo "Broken venv detected. Removing..."
    rm -rf "$HF_VENV"
fi

########################################
# Create venv only if missing
########################################

if [ ! -d "$HF_VENV" ]; then

    echo "Creating virtual environment..."

    $PYTHON_BIN -m venv "$HF_VENV"

    PYTHON="$HF_VENV/bin/python"
    PIP="$HF_VENV/bin/pip"

    echo "Upgrading pip..."
    $PIP install --upgrade pip setuptools wheel

    ########################################
    # Install PyTorch CUDA 12.1
    ########################################

    echo "Installing PyTorch..."

    $PIP install \
        torch==2.4.0 \
        torchvision==0.19.0 \
        torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    ########################################
    # Fix numpy compatibility
    ########################################

    $PIP install numpy==1.26.4

    ########################################
    # Install HuggingFace stack
    ########################################

    echo "Installing HuggingFace stack..."

    $PIP install \
        transformers==4.45.2 \
        accelerate==0.33.0 \
        huggingface_hub==0.36.2 \
        tokenizers==0.20.3 \
        safetensors \
        sentencepiece \
        protobuf \
        psutil \
        requests \
        tqdm \
        pyyaml \
        regex

    echo "Environment created."

else

    echo "Environment already exists."

    PYTHON="$HF_VENV/bin/python"
    PIP="$HF_VENV/bin/pip"

fi

########################################
# Verify
########################################

echo "Verifying..."

$PYTHON - <<EOF
import torch, transformers

print("OK")
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
EOF

echo "========================================"
echo "HF environment ready"
echo "========================================"