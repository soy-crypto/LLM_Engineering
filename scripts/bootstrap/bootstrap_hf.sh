#!/bin/bash
set -euo pipefail

HF_VENV="/workspace/.venv_hf"
PYTHON_BIN="python3.10"

echo "Bootstrapping HuggingFace backend"

########################################
# Verify Python 3.10 exists
########################################

if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "ERROR: python3.10 not found."
    echo "Install it first:"
    echo "apt update && apt install -y python3.10 python3.10-venv"
    exit 1
fi

########################################
# Remove incompatible env (Python 3.12)
########################################

if [ -d "$HF_VENV" ]; then
    echo "Removing old incompatible venv..."
    rm -rf "$HF_VENV"
fi

########################################
# Create clean venv with Python 3.10
########################################

$PYTHON_BIN -m venv "$HF_VENV"

source "$HF_VENV/bin/activate"

########################################
# Upgrade core tooling
########################################

pip install --upgrade pip setuptools wheel

########################################
# Install PyTorch (CUDA 12.1 stable)
########################################

pip install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

########################################
# Install HuggingFace stack
########################################

pip install \
    transformers==4.43.3 \
    huggingface_hub \
    accelerate \
    protobuf \
    triton==3.0.0

########################################
# Install Mamba dependencies
########################################

pip install \
    mamba-ssm==2.2.2 \
    causal-conv1d==1.4.0 \
    --no-build-isolation

########################################
# Verify installation
########################################

python - <<EOF
import torch
import transformers
import mamba_ssm
import causal_conv1d

print("Environment verification OK")
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
EOF

deactivate

echo "HF backend ready (Python 3.10, Torch 2.4, CUDA 12.1 compatible)"