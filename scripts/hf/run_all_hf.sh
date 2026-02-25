#!/bin/bash
set -euo pipefail

HF_VENV="/workspace/.venv_hf"
PYTHON_BIN="python3.10"

echo "========================================"
echo "Bootstrapping HuggingFace backend"
echo "========================================"

########################################
# Install Python 3.10 if missing
########################################

if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "Python 3.10 not found. Installing..."

    apt update
    apt install -y software-properties-common

    # Add repo if needed
    if ! apt-cache policy python3.10 | grep -q Candidate; then
        add-apt-repository ppa:deadsnakes/ppa -y
        apt update
    fi

    apt install -y python3.10 python3.10-venv python3.10-dev

    echo "Python 3.10 installed."
fi

########################################
# Remove incompatible env safely
########################################

if [ -d "$HF_VENV" ]; then
    echo "Removing old venv..."
    rm -rf "$HF_VENV"
fi

########################################
# Create clean Python 3.10 environment
########################################

echo "Creating new venv..."

$PYTHON_BIN -m venv "$HF_VENV"

PYTHON="$HF_VENV/bin/python"
PIP="$HF_VENV/bin/pip"

########################################
# Upgrade core build tools
########################################

echo "Upgrading pip and build tools..."

$PIP install --upgrade pip setuptools wheel

########################################
# Install PyTorch CUDA 12.1 stack
########################################

echo "Installing PyTorch..."

$PIP install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

########################################
# Install HuggingFace stack
########################################

echo "Installing HuggingFace stack..."

$PIP install \
    transformers==4.43.3 \
    huggingface_hub \
    accelerate \
    protobuf \
    triton==3.0.0

########################################
# Install Mamba dependencies
########################################

echo "Installing Mamba dependencies..."

$PIP install \
    mamba-ssm==2.2.2 \
    causal-conv1d==1.4.0 \
    --no-build-isolation

########################################
# Verify installation
########################################

echo "Verifying environment..."

$PYTHON - <<EOF
import torch
import transformers
import mamba_ssm
import causal_conv1d

print("Environment verification OK")
print("Python:", __import__("sys").version)
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("CUDA:", torch.version.cuda)
EOF

echo "========================================"
echo "HF backend ready"
echo "Venv: $HF_VENV"
echo "========================================"