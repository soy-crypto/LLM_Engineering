#!/bin/bash
set -euo pipefail

HF_VENV="/workspace/.venv_hf"
PYTHON_BIN="python3.10"

echo "========================================"
echo "Bootstrapping HuggingFace + Nemotron stack"
echo "========================================"

########################################
# Install Python 3.10 if missing
########################################

if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "Python 3.10 not found. Installing..."

    apt update
    apt install -y software-properties-common

    if ! apt-cache policy python3.10 | grep -q Candidate; then
        add-apt-repository ppa:deadsnakes/ppa -y
        apt update
    fi

    apt install -y python3.10 python3.10-venv python3.10-dev
fi

########################################
# Remove old venv
########################################

if [ -d "$HF_VENV" ]; then
    echo "Removing existing venv..."
    rm -rf "$HF_VENV"
fi

########################################
# Create venv
########################################

echo "Creating virtual environment..."

$PYTHON_BIN -m venv "$HF_VENV"

PYTHON="$HF_VENV/bin/python"
PIP="$HF_VENV/bin/pip"

########################################
# Upgrade build tools
########################################

echo "Upgrading pip and build tools..."

$PIP install --upgrade pip setuptools wheel

########################################
# Install PyTorch (force correct version)
########################################

echo "Installing PyTorch 2.4.0..."

$PIP install --no-cache-dir --force-reinstall \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

########################################
# Install Transformers stack (Nemotron compatible)
########################################

echo "Installing Transformers stack..."

$PIP install \
    transformers==4.45.2 \
    safetensors>=0.4.3 \
    huggingface_hub \
    accelerate \
    protobuf \
    triton==3.0.0

########################################
# Install causal-conv1d PREBUILT wheel
########################################

echo "Installing causal-conv1d..."

$PIP install \
https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/\
causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

########################################
# Install mamba-ssm
########################################

echo "Installing mamba-ssm..."

$PIP install mamba-ssm==2.2.2 --no-build-isolation

########################################
# Verify installation
########################################

echo "Verifying environment..."

$PYTHON - <<EOF
import sys
import torch
import transformers
import safetensors
import causal_conv1d
import mamba_ssm

print("Environment verification OK")
print("Python:", sys.version.split()[0])
print("Torch:", torch.__version__, "| CUDA ABI:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("Safetensors:", safetensors.__version__)
EOF

echo "========================================"
echo "HF backend ready"
echo "Venv: $HF_VENV"
echo "========================================"