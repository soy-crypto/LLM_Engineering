#!/bin/bash
set -euo pipefail

HF_VENV="/workspace/.venv_hf"
PYTHON_BIN="python3.10"

echo "========================================"
echo "Bootstrapping HuggingFace + Mamba stack"
echo "========================================"

########################################
# Verify Python 3.10 exists
########################################

if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "ERROR: python3.10 not found."
    echo "Install it first:"
    echo "apt update && apt install -y python3.10 python3.10-venv python3.10-dev"
    exit 1
fi

########################################
# Remove old venv
########################################

if [ -d "$HF_VENV" ]; then
    echo "Removing existing venv..."
    rm -rf "$HF_VENV"
fi

########################################
# Create clean venv
########################################

$PYTHON_BIN -m venv "$HF_VENV"

PYTHON="$HF_VENV/bin/python"
PIP="$HF_VENV/bin/pip"

########################################
# Upgrade build tools
########################################

$PIP install --upgrade pip setuptools wheel

########################################
# Install Torch (FORCE correct version)
########################################

$PIP install --no-cache-dir --force-reinstall \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

########################################
# Install Transformers stack (locked)
########################################

$PIP install \
    transformers==4.43.3 \
    huggingface_hub \
    accelerate \
    protobuf \
    triton==3.0.0

########################################
# Install causal-conv1d (PREBUILT WHEEL)
########################################

$PIP install \
https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/\
causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

########################################
# Install mamba-ssm (disable build isolation)
########################################

$PIP install mamba-ssm==2.2.2 --no-build-isolation

########################################
# Verify environment
########################################

$PYTHON - <<EOF
import torch
import transformers
import causal_conv1d
import mamba_ssm

print("Environment verification OK")
print("Python:", __import__("sys").version)
print("Torch:", torch.__version__, "CUDA ABI:", torch.version.cuda)
print("Transformers:", transformers.__version__)
EOF

echo "========================================"
echo "HF backend ready"
echo "Python 3.10 | Torch 2.4.0 | Transformers 4.43.3"
echo "========================================"