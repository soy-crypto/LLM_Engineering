#!/bin/bash
set -euo pipefail

HF_VENV="/workspace/.venv_hf"

echo "Bootstrapping HuggingFace backend"

# Create if not exists
if [ ! -d "$HF_VENV" ]; then
    python3 -m venv "$HF_VENV"
fi

# Always activate
source "$HF_VENV/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Core PyTorch (CUDA 12.1)
pip install --upgrade \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# HuggingFace stack
pip install --upgrade \
    transformers==4.43.3 \
    huggingface_hub \
    accelerate \
    protobuf \
    triton

# Required for Nemotron Mamba models
pip install --upgrade \
    mamba-ssm \
    causal-conv1d \
    --no-build-isolation

deactivate

echo "HF backend ready."