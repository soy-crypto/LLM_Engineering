#!/bin/bash
set -euo pipefail

HF_VENV="/workspace/.venv_hf"
PYTHON_BIN="python3.10"

echo "========================================"
echo "Bootstrapping Clean HF Stack"
echo "========================================"

rm -rf "$HF_VENV"

$PYTHON_BIN -m venv "$HF_VENV"

PYTHON="$HF_VENV/bin/python"
PIP="$HF_VENV/bin/pip"

########################################
# Upgrade base tools
########################################

$PIP install --upgrade pip setuptools wheel

########################################
# Install Torch (includes correct Triton)
########################################

$PIP install \
torch==2.4.0 \
torchvision==0.19.0 \
torchaudio==2.4.0 \
--index-url https://download.pytorch.org/whl/cu121

########################################
# Fix numpy version
########################################

$PIP install numpy==1.26.4

########################################
# Install HF stack ONLY
########################################

$PIP install \
transformers==4.45.2 \
accelerate==0.33.0 \
huggingface_hub==0.36.2 \
tokenizers==0.20.3 \
safetensors \
sentencepiece \
psutil

########################################
# Verify
########################################

$PYTHON - <<EOF
import torch
import transformers

print("===================================")
print("READY")
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
EOF