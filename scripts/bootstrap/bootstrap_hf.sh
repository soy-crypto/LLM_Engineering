#!/bin/bash
set -euo pipefail

HF_VENV="/workspace/.venv_hf"
PYTHON_BIN="python3.10"

echo "========================================"
echo "Bootstrapping HuggingFace inference stack"
echo "========================================"

########################################
# Clean env
########################################

rm -rf "$HF_VENV"

$PYTHON_BIN -m venv "$HF_VENV"

PYTHON="$HF_VENV/bin/python"
PIP="$HF_VENV/bin/pip"

########################################
# Upgrade build tools
########################################

$PIP install --upgrade pip setuptools wheel

########################################
# Install Torch stack (includes Triton 3.0.0)
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
# Install HF stack
########################################

$PIP install \
transformers==4.45.2 \
accelerate==0.33.0 \
huggingface_hub==0.36.2 \
tokenizers==0.20.3 \
safetensors \
sentencepiece \
hf-xet==1.3.1

########################################
# Install required runtime deps
########################################

$PIP install \
einops \
httpcore \
httpx \
requests \
tqdm \
pyyaml \
regex \
psutil

########################################
# Install causal-conv1d CORRECTLY
########################################

$PIP install causal-conv1d==1.4.0

########################################
# Install mamba AFTER causal-conv1d
########################################

$PIP install mamba-ssm==2.2.2 --no-deps

########################################
# Verify stack
########################################

echo "Verifying installation..."

$PYTHON - <<EOF
import torch
import transformers
import causal_conv1d
import mamba_ssm
import einops
import accelerate

print("===================================")
print("FULL STACK OK")
print("===================================")
print("Python:", __import__("sys").version)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
EOF

echo "========================================"
echo "HF inference stack ready"
echo "========================================"