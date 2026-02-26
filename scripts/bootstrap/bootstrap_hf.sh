#!/bin/bash
set -euo pipefail

HF_VENV="/workspace/.venv_hf"
PYTHON_BIN="python3.10"

echo "========================================"
echo "Bootstrapping HuggingFace inference stack"
echo "========================================"

########################################
# Remove old env
########################################

rm -rf "$HF_VENV"

########################################
# Create clean env
########################################

$PYTHON_BIN -m venv "$HF_VENV"

PYTHON="$HF_VENV/bin/python"
PIP="$HF_VENV/bin/pip"

########################################
# Upgrade tooling
########################################

$PIP install --upgrade pip setuptools wheel

########################################
# Install Torch (CUDA 12.1, Triton 2.3 compatible)
########################################

$PIP install \
torch==2.4.0 \
torchvision==0.19.0 \
torchaudio==2.4.0 \
--index-url https://download.pytorch.org/whl/cu121

########################################
# Fix numpy BEFORE transformers
########################################

$PIP install numpy==1.26.4

########################################
# Install core HF stack (STRICT versions)
########################################

$PIP install \
transformers==4.45.2 \
accelerate==0.33.0 \
huggingface_hub==0.36.2 \
tokenizers==0.20.3 \
safetensors \
sentencepiece

########################################
# Install networking deps
########################################

$PIP install \
httpcore \
httpx \
requests \
tqdm \
pyyaml \
regex \
psutil \
einops \
hf-xet==1.3.0

########################################
# Install Triton compatible with Torch 2.4
########################################

$PIP install triton==2.3.1

########################################
# Install causal-conv1d (PREBUILT CORRECT ABI)
########################################

$PIP install \
causal-conv1d==1.4.0+cu122torch2.4cxx11abifalse \
-f https://github.com/Dao-AILab/causal-conv1d/releases/expanded_assets/v1.4.0

########################################
# Install Mamba AFTER causal-conv1d
########################################

$PIP install mamba-ssm==2.2.2 --no-deps

########################################
# Verify environment
########################################

echo "Verifying installation..."

$PYTHON - <<EOF
import torch
import transformers
import mamba_ssm
import causal_conv1d
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