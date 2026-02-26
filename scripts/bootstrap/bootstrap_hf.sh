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
    apt update
    apt install -y software-properties-common
    add-apt-repository ppa:deadsnakes/ppa -y
    apt update
    apt install -y python3.10 python3.10-venv python3.10-dev
fi

########################################
# Clean environment
########################################

rm -rf "$HF_VENV"
$PYTHON_BIN -m venv "$HF_VENV"

PYTHON="$HF_VENV/bin/python"
PIP="$HF_VENV/bin/pip"

########################################
# Upgrade tools
########################################

$PIP install --upgrade pip setuptools wheel

########################################
# Install Torch FIRST and freeze it
########################################

$PIP install --no-cache-dir \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Freeze torch so pip cannot downgrade it
$PIP install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --no-deps

########################################
# Install Transformers stack (no torch touch)
########################################

########################################
# Install Transformers stack (PINNED)
########################################

$PIP install --no-deps \
    transformers==4.45.2 \
    tokenizers==0.20.3 \
    huggingface-hub==0.36.2 \
    safetensors==0.4.5 \
    accelerate==0.33.0 \
    protobuf \
    psutil

########################################
# Install required runtime deps
########################################

$PIP install \
    httpcore \
    httpx \
    requests \
    tqdm \
    pyyaml \
    regex \
    filelock \
    fsspec \
    packaging \
    numpy \
    certifi \
    charset_normalizer \
    urllib3 \
    idna \
    anyio \
    h11

########################################
# Install Triton matching Torch 2.4
########################################

$PIP install triton==2.3.1 --no-deps

########################################
# Install causal-conv1d wheel matching torch 2.4
########################################

$PIP install --no-deps \
https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/\
causal_conv1d-1.4.0+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

########################################
# Install mamba (no dependency resolution)
########################################

$PIP install mamba-ssm==2.2.2 --no-deps

########################################
# Verify
########################################

$PYTHON - <<EOF
import torch
import transformers
import causal_conv1d
import mamba_ssm
import triton

print("Torch:", torch.__version__, "| CUDA:", torch.version.cuda)
print("Transformers:", transformers.__version__)
print("Triton:", triton.__version__)
print("Full stack OK")
EOF

echo "========================================"
echo "Environment Ready"
echo "========================================"