#!/bin/bash
set -euo pipefail

########################################
# Config
########################################

VLLM_VENV="/workspace/.venv_vllm"
PYTHON_BIN="python3"

echo "========================================"
echo "Bootstrapping vLLM inference environment"
echo "Python: $($PYTHON_BIN --version)"
echo "========================================"

########################################
# Remove broken venv automatically
########################################

if [ -d "$VLLM_VENV" ] && [ ! -f "$VLLM_VENV/bin/python" ]; then
    echo "Broken venv detected. Removing..."
    rm -rf "$VLLM_VENV"
fi

########################################
# Create venv
########################################

if [ ! -d "$VLLM_VENV" ]; then

    echo "Creating virtual environment..."

    $PYTHON_BIN -m venv "$VLLM_VENV"

fi

########################################
# Activate
########################################

source "$VLLM_VENV/bin/activate"

########################################
# Upgrade pip
########################################

pip install --upgrade pip setuptools wheel

########################################
# Install PyTorch (CUDA 12.1)
########################################

echo "Installing PyTorch..."

pip install \
  torch==2.4.0 \
  torchvision==0.19.0 \
  torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121

########################################
# Install compatible numpy
########################################

pip install numpy==1.26.4

########################################
# Install vLLM and required dependencies
########################################

echo "Installing vLLM..."

pip install \
  vllm==0.6.3 \
  transformers==4.45.2 \
  accelerate==0.33.0 \
  huggingface_hub \
  outlines \
  pyairports \
  sentencepiece \
  protobuf \
  psutil \
  requests \
  tqdm \
  pyyaml \
  regex

########################################
# Verify installation
########################################

echo "Verifying installation..."

python - <<EOF
import torch
import vllm
import transformers
import outlines
import pyairports

print("OK")
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("Transformers:", transformers.__version__)
EOF

########################################
# Done
########################################

echo "========================================"
echo "vLLM environment ready"
echo "Activate with:"
echo "source $VLLM_VENV/bin/activate"
echo "========================================"