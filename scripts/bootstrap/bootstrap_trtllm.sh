#!/bin/bash
set -euo pipefail

############################################
# Config
############################################

VLLM_VENV="/workspace/.venv_vllm"
PYTHON_BIN="python3.10"

echo "========================================"
echo "Bootstrapping vLLM backend"
echo "========================================"

############################################
# Check Python
############################################

if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "ERROR: python3.10 not found."
    echo "Install with:"
    echo "apt update && apt install -y python3.10 python3.10-venv"
    exit 1
fi

############################################
# Create venv if missing
############################################

if [ ! -d "$VLLM_VENV" ]; then
    echo "Creating virtual environment..."
    $PYTHON_BIN -m venv "$VLLM_VENV"
fi

source "$VLLM_VENV/bin/activate"

############################################
# Upgrade pip
############################################

pip install --upgrade pip setuptools wheel packaging

############################################
# Install PyTorch CUDA 12.1
############################################

pip install torch==2.4.0 \
            torchvision==0.19.0 \
            torchaudio==2.4.0 \
            --index-url https://download.pytorch.org/whl/cu121

############################################
# Install vLLM
############################################

pip install vllm==0.5.4

############################################
# Install HF utilities
############################################

pip install transformers \
            huggingface_hub \
            accelerate \
            sentencepiece \
            safetensors

############################################
# Verify installation
############################################

python - <<EOF
import torch
import vllm

print("===================================")
print("vLLM environment ready")
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("===================================")
EOF

deactivate

echo "========================================"
echo "vLLM backend ready."
echo "Venv: $VLLM_VENV"
echo "========================================"