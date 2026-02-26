#!/bin/bash
set -euo pipefail

############################################
# Configuration
############################################

VLLM_VENV="/workspace/.venv_vllm"
PYTHON_BIN="python3.10"

echo "========================================"
echo "Bootstrapping vLLM inference environment"
echo "========================================"

############################################
# Verify Python
############################################

if ! command -v $PYTHON_BIN &> /dev/null; then
    echo "ERROR: python3.10 not found"
    echo "Install via:"
    echo "apt update && apt install -y python3.10 python3.10-venv"
    exit 1
fi

############################################
# Remove old environment
############################################

if [ -d "$VLLM_VENV" ]; then
    echo "Removing old vLLM venv..."
    rm -rf "$VLLM_VENV"
fi

############################################
# Create fresh environment
############################################

echo "Creating new vLLM environment..."

$PYTHON_BIN -m venv "$VLLM_VENV"

PYTHON="$VLLM_VENV/bin/python"
PIP="$VLLM_VENV/bin/pip"

############################################
# Upgrade build tools
############################################

echo "Upgrading pip and build tools..."

$PIP install --upgrade pip setuptools wheel packaging

############################################
# Install PyTorch CUDA 12.1 (required)
############################################

echo "Installing PyTorch CUDA 12.1..."

$PIP install \
    torch==2.4.0 \
    torchvision==0.19.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

############################################
# Install vLLM
############################################

echo "Installing vLLM..."

$PIP install vllm

############################################
# Install HF utilities
############################################

echo "Installing HuggingFace utilities..."

$PIP install \
    transformers \
    huggingface_hub \
    accelerate \
    sentencepiece \
    safetensors

############################################
# Verify installation
############################################

echo "Verifying vLLM installation..."

$PYTHON - <<EOF
import torch
import vllm

print("===================================")
print("vLLM environment ready")
print("Python:", __import__("sys").version)
print("Torch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("GPU available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("===================================")
EOF

echo ""
echo "========================================"
echo "vLLM bootstrap complete"
echo "Venv location: $VLLM_VENV"
echo "Activate with:"
echo "source $VLLM_VENV/bin/activate"
echo "========================================"