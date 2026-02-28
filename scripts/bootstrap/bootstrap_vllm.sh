#!/bin/bash
set -euo pipefail

VLLM_VENV="/workspace/.venv_vllm"
PYTHON_BIN="python3"

echo "========================================"
echo "Bootstrapping vLLM inference environment"
echo "========================================"

########################################
# Remove broken venv automatically
########################################

if [ -d "$VLLM_VENV" ] && [ ! -f "$VLLM_VENV/bin/python" ]; then
    echo "Broken venv detected. Removing..."
    rm -rf "$VLLM_VENV"
fi

########################################
# Create venv if missing
########################################

if [ ! -d "$VLLM_VENV" ]; then

    echo "Creating vLLM virtual environment..."

    $PYTHON_BIN -m venv "$VLLM_VENV"

    PYTHON="$VLLM_VENV/bin/python"
    PIP="$VLLM_VENV/bin/pip"

    $PIP install --upgrade pip setuptools wheel

    ########################################
    # Install PyTorch CUDA 12.1
    ########################################

    $PIP install \
        torch==2.4.0 \
        torchvision==0.19.0 \
        torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    ########################################
    # Install vLLM
    ########################################

    $PIP install vllm==0.5.4

    echo "vLLM environment created."

else

    echo "Environment already exists."

    PYTHON="$VLLM_VENV/bin/python"
fi

########################################
# Verify
########################################

echo "Verifying vLLM..."

$PYTHON - <<EOF
import vllm, torch
print("OK")
print("vLLM:", vllm.__version__)
print("Torch:", torch.__version__)
EOF

echo "========================================"
echo "vLLM environment ready"
echo "========================================"