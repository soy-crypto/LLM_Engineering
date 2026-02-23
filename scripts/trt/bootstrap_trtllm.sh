#!/bin/bash
set -euo pipefail

VLLM_VENV="/workspace/.venv_vllm"

echo "Bootstrapping vLLM backend"

if [ ! -d "$VLLM_VENV" ]; then

    python3 -m venv "$VLLM_VENV"
    source "$VLLM_VENV/bin/activate"

    pip install --upgrade pip

    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    pip install vllm==0.5.4

    deactivate
fi

echo "vLLM backend ready."