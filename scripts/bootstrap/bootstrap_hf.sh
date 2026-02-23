#!/bin/bash
set -euo pipefail

HF_VENV="/workspace/.venv_hf"

echo "Bootstrapping HuggingFace backend"

if [ ! -d "$HF_VENV" ]; then

    python3 -m venv "$HF_VENV"
    source "$HF_VENV/bin/activate"

    pip install --upgrade pip

    pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    pip install \
        transformers==4.43.3 \
        huggingface_hub \
        accelerate \
        protobuf

    deactivate
fi

echo "HF backend ready."