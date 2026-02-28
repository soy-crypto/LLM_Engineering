#!/bin/bash
set -euo pipefail

echo "========================================"
echo "Bootstrapping TensorRT-LLM backend"
echo "========================================"

if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: NVIDIA driver not available"
    exit 1
fi

nvidia-smi

if ! command -v trtllm-build >/dev/null; then
    echo "ERROR: trtllm-build not found."
    exit 1
fi

echo "TensorRT-LLM version:"
python3 - <<EOF
import tensorrt_llm
print(tensorrt_llm.__version__)
EOF

echo ""
echo "Installing non-core utilities only..."

python3 -m pip install \
    psutil \
    tqdm \
    sentencepiece \
    safetensors \
    --no-cache-dir

echo ""
echo "Environment ready."