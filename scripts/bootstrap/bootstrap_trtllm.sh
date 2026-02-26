#!/bin/bash
set -euo pipefail

echo "========================================"
echo "Bootstrapping TensorRT-LLM backend"
echo "========================================"

########################################
# Validate CUDA
########################################

if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: NVIDIA driver not available"
    exit 1
fi

nvidia-smi

########################################
# Validate Python
########################################

PYTHON_BIN="python3"

if ! command -v $PYTHON_BIN >/dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

########################################
# Validate TensorRT-LLM container
########################################

if ! command -v trtllm-build >/dev/null; then
    echo ""
    echo "ERROR: trtllm-build not found."
    echo ""
    echo "You must run inside NVIDIA TensorRT-LLM container:"
    echo ""
    echo "docker run --gpus all -it --rm \\"
    echo "  -v /workspace:/workspace \\"
    echo "  nvcr.io/nvidia/tensorrt-llm:latest"
    echo ""
    exit 1
fi

echo "TensorRT-LLM detected:"
trtllm-build --version || true

########################################
# Validate Python packages
########################################

echo "Checking tensorrt_llm module..."

$PYTHON_BIN - <<EOF
import tensorrt_llm
print("tensorrt_llm OK")
EOF

########################################
# Install optional tools (safe)
########################################

echo "Installing optional utilities..."

$PYTHON_BIN -m pip install --upgrade \
    transformers \
    huggingface_hub \
    accelerate \
    sentencepiece \
    protobuf \
    safetensors \
    numpy \
    psutil \
    tqdm

########################################
# Create engine directory
########################################

ENGINE_ROOT="/workspace/trt_engine"

mkdir -p "$ENGINE_ROOT"

echo "Engine directory ready:"
echo "$ENGINE_ROOT"

########################################
# GPU validation
########################################

echo ""
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

########################################
# Complete
########################################

echo ""
echo "========================================"
echo "TensorRT-LLM backend ready"
echo "========================================"

echo ""
echo "Next steps:"
echo ""
echo "1. Convert HF model → TRT engine:"
echo ""
echo "   trtllm-build \\"
echo "     --checkpoint_dir /workspace/hf_models/llama3_1_8b \\"
echo "     --output_dir /workspace/trt_engine/llama3_1_8b \\"
echo "     --dtype bfloat16"
echo ""
echo "2. Run benchmark:"
echo ""
echo "   ./run_all_trt.sh"
echo ""