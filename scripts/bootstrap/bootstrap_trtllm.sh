```bash
#!/bin/bash
set -euo pipefail

echo "========================================"
echo "Bootstrapping TensorRT-LLM backend"
echo "========================================"

########################################
# Validate NVIDIA driver
########################################

if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: NVIDIA driver not available"
    exit 1
fi

echo ""
echo "NVIDIA driver detected:"
nvidia-smi

########################################
# Validate Python
########################################

PYTHON_BIN="python3"

if ! command -v $PYTHON_BIN >/dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

echo ""
echo "Python detected:"
$PYTHON_BIN --version

########################################
# Validate TensorRT-LLM CLI
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

echo ""
echo "TensorRT-LLM CLI detected."

########################################
# Validate TensorRT-LLM Python module
########################################

echo ""
echo "Checking TensorRT-LLM Python module..."

$PYTHON_BIN - <<EOF
import tensorrt_llm
print("TensorRT-LLM version:", tensorrt_llm.__version__)
EOF

########################################
# Install compatible optional utilities
########################################

echo ""
echo "Installing optional utilities (TensorRT-LLM compatible versions)..."

$PYTHON_BIN -m pip install \
    transformers==4.57.1 \
    huggingface_hub==0.36.2 \
    accelerate==1.12.0 \
    numpy==1.26.4 \
    sentencepiece \
    protobuf \
    safetensors \
    psutil \
    tqdm \
    --no-cache-dir

########################################
# Verify compatibility
########################################

echo ""
echo "Verifying dependency versions..."

$PYTHON_BIN - <<EOF
import numpy, transformers, tensorrt_llm
print("numpy:", numpy.__version__)
print("transformers:", transformers.__version__)
print("tensorrt_llm:", tensorrt_llm.__version__)
EOF

########################################
# Create engine directory
########################################

ENGINE_ROOT="/workspace/trt_engine"

mkdir -p "$ENGINE_ROOT"

echo ""
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
echo "     --dtype bfloat16 \\"
echo "     --max_batch_size 8 \\"
echo "     --max_seq_len 4096"
echo ""
echo "2. Run benchmark:"
echo ""
echo "   ./run_all_trt.sh"
echo ""
```
