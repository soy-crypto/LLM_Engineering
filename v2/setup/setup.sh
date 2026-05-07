#!/bin/bash
# Single entrypoint to set up the full LLM Engineering environment.
# Run this once on a fresh GPU server after cloning the repo.

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT/v2/setup/.env"

echo "========================================"
echo "LLM Engineering — Full Setup"
echo "Project: $PROJECT"
echo "========================================"

# -------------------------------------------------------
# Step 0: Load .env
# -------------------------------------------------------
if [ ! -f "$ENV_FILE" ]; then
    echo ""
    echo "No v2/setup/.env found. Creating from template..."
    cp "$PROJECT/v2/setup/.env.template" "$ENV_FILE"
    echo ""
    echo "  ACTION REQUIRED:"
    echo "  Edit env/.env and set your HUGGINGFACE_HUB_TOKEN, then re-run this script."
    echo ""
    exit 1
fi

source "$ENV_FILE"
export HUGGINGFACE_HUB_TOKEN
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "Loaded env/.env"

# -------------------------------------------------------
# Step 1: Host setup (drivers, Docker, NVIDIA toolkit)
# -------------------------------------------------------
echo ""
echo "[ Step 1/5 ] Host setup"

if command -v nvidia-smi &>/dev/null && command -v docker &>/dev/null; then
    echo "  Host already configured (nvidia-smi and docker found)"
else
    echo "  Running bootstrap_host.sh..."
    bash "$PROJECT/scripts/bootstrap_host.sh"
fi

# -------------------------------------------------------
# Step 2: Download models
# -------------------------------------------------------
echo ""
echo "[ Step 2/5 ] Download models"

if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "  ERROR: HUGGINGFACE_HUB_TOKEN not set in env/.env"
    exit 1
fi

bash "$PROJECT/scripts/download_models.sh"

# -------------------------------------------------------
# Step 3: HuggingFace environment
# -------------------------------------------------------
echo ""
echo "[ Step 3/5 ] HuggingFace environment"
bash "$PROJECT/scripts/bootstrap/bootstrap_hf.sh"

# -------------------------------------------------------
# Step 4: vLLM environment
# -------------------------------------------------------
echo ""
echo "[ Step 4/5 ] vLLM environment"
bash "$PROJECT/scripts/bootstrap/bootstrap_vllm.sh"

# -------------------------------------------------------
# Step 5: Serving environment
# -------------------------------------------------------
echo ""
echo "[ Step 5/5 ] Serving environment"
bash "$PROJECT/scripts/bootstrap/bootstrap_serving.sh"

# -------------------------------------------------------
# TRT-LLM note
# -------------------------------------------------------
echo ""
echo "========================================"
echo "Setup complete."
echo ""
echo "TRT-LLM engines must be built separately inside the container:"
echo ""
echo "  docker run -it --gpus all --shm-size=32g \\"
echo "    --name trt_llm_big \\"
echo "    -v /ephemeral/llm_workspace:/workspace \\"
echo "    -v /ephemeral/hf_models:/workspace/hf_models \\"
echo "    -v /ephemeral/trt_engine:/workspace/trt_engine \\"
echo "    -v $PROJECT:/workspace/LLM_Engineering \\"
echo "    nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3 bash"
echo ""
echo "  Then inside the container:"
echo "  bash scripts/bootstrap/trtllm/bootstrap_all_trt.sh"
echo ""
echo "Run environment check:"
echo "  bash scripts/check_env.sh"
echo "========================================"
