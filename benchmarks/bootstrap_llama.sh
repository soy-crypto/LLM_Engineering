#!/bin/bash
set -e

PROJECT="/workspace/LLM_Engineering"
MODEL_ID="meta-llama/Llama-3.1-8B"

HF_MODEL_DIR="$PROJECT/hf_models/llama3_1_8b"
CKPT_DIR="$PROJECT/trt_ckpt/llama3_1_8b_bf16_1gpu"
ENGINE_DIR="$PROJECT/trt_engine/llama3_1_8b_bf16_b16_s2560"

echo "================================="
echo "TensorRT-LLM Bootstrap (Llama-3.1-8B)"
echo "================================="

############################################
# 1️⃣ Require HF Token
############################################
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set."
    echo "Run on host:"
    echo "export HF_TOKEN=your_read_token"
    exit 1
fi

############################################
# 2️⃣ Clean incomplete model folder
############################################
if [ -d "$HF_MODEL_DIR" ] && [ ! -f "$HF_MODEL_DIR/config.json" ]; then
    echo "Incomplete model detected. Cleaning..."
    rm -rf "$HF_MODEL_DIR"
fi

############################################
# 3️⃣ Download model
############################################
if [ ! -d "$HF_MODEL_DIR" ]; then
    echo "Downloading model: $MODEL_ID"
    mkdir -p "$PROJECT/hf_models"

    python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$HF_MODEL_DIR",
    token="$HF_TOKEN",
    local_dir_use_symlinks=False
)
EOF

    echo "Model download complete."
else
    echo "Model already exists."
fi

############################################
# 4️⃣ Ensure running inside TRT container
############################################
if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
    echo "ERROR: Must run inside NVIDIA TensorRT-LLM container."
    exit 1
fi

echo "TensorRT-LLM version:"
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"

############################################
# 5️⃣ Verify config integrity
############################################
if ! grep -q "architectures" "$HF_MODEL_DIR/config.json"; then
    echo "ERROR: architectures field missing in config.json"
    exit 1
fi

############################################
# 6️⃣ Convert HF → TRT checkpoint
############################################
if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "Converting HF → TRT checkpoint (bf16)..."
    mkdir -p "$CKPT_DIR"

    python3 /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
        --model_dir "$HF_MODEL_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16
fi

############################################
# 7️⃣ Build TensorRT engine
############################################
if [ ! -f "$ENGINE_DIR/rank0.engine" ]; then
    echo "Building TensorRT engine (bf16, paged KV)..."
    mkdir -p "$ENGINE_DIR"

    trtllm-build \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$ENGINE_DIR" \
        --max_batch_size 16 \
        --max_seq_len 8192 \
        --kv_cache_type paged \
        --gemm_plugin bfloat16 \
        --gpt_attention_plugin bfloat16
fi

echo "================================="
echo "Bootstrap complete."
echo "================================="
