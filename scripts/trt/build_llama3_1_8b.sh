```bash
#!/bin/bash --noprofile --norc
set -euo pipefail

MODEL_NAME="llama3_1_8b"

HF_DIR="/workspace/hf_models/$MODEL_NAME"
CKPT_DIR="/workspace/trt_ckpt/$MODEL_NAME"
ENGINE_DIR="/workspace/trt_engine/$MODEL_NAME"

echo "================================="
echo "Building TensorRT-LLM for $MODEL_NAME"
echo "================================="

mkdir -p "$CKPT_DIR"
mkdir -p "$ENGINE_DIR"

########################################
# Step 1: Convert checkpoint
########################################

if [ ! -f "$CKPT_DIR/config.json" ]; then

    echo "Converting HF checkpoint → TRT checkpoint"

    python3 -u \
    /usr/local/lib/python3.12/dist-packages/tensorrt_llm/examples/llama/convert_checkpoint.py \
        --model_dir "$HF_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16

else

    echo "Checkpoint already exists"

fi

########################################
# Step 2: Build engine
########################################

if ! ls "$ENGINE_DIR"/*.engine >/dev/null 2>&1; then

    echo "Building TRT engine"

    trtllm-build \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$ENGINE_DIR" \
        --max_batch_size 8 \
        --max_seq_len 4096 \
        --gpt_attention_plugin bfloat16 \
        --gemm_plugin bfloat16 \
        --context_fmha enable \
        --remove_input_padding enable \
        --kv_cache_type paged

else

    echo "Engine already exists"

fi

echo "================================="
echo "DONE: $MODEL_NAME"
echo "================================="
```
