```bash
#!/bin/bash --noprofile --norc
set -euo pipefail

MODEL_NAME="gemma_7b"

HF_DIR="/workspace/hf_models/$MODEL_NAME"
CKPT_DIR="/workspace/trt_ckpt/$MODEL_NAME"
ENGINE_DIR="/workspace/trt_engine/$MODEL_NAME"

echo "================================="
echo "Building TensorRT-LLM for $MODEL_NAME"
echo "================================="

mkdir -p "$CKPT_DIR"
mkdir -p "$ENGINE_DIR"

########################################
# Step 1: Convert HF → TRT checkpoint
########################################

if [ ! -f "$CKPT_DIR/config.json" ]; then

    echo "Converting HF checkpoint → TRT checkpoint"

    python3 -u \
    /usr/local/lib/python3.12/dist-packages/tensorrt_llm/examples/gemma/convert_checkpoint.py \
        --model_dir "$HF_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype bfloat16

else

    echo "Checkpoint already exists"

fi

########################################
# Step 2: Build TRT engine
###########################
```
