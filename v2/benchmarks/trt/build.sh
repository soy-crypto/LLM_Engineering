#!/bin/bash
set -euo pipefail

########################################
# Defaults
########################################

DTYPE="bfloat16"
MAX_BATCH_SIZE=16
MAX_SEQ_LEN=4096

########################################
# Parse args
########################################

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_id)       MODEL_ID="$2";       shift 2 ;;
        --engine_dir)     ENGINE_DIR="$2";     shift 2 ;;
        --dtype)          DTYPE="$2";          shift 2 ;;
        --max_batch_size) MAX_BATCH_SIZE="$2"; shift 2 ;;
        --max_seq_len)    MAX_SEQ_LEN="$2";    shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL_ID:-}" || -z "${ENGINE_DIR:-}" ]]; then
    echo "Usage: $0 --model_id <id> --engine_dir <dir> [--dtype bfloat16] [--max_batch_size 16] [--max_seq_len 4096]"
    exit 1
fi

########################################
# Validate environment
########################################

command -v trtllm-build >/dev/null || { echo "ERROR: Must run inside TensorRT-LLM container"; exit 1; }
python3 -c "import tensorrt_llm" 2>/dev/null || { echo "ERROR: tensorrt_llm Python module not available"; exit 1; }

########################################
# Derive paths and converter from model family
########################################

MODEL_SLUG=$(echo "$MODEL_ID" | awk -F/ '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '_' | sed 's/_*$//')
MODEL_DIR="/workspace/hf_models/${MODEL_SLUG}"
CKPT_DIR="/workspace/trt_ckpt/${MODEL_SLUG}_${DTYPE}_1gpu"

MODEL_ID_LOWER=$(echo "$MODEL_ID" | tr '[:upper:]' '[:lower:]')
if   echo "$MODEL_ID_LOWER" | grep -qE "llama|mistral"; then FAMILY="llama"
elif echo "$MODEL_ID_LOWER" | grep -q "gemma";           then FAMILY="gemma"
elif echo "$MODEL_ID_LOWER" | grep -q "phi";             then FAMILY="phi"
elif echo "$MODEL_ID_LOWER" | grep -q "qwen";            then FAMILY="qwen"
else echo "ERROR: Unsupported model family for $MODEL_ID"; exit 1
fi

CONVERT_SCRIPT="/app/tensorrt_llm/examples/models/core/${FAMILY}/convert_checkpoint.py"

echo "========================================"
echo "TRT Build: $MODEL_ID"
echo "  family:  $FAMILY"
echo "  dtype:   $DTYPE"
echo "  engine:  $ENGINE_DIR"
echo "========================================"

########################################
# Download HF model
########################################

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "Downloading $MODEL_ID..."
    mkdir -p "$MODEL_DIR"
    huggingface-cli download "$MODEL_ID" \
        --local-dir "$MODEL_DIR" \
        --local-dir-use-symlinks False
fi
echo "HF model ready: $MODEL_DIR"

########################################
# Convert checkpoint
########################################

if [ ! -f "$CKPT_DIR/config.json" ]; then
    echo "Converting checkpoint..."
    mkdir -p "$CKPT_DIR"
    python3 "$CONVERT_SCRIPT" \
        --model_dir "$MODEL_DIR" \
        --output_dir "$CKPT_DIR" \
        --dtype "$DTYPE"
fi
echo "Checkpoint ready: $CKPT_DIR"

########################################
# Build engine
########################################

if [ ! -f "$ENGINE_DIR/rank0.engine" ]; then
    echo "Building engine..."
    mkdir -p "$ENGINE_DIR"
    trtllm-build \
        --checkpoint_dir "$CKPT_DIR" \
        --output_dir "$ENGINE_DIR" \
        --max_batch_size "$MAX_BATCH_SIZE" \
        --max_seq_len "$MAX_SEQ_LEN" \
        --kv_cache_type paged \
        --gemm_plugin "$DTYPE" \
        --gpt_attention_plugin "$DTYPE" \
        --context_fmha enable \
        --use_paged_context_fmha enable \
        --remove_input_padding enable
fi
echo "Engine ready: $ENGINE_DIR"

echo "DONE"
