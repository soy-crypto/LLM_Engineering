#!/bin/bash

set -e

# Change this if needed
BASE_DIR=/workspace/hf_models

echo "==================================="
echo "Creating model directory..."
echo "==================================="

mkdir -p $BASE_DIR
cd $BASE_DIR

# Check huggingface-cli
if ! command -v huggingface-cli &> /dev/null
then
    echo "Installing huggingface_hub..."
    pip install -U huggingface_hub
fi

# Function to download safely
download_model () {
    MODEL_NAME=$1
    LOCAL_DIR=$2

    if [ -d "$LOCAL_DIR" ]; then
        echo "Skipping $MODEL_NAME (already exists)"
    else
        echo "Downloading $MODEL_NAME..."
        huggingface-cli download $MODEL_NAME \
            --local-dir $LOCAL_DIR \
            --local-dir-use-symlinks False
    fi
}

echo "==================================="
echo "Starting downloads..."
echo "==================================="

download_model meta-llama/Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct
download_model Qwen/Qwen2.5-7B-Instruct Qwen2.5-7B-Instruct
download_model nvidia/Nemotron-3-8B-Instruct Nemotron-3-8B-Instruct
download_model mistralai/Mistral-7B-Instruct-v0.3 Mistral-7B-Instruct
download_model mistralai/Mixtral-8x7B-Instruct-v0.1 Mixtral-8x7B
download_model microsoft/phi-3-mini-4k-instruct Phi-3-mini
download_model google/gemma-7b-it Gemma-7B

echo "==================================="
echo "All models ready in $BASE_DIR"
echo "==================================="
