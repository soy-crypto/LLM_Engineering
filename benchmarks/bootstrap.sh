#!/bin/bash
set -e

echo "================================="
echo "Setting up benchmark environments"
echo "================================="

PROJECT="/workspace/LLM_Engineering"
PYTHON_BIN="python3"

# HF venv
if [ ! -d "$PROJECT/.venv_hf" ]; then
    echo "Creating HF venv..."
    $PYTHON_BIN -m venv $PROJECT/.venv_hf
    source $PROJECT/.venv_hf/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers
    deactivate
fi

# vLLM venv
if [ ! -d "$PROJECT/.venv_vllm" ]; then
    echo "Creating vLLM venv..."
    $PYTHON_BIN -m venv $PROJECT/.venv_vllm
    source $PROJECT/.venv_vllm/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install vllm transformers
    deactivate
fi

echo "Environment setup complete."
