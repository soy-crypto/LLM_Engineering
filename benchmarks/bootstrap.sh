#!/bin/bash
set -e

echo "================================="
echo "Setting up benchmark environments"
echo "================================="

# Use system Python (3.12) for venv
PYTHON_BIN="python3"

# HF venv
if [ ! -d "/workspace/.venv_hf" ]; then
    echo "Creating HF venv..."
    $PYTHON_BIN -m venv /workspace/.venv_hf
    source /workspace/.venv_hf/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers
    deactivate
fi

# vLLM venv
if [ ! -d "/workspace/.venv_vllm" ]; then
    echo "Creating vLLM venv..."
    $PYTHON_BIN -m venv /workspace/.venv_vllm
    source /workspace/.venv_vllm/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install vllm transformers
    deactivate
fi

echo "Environment setup complete."
