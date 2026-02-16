#!/bin/bash
set -e

echo "================================="
echo "Setting up benchmark environments"
echo "================================="

# Install Python 3.10 if not present
if ! command -v python3.10 &> /dev/null; then
    echo "Installing Python 3.10..."
    apt update
    apt install -y python3.10 python3.10-venv
fi

# HF venv
if [ ! -d "/workspace/.venv_hf" ]; then
    echo "Creating HF venv..."
    python3.10 -m venv /workspace/.venv_hf
    source /workspace/.venv_hf/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install transformers
    deactivate
fi

# vLLM venv
if [ ! -d "/workspace/.venv_vllm" ]; then
    echo "Creating vLLM venv..."
    python3.10 -m venv /workspace/.venv_vllm
    source /workspace/.venv_vllm/bin/activate
    pip install --upgrade pip
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    pip install vllm transformers
    deactivate
fi

echo "Environment setup complete."
