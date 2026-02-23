#!/bin/bash
set -e

VENV="/workspace/.venv_vllm"
SCRIPT="/workspace/LLM_Engineering/experiments/scaling_study/run_vllm_scaling.py"
MODEL="/workspace/hf_models/llama3_1_8b"

source "$VENV/bin/activate"

python "$SCRIPT" \
    --model "$MODEL" \
    --batches 1,2,4,8 \
    --context 512 \
    --decode 128

deactivate