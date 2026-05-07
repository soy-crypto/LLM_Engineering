#!/bin/bash
set -euo pipefail

echo "========================================"
echo "TensorRT-LLM: Build All Models"
echo "========================================"

BASE_DIR="/workspace/LLM_Engineering/scripts_v2/bootstrap/trtllm"

run_model () {
    local SCRIPT="$1"

    echo ""
    echo "----------------------------------------"
    echo "Running $SCRIPT"
    echo "----------------------------------------"

    bash "$BASE_DIR/$SCRIPT"

    echo "Finished $SCRIPT"
}

run_model "bootstrap_llama3_1_8b.sh"
run_model "bootstrap_mistral_7b.sh"
run_model "bootstrap_gemma_7b.sh"
run_model "bootstrap_qwen2_5_7b.sh"
run_model "bootstrap_phi_3_mini.sh"
run_model "bootstrap_deepseek_moe_16b.sh"

echo ""
echo "========================================"
echo "All TRT engines built."
echo "========================================"