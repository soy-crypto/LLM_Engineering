#!/bin/bash
set -euo pipefail

echo "================================="
echo "TensorRT-LLM Full Build Pipeline"
echo "GPU: L40S"
echo "================================="

BASE_DIR="/workspace/LLM_Engineering/scripts/trt"

echo ""
echo "Step 1: LLaMA 3.1 8B"
bash --noprofile --norc "$BASE_DIR/build_llama3_1_8b.sh"

echo ""
echo "Step 2: Mistral 7B"
bash --noprofile --norc "$BASE_DIR/build_mistral_7b.sh"

echo ""
echo "Step 3: Gemma 7B"
bash --noprofile --norc "$BASE_DIR/build_gemma_7b.sh"

echo ""
echo "Step 4: Qwen 2.5 7B"
bash --noprofile --norc "$BASE_DIR/build_qwen2_5_7b.sh"

echo ""
echo "Step 5: Phi-3 Mini"
bash --noprofile --norc "$BASE_DIR/build_phi_3_mini.sh"

echo ""
echo "Step 6: DeepSeek MoE 16B"
bash --noprofile --norc "$BASE_DIR/build_deepseek_moe_16b.sh"

echo ""
echo "================================="
echo "All TensorRT engines built successfully"
echo "================================="