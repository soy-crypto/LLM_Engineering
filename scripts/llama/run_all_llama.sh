#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"

RESULTS_DIR="$PROJECT/results"

echo "================================="
echo "Running All Backends"
echo "Workspace: $WORKSPACE"
echo "Project: $PROJECT"
echo "================================="

# Ensure results directory exists
mkdir -p "$RESULTS_DIR"

########################################
# Validate required artifacts
########################################

MODEL_DIR="$WORKSPACE/hf_models/llama3_1_8b"
ENGINE_DIR="$WORKSPACE/trt_engine/llama3_1_8b_bf16_b16_s4096"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

if [ ! -d "$ENGINE_DIR" ]; then
    echo "ERROR: TRT engine directory not found: $ENGINE_DIR"
    exit 1
fi

########################################
# Run HF benchmark
########################################

echo ""
echo "[HF] Running benchmark..."
bash "$PROJECT/benchmarks/hf/run_llama.sh"

########################################
# Run vLLM benchmark
########################################

echo ""
echo "[vLLM] Running benchmark..."
bash "$PROJECT/benchmarks/vllm/run_llama.sh"

########################################
# Run TensorRT-LLM benchmark
########################################

echo ""
echo "[TensorRT-LLM] Running benchmark..."
bash "$PROJECT/benchmarks/trt/run_llama.sh"

########################################
# Done
########################################

echo ""
echo "================================="
echo "All benchmarks complete."
echo "Results stored in:"
echo "$RESULTS_DIR"
echo "================================="