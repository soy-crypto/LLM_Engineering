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
# Validate benchmark scripts exist
########################################

HF_SCRIPT="$PROJECT/benchmarks/hf/run_hf.sh"
VLLM_SCRIPT="$PROJECT/benchmarks/vllm/run_vllm.sh"
TRT_SCRIPT="$PROJECT/benchmarks/trt/run_trt.sh"

if [ ! -f "$HF_SCRIPT" ]; then
    echo "ERROR: HF benchmark script not found: $HF_SCRIPT"
    exit 1
fi

if [ ! -f "$VLLM_SCRIPT" ]; then
    echo "ERROR: vLLM benchmark script not found: $VLLM_SCRIPT"
    exit 1
fi

if [ ! -f "$TRT_SCRIPT" ]; then
    echo "ERROR: TRT benchmark script not found: $TRT_SCRIPT"
    exit 1
fi

########################################
# Run benchmarks
########################################

echo ""
echo "[HF] Running..."
bash "$HF_SCRIPT"

echo ""
echo "[vLLM] Running..."
bash "$VLLM_SCRIPT"

echo ""
echo "[TensorRT-LLM] Running..."
bash "$TRT_SCRIPT"

########################################
# Done
########################################

echo ""
echo "================================="
echo "All benchmarks complete."
echo "Results located at:"
echo "$RESULTS_DIR"
echo "================================="