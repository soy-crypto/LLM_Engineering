#!/bin/bash
set -e

PROJECT="/workspace/LLM_Engineering"

echo "================================="
echo "Running All Backends"
echo "================================="

bash $PROJECT/benchmarks/hf/run_hf.sh
bash $PROJECT/benchmarks/vllm/run_vllm.sh
bash $PROJECT/benchmarks/trt/run_trt.sh

echo "================================="
echo "All benchmarks complete."
echo "Results in: $PROJECT/results"
echo "================================="
