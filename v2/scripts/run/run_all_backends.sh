#!/bin/bash
set -e

echo "Running HF benchmarks"
./v2/scripts/hf/run_all_hf.sh

echo "Running vLLM benchmarks"
./v2/scripts/vllm/run_all_vllm.sh

echo "Running TensorRT benchmarks"
./v2/scripts/trt/run_all_trt.sh

echo "Aggregating results"
python v2/benchmarks/analyze.py