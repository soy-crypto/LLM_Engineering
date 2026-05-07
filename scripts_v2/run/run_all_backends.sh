#!/bin/bash
set -e

echo "Running HF benchmarks"
./scripts_v2/hf/run_all_hf.sh

echo "Running vLLM benchmarks"
./scripts_v2/vllm/run_all_vllm.sh

echo "Running TensorRT benchmarks"
./scripts_v2/trt/run_all_trt.sh

echo "Aggregating results"
python benchmarks_v2/analyze.py