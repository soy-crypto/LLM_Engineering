#!/bin/bash
set -e

echo "Running HF benchmarks"
./scripts/run/run_all_hf.sh

echo "Running vLLM benchmarks"
./scripts/run/run_all_vllm.sh

echo "Running TensorRT benchmarks"
./scripts/run/run_all_trt.sh

echo "Aggregating results"
python analysis/aggregate.py