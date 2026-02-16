#!/bin/bash
set -e

echo "================================="
echo "Running TensorRT-LLM Benchmark"
echo "================================="

docker run --gpus all --rm \
  -v $PWD:/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3 \
  python benchmarks/trt/bm_trtllm.py

echo "TensorRT benchmark completed."