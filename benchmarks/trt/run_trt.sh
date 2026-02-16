#!/bin/bash

echo "Running TensorRT-LLM benchmark..."

docker run --gpus all --rm \
  -v $PWD:/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3 \
  python benchmarks/trt/bm_trtllm.py