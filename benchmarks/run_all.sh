#!/bin/bash
set -e

./benchmarks/hf/run_hf.sh
./benchmarks/vllm/run_vllm.sh
./benchmarks/trt/run_trt.sh

echo "All backends finished successfully."
