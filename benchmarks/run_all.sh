#!/bin/bash
set -e

/workspace/benchmarks/hf/run_hf.sh
/workspace/benchmarks/vllm/run_vllm.sh
/workspace/benchmarks/trt/run_trt.sh
