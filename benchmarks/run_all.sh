#!/bin/bash
set -e

#/workspace/LLM_Engineering/benchmarks/hf/run_hf.sh
/workspace/LLM_Engineering/benchmarks/vllm/run_vllm.sh
/workspace/LLM_Engineering/benchmarks/trt/run_trt.sh
