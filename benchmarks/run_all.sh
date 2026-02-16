#!/bin/bash
set -e

/workspace/LLM_Engineeringbenchmarks/hf/run_hf.sh
/workspace/LLM_Engineeringbenchmarks/vllm/run_vllm.sh
/workspace/LLM_Engineeringbenchmarks/trt/run_trt.sh
