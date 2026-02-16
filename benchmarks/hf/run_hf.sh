#!/bin/bash
set -e

source /workspace/.venv_hf/bin/activate


python /workspace/LLM_Engineering/benchmarks/hf/bm_hf.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompts /workspace/LLM_Engineering/prompts/prompts_mid.txt \
  --batch_size 1,2,4,8,16 \
  --max_new_tokens 512 \
  --dtype bfloat16 \
  --runs 3 \
  --out_csv /workspace/results/hf_results.csv

deactivate