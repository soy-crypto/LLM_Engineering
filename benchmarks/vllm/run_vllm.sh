#!/bin/bash
set -e

source /workspace/LLM_Engineering/.venv_vllm/bin/activate

python /workspace/LLM_Engineering/benchmarks/vllm/bm_vllm.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --prompts /workspace/LLM_Engineering/prompts/prompts_mid.txt \
  --batch_size 1,2,4,8,16 \
  --max_new_tokens 512 \
  --runs 3 \
  --out_csv /workspace/LLM_Engineering/results/vllm_results.csv

deactivate
