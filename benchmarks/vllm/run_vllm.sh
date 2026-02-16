#!/bin/bash
set -e

MODEL="Qwen/Qwen2.5-7B-Instruct"
PROMPTS="prompts/prompts_mid.txt"
BATCH="1,2,4,8,16"
MAX_NEW=512

echo "================================="
echo "Running vLLM Benchmark"
echo "================================="

source .venv_vllm/bin/activate

python benchmarks/vllm/bm_vllm.py \
  --model $MODEL \
  --prompts $PROMPTS \
  --batch_size $BATCH \
  --max_new_tokens $MAX_NEW \
  --dtype bfloat16 \
  --runs 3 \
  --out_csv results/vllm_results.csv

deactivate

echo "vLLM benchmark completed."
