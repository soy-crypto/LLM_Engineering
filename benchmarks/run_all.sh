#!/bin/bash

MODEL="Qwen/Qwen2.5-7B-Instruct"
PROMPTS="prompts/prompts_mid.txt"
BATCH="1,2,4,8,16"
MAX_NEW=512

echo "Running HuggingFace benchmark..."
python benchmarks/bm_hf.py \
  --model $MODEL \
  --prompts $PROMPTS \
  --batch_size $BATCH \
  --max_new_tokens $MAX_NEW \
  --dtype bfloat16 \
  --runs 3 \
  --out_csv results/hf_results.csv

echo "Running vLLM benchmark..."
python benchmarks/bm_vllm.py \
  --model $MODEL \
  --prompts $PROMPTS \
  --batch_size $BATCH \
  --max_new_tokens $MAX_NEW \
  --dtype bfloat16 \
  --runs 3 \
  --out_csv results/vllm_results.csv

echo "Running TensorRT-LLM benchmark..."
python benchmarks/bm_trtllm.py

echo "Done."
