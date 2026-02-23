#!/bin/bash
set -e

MODEL="meta-llama/Meta-Llama-3-8B"   # change if needed
CONTEXT=512
DECODE=128
BATCH_SIZES=(1 2 4 8 16)

OUTPUT="results_scaling.csv"

echo "backend,batch,context,decode,total_latency_ms,tokens_per_sec,gpu_mem_mb" > $OUTPUT

for B in "${BATCH_SIZES[@]}"
do
    echo "Running batch size $B..."

    python run_vllm_scaling.py \
        --model $MODEL \
        --batch $B \
        --context $CONTEXT \
        --decode $DECODE \
        >> $OUTPUT
done

echo "Scaling study complete."
echo "Results saved to $OUTPUT"