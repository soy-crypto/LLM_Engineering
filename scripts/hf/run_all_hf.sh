#!/bin/bash
set -euo pipefail

CONFIG="/workspace/LLM_Engineering/scripts/config/models.conf"
MODEL_DIR="/workspace/hf_models"
PROMPTS="/workspace/LLM_Engineering/prompts/prompts_mid.txt"
OUT="/workspace/LLM_Engineering/results/results_hf.csv"
PYTHON="/workspace/.venv_hf/bin/python"
BENCH="/workspace/LLM_Engineering/benchmarks/hf/bm_hf.py"

echo "Using config: $CONFIG"
echo ""

while IFS= read -r line || [[ -n "$line" ]]; do

    # skip empty or comment
    [[ -z "$line" ]] && continue
    [[ "$line" =~ ^# ]] && continue

    ########################################
    # split alias and HF model id
    ########################################

    alias="${line%%|*}"
    hf_id="${line##*|}"

    model_path="$MODEL_DIR/$alias"

    echo "========================================"
    echo "Running HuggingFace benchmark: $alias"
    echo "HF model id: $hf_id"
    echo "Local path: $model_path"
    echo "========================================"

    ########################################
    # download if missing
    ########################################

    if [ ! -d "$model_path" ]; then

        echo "Downloading model..."

        "$PYTHON" - <<EOF
from transformers import AutoModelForCausalLM, AutoTokenizer

model = "$hf_id"
path = "$model_path"

AutoTokenizer.from_pretrained(model).save_pretrained(path)
AutoModelForCausalLM.from_pretrained(
    model,
    torch_dtype="auto",
    device_map="cpu"
).save_pretrained(path)

print("Downloaded:", model)
EOF

    fi

    ########################################
    # run benchmark
    ########################################

    "$PYTHON" "$BENCH" \
        --model "$model_path" \
        --prompts "$PROMPTS" \
        --out_csv "$OUT" \
        --backend HF

    echo "Completed: $alias"
    echo ""

    sleep 2

done < "$CONFIG"

echo "========================================"
echo "All HF benchmarks completed"
echo "========================================"