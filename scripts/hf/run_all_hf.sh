#!/bin/bash
set -euo pipefail

# --------------------------------------------------
# Prevent interactive remote-code prompt
# --------------------------------------------------
export TRANSFORMERS_TRUST_REMOTE_CODE=1

# --------------------------------------------------
# Paths
# --------------------------------------------------
CONFIG="/workspace/LLM_Engineering/scripts/config/models.conf"
MODEL_DIR="/workspace/hf_models"
PROMPTS="/workspace/LLM_Engineering/prompts/prompts_mid.txt"
OUT="/workspace/LLM_Engineering/results/results_hf.csv"
PYTHON="/workspace/.venv_hf/bin/python"
BENCH="/workspace/LLM_Engineering/benchmarks/hf/bm_hf.py"

echo "Using config: $CONFIG"
echo ""

# --------------------------------------------------
# Validate config file
# --------------------------------------------------
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

# --------------------------------------------------
# Read config line-by-line
# Format:
# alias|huggingface_model_id
# --------------------------------------------------
while IFS="|" read -r alias hf_id || [[ -n "$alias" ]]; do

    # Skip empty lines
    [[ -z "$alias" ]] && continue

    # Skip comments
    [[ "$alias" =~ ^# ]] && continue

    model_path="$MODEL_DIR/$alias"

    echo "========================================"
    echo "Running HuggingFace benchmark: $alias"
    echo "HF model id: $hf_id"
    echo "Local path: $model_path"
    echo "========================================"

    # --------------------------------------------------
    # Download model if missing
    # --------------------------------------------------
    if [ ! -d "$model_path" ]; then

        echo "Model not found locally. Downloading..."

        "$PYTHON" - <<EOF
from transformers import AutoTokenizer, AutoModelForCausalLM

hf_id = "$hf_id"
model_path = "$model_path"

print("Downloading:", hf_id)

tokenizer = AutoTokenizer.from_pretrained(
    hf_id,
    trust_remote_code=True
)
tokenizer.save_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    hf_id,
    torch_dtype="auto",
    device_map="cpu",
    trust_remote_code=True
)
model.save_pretrained(model_path)

print("Download complete:", hf_id)
EOF

    fi

    # --------------------------------------------------
    # Run benchmark
    # --------------------------------------------------
    "$PYTHON" "$BENCH" \
        --model "$model_path" \
        --prompts "$PROMPTS" \
        --out_csv "$OUT" \
        --backend HF

    echo "Completed: $alias"
    echo ""

    # --------------------------------------------------
    # GPU cleanup safety
    # --------------------------------------------------
    sleep 2

done < "$CONFIG"

echo "========================================"
echo "All HF benchmarks completed"
echo "Results saved to: $OUT"
echo "========================================"