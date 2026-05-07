#!/bin/bash
set -euo pipefail

export TRANSFORMERS_TRUST_REMOTE_CODE=1

CONFIG="/workspace/LLM_Engineering/scripts_v2/config/models.conf"
MODEL_DIR="/workspace/hf_models"
PROMPTS="/workspace/LLM_Engineering/prompts/prompts_mid.txt"
OUT="/workspace/LLM_Engineering/results/results_hf.csv"
PYTHON="/workspace/.venv_hf/bin/python"
BENCH="/workspace/LLM_Engineering/benchmarks_v2/hf/benchmark.py"

echo "Using config: $CONFIG"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

while IFS="|" read -r alias hf_id || [[ -n "$alias" ]]; do

    [[ -z "$alias" ]] && continue
    [[ "$alias" =~ ^# ]] && continue

    model_path="$MODEL_DIR/$alias"

    echo "========================================"
    echo "Running HuggingFace benchmark: $alias"
    echo "HF model id: $hf_id"
    echo "Local path: $model_path"
    echo "========================================"

    if [ ! -d "$model_path" ]; then

        echo "Downloading model..."

"$PYTHON" - <<EOF
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="$hf_id",
    local_dir="$model_path",
    local_dir_use_symlinks=False,
    resume_download=True,
    max_workers=8
)

print("Download complete")
EOF

    fi

    "$PYTHON" "$BENCH" \
        --model "$model_path" \
        --prompts "$PROMPTS" \
        --out_csv "$OUT" \
        --backend HF

    echo "Completed: $alias"
    echo ""

    # CRITICAL: free GPU memory
"$PYTHON" - <<EOF
import torch, gc
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
EOF

    sleep 2

done < "$CONFIG"

echo "========================================"
echo "All HF benchmarks completed"
echo "Results saved to: $OUT"
echo "========================================"