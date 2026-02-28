#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
MODELS_DIR="$WORKSPACE/hf_models"
CONFIG_FILE="$WORKSPACE/LLM_Engineering/scripts/config/models.conf"

echo "================================="
echo "Downloading models from config"
echo "================================="

mkdir -p "$MODELS_DIR"

########################################
# Require token
########################################

if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
    echo "ERROR: HUGGINGFACE_HUB_TOKEN not set"
    exit 1
fi

########################################
# Download models
########################################

while IFS="|" read -r NAME MODEL_ID
do
    # Skip empty lines
    [ -z "$NAME" ] && continue

    # Skip commented lines
    [[ "$NAME" =~ ^# ]] && continue

    MODEL_DIR="$MODELS_DIR/$NAME"

    if [ -f "$MODEL_DIR/config.json" ]; then
        echo "Skipping $NAME (already exists)"
        continue
    fi

    echo ""
    echo "Downloading $NAME"
    echo "Model ID: $MODEL_ID"

    python3 - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="$MODEL_ID",
    local_dir="$MODEL_DIR",
    token="$HUGGINGFACE_HUB_TOKEN",
    local_dir_use_symlinks=False
)
EOF

    echo "Done: $NAME"

done < "$CONFIG_FILE"

echo ""
echo "All models ready in:"
echo "$MODELS_DIR"