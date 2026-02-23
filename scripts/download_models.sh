#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
MODELS_DIR="$WORKSPACE/hf_models"
CONFIG_FILE="$WORKSPACE/LLM_Engineering/scripts/config/models.conf"

HF_VENV="$WORKSPACE/.venv_hf"

echo "================================="
echo "Downloading models from config"
echo "================================="

mkdir -p "$MODELS_DIR"

########################################
# Ensure HF venv exists
########################################

if [ ! -d "$HF_VENV" ]; then

    python3 -m venv "$HF_VENV"

    source "$HF_VENV/bin/activate"

    pip install --upgrade pip
    pip install huggingface_hub

    deactivate
fi

########################################
# Activate venv
########################################

source "$HF_VENV/bin/activate"

########################################
# Check login
########################################

if ! huggingface-cli whoami &>/dev/null; then

    echo ""
    echo "ERROR: HuggingFace login required."
    echo "Run: huggingface-cli login"
    echo ""

    exit 1
fi

########################################
# Download models
########################################

while IFS="|" read -r NAME MODEL_ID
do

    MODEL_DIR="$MODELS_DIR/$NAME"

    if [ -f "$MODEL_DIR/config.json" ]; then

        echo "Skipping $NAME (already exists)"

    else

        echo ""
        echo "Downloading $NAME"
        echo "Model ID: $MODEL_ID"

        mkdir -p "$MODEL_DIR"

        huggingface-cli download "$MODEL_ID" \
            --local-dir "$MODEL_DIR" \
            --local-dir-use-symlinks False

        echo "Done: $NAME"
    fi

done < "$CONFIG_FILE"

deactivate

echo ""
echo "All models ready in:"
echo "$MODELS_DIR"