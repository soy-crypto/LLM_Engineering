#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"
RESULTS_DIR="$PROJECT/results"
CONFIG_FILE="$PROJECT/scripts/config/models.conf"

PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8"
MAX_NEW_TOKENS=512
DTYPE="bfloat16"

mkdir -p "$RESULTS_DIR"

########################################
# Validate paths
########################################

if [ ! -f "$PROMPTS" ]; then
  echo "ERROR: Prompts file not found at $PROMPTS"
  exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: Config file not found at $CONFIG_FILE"
  exit 1
fi

########################################
# Activate HF environment
########################################

source "$WORKSPACE/.venv_hf/bin/activate"

pip install -q accelerate

########################################
# Run model function
########################################

run_model () {

  local NAME="$1"
  local MODEL_PATH="$2"
  local OUT_CSV="$3"

  echo "================================="
  echo "Running HuggingFace benchmark ($NAME)"
  echo "================================="

  if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
  fi

  python "$PROJECT/benchmarks/hf/bm_hf.py" \
    --model "$MODEL_PATH" \
    --prompts "$PROMPTS" \
    --batch_size "$BATCH_SIZES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --dtype "$DTYPE" \
    --out_csv "$OUT_CSV" \
    --backend HF

  echo "Done: $NAME"
}

########################################
# Loop through models.conf
########################################

while IFS="|" read -r NAME MODEL_ID
do

  MODEL_PATH="$WORKSPACE/hf_models/$NAME"
  OUT_CSV="$RESULTS_DIR/hf_${NAME}_results.csv"

  run_model "$NAME" "$MODEL_PATH" "$OUT_CSV"

done < "$CONFIG_FILE"

deactivate

echo "================================="
echo "All HF benchmarks complete."
echo "Results directory: $RESULTS_DIR"
echo "================================="