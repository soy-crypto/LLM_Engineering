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

PYTHON="$WORKSPACE/.venv_hf/bin/python"
PIP="$WORKSPACE/.venv_hf/bin/pip"

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

if [ ! -x "$PYTHON" ]; then
  echo "ERROR: Python venv not found at $PYTHON"
  exit 1
fi

########################################
# Ensure required packages exist
########################################

echo "Verifying HF environment..."

$PIP install -q accelerate
$PIP install -q mamba-ssm causal-conv1d --no-build-isolation

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

  $PYTHON "$PROJECT/benchmarks/hf/bm_hf.py" \
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
  [ -z "$NAME" ] && continue
  [[ "$NAME" =~ ^# ]] && continue

  MODEL_PATH="$WORKSPACE/hf_models/$NAME"
  OUT_CSV="$RESULTS_DIR/hf_${NAME}_results.csv"

  run_model "$NAME" "$MODEL_PATH" "$OUT_CSV"

done < "$CONFIG_FILE"

echo "================================="
echo "All HF benchmarks complete."
echo "Results directory: $RESULTS_DIR"
echo "================================="