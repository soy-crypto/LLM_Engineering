#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"
RESULTS_DIR="$PROJECT/results"

PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8"
MAX_NEW_TOKENS=512
DTYPE="bfloat16"

mkdir -p "$RESULTS_DIR"

########################################
# Validate common paths
########################################
if [ ! -f "$PROMPTS" ]; then
  echo "ERROR: Prompts file not found at $PROMPTS"
  exit 1
fi

########################################
# Activate HF environment
########################################
source "$WORKSPACE/.venv_hf/bin/activate"

# Ensure dependency exists (safe if already installed)
pip install -q accelerate

########################################
# Helper to run one model
########################################
run_model () {
  local NAME="$1"
  local MODEL_PATH="$2"
  local OUT_CSV="$3"

  echo "================================="
  echo "Running HuggingFace benchmark ($NAME)"
  echo "================================="

  if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "ERROR: Model not found at $MODEL_PATH (missing config.json)"
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

  echo ""
  echo "Done: $NAME"
  echo "Saved: $OUT_CSV"
  echo ""
}

########################################
# Model paths (EDIT if your folder names differ)
########################################
LLAMA_PATH="$WORKSPACE/hf_models/llama3_1_8b"
QWEN_PATH="$WORKSPACE/hf_models/qwen2_5_7b"
NEMOTRON_PATH="$WORKSPACE/hf_models/nemotron_3_8b"
MISTRAL_PATH="$WORKSPACE/hf_models/mistral_7b"
MIXTRAL_PATH="$WORKSPACE/hf_models/mixtral_8x7b"
PHI_PATH="$WORKSPACE/hf_models/phi_3_mini"
GEMMA_PATH="$WORKSPACE/hf_models/gemma_7b"

########################################
# Run benchmarks (comment out what you don't want)
########################################
run_model "LLaMA-3.1-8B"          "$LLAMA_PATH"     "$RESULTS_DIR/hf_llama3_1_8b_results.csv"
run_model "Qwen2.5-7B"            "$QWEN_PATH"      "$RESULTS_DIR/hf_qwen2_5_7b_results.csv"
run_model "Nemotron-3-8B"         "$NEMOTRON_PATH"  "$RESULTS_DIR/hf_nemotron_3_8b_results.csv"
run_model "Mistral-7B"            "$MISTRAL_PATH"   "$RESULTS_DIR/hf_mistral_7b_results.csv"
run_model "Mixtral-8x7B"          "$MIXTRAL_PATH"   "$RESULTS_DIR/hf_mixtral_8x7b_results.csv"
run_model "Phi-3-Mini"            "$PHI_PATH"       "$RESULTS_DIR/hf_phi_3_mini_results.csv"
run_model "Gemma-7B"              "$GEMMA_PATH"     "$RESULTS_DIR/hf_gemma_7b_results.csv"

deactivate

echo "================================="
echo "All HF benchmarks complete."
echo "Results directory: $RESULTS_DIR"
echo "================================="