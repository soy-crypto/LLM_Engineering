#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"

VLLM_VENV="$WORKSPACE/.venv_vllm"
RESULTS_DIR="$PROJECT/results"

PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES=(1 2 4 8)
MAX_NEW_TOKENS=512
DTYPE="bfloat16"

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running vLLM Scaling Study (ALL MODELS)"
echo "================================="

#######################################
# Validate prompts
#######################################

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: Prompts file not found at $PROMPTS"
    exit 1
fi

#######################################
# Ensure venv exists
#######################################

if [ ! -d "$VLLM_VENV" ]; then
    echo "ERROR: vLLM venv not found at $VLLM_VENV"
    exit 1
fi

#######################################
# Activate vLLM environment
#######################################

source "$VLLM_VENV/bin/activate"

#######################################
# Helper function
#######################################

run_model () {

    MODEL_NAME="$1"
    MODEL_PATH="$2"
    OUTPUT="$3"

    echo ""
    echo "================================="
    echo "Running vLLM Scaling Study ($MODEL_NAME)"
    echo "================================="

    if [ ! -d "$MODEL_PATH" ]; then
        echo "WARNING: Skipping $MODEL_NAME (model not found)"
        return
    fi

    echo "backend,batch_size,max_new_tokens,total_latency_ms,tokens_per_sec,gpu_mem_mb" > "$OUTPUT"

    for B in "${BATCH_SIZES[@]}"
    do
        echo "Batch size $B..."

        python "$PROJECT/benchmarks/vllm/bm_vllm.py" \
          --model "$MODEL_PATH" \
          --prompts "$PROMPTS" \
          --batch_size "$B" \
          --max_new_tokens "$MAX_NEW_TOKENS" \
          --dtype "$DTYPE" \
          --append_csv "$OUTPUT" \
          --backend vLLM
    done

    echo "Done: $MODEL_NAME"
    echo "Saved: $OUTPUT"
}

#######################################
# Model paths
#######################################

LLAMA_PATH="$WORKSPACE/hf_models/llama3_1_8b"
QWEN_PATH="$WORKSPACE/hf_models/qwen2_5_7b"
NEMOTRON_PATH="$WORKSPACE/hf_models/nemotron_3_8b"
MISTRAL_PATH="$WORKSPACE/hf_models/mistral_7b"
MIXTRAL_PATH="$WORKSPACE/hf_models/mixtral_8x7b"
PHI_PATH="$WORKSPACE/hf_models/phi_3_mini"
GEMMA_PATH="$WORKSPACE/hf_models/gemma_7b"

#######################################
# Run all models
#######################################

run_model "LLaMA-3.1-8B" \
"$LLAMA_PATH" \
"$RESULTS_DIR/vllm_llama3_1_scaling.csv"

run_model "Qwen2.5-7B" \
"$QWEN_PATH" \
"$RESULTS_DIR/vllm_qwen2_5_7b_scaling.csv"

run_model "Nemotron-3-8B" \
"$NEMOTRON_PATH" \
"$RESULTS_DIR/vllm_nemotron_3_8b_scaling.csv"

run_model "Mistral-7B" \
"$MISTRAL_PATH" \
"$RESULTS_DIR/vllm_mistral_7b_scaling.csv"

run_model "Mixtral-8x7B" \
"$MIXTRAL_PATH" \
"$RESULTS_DIR/vllm_mixtral_8x7b_scaling.csv"

run_model "Phi-3-Mini" \
"$PHI_PATH" \
"$RESULTS_DIR/vllm_phi_3_mini_scaling.csv"

run_model "Gemma-7B" \
"$GEMMA_PATH" \
"$RESULTS_DIR/vllm_gemma_7b_scaling.csv"

#######################################

deactivate

echo ""
echo "================================="
echo "All vLLM scaling studies complete."
echo "Results saved in:"
echo "$RESULTS_DIR"
echo "================================="