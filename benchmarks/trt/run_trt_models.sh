#!/bin/bash
set -euo pipefail

WORKSPACE="/workspace"
PROJECT="$WORKSPACE/LLM_Engineering"
RESULTS_DIR="$PROJECT/results"

PROMPTS="$PROJECT/prompts/prompts_mid.txt"

BATCH_SIZES="1,2,4,8,16"
MAX_NEW_TOKENS=512

mkdir -p "$RESULTS_DIR"

echo "================================="
echo "Running TensorRT-LLM benchmarks (ALL MODELS)"
echo "================================="

########################################
# Validate prompts
########################################

if [ ! -f "$PROMPTS" ]; then
    echo "ERROR: Prompts file not found at $PROMPTS"
    exit 1
fi

########################################
# Validate TensorRT-LLM environment
########################################

if ! command -v trtllm-build >/dev/null; then
    echo "ERROR: Must run inside NVIDIA TensorRT-LLM container."
    exit 1
fi

if ! python3 -c "import tensorrt_llm" 2>/dev/null; then
    echo "ERROR: tensorrt_llm Python module not available."
    exit 1
fi

########################################
# Helper function
########################################

run_model () {

    MODEL_NAME="$1"
    ENGINE_DIR="$2"
    MODEL_ID="$3"
    OUT_CSV="$4"

    echo "---------------------------------"
    echo "Running TensorRT-LLM benchmark ($MODEL_NAME)"
    echo "---------------------------------"

    if [ ! -d "$ENGINE_DIR" ]; then
        echo "WARNING: Skipping $MODEL_NAME (engine not found)"
        return
    fi

    python "$PROJECT/benchmarks/trt/bm_trtllm.py" \
      --engine_dir "$ENGINE_DIR" \
      --model_id "$MODEL_ID" \
      --prompts "$PROMPTS" \
      --batch_size "$BATCH_SIZES" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --out_csv "$OUT_CSV" \
      --backend "TensorRT-LLM"

    echo "Done: $MODEL_NAME"
    echo ""
}

########################################
# Engine directories (edit if needed)
########################################

LLAMA_ENGINE="$WORKSPACE/trt_engine/llama3_1_8b_bf16_b16_s4096"
QWEN_ENGINE="$WORKSPACE/trt_engine/qwen2_5_7b_bf16_b16_s4096"
NEMOTRON_ENGINE="$WORKSPACE/trt_engine/nemotron_3_8b_bf16_b16_s4096"
MISTRAL_ENGINE="$WORKSPACE/trt_engine/mistral_7b_bf16_b16_s4096"
MIXTRAL_ENGINE="$WORKSPACE/trt_engine/mixtral_8x7b_bf16_b16_s4096"
PHI_ENGINE="$WORKSPACE/trt_engine/phi_3_mini_bf16_b16_s4096"
GEMMA_ENGINE="$WORKSPACE/trt_engine/gemma_7b_bf16_b16_s4096"

########################################
# Run all benchmarks
########################################

run_model "LLaMA-3.1-8B" \
"$LLAMA_ENGINE" \
"meta-llama/Llama-3.1-8B" \
"$RESULTS_DIR/trt_llama3_1_results.csv"

run_model "Qwen2.5-7B" \
"$QWEN_ENGINE" \
"Qwen/Qwen2.5-7B-Instruct" \
"$RESULTS_DIR/trt_qwen2_5_7b_results.csv"

run_model "Nemotron-3-8B" \
"$NEMOTRON_ENGINE" \
"nvidia/Nemotron-3-8B-Instruct" \
"$RESULTS_DIR/trt_nemotron_3_8b_results.csv"

run_model "Mistral-7B" \
"$MISTRAL_ENGINE" \
"mistralai/Mistral-7B-Instruct-v0.3" \
"$RESULTS_DIR/trt_mistral_7b_results.csv"

run_model "Mixtral-8x7B" \
"$MIXTRAL_ENGINE" \
"mistralai/Mixtral-8x7B-Instruct-v0.1" \
"$RESULTS_DIR/trt_mixtral_8x7b_results.csv"

run_model "Phi-3-Mini" \
"$PHI_ENGINE" \
"microsoft/phi-3-mini-4k-instruct" \
"$RESULTS_DIR/trt_phi_3_mini_results.csv"

run_model "Gemma-7B" \
"$GEMMA_ENGINE" \
"google/gemma-7b-it" \
"$RESULTS_DIR/trt_gemma_7b_results.csv"

########################################

echo "================================="
echo "All TensorRT-LLM benchmarks complete."
echo "Results saved in:"
echo "$RESULTS_DIR"
echo "================================="