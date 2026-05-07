#!/bin/bash
# Verify the full environment is ready before running benchmarks.

set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$PROJECT/v2/setup/.env"

PASS=0
FAIL=0

green()  { echo -e "\033[0;32m  OK   $1\033[0m"; }
red()    { echo -e "\033[0;31m  FAIL $1\033[0m"; FAIL=$((FAIL+1)); }
yellow() { echo -e "\033[0;33m  WARN $1\033[0m"; }

echo "========================================"
echo "Environment Check"
echo "========================================"

# -------------------------------------------------------
# .env file
# -------------------------------------------------------
echo ""
echo "[ .env ]"
if [ -f "$ENV_FILE" ]; then
    green ".env file exists"
    source "$ENV_FILE"
else
    red ".env not found — copy env/.env.template to env/.env"
fi

# -------------------------------------------------------
# NVIDIA driver
# -------------------------------------------------------
echo ""
echo "[ GPU ]"
if command -v nvidia-smi &>/dev/null; then
    green "nvidia-smi found"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while read line; do
        echo "       $line"
    done
else
    red "nvidia-smi not found — NVIDIA driver not installed"
fi

# -------------------------------------------------------
# CUDA
# -------------------------------------------------------
echo ""
echo "[ CUDA ]"
if command -v nvcc &>/dev/null; then
    green "nvcc: $(nvcc --version | grep release | awk '{print $5}' | tr -d ',')"
else
    yellow "nvcc not in PATH (may still work if torch has CUDA)"
fi

# -------------------------------------------------------
# Docker
# -------------------------------------------------------
echo ""
echo "[ Docker ]"
if command -v docker &>/dev/null; then
    green "docker: $(docker --version)"
    if docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        green "GPU accessible inside Docker"
    else
        red "GPU not accessible inside Docker — check nvidia-container-toolkit"
    fi
else
    yellow "docker not found (only needed for TRT-LLM)"
fi

# -------------------------------------------------------
# HuggingFace token
# -------------------------------------------------------
echo ""
echo "[ HuggingFace ]"
HF_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}"
if [ -n "$HF_TOKEN" ]; then
    green "HUGGINGFACE_HUB_TOKEN is set"
else
    red "HUGGINGFACE_HUB_TOKEN not set — required for downloading gated models"
fi

# -------------------------------------------------------
# Python venvs
# -------------------------------------------------------
echo ""
echo "[ Python environments ]"

check_venv() {
    local NAME="$1"
    local VENV="$2"
    if [ -f "$VENV/bin/python" ]; then
        green "$NAME venv: $VENV"
    else
        red "$NAME venv not found: $VENV — run scripts/bootstrap/bootstrap_${NAME}.sh"
    fi
}

check_venv "hf"      "${VENV_HF:-/workspace/.venv_hf}"
check_venv "vllm"    "${VENV_VLLM:-/workspace/.venv_vllm}"
check_venv "serving" "${VENV_SERVING:-/workspace/.venv_serving}"

# -------------------------------------------------------
# Downloaded models
# -------------------------------------------------------
echo ""
echo "[ Models ]"
MODELS_DIR="${HF_MODELS_DIR:-/workspace/hf_models}"
CONFIG="$PROJECT/scripts/config/models.conf"

if [ ! -f "$CONFIG" ]; then
    red "models.conf not found: $CONFIG"
else
    while IFS="|" read -r NAME MODEL_ID; do
        [[ -z "$NAME" || "$NAME" =~ ^# ]] && continue
        MODEL_PATH="$MODELS_DIR/$NAME"
        if [ -f "$MODEL_PATH/config.json" ]; then
            green "model: $NAME"
        else
            red "model missing: $NAME ($MODEL_PATH) — run scripts/download_models.sh"
        fi
    done < "$CONFIG"
fi

# -------------------------------------------------------
# TRT engines
# -------------------------------------------------------
echo ""
echo "[ TRT Engines ]"
TRT_DIR="${TRT_ENGINE_DIR:-/workspace/trt_engine}"
if [ -d "$TRT_DIR" ]; then
    ENGINE_COUNT=$(find "$TRT_DIR" -name "*.engine" 2>/dev/null | wc -l)
    if [ "$ENGINE_COUNT" -gt 0 ]; then
        green "$ENGINE_COUNT engine file(s) found in $TRT_DIR"
    else
        yellow "No .engine files found — run scripts/bootstrap/trtllm/bootstrap_all_trt.sh inside the TRT container"
    fi
else
    yellow "TRT engine dir not found: $TRT_DIR"
fi

# -------------------------------------------------------
# Results dir
# -------------------------------------------------------
echo ""
echo "[ Results ]"
RESULTS="${RESULTS_DIR:-$PROJECT/results}"
if [ -d "$RESULTS" ]; then
    CSV_COUNT=$(find "$RESULTS" -maxdepth 1 -name "*.csv" 2>/dev/null | wc -l)
    green "results dir exists ($CSV_COUNT CSV files)"
else
    yellow "results dir not found — will be created on first run"
fi

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "========================================"
if [ "$FAIL" -eq 0 ]; then
    echo -e "\033[0;32mAll checks passed. Environment is ready.\033[0m"
else
    echo -e "\033[0;31m$FAIL check(s) failed. Fix issues above before running benchmarks.\033[0m"
fi
echo "========================================"
exit $FAIL
