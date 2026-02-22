
#!/bin/bash
set -e

PROJECT="/workspace/LLM_Engineering"

echo "================================="
echo "Running All Backends"
echo "================================="

bash $PROJECT/benchmarks/hf/run_llama.sh
bash $PROJECT/benchmarks/vllm/run_llama.sh
bash $PROJECT/benchmarks/trt/run_llama.sh

echo "================================="
echo "All benchmarks complete."
echo "Results in: $PROJECT/results"
echo "================================="
