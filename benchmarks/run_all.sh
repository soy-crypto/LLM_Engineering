#!/bin/bash

set -e  # Stop on first error

echo "======================================"
echo "Starting Full Benchmark Suite"
echo "======================================"

./benchmarks/hf/run_hf.sh
./benchmarks/vllm/run_vllm.sh
./benchmarks/trt/run_trt.sh

echo ""
echo "Aggregating results..."
python benchmarks/aggregate.py

echo ""
echo "Generating plots..."
python benchmarks/plot_results.py

echo ""
echo "All benchmarks completed successfully."
