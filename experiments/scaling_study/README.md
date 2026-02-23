# LLM Inference Scaling Study

## Objective

Measure scaling behavior of vLLM inference as batch size increases.

## Fixed Parameters

- Context length: 512
- Decode length: 128
- Model: Meta-Llama-3-8B
- Single GPU

## Metrics Collected

- Total latency (ms)
- Tokens/sec
- GPU memory usage (MB)

## How to Run

```bash
cd experiments/scaling_study
./run_scaling.sh