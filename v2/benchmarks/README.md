# Benchmarks V2

Refactored benchmarking suite for HuggingFace, vLLM, and TensorRT-LLM backends.

---

## What Changed from V1

| Issue | V1 | V2 |
|---|---|---|
| Shared utilities | Copy-pasted across files | Single `common.py` |
| CSV column names | Different per backend | Unified schema |
| Results path | Hardcoded `/workspace/...` | Relative to project root |
| GPU memory in TRT | Not measured | Added |
| CSV header in TRT | Written by shell script | Handled by `CsvWriter` |

---

## Structure

```
v2/benchmarks/
├── common.py          # Shared utilities: load_prompts, CsvWriter, etc.
├── hf/
│   └── benchmark.py  # HuggingFace benchmark
├── vllm/
│   └── benchmark.py  # vLLM benchmark
├── trt/
│   └── benchmark.py  # TensorRT-LLM benchmark
└── analyze.py        # Aggregate results and generate plots
```

---

## Unified CSV Schema

All three backends write the same columns:

| Column | Description |
|---|---|
| `backend` | HF / vLLM / TensorRT-LLM |
| `model` | Model name or path |
| `dtype` | Precision (bfloat16, float16, float32) |
| `batch_size` | Number of concurrent requests |
| `avg_ttft_s` | Average time to first token (seconds) |
| `avg_latency_s` | Average total generation latency (seconds) |
| `tokps_new` | New tokens generated per second |
| `gpu_mem_mb` | GPU memory allocated at end of run (MB) |

---

## Running Benchmarks

### HuggingFace

```bash
python v2/benchmarks/hf/benchmark.py \
  --model /workspace/hf_models/llama3_1_8b \
  --prompts prompts/prompts_mid.txt \
  --batch_size 1,2,4,8 \
  --max_new_tokens 512 \
  --dtype bfloat16 \
  --out_csv results/hf_llama.csv
```

### vLLM

```bash
python v2/benchmarks/vllm/benchmark.py \
  --model /workspace/hf_models/llama3_1_8b \
  --prompts prompts/prompts_mid.txt \
  --batch_size 1,2,4,8 \
  --max_new_tokens 512 \
  --dtype bfloat16 \
  --out_csv results/vllm_llama.csv
```

### TensorRT-LLM

Must run inside the TensorRT-LLM container.

```bash
python v2/benchmarks/trt/benchmark.py \
  --engine_dir /workspace/trt_engine/llama3_1_8b_bf16_b16_s4096 \
  --model_id meta-llama/Llama-3.1-8B \
  --prompts prompts/prompts_mid.txt \
  --batch_size 1,2,4,8,16 \
  --max_new_tokens 512 \
  --out_csv results/trt_llama.csv
```

### All Backends (via scripts)

```bash
./scripts/run/run_all_backends.sh
```

---

## Analyzing Results

```bash
python v2/benchmarks/analyze.py
```

Reads all `*.csv` files from `results/`, produces:

- `results/aggregate/aggregate_results.csv` — combined table
- `results/aggregate/<model>_tokps_new.png` — throughput plots
- `results/aggregate/<model>_avg_latency_s.png` — latency plots
- `results/aggregate/<model>_avg_ttft_s.png` — TTFT plots
- Summary printed to stdout
