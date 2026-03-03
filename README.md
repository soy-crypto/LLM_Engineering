Understood.

Below is a **complete, fully detailed README** that:

* Preserves **all technical details** from your original README
* Preserves the **full workflow section**
* Preserves the **container instructions**
* Preserves the **empirical tables**
* Preserves the **KV math**
* Preserves the **regime summary**
* Matches your **actual `tree` structure**
* Keeps every script and directory represented

Nothing important is removed.
Only reorganized for clarity and professionalism.

You can paste this directly as your `README.md`.

---

# 🚀 LLM Inference Systems Study

**KV Cache Dynamics · Compute vs Bandwidth Regimes · Backend Engineering**

A GPU-native investigation of decoder-only LLM inference performance across HuggingFace, vLLM, and TensorRT-LLM, with explicit KV cache modeling and backend-aware benchmarking.

This project studies how architectural parameters, GPU memory systems, and backend implementations determine real-world decode performance on 48GB-class GPUs.

This is **inference systems engineering**, not model fine-tuning.

---

# 🎯 Objective

Understand when LLM inference is:

* **Compute-bound**
* **Memory-bandwidth-bound**
* **Memory-capacity-constrained**
* **Backend-limited**

Rather than just measuring tokens/sec, this project models and validates the structural causes of performance behavior.

---

# 🏗 Experimental Environment

## Hardware

* 48GB VRAM GPU (RTX 6000 Ada / A6000 class)
* CUDA 12.x
* Single GPU
* bfloat16 precision

48GB VRAM enables:

* 8K–16K context analysis
* Large batch scaling without early OOM
* Clear separation between bandwidth vs capacity limits

---

# 🧠 Models Evaluated

* Qwen2.5-0.5B-Instruct
* Qwen2.5-1.5B
* Qwen2.5-7B
* Llama-3.1-8B
* Mistral-7B
* Gemma-7B
* Phi-3-Mini

All experiments use **bfloat16** unless otherwise specified.

---

# ⚙️ Backends Compared

| Backend      | KV Strategy | Execution Model      |
| ------------ | ----------- | -------------------- |
| HuggingFace  | Dynamic KV  | PyTorch eager        |
| vLLM         | Paged KV    | Custom CUDA kernels  |
| TensorRT-LLM | Paged KV    | Engine-compiled CUDA |

This comparison isolates:

* Kernel fusion impact
* KV memory layout strategy
* Scheduling differences
* Engine compilation advantages

---

# 🧮 KV Cache Model

For decoder-only transformers:

```
KV_MB_per_token =
(2 × L × hidden_size × bytes) / 1024²
```

Where:

* L = number of layers
* hidden_size = model width
* bytes = dtype size (bf16 = 2)

Total KV memory:

```
KV_total_MB =
batch × tokens × KV_MB_per_token
```

Example (7B model):

≈ **0.38 MB per token (batch = 1)**

KV memory scales linearly with:

* Batch size
* Generated tokens
* Model depth
* Hidden dimension

This model allows prediction of memory regime transitions.

Implementation:

```
analysis/model_kv_theory.py
```

Supporting documents:

* `analysis/scaling_design.md`
* `analysis/production_tradeoffs.md`
* `analysis/serving_design.md`
* `analysis/multi_gpu_scaling.md`
* `analysis/findings.md`

---

# 📊 Empirical Results (Measured)

All results collected on a 48GB GPU.

---

## 1️⃣ Sequence Length Scaling

Model: Llama-3.1-8B
Batch = 1
Decode tokens = 128
bf16 precision

| Context | Prefill (s) | Decode (s) | Per-token (ms) | Throughput (tok/s) | GPU Mem (MB) |
| ------- | ----------- | ---------- | -------------- | ------------------ | ------------ |
| 512     | 0.28        | 3.06       | 23.94          | 41.77              | 15543        |
| 4096    | 0.59        | 3.24       | 25.31          | 39.50              | 16871        |

### Observations

* KV memory increased ~1.3GB
* Per-token decode latency increased moderately
* Throughput degradation was mild
* Decode did not collapse under longer context

### Interpretation

At 48GB memory headroom:

* Decode remains largely compute-efficient
* KV growth measurable but bandwidth not saturated
* Arithmetic intensity remains sufficient

Raw data:

```
benchmarks/results/raw_data/sequence_scaling.csv
```

---

## 2️⃣ Batch Scaling

Model: Llama-3.1-8B
Decode tokens = 128
bf16 precision

| Batch | Decode (s) | Per-token (ms) | Throughput (tok/s) | GPU Mem (MB) |
| ----- | ---------- | -------------- | ------------------ | ------------ |
| 1     | 3.24       | 25.30          | 39.52              | 16871        |
| 2     | 3.59       | 28.03          | 71.36              | 18417        |
| 4     | 4.01       | 31.33          | 127.73             | 21510        |
| 8     | 5.38       | 42.03          | 190.33             | 27696        |
| 16    | OOM        | —              | —                  | >40GB        |

### Observations

* Throughput scales strongly up to batch 8
* Latency increases with batch
* Memory scales linearly
* OOM at batch 16

### Interpretation

* Batch 1–4: compute-efficient regime
* Batch 8: bandwidth pressure begins
* Batch 16: memory capacity limit

---

## 3️⃣ Precision Scaling

Model: Llama-3.1-8B
Batch = 1

| Dtype | Decode (s) | Per-token (ms) | Throughput (tok/s) | GPU Mem (MB) |
| ----- | ---------- | -------------- | ------------------ | ------------ |
| FP16  | 3.24       | 25.29          | 39.54              | 16871        |
| FP32  | 8.31       | 64.94          | 15.40              | 36794        |

### Interpretation

FP16 enables:

* Tensor Core acceleration
* Reduced memory footprint
* Higher effective bandwidth

FP32:

* Doubles memory
* Reduces compute throughput
* Pushes decode toward bandwidth stress

---

## 4️⃣ Serving Load Test (20 Concurrent Users)

* Average latency: 3.92 sec
* P50 latency: 3.34 sec
* P95 latency: 4.97 sec
* Max latency: 5.14 sec

Dynamic batching introduces:

* Improved GPU utilization
* Increased tail latency
* Latency variance due to scheduling

Serving implementation:

```
serving/
```

Includes:

* `server.py`
* `hf_server.py`
* `vllm_server.py`
* `trt_server.py`
* `benchmark_client.py`
* `load_test.py`

---

# 📁 Full Repository Structure

```
LLM_Engineering/
├── README.md
├── run.py
├── requirements.txt
├── image.png
│
├── analysis/
├── benchmarks/
│   ├── hf/
│   ├── vllm/
│   ├── trt/
│   ├── run_analysis.py
│   └── results/raw_data/
│
├── experiments/
│   └── scaling_study/
│
├── prompts/
│
├── env/
│   ├── hf_requirements.txt
│   ├── vllm_requirements.txt
│   ├── trt_requirements.txt
│   └── serving_requirements.txt
│
├── scripts/
│   ├── bootstrap_host.sh
│   ├── download_models.sh
│   ├── config/models.conf
│   ├── bootstrap/
│   ├── hf/
│   ├── vllm/
│   ├── trt/
│   └── run/
│
├── serving/
│
└── results/
    ├── per-backend CSV outputs
    └── aggregate/
```

---

# 🚀 Environment Setup

## System Requirements

* Linux (Ubuntu 22.04 recommended)
* NVIDIA driver ≥ 535
* CUDA 12.x
* Python 3.10+
* 48GB GPU recommended

---

## Install NVIDIA & Docker

```bash
chmod +x bootstrap_host.sh
./bootstrap_host.sh
```

Installs:

* NVIDIA driver
* Docker
* NVIDIA Container Toolkit
* CUDA validation

---

# Backend Environments

## HuggingFace

```bash
python3 -m venv .venv_hf
source .venv_hf/bin/activate
pip install -r env/hf_requirements.txt
```

## vLLM

```bash
python3 -m venv .venv_vllm
source .venv_vllm/bin/activate
pip install -r env/vllm_requirements.txt
```

## Serving

```bash
python3 -m venv .venv_serving
source .venv_serving/bin/activate
pip install -r env/serving_requirements.txt
```

---

# TensorRT-LLM Setup

Pull container:

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

Allocate large shared memory container:

```bash
docker run -it --gpus all --shm-size=32g \
  --name trt_llm_big \
  -v /ephemeral/llm_workspace:/workspace \
  -v /ephemeral/hf_models:/workspace/hf_models \
  -v /ephemeral/trt_engine:/workspace/trt_engine \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3 \
  bash
```

Convert checkpoint:

```bash
python convert_checkpoint.py \
  --model_dir hf_models/llama3_1_8b \
  --output_dir trt_ckpt/llama3_1_8b \
  --dtype bfloat16
```

Build engine:

```bash
trtllm-build \
  --checkpoint_dir trt_ckpt/llama3_1_8b \
  --output_dir trt_engine/llama3_1_8b \
  --max_batch_size 32 \
  --max_seq_len 8192
```

---

# 🔁 End-to-End Workflow

## One-Time Setup

```
./scripts/bootstrap_host.sh
./scripts/download_models.sh
```

## Setup Backends

```
./scripts/bootstrap/bootstrap_hf.sh
./scripts/bootstrap/bootstrap_vllm.sh
./scripts/bootstrap/bootstrap_serving.sh
./scripts/bootstrap/trtllm/bootstrap_all_trt.sh
```

## Run Benchmarks

```
./scripts/run/run_all_backends.sh
```

## Aggregate

```
python benchmarks/run_analysis.py
```

## Fully Automated

```
./scripts/run/run_full_experiment.sh
```

---

# 📂 Results

All outputs stored in:

```
results/
```

Includes:

* Throughput scaling
* KV growth validation
* Precision comparison
* Backend comparison
* Aggregate plots
* Experiment metadata

---

# 🔭 Future Work

* Multi-GPU tensor parallel study
* INT8 / FP8 quantization comparison
* Roofline intensity modeling
* Communication-bound regime analysis
* Triton production deployment
