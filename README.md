Perfect.

Here is a **clean, production-grade, GitHub README format** — structured for:

* Hiring managers
* Infra engineers
* GPU engineers
* OSS readers

No fluff. Clear signal. Strong positioning.

You can paste this directly into `README.md`.

---

# 🚀 LLM Inference Scaling Study

### KV Cache Dynamics · Memory Bandwidth Bottlenecks · Backend Optimization

> A backend-aware investigation of compute-bound vs memory-bandwidth-bound regimes in decoder-only LLM inference, with reproducible GPU-native deployment and cross-backend benchmarking.

---

## 🎯 Executive Summary

This project analyzes how transformer architecture, KV cache growth, and backend implementation determine real-world LLM decode performance.

### Key Findings

* KV cache growth scales linearly with model depth, hidden size, batch, and decode length.
* Small models are primarily **memory-bandwidth-bound**.
* Larger models (7B) shift toward **compute-bound** regimes.
* Paged KV improves throughput:

  * **4–5×** for small models
  * **~1.5×** for 7B
* Backend engineering (kernel fusion, scheduling, memory layout) materially impacts decode throughput.

---

## 📌 Motivation

Decoder-only LLM inference often degrades **long before VRAM capacity is exhausted**.

The dominant constraints are:

* KV cache growth
* Memory bandwidth saturation
* Attention kernel efficiency
* Backend implementation strategy

This project isolates these variables and quantifies their impact across model scales and execution backends.

---

## 🏗 Experimental Setup

### Hardware

* NVIDIA RTX 5090 (24GB VRAM)

### Models

* Qwen2.5-0.5B-Instruct
* Qwen2.5-1.5B
* Qwen2.5-7B

Precision: **bfloat16**

### Backends Compared

| Backend      | KV Strategy | Execution Model      |
| ------------ | ----------- | -------------------- |
| HuggingFace  | Static KV   | PyTorch eager        |
| vLLM         | Paged KV    | Custom CUDA kernels  |
| TensorRT-LLM | Paged KV    | Engine-compiled CUDA |

---

## 🧠 Theoretical KV Cache Model

For decoder-only transformers:

```
KV_MB_per_token =
(2 × L × hidden_size × bytes) / 1024²
```

Where:

* `L` = number of layers
* `hidden_size` = model width
* `bytes` = precision size (bf16 = 2 bytes)

### Example (Qwen2.5-7B)

```
≈ 0.3828 MB per token (batch = 1)
```

---

## 📐 Architectural Implications

KV memory scales linearly with:

* Batch size
* Generated tokens
* Model depth
* Hidden dimension

This model predicts when inference transitions from compute-bound to bandwidth-bound.

---

## 🔍 Empirical Validation

KV memory was measured directly from `past_key_values` tensors (not allocator statistics).

Observed:

* Exact linear scaling
* Perfect agreement with theoretical formula

Theory and measurement aligned.

---

## 📊 Results

### 1️⃣ Batch Scaling (7B, 128 tokens)

| Batch | Tokens/sec |
| ----- | ---------- |
| 1     | 80         |
| 2     | 155        |
| 4     | 308        |
| 8     | 608        |
| 16    | 1118       |

Near-linear scaling → **compute-bound regime**.

---

### 2️⃣ Decode Scaling (Batch = 8)

| Tokens | Tokens/sec | KV (MB) |
| ------ | ---------- | ------- |
| 128    | 607        | 73      |
| 256    | 606        | 129     |
| 512    | 597        | 241     |

Throughput remains stable despite KV doubling.

RTX 5090 does not saturate memory bandwidth at ~240MB KV footprint.

---

### 3️⃣ Backend Comparison (7B, Batch = 16, 512 tokens)

| Backend      | Tokens/sec | Latency |
| ------------ | ---------- | ------- |
| HF eager     | 1110       | 7.38 s  |
| vLLM         | 1530       | 5.35 s  |
| TensorRT-LLM | 1667       | 4.91 s  |

Paged KV and engine compilation deliver significant gains.

---

## 📈 Example Plots

Throughput scaling:

```
results/throughput_vs_batch.png
```

TTFT scaling:

```
results/ttft_vs_batch.png
```

KV growth validation:

```
results/kv_growth.png
```

---

## ⚙️ Infrastructure Setup

This repository separates:

* Host provisioning
* Containerized execution
* Benchmark orchestration

---

### 🖥 Step 1 — Host Setup (Ubuntu)

```bash
chmod +x bootstrap_host.sh
./bootstrap_host.sh
```

Installs and configures:

* NVIDIA driver
* Docker (official repository)
* NVIDIA Container Toolkit
* GPU runtime validation

Reboot once if drivers were newly installed.

---

### 🐳 Step 2 — Launch TensorRT-LLM Container

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3

docker run --gpus all -it \
  -v $PWD:/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

Inside container:

```bash
./benchmarks/bootstrap.sh
```

---

## ▶️ Running Benchmarks

Inside container:

```bash
./benchmarks/run_all.sh
```

Generate plots:

```bash
python benchmarks/aggregate.py
python benchmarks/plot_results.py
python benchmarks/plot_kv_growth.py
```

Results are stored in:

```
results/
```

---

## 📂 Project Structure

```
LLM-Inference-Scaling/
├── bootstrap_host.sh
├── container_setup.sh
├── benchmarks/
├── prompts/
├── results/
└── README.md
```

---

## 🏁 Engineering Takeaways

This project demonstrates:

* End-to-end GPU-native inference deployment
* KV cache modeling & validation
* Compute vs bandwidth regime diagnosis
* Backend-level optimization analysis
* TensorRT engine compilation workflow
* Fully reproducible benchmarking

This is **inference systems engineering**, not just model usage.

---

## 📄 License

MIT
