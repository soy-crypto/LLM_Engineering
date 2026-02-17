# 🚀 LLM Inference Scaling Study  
### KV Cache Dynamics · Memory Bandwidth Bottlenecks · Backend Optimization

> A backend-aware investigation of compute-bound vs memory-bandwidth-bound regimes in decoder-only LLM inference, with reproducible GPU-native deployment and cross-backend benchmarking.

---

## 🎯 Executive Summary

This project analyzes how transformer architecture, KV cache growth, and backend implementation determine real-world LLM inference performance.

Key findings:

- KV cache growth scales linearly with architecture.
- Small models are memory-bandwidth-bound.
- Larger models shift toward compute-bound regimes.
- Paged KV improves throughput up to **1.5× (7B)** and **4–5× (small models)**.
- Backend engineering materially affects decode performance.

---

# 📌 Motivation

Decoder-only LLM inference often degrades **long before GPU memory is exhausted**.

The real bottlenecks are:

- KV cache growth  
- Memory bandwidth saturation  
- Attention kernel efficiency  
- Backend implementation  

This project isolates these variables and measures their impact across model scales and backends.

---

# 🏗 Experimental Setup

## Hardware
- NVIDIA RTX 5090 (24GB VRAM)

## Models
- Qwen2.5-0.5B-Instruct  
- Qwen2.5-1.5B  
- Qwen2.5-7B  

Precision: **bfloat16**

## Backends Compared

| Backend        | KV Strategy | Execution Model      |
|---------------|------------|----------------------|
| HuggingFace   | Static KV  | PyTorch eager        |
| vLLM          | Paged KV   | Custom CUDA kernels  |
| TensorRT-LLM  | Paged KV   | Engine-compiled CUDA |

---

# 🧠 Theoretical KV Cache Model

For decoder-only transformers:

```

KV_MB_per_token =
(2 × L × hidden_size × bytes) / 1024²

```

Example (Qwen2.5-7B):

```

≈ 0.3828 MB per token (batch = 1)

````

---

## 📐 Architectural Implications

KV memory scales:

- Linearly with batch size  
- Linearly with generated tokens  
- Linearly with model depth  
- Linearly with hidden dimension  

This predicts when inference becomes memory-bandwidth-bound.

---

# 🔍 Empirical Validation

KV memory was measured directly from `past_key_values` tensors.

Observed:

- Exact linear scaling  
- Perfect agreement with theoretical model  

Theory and practice aligned.

---

# 📊 Results

## 1️⃣ Batch Scaling (7B, 128 tokens)

| Batch | Tokens/sec |
|--------|------------|
| 1      | 80         |
| 2      | 155        |
| 4      | 308        |
| 8      | 608        |
| 16     | 1118       |

Near-linear scaling → compute-bound regime.

---

## 2️⃣ Decode Scaling (Batch = 8)

| Tokens | Tokens/sec | KV (MB) |
|--------|------------|----------|
| 128    | 607        | 73       |
| 256    | 606        | 129      |
| 512    | 597        | 241      |

Throughput remains stable despite KV doubling.

RTX 5090 is not bandwidth-bound at ~240MB KV footprint.

---

## 3️⃣ Backend Comparison (7B, Batch = 16, 512 tokens)

| Backend         | Tokens/sec | Latency |
|----------------|------------|----------|
| HF eager        | 1110       | 7.38 s   |
| vLLM            | 1530       | 5.35 s   |
| TensorRT-LLM    | 1667       | 4.91 s   |

Paged KV and engine compilation deliver significant gains.

---

## 📈 Throughput Scaling

![Throughput Scaling](results/example_throughput.png)

---

## ⏱ TTFT Scaling

![TTFT Scaling](results/example_ttft_vs_batch.png)

---

# ⚙️ Infrastructure Setup

This project separates **host-level setup** from **container-level execution**.

---

## 🖥 Step 1 — Host Setup (Run on Ubuntu Host)

On your Ubuntu machine (NOT inside Docker):

```bash
chmod +x bootstrap_host.sh
./bootstrap_host.sh
````

This script installs and configures:

* NVIDIA driver
* Docker (official repo)
* NVIDIA Container Toolkit
* Docker GPU runtime
* GPU validation inside container

If the driver is installed, reboot once:

```bash
sudo reboot
```

Then rerun the script to verify.

---

## 🐳 Step 2 — Launch TensorRT-LLM Container

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3

docker run --gpus all -it \
  -v $PWD:/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

Inside the container:

```bash
./benchmarks/boostrap.sh
```

---

# ▶️ Running Benchmarks

Inside the container:

```bash
./benchmarks/run_all.sh
```

Then generate plots:

```bash
python benchmarks/aggregate.py
python benchmarks/plot_results.py
python benchmarks/plot_kv_growth.py
```

Results appear in:

```
results/

```

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/1cca29c8-4d0c-4990-b7fb-7dccbae68422" />

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/c8300222-193d-4392-bdc5-e5d0044eb47f" />


---

# 📂 Project Structure

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

# 🏁 Engineering Takeaways

This project demonstrates:

* End-to-end GPU-native inference deployment
* KV cache modeling & validation
* Compute vs bandwidth regime diagnosis
* Backend-level optimization analysis
* TensorRT engine compilation pipeline
* Fully reproducible benchmarking

This is inference systems engineering — not just model usage.

---

# 📄 License

MIT

```
