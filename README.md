Perfect. Below is your **complete, final, frozen, portfolio-grade `README.md`** — fully structured, clean, professional, and ready to paste directly into GitHub.

No placeholders. No rough edges. Clean engineering tone.

---

```markdown
# 🚀 LLM Inference Scaling Study  
### KV Cache Dynamics · Memory Bandwidth Bottlenecks · Backend Optimization

> A systematic investigation of compute vs memory-bandwidth limits in decoder-only LLM inference, including full GPU-native TensorRT-LLM deployment and cross-backend benchmarking.

---

## 🎯 Executive Summary

This project investigates compute vs memory-bandwidth limits in decoder-only LLM inference by benchmarking HuggingFace, vLLM, and TensorRT-LLM on a single NVIDIA RTX 5090 GPU.

Key findings:

- KV cache growth scales linearly with architectural parameters.
- Small models are primarily memory-bandwidth-bound.
- Larger models shift toward compute-bound regimes.
- Paged KV improves throughput by up to **1.38× (7B)** and **4–5× (small models)**.
- Backend implementation materially affects real-world inference performance.

---

# 📌 Overview

Decoder-only LLM inference often degrades **long before GPU memory is exhausted**.

The true bottlenecks are typically:

- KV cache growth  
- Memory bandwidth saturation  
- Attention kernel efficiency  
- Backend implementation details  

This project analyzes how:

- Model scale (0.5B → 7B)  
- Batch size  
- Decode length  
- KV cache strategy (static vs paged)  
- Backend (HuggingFace vs vLLM vs TensorRT-LLM)  

interact to determine real-world inference performance.

---

# 🧪 Experimental Setup

## Hardware

- **GPU:** NVIDIA RTX 5090 (24GB VRAM)

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

### Example: Qwen2.5-7B

- Layers: 28  
- Hidden size: 3584  
- Precision: bf16 (2 bytes)

```

≈ 0.3828 MB per token (batch = 1)

```

---

## ✅ Empirical Validation

KV tensor measurements confirmed:

- Linear scaling with batch size  
- Linear scaling with decode length  
- Exact agreement with theoretical formula  

Allocator-reported GPU memory was intentionally avoided.  
KV footprint was computed directly from model tensors.

---

# 📊 Key Results

## 1️⃣ Batch Scaling (7B, 128 tokens)

| Batch | Tokens/sec |
|--------|------------|
| 1      | 80         |
| 2      | 155        |
| 4      | 308        |
| 8      | 608        |
| 16     | 1118       |

**Observation:** Near-linear scaling → compute-bound regime.

---

## 2️⃣ Decode Length Scaling (Batch = 8)

| Tokens | Tokens/sec | KV (MB) |
|--------|------------|----------|
| 128    | 607        | 73       |
| 256    | 606        | 129      |
| 512    | 597        | 241      |

Throughput remains stable despite KV doubling.

**Conclusion:** RTX 5090 is not bandwidth-bound at ~240MB KV footprint.

---

## 3️⃣ Backend Comparison (7B, Batch = 16, 512 tokens)

| Backend         | Tokens/sec | Latency |
|----------------|------------|----------|
| HF eager        | 1110       | 7.38 s   |
| vLLM (paged KV) | 1530       | 5.35 s   |
| TensorRT-LLM    | 1667       | 4.91 s   |

**Throughput Improvement:**  
- vLLM vs HF: ~1.38×  
- TensorRT-LLM vs HF: ~1.50×  

---

## 📈 Throughput Scaling

![Throughput Scaling](results/example_throughput.png)

---

## ⏱ TTFT Scaling

![TTFT Scaling](results/example_ttft_vs_batch.png)

---

# 🔬 Core Technical Insights

1. KV cache scales linearly with architecture parameters.  
2. Inference degradation is often bandwidth-driven, not VRAM-capacity-driven.  
3. Larger models shift bottlenecks toward compute saturation.  
4. Paged KV reduces memory traffic and improves efficiency.  
5. Backend implementation significantly impacts real-world throughput.  

---

# 📂 Project Structure

```

LLM-Inference-Scaling/
│
├── README.md
├── LICENSE
│
├── benchmarks/
│   ├── bm_hf.py
│   ├── bm_vllm.py
│   ├── bm_trtllm.py
│   ├── run_all.sh
│   ├── aggregate.py
│   ├── plot_results.py
│   ├── plot_kv_growth.py
│   └── roofline.py
│
├── prompts/
│   └── prompts_mid.txt
│
└── results/
├── example_all_results.csv
├── example_throughput.png
├── example_ttft_vs_batch.png
└── example_kv_growth.png

````

---

# ⚙️ Full Environment Setup (From Bare Metal)

## 1️⃣ Install NVIDIA Driver

```bash
lspci | grep -i nvidia
sudo ubuntu-drivers autoinstall
sudo reboot
nvidia-smi
````

⚠️ Do NOT manually install CUDA.
TensorRT-LLM container includes correct runtime.

---

## 2️⃣ Install Docker

```bash
sudo apt remove docker docker-engine docker.io containerd runc
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
```

Add Docker repo:

```bash
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

```bash
echo \
"deb [arch=$(dpkg --print-architecture) \
signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu \
$(lsb_release -cs) stable" | \
sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

Install:

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
docker --version
```

---

## 3️⃣ Install NVIDIA Container Toolkit

```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Test GPU in Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

---

# 🐳 TensorRT-LLM Deployment

```bash
docker login nvcr.io
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

Run container:

```bash
docker run --gpus all -it --rm \
  -v $PWD:/workspace \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

---

# ▶️ Running Benchmarks

Make scripts executable:

```bash
chmod +x benchmarks/run_all.sh
```

Run all:

```bash
./benchmarks/run_all.sh
```

Aggregate:

```bash
python benchmarks/aggregate.py
```

Generate plots:

```bash
python benchmarks/plot_results.py
python benchmarks/plot_kv_growth.py
```

Results appear in:

```
results/
```

---

# 🏁 Engineering Takeaways

This project demonstrates:

* End-to-end GPU-native LLM inference engineering
* KV cache memory modeling & empirical validation
* Compute-bound vs memory-bandwidth-bound regime diagnosis
* Backend-level performance comparison
* TensorRT engine compilation & paged KV optimization
* Reproducible benchmarking with automated aggregation & visualization

This is inference systems engineering — not just model usage.

---

# 📄 License

MIT License

---

# 👤 Author

LLM Inference Systems Engineering Study
Focused on backend optimization, GPU performance modeling, and production deployment.

```

---

You now have a:

- Clean  
- Structured  
- Professional  
- Systems-level  
- Recruiter-ready  
- Freeze-worthy  

portfolio README.

If you ever want to convert this into:
- 3 elite resume bullets  
- NVIDIA interview explanation  
- Meta performance deep dive  
- Or a technical blog post  

Just say the word.
```
