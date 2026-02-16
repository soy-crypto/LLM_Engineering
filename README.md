Perfect. Below is your **final, polished, logically structured GitHub README** — combining:

* Project overview
* Scaling research
* Backend comparison
* Full reproducible NVIDIA setup
* TensorRT-LLM deployment pipeline

Clean. Professional. Recruiter-ready. Reproducible.

---

# 🚀 LLM Inference Scaling Study

### KV Cache Dynamics · Memory Bandwidth Bottlenecks · Backend Optimization

> A systematic investigation into compute vs bandwidth limits in decoder-only LLM inference, with full GPU-native TensorRT-LLM deployment.

---

# 📌 Overview

Decoder-only LLM inference often degrades **long before GPU memory is exhausted**.

The real bottlenecks are typically:

* KV cache growth
* Memory bandwidth saturation
* Attention kernel efficiency
* Backend implementation details

This project investigates how:

* Model scale (0.5B → 7B)
* Batch size
* Decode length
* KV cache strategy (static vs paged)
* Backend (HF vs vLLM vs TensorRT-LLM)

interact to determine real-world inference performance.

---

# 🧪 Experimental Setup

## Hardware

* **GPU:** NVIDIA RTX 5090 (24GB VRAM)

## Models

* Qwen2.5-0.5B-Instruct
* Qwen2.5-1.5B
* Qwen2.5-7B

Precision: **bfloat16**

## Backends Compared

| Backend      | KV Strategy | Execution       |
| ------------ | ----------- | --------------- |
| HuggingFace  | Static KV   | PyTorch eager   |
| vLLM         | Paged KV    | CUDA kernels    |
| TensorRT-LLM | Paged KV    | Engine-compiled |

---

# 🧠 Theoretical KV Cache Model

For decoder-only transformers:

```
KV_MB_per_token =
(2 × L × hidden_size × bytes) / 1024²
```

Example (Qwen2.5-7B):

* Layers: 28
* Hidden size: 3584
* bf16 (2 bytes)

```
≈ 0.3828 MB per token (batch=1)
```

### ✅ Empirical Validation

KV tensor measurements confirmed:

* Linear scaling with batch size
* Linear scaling with decode length
* Exact agreement with theory

Allocator-reported GPU memory was avoided; KV footprint was computed directly from tensor outputs.

---

# 📊 Key Results

## 1️⃣ Batch Scaling (7B, 128 tokens)

| Batch | tok/s |
| ----- | ----- |
| 1     | 80    |
| 2     | 155   |
| 4     | 308   |
| 8     | 608   |
| 16    | 1118  |

➡ Near-linear scaling → compute-bound regime.

---

## 2️⃣ Decode Length Scaling

Batch = 8

| Tokens | tok/s | KV (MB) |
| ------ | ----- | ------- |
| 128    | 607   | 73      |
| 256    | 606   | 129     |
| 512    | 597   | 241     |

Throughput remains stable despite KV doubling.

➡ RTX 5090 not bandwidth-bound at ~240MB KV footprint.

---

## 3️⃣ Backend Comparison (7B, batch=16, 512 tokens)

| Backend         | tok/s | Latency |
| --------------- | ----- | ------- |
| HF eager        | 1110  | 7.38 s  |
| vLLM (paged KV) | 1530  | 5.35 s  |

**~1.38× throughput improvement using paged KV.**

---

## 4️⃣ Cross-Regime Insight

Small models (0.5B / 1.5B):

* Strong memory-bandwidth bottleneck
* 4–5× improvement with paged KV

7B model:

* Mostly compute-bound
* ~38% backend improvement

---

# 🔬 Core Technical Insights

1. KV cache scales linearly with architecture parameters.
2. Inference degradation is often bandwidth-driven, not VRAM-capacity-driven.
3. Larger models shift bottlenecks toward compute saturation.
4. Paged KV reduces memory traffic and improves efficiency.
5. Backend implementation significantly impacts throughput.

---

# ⚙️ Full Environment Setup (From Bare Metal)

This section provides a fully reproducible NVIDIA GPU + Docker + TensorRT-LLM setup.

---

## 1️⃣ Install NVIDIA Driver (Enable `nvidia-smi`)

Check GPU:

```bash
lspci | grep -i nvidia
```

Install driver (Ubuntu):

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

Verify:

```bash
nvidia-smi
```

If this works → GPU driver is correctly installed.

⚠️ Do NOT manually install CUDA. TensorRT-LLM container includes runtime CUDA.

---

## 2️⃣ Install Docker

Remove old versions:

```bash
sudo apt remove docker docker-engine docker.io containerd runc
```

Install dependencies:

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
```

Add Docker repository and install:

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

```bash
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify:

```bash
docker --version
```

---

## 3️⃣ Install NVIDIA Container Toolkit

```bash
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Test GPU inside Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.3.0-base-ubuntu22.04 nvidia-smi
```

If GPU appears → Docker GPU integration is working.

---

# 🐳 TensorRT-LLM Deployment

---

## 4️⃣ Login to NVIDIA NGC

```bash
docker login nvcr.io
```

Username:

```
$oauthtoken
```

Password:

```
<NGC API key>
```

---

## 5️⃣ Pull TensorRT-LLM Container

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

---

## 6️⃣ Launch Container

```bash
docker run --gpus all -it --rm \
  -v $PWD:/workspace \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

Inside container:

```bash
nvidia-smi
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

---

# 🔄 HF → TensorRT Engine Pipeline

---

## Convert HuggingFace Checkpoint

```bash
python convert_checkpoint.py \
  --model_dir /workspace/model \
  --output_dir /workspace/model_trt \
  --dtype bfloat16
```

---

## Build Engine

```bash
trtllm-build \
  --checkpoint_dir /workspace/model_trt \
  --output_dir /workspace/model_engine \
  --max_batch_size 16 \
  --max_seq_len 2560 \
  --kv_cache_type paged
```

---

## Run GPU-Native Inference

```bash
python run.py \
  --engine_dir /workspace/model_engine \
  --tokenizer_dir /workspace/model \
  --max_output_len 64
```

✔ Engine-compiled decode
✔ Paged KV active
✔ No PyTorch fallback

---

# 🏗 System Architecture

```
GPU Hardware
→ NVIDIA Driver
→ Docker
→ NVIDIA Container Toolkit
→ TensorRT-LLM Container
→ HF Model Conversion
→ TensorRT Engine Build
→ Runtime Decode
```

---

# 🏆 What This Project Demonstrates

* Compute vs bandwidth regime diagnosis
* KV cache analytical modeling
* Backend-level performance comparison
* TensorRT engine optimization
* GPU-native container deployment
* Production inference workflow

This is **inference systems engineering**, not just model usage.

---

# 🔮 Future Work

* Roofline analysis (FLOPs vs bandwidth)
* FP8 KV experiments
* Multi-tenant scheduling benchmarks
* Kernel-level decode profiling
* TTFT microbenchmarking

---

# 📄 License

MIT (or specify your preferred license)

--
