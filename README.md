Here is your **complete README.md in clean GitHub markdown format** — ready to paste directly into your repository.

---

```markdown
# 🚀 LLM Inference Scaling Study

### KV Cache Dynamics · Memory Bandwidth Bottlenecks · Backend Optimization

> A systematic investigation into compute vs memory bandwidth limits in decoder-only LLM inference, with full GPU-native TensorRT-LLM deployment and cross-backend benchmarking.

---

# 📌 Overview

Decoder-only LLM inference often degrades **long before GPU memory is exhausted**.

The real bottlenecks are typically:

- KV cache growth  
- Memory bandwidth saturation  
- Attention kernel efficiency  
- Backend implementation details  

This project investigates how:

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
| HuggingFace  | Static KV  | PyTorch eager        |
| vLLM         | Paged KV   | Custom CUDA kernels  |
| TensorRT-LLM | Paged KV   | Engine-compiled CUDA |

---

# 🧠 Theoretical KV Cache Model

For decoder-only transformers:

```

KV_MB_per_token =
(2 × L × hidden_size × bytes) / 1024²

```

Example: **Qwen2.5-7B**

- Layers: 28  
- Hidden size: 3584  
- Precision: bf16 (2 bytes)

```

≈ 0.3828 MB per token (batch=1)

````

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

## 2️⃣ Decode Length Scaling (Batch=8)

| Tokens | Tokens/sec | KV (MB) |
|--------|------------|----------|
| 128    | 607        | 73       |
| 256    | 606        | 129      |
| 512    | 597        | 241      |

Throughput remains stable despite KV doubling.

**Conclusion:** RTX 5090 is not bandwidth-bound at ~240MB KV footprint.

---

## 3️⃣ Backend Comparison (7B, batch=16, 512 tokens)

| Backend         | Tokens/sec | Latency |
|----------------|------------|----------|
| HF eager        | 1110       | 7.38 s   |
| vLLM (paged KV) | 1530       | 5.35 s   |

**~1.38× throughput improvement using paged KV.**

---

## 4️⃣ Cross-Regime Insight

Small models (0.5B / 1.5B):

- Strong memory-bandwidth bottleneck  
- 4–5× improvement with paged KV  

7B model:

- Mostly compute-bound  
- ~38% backend improvement  

---

# 🔬 Core Technical Insights

1. KV cache scales linearly with architecture parameters.  
2. Inference degradation is often bandwidth-driven, not VRAM-capacity-driven.  
3. Larger models shift bottlenecks toward compute saturation.  
4. Paged KV reduces memory traffic and improves efficiency.  
5. Backend implementation significantly impacts real-world throughput.  

---

# ⚙️ Full Environment Setup (From Bare Metal)

Fully reproducible NVIDIA GPU + Docker + TensorRT-LLM deployment.

---

## 1️⃣ Install NVIDIA Driver (Enable `nvidia-smi`)

Check GPU:

```bash
lspci | grep -i nvidia
````

Install driver (Ubuntu):

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

Verify:

```bash
nvidia-smi
```

⚠️ Do NOT manually install CUDA. The TensorRT-LLM container includes CUDA runtime.

---

## 2️⃣ Install Docker

```bash
sudo apt remove docker docker-engine docker.io containerd runc
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
```

Add Docker repository:

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

Install Docker:

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

---

# 🐳 TensorRT-LLM Deployment

## Login to NVIDIA NGC

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

## Pull TensorRT-LLM Container

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

---

## Launch Container

```bash
docker run --gpus all -it --rm \
  -v $PWD:/workspace \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

Verify inside container:

```bash
nvidia-smi
python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

---

# ▶️ Running the Benchmarks

```
LLM_Engineering/
 ├── benchmarks/
 │   ├── bootstrap.sh
 │   ├── run_all.sh
 │   ├── hf/run_hf.sh
 │   ├── vllm/run_vllm.sh
 │   └── trt/run_trt.sh
 ├── hf_models/
 ├── trt_ckpt/
 ├── trt_engine/
 ├── prompts/
 └── results/
```

## Make Scripts Executable

```bash
chmod +x benchmarks/bootstrap.sh
chmod +x benchmarks/run_all.sh
chmod +x benchmarks/hf/run_hf.sh
chmod +x benchmarks/vllm/run_vllm.sh
chmod +x benchmarks/trt/run_trt.sh
```

## Bootstrap (One-Time Setup)

```bash
./benchmarks/bootstrap.sh
```

## Run All Benchmarks

```bash
./benchmarks/run_all.sh
```

Results will be saved to:

```
results/
 ├── hf_results.csv
 ├── vllm_results.csv
 └── trt_results.csv
```

---

# 🏁 What This Project Demonstrates

* End-to-end GPU-native LLM inference engineering
* KV cache memory modeling and validation
* Compute-bound vs bandwidth-bound regime analysis
* Backend-level performance comparison
* Production-grade NVIDIA container deployment
* Engine-based inference optimization (TensorRT-LLM)

---

## 📎 License

MIT

```

---

This is now:

- Clean  
- GitHub-native  
- Structured  
- Professional  
- Ready to publish  

If you'd like, I can next provide:
- A diagram version (architecture visual section)
- A resume bullet summary
- Or a condensed “Recruiter 60-second skim” version
```
