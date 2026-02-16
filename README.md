Excellent — you have **strong raw material**, but it currently reads like a running lab notebook.

I’m going to rewrite everything into a **clean, logically structured, production-grade technical document** with:

* Clear narrative flow
* No duplication
* Separation of research vs deployment
* Professional tone
* Portfolio-ready structure

---

# 🚀 LLM Inference Scaling Study

### KV Cache Dynamics · Memory Bandwidth Bottlenecks · Backend Optimization

> A systematic investigation into how transformer architecture, KV cache growth, GPU hardware, and backend implementation determine real-world LLM inference performance.

---

# I. Research Study — Performance Scaling Analysis

---

## 1️⃣ Problem Statement

Decoder-only LLM inference frequently degrades **before GPU memory capacity is exhausted**.

The real bottlenecks are often:

* KV cache growth
* Memory bandwidth saturation
* Attention kernel implementation
* Backend scheduling efficiency
* Memory layout and allocation strategy

This study isolates those variables to understand:

* When inference is compute-bound
* When it becomes memory-bandwidth-bound
* How paged KV cache changes system behavior
* How backend implementations impact throughput

---

## 2️⃣ Experimental Setup

### Hardware

* **GPU:** NVIDIA RTX 5090 (24GB VRAM)

### Container Runtime

* TensorRT-LLM container:

  ```
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3
  ```

### Models Evaluated

* Qwen2.5-0.5B-Instruct
* Qwen2.5-1.5B
* Qwen2.5-7B

Precision: **bfloat16**

---

## 3️⃣ Backends Compared

| Backend      | KV Strategy | Execution Mode  |
| ------------ | ----------- | --------------- |
| HuggingFace  | Static KV   | PyTorch eager   |
| vLLM         | Paged KV    | Custom CUDA     |
| TensorRT-LLM | Paged KV    | Engine-compiled |

---

## 4️⃣ Theoretical KV Cache Scaling

For decoder-only transformers:

[
KV_{MB/token} = \frac{2 \times L \times hidden_size \times bytes}{1024^2}
]

Where:

* L = number of layers
* hidden_size = model width
* bytes = precision size (bf16 = 2 bytes)

---

### Example: Qwen2.5-7B

* Layers: 28
* Hidden size: 3584
* Precision: bf16

Result:

```
KV ≈ 0.3828 MB per token (batch=1)
```

---

### Empirical Validation

Direct tensor-based measurement confirmed:

* Linear scaling with batch size
* Linear scaling with decode length
* Exact match with theoretical formula

This validates the architectural KV model.

---

## 5️⃣ Ground-Truth KV Measurement

Allocator-reported GPU memory is noisy due to caching.

Instead, KV memory was computed directly:

```python
past = outputs.past_key_values
kv_bytes = 0

for k, v in past:
    kv_bytes += k.numel() * k.element_size()
    kv_bytes += v.numel() * v.element_size()

kv_mb = kv_bytes / (1024 ** 2)
```

This reflects true KV tensor footprint.

---

## 6️⃣ Phase I — Batch Scaling (7B Model)

**max_new_tokens = 128**

| Batch | tok/s |
| ----- | ----- |
| 1     | 80    |
| 2     | 155   |
| 4     | 308   |
| 8     | 608   |
| 16    | 1118  |

### Interpretation

* Near-linear scaling to batch=8
* Slight curvature at batch=16
* Increasing GPU compute utilization

**Conclusion:** System remains compute-bound in this region.

---

## 7️⃣ Phase II — Decode Length Scaling

### Batch = 8

| Tokens | tok/s | KV (MB) |
| ------ | ----- | ------- |
| 128    | 607   | 73      |
| 256    | 606   | 129     |
| 512    | 597   | 241     |

Throughput remains stable despite KV doubling.

**RTX 5090 is not bandwidth-bound at ~240MB KV.**

---

### Batch = 16

| Tokens | tok/s | KV (MB) |
| ------ | ----- | ------- |
| 256    | 1122  | 259     |
| 512    | 1110  | 483     |

Slight degradation, but still compute-dominant.

---

## 8️⃣ Backend Comparison (7B, batch=16, 512 tokens)

| Backend         | tok/s | Latency |
| --------------- | ----- | ------- |
| HF eager        | 1110  | 7.38 s  |
| vLLM (paged KV) | 1530  | 5.35 s  |

### Performance Gain

~1.38× throughput improvement.

### Explanation

Paged KV improves:

* Memory locality
* Reduced redundant memory traffic
* Kernel efficiency
* Scheduling

Even in compute-dominant regimes, backend implementation materially impacts performance.

---

## 9️⃣ Cross-Regime Behavior

### Small Models (0.5B / 1.5B)

* Strong memory-bandwidth bottleneck
* 4–5× improvement with paged KV

### 7B Model

* Mostly compute-bound
* ~38% backend improvement

---

## 🔬 Key Insight

Inference bottlenecks depend on:

* Model scale
* KV footprint
* GPU bandwidth
* Backend memory layout
* Kernel implementation

There is no universal bottleneck — regime is architecture + hardware dependent.

---

## 🧠 Technical Conclusions

1. KV cache scales linearly with architecture parameters.
2. Inference degradation is often bandwidth-driven, not VRAM-capacity-driven.
3. Larger models shift bottlenecks toward compute saturation.
4. Paged KV reduces memory traffic and improves throughput.
5. Backend engineering significantly impacts real-world performance.

---

# II. Production Deployment — From Bare Metal to TensorRT-LLM

This section documents a **fully reproducible GPU inference deployment pipeline**.

---

## 1️⃣ System Stack (Mental Model)

```
GPU Hardware
↓
NVIDIA Driver
↓
Docker Engine
↓
NVIDIA Container Toolkit
↓
NGC TensorRT-LLM Container
↓
HF Model Conversion
↓
TensorRT Engine Build
↓
Runtime Inference
```

---

## 2️⃣ Host Machine Preparation

### Install NVIDIA Driver

Verify:

```bash
nvidia-smi
```

If not installed (Ubuntu):

```bash
sudo ubuntu-drivers autoinstall
sudo reboot
```

GPU must work before Docker.

---

## 3️⃣ Install Docker

```bash
sudo apt remove docker docker-engine docker.io containerd runc
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release
```

Add repository + install Docker Engine.

Verify:

```bash
docker --version
```

---

## 4️⃣ Install NVIDIA Container Toolkit

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

## 5️⃣ Pull TensorRT-LLM Container

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

Launch:

```bash
docker run --gpus all -it --rm \
  -v $PWD:/workspace \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3
```

---

## 6️⃣ Convert HuggingFace Model

```bash
python convert_checkpoint.py \
  --model_dir /workspace/model \
  --output_dir /workspace/model_trt \
  --dtype bfloat16
```

---

## 7️⃣ Build TensorRT Engine

```bash
trtllm-build \
  --checkpoint_dir /workspace/model_trt \
  --output_dir /workspace/model_engine \
  --max_batch_size 16 \
  --max_seq_len 2560 \
  --kv_cache_type paged
```

---

## 8️⃣ Run GPU-Native Inference

```bash
python run.py \
  --engine_dir /workspace/model_engine \
  --tokenizer_dir /workspace/model \
  --max_output_len 64
```

✔ Paged KV operational
✔ No PyTorch fallback
✔ Engine-compiled decode

---

# 🏆 Engineering Outcomes

You now demonstrate:

* Compute vs bandwidth regime diagnosis
* KV cache analytical modeling
* Backend-level performance comparison
* Engine-level optimization via TensorRT-LLM
* Full GPU containerized deployment
* Production inference workflow

This is **inference systems engineering**, not model usage.

---

# 🔮 Future Extensions

* Roofline analysis (FLOPs vs bandwidth)
* FP8 KV experiments
* Multi-tenant workload scheduling
* Kernel-level decode profiling
* TTFT micro-benchmarking

---

If you'd like, next I can:

* Turn this into a GitHub README (clean + concise)
* Convert into a 1-page PDF research brief
* Generate 3 high-impact resume bullets
* Prepare an interview explanation script
* Add a roofline performance diagram explanation

You’ve built something real here.
