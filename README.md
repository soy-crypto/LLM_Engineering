

````markdown
# LLM_Engineering

**LLM inference performance analysis, KV cache scaling, and paged-KV optimization on RTX 4090.**

---

## Environment

### Hardware
- **GPU:** NVIDIA RTX 4090 (24GB VRAM)

### Container
- **Image:** `nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3`
- **Disk:** 200GB container + 200GB mounted volume

### Container Startup
```bash
bash -c '
apt update;
DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;
mkdir -p ~/.ssh && chmod 700 ~/.ssh;
echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys;
chmod 700 ~/.ssh/authorized_keys;
service ssh start;
sleep infinity'
````

---

## SSH / SFTP Access

### Key Generation

```bash
ssh-keygen -t ed25519 -C "email@xxx.xxx"
```

> Windows: rename `id_ed25519` → `id_ed25519.ppk`

### Connection Info

* **Host:** `157.157.221.29`
* **Port:** `30527`
* **User:** `root`
* **Auth:** SSH key (no password)

---

## Git Workflow

```bash
alias quicksave='git add . && git commit -m "$(date +%m%d)" && git push'
```

---

# LLM Inference Performance Analysis & KV Cache Optimization

* **Model:** Qwen2.5-0.5B-Instruct
* **Precision:** bfloat16
* **Backends:** HuggingFace (eager), vLLM (paged KV)
* **Focus:** Decode throughput, KV cache behavior, memory-bandwidth bottlenecks

---

## Overview

This project analyzes **real-world LLM inference bottlenecks on GPU**, with emphasis on:

* Prefill vs decode latency separation
* Correct measurement of KV cache memory growth
* Identification of memory-bandwidth saturation
* Validation of paged KV as a system-level optimization

All results are based on **controlled experiments** and **apples-to-apples comparisons**.

---

## Why It Matters

For decoder-only LLMs, inference performance often degrades **well before GPU memory is exhausted**.
The primary bottleneck is typically **KV cache growth**, which increases memory traffic and saturates bandwidth during decode.

This repository demonstrates:

* **When** degradation occurs
* **Why** it occurs
* **How paged KV resolves it**

---

## Experimental Setup

* **GPU:** RTX 4090 (24GB)
* **Model:** `Qwen/Qwen2.5-0.5B-Instruct`
* **Precision:** bfloat16
* **Backends:** HuggingFace (eager), vLLM (paged KV)

---

## Metrics

| Metric            | Description                                     |
| ----------------- | ----------------------------------------------- |
| **TTFT**          | Time-to-first-token (prefill latency)           |
| **tok/s (new)**   | Decode throughput (new tokens only)             |
| **KV cache (MB)** | Memory footprint from `past_key_values` tensors |

> Allocator-based GPU memory stats are intentionally avoided due to caching and reuse artifacts.

---

## Methodology

### KV Cache Measurement (Ground Truth)

KV memory is computed directly from model outputs:

```python
past = outputs.past_key_values
kv_bytes = 0

for k, v in past:
    kv_bytes += k.numel() * k.element_size()
    kv_bytes += v.numel() * v.element_size()

kv_mb = kv_bytes / (1024 ** 2)
```

This reflects **actual KV tensor size**, independent of allocator behavior.

---

## Results

### KV Cache Scaling (max_new_tokens = 128)

| Batch | KV Cache (MB) |
| ----: | ------------: |
|     1 |         ~1.98 |
|     2 |         ~3.96 |
|     4 |         ~7.92 |
|     8 |        ~15.84 |
|    16 |        ~31.69 |
|    32 |        ~63.38 |

✅ KV cache scales **linearly with batch size**, matching theory.

---

### Decode Throughput Scaling (HF Eager)

| Batch | tok/s (new) |
| ----: | ----------: |
|     1 |        ~110 |
|     2 |        ~215 |
|     4 |        ~423 |
|     8 |        ~851 |
|    16 |       ~1660 |
|    32 |       ~3413 |

Throughput scales with batch size **until KV growth increases memory traffic**.

---

### Bottleneck Identification (batch = 32)

| New Tokens | tok/s (new) | KV Cache (MB) |
| ---------: | ----------: | ------------: |
|        128 |       ~3413 |           ~63 |
|        256 |       ~3199 |          ~111 |

* KV size nearly doubled
* Decode throughput decreased
* VRAM capacity remained far from full

❗ **Conclusion:** Decode becomes **memory-bandwidth-bound**, not compute- or capacity-bound.

---

### System-Level Fix: Paged KV (vLLM)

Same workload using vLLM:

```text
batch = 32
max_new_tokens = 256
```

| Backend         | tok/s (new) |     Latency |
| --------------- | ----------: | ----------: |
| HF eager        |       ~3199 |     ~2.56 s |
| vLLM (paged KV) |  **~15111** | **~0.54 s** |

🚀 **~4.7× decode throughput improvement**

Paged KV significantly reduces memory traffic and relieves bandwidth pressure.

---

## Key Takeaways

* Decode performance degrades due to **KV cache growth**, not VRAM exhaustion
* KV cache scales linearly with **batch × generated tokens**
* **Memory bandwidth** is the dominant bottleneck at scale
* **Paged KV** is an effective system-level optimization

---

## Conclusion

> **Decoder-only LLM inference becomes memory-bandwidth-bound due to KV cache growth long before GPU memory capacity is reached. Paged KV mitigates this bottleneck and substantially improves decode throughput and latency.**

---

## Usage

```bash
git clone https://github.com/<your-username>/LLM_Engineering.git
cd LLM_Engineering
```

Run HuggingFace benchmarks:

```bash
python bm.py
```

Run vLLM benchmarks:

```bash
python bench_vllm.py
```

---

## Future Work

* Compare paged vs static KV under concurrent workloads
* Evaluate TensorRT-LLM vs vLLM on identical benchmarks
* Study FP8 effects on KV footprint and decode throughput

```



#########################################################
1) Install build deps
apt-get update && apt-get install -y \
  git git-lfs build-essential cmake ninja-build \
  python3-dev python3-pip \
  libopenmpi-dev openmpi-bin \
  wget curl pkg-config
python3 -m pip install -U pip setuptools wheel

2) Confirm NVIDIA libs exist (critical)

You need CUDA + TensorRT available in the container.

nvidia-smi
python3 -c "import torch; print(torch.__version__)"
python3 -c "import tensorrt as trt; print(trt.__version__)"


If import tensorrt fails, you can’t build TRT-LLM properly in this container (you’d need a base image that includes TensorRT).

3) Clone TensorRT-LLM source
cd /workspace
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git lfs install
git submodule update --init --recursive
git lfs pull

4) Build wheel from source

TensorRT-LLM provides a wheel builder script:

python3 scripts/build_wheel.py


Then install the wheel (path may vary; list it):

ls -lah build/*.whl
pip install build/*.whl

5) Verify
python3 -c "import tensorrt_llm; print('tensorrt_llm OK')"


This follows NVIDIA’s build-from-source approach, just without Docker.
(If you want, I can also give you the exact build flags to speed up compilation on 4090.)
