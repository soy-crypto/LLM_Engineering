Got it 👍
Below is **everything consolidated into one clean Markdown document**, ready to paste into **GitHub** (either as `README.md` or split later if you want).

You can copy **exactly this**.

---

````markdown
# LLM_Engineering

LLM inference performance analysis, KV cache scaling, and paged-KV optimization on RTX 4090.

---

## Environment & Pod Configuration

### Hardware
- **GPU:** NVIDIA RTX 4090 (24GB VRAM)

### Container
- **Image:** `nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3`
- **Container Disk:** 200GB
- **Volume Disk:** 200GB (mounted)

### Container Start Command
```bash
bash -c '
apt update;
DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;
mkdir -p ~/.ssh;
cd ~/.ssh;
chmod 700 ~/.ssh;
echo "$PUBLIC_KEY" >> authorized_keys;
chmod 700 authorized_keys;
service ssh start;
sleep infinity'
````

---

## SSH & SFTP Usage

### Step 1: Generate SSH Key

```bash
ssh-keygen -t ed25519 -C "email@xxx.xxx"
```

### Step 2: Rename Key (Windows)

Rename:

```
id_ed25519 → id_ed25519.ppk
```

### Step 3: SCP / SFTP Connection

* **Host:** `157.157.221.29`
* **Port:** `30527`
* **Username:** `root`
* **Authentication:** Private key only (no password)

### Step 4: Connect

Use an SCP/SFTP client (e.g., WinSCP) to connect.

---

## Git Workflow

### Quick Commit Alias

```bash
alias quicksave='git add . && git commit -m "$(date +%m%d)" && git push'
```

---

# LLM Inference Performance Analysis & KV Cache Optimization

* **GPU:** RTX 4090
* **Model:** Qwen2.5-0.5B-Instruct
* **Backends:** HuggingFace (eager), vLLM (paged KV)
* **Focus:** Decode throughput, KV cache behavior, memory-bandwidth bottlenecks

---

## Overview

This project investigates **real-world LLM inference bottlenecks on GPU**, focusing on:

* Separating **prefill vs decode latency**
* Measuring **KV cache memory growth correctly**
* Identifying **memory-bandwidth saturation** as a decode bottleneck
* Demonstrating **paged KV** as a system-level optimization

All conclusions are based on **controlled experiments**, **ground-truth measurements**, and **apples-to-apples backend comparisons**.

---

## Why This Matters

For decoder-only LLMs, inference performance often degrades **long before GPU memory is exhausted**.
The limiting factor is frequently **KV cache growth**, which increases memory traffic and saturates bandwidth during decode.

This repository shows:

* **When** that happens
* **Why** it happens
* **How paged KV fixes it**

---

## Experimental Setup

### Hardware

* NVIDIA RTX 4090 (24GB VRAM)

### Model

* `Qwen/Qwen2.5-0.5B-Instruct`

### Precision

* `bfloat16`

### Backends

* HuggingFace Transformers (eager execution)
* vLLM (paged KV)

---

## Metrics (Used Correctly)

| Metric            | Meaning                                                           |
| ----------------- | ----------------------------------------------------------------- |
| **TTFT**          | Time-to-first-token (prefill latency)                             |
| **tok/s (new)**   | Decode throughput (new tokens only)                               |
| **KV cache (MB)** | Memory footprint computed directly from `past_key_values` tensors |

> ⚠️ Allocator-based GPU memory stats were intentionally avoided for KV analysis, as they obscure true KV growth due to caching and reuse.

---

## Methodology

### KV Cache Measurement (Ground Truth)

KV memory is computed directly from model outputs:

```python
past = outputs.past_key_values
kv_bytes = 0

for layer in past:
    k, v = layer
    kv_bytes += k.numel() * k.element_size()
    kv_bytes += v.numel() * v.element_size()

kv_mb = kv_bytes / (1024 ** 2)
```

This avoids allocator artifacts and reflects **actual KV tensor size**.

---

## Results

### KV Cache Scaling (Batch × Tokens)

**max_new_tokens = 128**

| Batch | KV Cache (MB) |
| ----: | ------------: |
|     1 |         ~1.98 |
|     2 |         ~3.96 |
|     4 |         ~7.92 |
|     8 |        ~15.84 |
|    16 |        ~31.69 |
|    32 |        ~63.38 |

✅ KV cache scales **linearly with batch size**, matching theoretical expectations.

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

### Bottleneck Identification

At **batch = 32**:

| New Tokens | tok/s (new) | KV Cache (MB) |
| ---------: | ----------: | ------------: |
|        128 |       ~3413 |           ~63 |
|        256 |       ~3199 |          ~111 |

* KV size nearly doubled
* Throughput **decreased**
* VRAM capacity remained far from full

❗ **Conclusion:** Decode becomes **memory-bandwidth-bound**, not compute-bound or capacity-bound.

---

### System-Level Fix: Paged KV (vLLM)

Same workload using **vLLM**:

```text
batch = 32
max_new_tokens = 256
```

| Backend         | tok/s (new) |     Latency |
| --------------- | ----------: | ----------: |
| HF eager        |       ~3199 |     ~2.56 s |
| vLLM (paged KV) |  **~15111** | **~0.54 s** |

🚀 **~4.7× throughput improvement**

Paged KV dramatically reduces memory traffic, relieving bandwidth pressure during decode.

---

## Key Takeaways

* Decode performance degrades due to **KV cache growth**, not VRAM exhaustion
* KV cache size scales linearly with **batch × generated tokens**
* **Memory-bandwidth saturation** is the primary bottleneck at scale
* **Paged KV** is a highly effective system-level optimization

---

## Conclusion

> **Inference performance for decoder-only LLMs becomes memory-bandwidth-bound due to KV cache growth long before GPU memory capacity is reached. Paged KV mitigates this bottleneck and significantly improves decode throughput and latency under high batch and long-sequence workloads.**

---

## How to Use This Repository

```bash
git clone https://github.com/<your-username>/LLM_Engineering.git
cd LLM_Engineering
```

Run HF benchmarks:

```bash
python bm.py
```

Run vLLM benchmarks:

```bash
python bench_vllm.py
```

---

## Resume-Ready Summary

* Benchmarked LLM inference on RTX 4090, decomposing prefill vs decode latency and identifying memory-bandwidth-bound decode behavior.
* Quantified KV cache memory directly from `past_key_values`, validating linear scaling with batch size and generated tokens.
* Demonstrated a ~4.7× decode throughput improvement by replacing HuggingFace eager inference with paged KV using vLLM.

---

## Future Work

* Compare paged KV vs static KV under multi-request concurrency
* Evaluate TensorRT-LLM vs vLLM on the same workload
* Study FP8 effects on KV footprint and decode throughput

---

````

---

### ✅ What to do next
1. Paste this into `README.md`
2. Commit & push:
```bash
quicksave
````

3. Pin the repo on GitHub

If you want next, I can:

* split this into **README + docs/**
* add **plots**
* tailor wording for **NVIDIA job descriptions**
* or structure a **portfolio page**

Just say the word.
