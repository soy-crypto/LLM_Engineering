

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
1) Basic pip install (fast path)
python3 -m pip install -U pip setuptools wheel
python3 -m pip install tensorrt_llm


Verify:

python3 -c "import tensorrt_llm; print('tensorrt_llm OK')"




Next (minimal): confirm build tools + CUDA compiler

Run:

trtllm-build --help | head
nvcc --version


Next (simple): clone TensorRT-LLM repo (just for examples)
cd /workspace
apt-get update && apt-get install -y git git-lfs
git lfs install
git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull


Quick check:

ls -lah examples | head

Then we run the exact engine build flow (TinyLlama first)
1) Download model
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('TinyLlama/TinyLlama-1.1B-Chat-v1.0', local_dir='/workspace/models/tinyllama', local_dir_use_symlinks=False)"

2) Convert checkpoint
python3 examples/llama/convert_checkpoint.py \
  --model_dir /workspace/models/tinyllama \
  --output_dir /workspace/trt_ckpt_tinyllama \
  --dtype float16

3) Build engine
trtllm-build \
  --checkpoint_dir /workspace/trt_ckpt_tinyllama \
  --output_dir /workspace/trt_engine_tinyllama_b32_o256 \
  --max_batch_size 32 \
  --max_input_len 1024 \
  --max_output_len 256

4) Benchmark
python3 examples/run.py \
  --engine_dir /workspace/trt_engine_tinyllama_b32_o256 \
  --batch_size 32 \
  --max_output_len 256

If you want ultra-minimal right now

Run only this first and paste output:

python3 -c "import tensorrt_llm; import inspect; print(tensorrt_llm.__file__)"


But you don’t need it — cloning repo is the cleanest path.





##################################################################
🚀 LLM Inference Scaling Study: Architecture × Hardware × Backend

GPU: NVIDIA RTX 5090
Precision: bfloat16
Models: Qwen2.5 (0.5B / 1.5B / 7B)
Backends: HuggingFace (eager) vs vLLM (paged KV)

📌 Objective

This project investigates how:

Transformer architecture

KV cache growth

Batch size

Sequence length

Backend implementation

GPU hardware

interact to determine real-world LLM inference performance.

The goal is to identify:

When inference is compute-bound

When it becomes memory-bandwidth-bound

How paged KV affects performance across regimes

🧠 Phase 1 — Theoretical KV Analysis (ML Layer)

For decoder-only transformers:

𝐾
𝑉
_
𝑀
𝐵
/
𝑡
𝑜
𝑘
𝑒
𝑛
=
2
×
𝐿
×
ℎ
𝑖
𝑑
𝑑
𝑒
𝑛
_
𝑠
𝑖
𝑧
𝑒
×
𝑏
𝑦
𝑡
𝑒
𝑠
/
1024
2
KV_MB/token=2×L×hidden_size×bytes/1024
2
7B Model Config

Layers = 28

Hidden size = 3584

bf16 (2 bytes)

𝐾
𝑉
≈
0.3828
 MB/token (batch=1)
KV≈0.3828 MB/token (batch=1)

Empirical KV measurements matched linear scaling across:

batch

token length

This validated architectural reasoning.

⚙️ Phase 2 — Batch Scaling
Qwen 7B — max_new_tokens=128
Batch	tok/s(new)	KV MB
1	80	
2	155	
4	308	
8	608	
16	1118	

Scaling is near-linear up to batch=8, with mild deviation at batch=16.

This indicates:

Increasing compute pressure

Beginning of scaling curvature

📈 Phase 3 — Token Scaling (Decode Behavior)
Qwen 7B — batch=8
Tokens	tok/s(new)	KV MB
128	607	73
256	606	129
512	597	241

Throughput remains nearly flat despite KV doubling.

Conclusion:

RTX 5090 is not bandwidth-bound at ~240MB KV.

Qwen 7B — batch=16
Tokens	tok/s(new)	KV MB
256	1122	259
512	1110	483

Slight throughput reduction, but still mostly compute-bound.

🚀 Phase 4 — Backend Comparison (HF vs vLLM)
7B — batch=16, new_tokens=512
Backend	tok/s(new)	Latency
HF eager	1110	7.38s
vLLM (paged KV)	1530	5.35s

Throughput improvement:

 
1.38
×
 1.38×

Interpretation:

Memory pressure exists but is not dominant

Compute cost is primary bottleneck

Paged KV still reduces memory traffic and improves efficiency

🔥 Cross-Regime Insight

Earlier small-model experiments showed:

Strong memory-bound behavior

4–5× gains from paged KV

With 7B on RTX 5090:

System is largely compute-bound

Paged KV provides moderate gains (~38%)

This demonstrates:

Optimization impact depends on bottleneck regime.

🎯 Final Conclusions

KV cache scales linearly with architecture parameters.

RTX 5090 sustains near-linear decode scaling for 7B models up to ~480MB KV.

Large models shift bottleneck from memory bandwidth to compute.

Paged KV improves performance in both regimes, but magnitude depends on hardware and model scale.






##########################TensorRT-LLM##################################
#Resolve your local HF model directory (no redownload)
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_files_only=True))"
export HF_MODEL_DIR="</path/printed/by/command>"

#Convert HF → TensorRT-LLM checkpoint (Qwen converter)
python3 -c "import tensorrt_llm, os; print(os.path.dirname(tensorrt_llm.__file__))"

#Set TRTLLM_ROOT
export TRTLLM_ROOT="/app/tensorrt_llm"



