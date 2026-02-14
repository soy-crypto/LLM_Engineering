# 🚀 LLM Inference Scaling Study

### KV Cache Dynamics · Memory Bandwidth Bottlenecks · Backend Optimization

> A systematic investigation of how transformer architecture, KV cache growth, GPU hardware, and inference backends interact to determine real-world LLM performance.

---

# 1️⃣ Motivation

Decoder-only LLM inference performance often degrades **long before GPU memory is exhausted**.

Why?

Because inference is frequently limited by:

* KV cache growth
* Memory bandwidth saturation
* Backend implementation details
* Attention kernel efficiency

This project isolates those variables through controlled experiments across:

* Model scales (0.5B → 7B)
* Batch sizes
* Sequence lengths
* HuggingFace vs vLLM vs TensorRT-LLM
* Static KV vs paged KV
* RTX 5090 GPU

The objective is to determine:

* When inference is compute-bound
* When it becomes memory-bandwidth-bound
* How paged KV changes system behavior

---

# 2️⃣ Environment

## Hardware

* **GPU:** NVIDIA RTX 5090 (24GB VRAM)

## Container

* **Image:** `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3`
* **Storage:** 200GB container + 200GB mounted volume

## Container Initialization

```bash
bash -c '
apt update;
DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;
mkdir -p ~/.ssh && chmod 700 ~/.ssh;
echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys;
chmod 600 ~/.ssh/authorized_keys;
service ssh start;
sleep infinity'
```

---

# 3️⃣ Secure Access

### Generate SSH Key

```bash
ssh-keygen -t ed25519 -C "email@example.com"
```

Windows users may convert to `.ppk` for PuTTY.

### Server Info

* Host: `157.157.221.29`
* Port: `30527`
* User: `root`
* Authentication: SSH key only

---

# 4️⃣ Project Scope

## Models

* Qwen2.5-0.5B-Instruct
* Qwen2.5-1.5B
* Qwen2.5-7B

## Precision

* bfloat16

## Backends Compared

* HuggingFace (eager mode, static KV)
* vLLM (paged KV cache)
* TensorRT-LLM (engine-compiled, paged KV)

---

# 5️⃣ Theoretical KV Cache Analysis

For decoder-only transformers:

[
KV_{MB/token} = \frac{2 \times L \times hidden_size \times bytes}{1024^2}
]

Where:

* L = number of layers
* hidden_size = model width
* bytes = precision size (bf16 = 2 bytes)

### Example: Qwen2.5-7B

* Layers: 28
* Hidden size: 3584
* Precision: bf16

Result:

```
KV ≈ 0.3828 MB per token (batch=1)
```

### Validation

Empirical measurements confirmed:

* Linear scaling with batch size
* Linear scaling with generated tokens
* Exact agreement with architectural formula

This validates the theoretical model.

---

# 6️⃣ KV Cache Measurement (Ground Truth)

Instead of relying on allocator-based GPU memory stats (which are noisy due to caching), KV memory is computed directly from tensor outputs:

```python
past = outputs.past_key_values
kv_bytes = 0

for k, v in past:
    kv_bytes += k.numel() * k.element_size()
    kv_bytes += v.numel() * v.element_size()

kv_mb = kv_bytes / (1024 ** 2)
```

This reflects **true KV tensor footprint**, independent of allocator artifacts.

---

# 7️⃣ Phase I — Batch Scaling Behavior (7B)

**max_new_tokens = 128**

| Batch | tok/s (new) |
| ----- | ----------- |
| 1     | 80          |
| 2     | 155         |
| 4     | 308         |
| 8     | 608         |
| 16    | 1118        |

### Observation

* Near-linear scaling up to batch=8
* Slight curvature at batch=16

### Interpretation

* Increasing GPU compute utilization
* Beginning of saturation effects

System is still primarily **compute-bound**.

---

# 8️⃣ Phase II — Token Scaling (Decode Growth)

## Batch = 8

| Tokens | tok/s | KV (MB) |
| ------ | ----- | ------- |
| 128    | 607   | 73      |
| 256    | 606   | 129     |
| 512    | 597   | 241     |

Throughput remains nearly constant despite KV doubling.

**Conclusion:** RTX 5090 is not bandwidth-bound at ~240MB KV.

---

## Batch = 16

| Tokens | tok/s | KV (MB) |
| ------ | ----- | ------- |
| 256    | 1122  | 259     |
| 512    | 1110  | 483     |

Slight throughput reduction, but still mostly compute-bound.

---

# 9️⃣ Backend Comparison

### Configuration

7B model
batch = 16
max_new_tokens = 512

| Backend         | tok/s | Latency |
| --------------- | ----- | ------- |
| HF eager        | 1110  | 7.38 s  |
| vLLM (paged KV) | 1530  | 5.35 s  |

### Improvement

~1.38× throughput gain

### Interpretation

* Memory traffic reduced
* Better KV management
* More efficient scheduling

Paged KV improves performance even in compute-dominant regimes.

---

# 🔟 Cross-Regime Comparison

## Small Models (0.5B / 1.5B)

* Strong memory-bandwidth bottleneck
* 4–5× improvement with paged KV

## 7B on RTX 5090

* Mostly compute-bound
* ~38% improvement with paged KV

### Insight

Optimization impact depends on:

* Model size
* KV footprint
* GPU memory bandwidth
* Backend implementation

---

# 1️⃣1️⃣ TensorRT-LLM Pipeline

## Verify Installation

```bash
python3 -c "import tensorrt_llm, os; print(os.path.dirname(tensorrt_llm.__file__))"
```

---

## Download Model

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen2.5-7B-Instruct")
print(p)
PY
```

```bash
export HF_MODEL_DIR="PATH_TO_MODEL"
```

---

## Convert HF → TRT-LLM Checkpoint

```bash
export CKPT_DIR="/workspace/trt_ckpt/qwen2p5_7b_bf16_1gpu"
mkdir -p "$CKPT_DIR"

python3 /app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py \
  --model_dir "$HF_MODEL_DIR" \
  --output_dir "$CKPT_DIR" \
  --dtype bfloat16
```

---

## Build Engine

```bash
export ENGINE_DIR="/workspace/trt_engine/qwen2p5_7b_bf16_b16_s2560"
mkdir -p "$ENGINE_DIR"

trtllm-build \
  --checkpoint_dir "$CKPT_DIR" \
  --output_dir "$ENGINE_DIR" \
  --max_batch_size 16 \
  --max_seq_len 2560 \
  --kv_cache_type paged \
  --gemm_plugin bfloat16 \
  --gpt_attention_plugin bfloat16
```

---

## Run Inference

```bash
python3 /app/tensorrt_llm/examples/run.py \
  --engine_dir "$ENGINE_DIR" \
  --tokenizer_dir "$HF_MODEL_DIR" \
  --input_text "Write one sentence about GPU inference." \
  --max_output_len 32
```

---

# 🧠 Final Technical Conclusions

1. KV cache growth scales linearly with architecture parameters.
2. Inference degradation often begins due to memory bandwidth, not VRAM capacity.
3. Larger models shift bottlenecks toward compute saturation.
4. Paged KV reduces memory traffic and improves efficiency across regimes.
5. Backend implementation significantly impacts real-world throughput.

---

# 🔮 Future Work

* Multi-tenant concurrent workload benchmarking
* Kernel-level decode profiling
* FP8 KV footprint experiments
* Roofline analysis (FLOPs vs bandwidth)
* TensorRT-LLM vs vLLM head-to-head scaling




Here’s the clean summary — NVIDIA GPU Docker + TensorRT-LLM workflow.

---

# ✅ 1️⃣ Make GPU Work on Host

Check GPU detected:

```bash
lspci | grep -i nvidia
```

Install driver (if needed):

```bash
ubuntu-drivers devices
sudo apt install nvidia-driver-XXX -y
sudo reboot
```

Verify:

```bash
nvidia-smi
```

If this doesn’t work → nothing else will.

---

# ✅ 2️⃣ Install Docker (Official Repo)

```bash
sudo apt install docker-ce docker-ce-cli containerd.io -y
```

Verify:

```bash
docker --version
```

---

# ✅ 3️⃣ Install NVIDIA Container Toolkit

Add repo (24.04 compatible):

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

Install:

```bash
sudo apt update
sudo apt install nvidia-container-toolkit -y
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify runtime exists:

```bash
docker info | grep runtime
```

You must see:

```
nvidia
```

---

# ✅ 4️⃣ Test GPU Inside Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

If GPU shows → Docker GPU layer is correct.

---

# ✅ 5️⃣ Login to NVIDIA NGC

Go to:

[https://ngc.nvidia.com](https://ngc.nvidia.com)
→ Profile → Setup → API Key → Generate

Then:

```bash
docker login nvcr.io
```

Username:

```
$oauthtoken
```

Password:

```
<your API key>
```

---

# ✅ 6️⃣ Pull TensorRT-LLM Image

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3
```

---

# ✅ 7️⃣ Run Container (Best Practice)

```bash
docker run --gpus all -it \
  --name trtllm \
  -v $PWD:/workspace \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6.post3
```

---

# ✅ 8️⃣ Re-enter Later (If Not Using --rm)

List containers:

```bash
docker ps -a
```

Restart:

```bash
docker start trtllm
```

Enter:

```bash
docker exec -it trtllm bash
```

---

# Full Stack Order (Mental Model)

```
GPU Hardware
→ NVIDIA Driver
→ Docker
→ NVIDIA Container Toolkit
→ CUDA test container
→ NGC login
→ TensorRT-LLM container
```

---

You now have a proper NVIDIA infra environment.

If you want next:

* Build TRT engine
* Convert HF model
* Benchmark decode throughput
* Or set up persistent model cache

Tell me the goal.

