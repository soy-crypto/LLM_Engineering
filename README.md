# LLM_Engineering
##############Pod configuration.##################

GPU: RTX4090
Container image: nvcr.io/nvidia/tritonserver:24.07-trtllm-python-py3
Container Start Command:
bash -c 'apt update; \
DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y; \
mkdir -p ~/.ssh; \
cd ~/.ssh; \
chmod 700 ~/.ssh; \
echo "$PUBLIC_KEY" >> authorized_keys; \
chmod 700 authorized_keys; \
service ssh start; \
sleep infinity'

Container disk: 200GB Volume disk 200GB Volume mount.

##########How to use ssh and sftp##########
step1:
SSH key genarate:
ssh-keygen -t ed25519 -C "email@xxx.xxx"
step2: rename ed25519 to ed25519.ppk (OS is windows)
step3: open SCP, host: 157.157.221.29 port: 30527 username: root. no need password, as we have private key in our machine.
step4: connect the host.


###########git operation#########
alias quicksave='git add . && git commit -m "$(date +%m%d)" && git push'

'''



LLM Inference Performance Analysis & KV Cache Optimization

GPU: RTX 4090
Model: Qwen2.5-0.5B-Instruct
Backends: HuggingFace (eager), vLLM (paged KV)
Focus: Decode throughput, KV cache behavior, memory-bandwidth bottlenecks

Overview

This project investigates real-world LLM inference bottlenecks on GPU, with a focus on:

Separating prefill vs decode latency

Measuring KV cache memory growth correctly

Identifying memory-bandwidth saturation as a decode bottleneck

Demonstrating paged KV as a system-level optimization

All conclusions are based on controlled experiments, ground-truth measurements, and apples-to-apples backend comparisons.

Why This Matters

For decoder-only LLMs, inference performance often degrades long before GPU memory is exhausted.
The limiting factor is frequently KV cache growth, which increases memory traffic and saturates bandwidth during decode.

This repository shows:

when that happens,

why it happens,

and how paged KV fixes it.

Experimental Setup
Hardware

NVIDIA RTX 4090 (24 GB VRAM)

Model

Qwen/Qwen2.5-0.5B-Instruct

Precision

bfloat16

Backends

HuggingFace Transformers (eager execution)

vLLM (paged KV)

Metrics (Used Correctly)
Metric	Meaning
TTFT	Time-to-first-token (prefill latency)
tok/s (new)	Decode throughput (new tokens only)
KV cache (MB)	Memory footprint computed directly from past_key_values tensors

⚠️ Allocator-based GPU memory stats were intentionally avoided for KV analysis, as they obscure true KV growth due to caching and reuse.

Methodology
KV Cache Measurement (Ground Truth)

KV memory is computed directly from model outputs:

past = outputs.past_key_values
kv_bytes = 0
for layer in past:
    k, v = layer
    kv_bytes += k.numel() * k.element_size()
    kv_bytes += v.numel() * v.element_size()
kv_mb = kv_bytes / (1024 ** 2)


This avoids allocator artifacts and reflects actual KV tensor size.

Results
1. KV Cache Scaling (Batch × Tokens)

max_new_tokens = 128

Batch	KV Cache (MB)
1	~1.98
2	~3.96
4	~7.92
8	~15.84
16	~31.69
32	~63.38

✅ KV cache scales linearly with batch size, matching theoretical expectations.

2. Decode Throughput Scaling (HF Eager)
Batch	tok/s (new)
1	~110
2	~215
4	~423
8	~851
16	~1660
32	~3413

Throughput scales with batch size until KV growth increases memory traffic.

3. Bottleneck Identification

At batch = 32:

New Tokens	tok/s (new)	KV Cache (MB)
128	~3413	~63
256	~3199	~111

KV size nearly doubled

Throughput decreased

VRAM capacity remained far from full

❗ Conclusion: Decode becomes memory-bandwidth-bound, not compute-bound or capacity-bound.

4. System-Level Fix: Paged KV (vLLM)

Same workload using vLLM:

batch = 32
max_new_tokens = 256

Backend	tok/s (new)	Latency
HF eager	~3199	~2.56 s
vLLM (paged KV)	~15111	~0.54 s

🚀 ~4.7× throughput improvement

Paged KV dramatically reduces memory traffic, relieving bandwidth pressure during decode.

Key Takeaways

Decode performance degrades due to KV cache growth, not VRAM exhaustion

KV cache size scales linearly with batch × generated tokens

Memory-bandwidth saturation is the primary bottleneck at scale

Paged KV is a highly effective system-level optimization

Conclusion

Inference performance for decoder-only LLMs becomes memory-bandwidth-bound due to KV cache growth long before GPU memory capacity is reached. Paged KV mitigates this bottleneck and significantly improves decode throughput and latency under high batch and long-sequence workloads.