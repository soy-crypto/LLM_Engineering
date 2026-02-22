📘 LLM Inference Systems Findings
Overview

This project benchmarks meta-llama/Llama-3.1-8B to analyze inference bottlenecks across:

Sequence length scaling

Batch scaling

Precision scaling

Prefill vs decode behavior

The goal is to understand compute-bound vs memory-bound phases and derive production-level system implications.

1️⃣ Sequence Length Scaling
Observation

Per-token decode latency increased from ~25 ms (512 tokens) to ~30 ms (4096 tokens).

GPU memory increased linearly with sequence length (~1.3 GB growth).

Throughput decreased gradually as context length increased.

Prefill time grew faster than decode time.

Interpretation

Decode phase exhibits linear scaling with sequence length due to KV cache growth.

Prefill phase scales worse due to quadratic attention complexity (O(n²)).

System Insight

Longer context increases memory bandwidth pressure.

Decode becomes progressively memory-sensitive.

Long-context applications (RAG, document QA) will increase TTFT significantly.

Context window must be bounded or distributed across GPUs in production.

2️⃣ Prefill vs Decode Behavior
Observation

At 512 tokens:

Prefill ≈ 0.28s

Decode ≈ 3.06s

At 4096 tokens:

Prefill ≈ 0.59s

Decode ≈ 3.24s

Interpretation

Prefill is compute-bound.

Decode is memory-bound.

Decode dominates total latency for chat-style workloads.

Prefill becomes more significant for long-context workloads.

System Insight

Optimizing decode improves serving throughput.

Optimizing prefill improves TTFT.

Workload type determines optimization priority.

3️⃣ Batch Scaling
Observation

Throughput increased sublinearly:

Batch 1 → 39 tok/s

Batch 2 → 71 tok/s

Batch 4 → 127 tok/s

Batch 8 → 190 tok/s

Per-token latency increased significantly at batch 8.

OOM occurred at batch 16.

Memory scaled linearly with batch size.

Interpretation

Small batch sizes improve GPU utilization.

Larger batch sizes increase memory bandwidth pressure.

At higher batch sizes, decode becomes bandwidth-limited.

Memory capacity becomes the hard ceiling.

System Insight

Optimal batch range ≈ 2–8 for this model and context.

Dynamic batching must balance latency vs throughput.

Memory headroom determines maximum batch capacity.

4️⃣ Precision Scaling (FP16 vs FP32)
Observation

FP32 vs FP16:

Memory: ~17 GB → ~37 GB

Prefill slowed ~3×

Decode slowed ~2.6×

Throughput reduced ~2.5×

Interpretation

FP16 benefits from Tensor Core acceleration.

FP16 reduces memory footprint and bandwidth pressure.

FP32 significantly increases memory traffic and compute cost.

System Insight

FP16 (or lower precision) is mandatory for production inference.

Precision directly affects serving cost and scaling headroom.

5️⃣ Production Implications

From these experiments:

Decode dominates latency for chat workloads.

Prefill dominates for long-context tasks.

Throughput scaling is limited by memory bandwidth.

Memory capacity constrains batch size and context.

FP16 significantly improves cost-efficiency.

Optimal batch must balance latency SLO and GPU efficiency.


