Serving System Architecture Design

Create:

analysis/serving_design.md


I’ll walk you through the structure. You write it as we go.

🏗 1️⃣ High-Level Architecture

Start with this:

Overview

This section describes a minimal LLM serving system designed to:

Handle concurrent user requests

Optimize GPU utilization via batching

Balance latency and throughput

Respect memory constraints

High-Level Flow

Client → API Layer → Request Queue → Dynamic Batcher → GPU Worker → Response

Explain each:

1. API Layer

Receives HTTP/gRPC requests

Validates input

Pushes requests to queue

2. Request Queue

FIFO structure

Stores pending inference jobs

Enables batching

3. Dynamic Batcher

Groups requests within a time window (e.g., 10–20ms)

Max batch size cap (e.g., 8)

Prevents GPU underutilization

4. GPU Worker

Executes inference

Uses FP16

Maintains KV cache per request

5. Response Handler

Maps outputs back to users

Returns generated tokens

🧠 Why This Matters

This shows you understand:

Queueing

Scheduling

Batching tradeoffs

GPU constraints

⚖️ 2️⃣ Dynamic Batching Strategy

Add a section:

Dynamic Batching Policy

Parameters:

max_batch_size = 8

max_wait_time = 20 ms

Logic:

Collect incoming requests.

If batch reaches max size → dispatch immediately.

If wait time exceeds threshold → dispatch partial batch.

Otherwise continue collecting.

Tradeoff:

Larger batches → better throughput

Smaller batches → lower latency

Explain clearly:

Batch size selection must consider KV memory growth and bandwidth saturation.

You proved this experimentally.

🧮 3️⃣ Resource Constraints

Add:

Memory Constraints

Given:

KV ∝ batch × sequence × layers × hidden

Implications:

Larger batch reduces memory headroom.

Long context reduces maximum batch.

FP16 required to maintain feasible memory footprint.

Bandwidth Constraints

Decode is memory-bound.

At high batch:

Per-token latency increases.

Throughput scaling flattens.

System must cap batch before bandwidth saturation.

🚀 4️⃣ Scaling Beyond One GPU

Add:

Multi-GPU Scaling Strategy

Option 1: Data Parallel

Replicate model across GPUs.

Shard requests across devices.

Linear throughput scaling until network becomes bottleneck.

Option 2: Tensor Parallel

Split model layers across GPUs.

Required for larger models (70B+).

Introduces communication overhead.

Network Bottlenecks

AllReduce overhead in tensor parallel.

PCIe vs NVLink bandwidth differences.

Latency increases with inter-GPU communication.

🎯 5️⃣ Failure Modes

This is what makes you senior.

Add:

Failure Scenarios

OOM due to large batch or long context.

Latency spikes from excessive batching.

GPU underutilization from low traffic.

Memory fragmentation.

Mitigation:

Batch cap

Context cap

Admission control

Horizontal scaling