1️⃣ If Traffic Increases 10×, What Breaks First?

First thing that breaks:

➤ GPU saturation

What happens:

Request queue grows

Batch sizes increase

Per-token latency increases

P95 latency spikes

Why?

From your experiments:

Decode is memory-bound

Batch scaling eventually saturates bandwidth

Throughput scales sublinearly

So at 10× traffic:

Single GPU cannot keep up.
Queue delay dominates latency.

Fix:

Data parallel replicas

Horizontal scaling

2️⃣ If Model Size Increases (e.g., 70B), What Breaks First?

First thing that breaks:

➤ Memory capacity

Why?

From your FP16 experiment:

8B model already uses ~17GB.

70B model ≈ ~8–9× bigger.

Even FP16 won’t fit on single GPU.

You must:

Use tensor parallel

Shard model weights

Accept communication overhead

New bottleneck:

Inter-GPU bandwidth (NVLink / PCIe)

AllReduce latency

3️⃣ If Context Length Increases to 32k, What Breaks First?

First thing that breaks:

➤ KV cache memory

Because:

KV ∝ batch × sequence

Sequence increases 8× (4k → 32k)

KV explodes.

Memory pressure increases.
Bandwidth pressure increases.

Decode latency increases dramatically.

Fix:

Limit context

Use paged KV (vLLM-style)

Reduce batch

Add more GPUs

4️⃣ If Latency SLO Must Be <1 Second, What Breaks?

Batching strategy breaks.

Why?

From load test:

Single decode ≈ 3.2s

Batching adds queue delay

To hit <1s:

Must reduce max_new_tokens

Must reduce batch size

Must reduce context

Must increase GPU replicas

Tradeoff:

Lower latency → lower throughput efficiency → higher cost.

5️⃣ If Requests Are Uneven (Short + Long Prompts Mixed)?

Batch inefficiency appears.

Why?

Long prompt:

Long prefill

High KV growth

Short prompt:

Finishes quickly

If batched together:

Short request waits for long one

Tail latency increases

Fix:

Length-aware batching

Separate queues

Priority scheduling

This is advanced systems thinking.

🔥 What You Should Notice

Every scenario ties back to:

Compute limits

Memory capacity

Memory bandwidth

Communication overhead

Scheduling policy

That’s ML Systems thinking.

🎯 The Real Level-Up

You now understand:

Single GPU inference behavior
Batch scaling limits
Precision impact
Memory-bound decode
Data parallel vs tensor parallel tradeoffs

That is a full-stack inference systems foundation.

If you stopped here and interviewed,
you would already be above many candidates.

Now the only thing missing is:

Clear, confident articulation.

If you want, I can now compress everything you built into a:

30-second summary

2-minute explanation

5-minute deep walkthrough

That’s the final transformation step.