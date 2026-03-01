# vLLM Llama-3.1-8B Scaling Study on L40S

## Setup
- GPU: NVIDIA L40S (48GB)
- Backend: vLLM (v0.16.0)
- Model: Llama-3.1-8B (BF16)
- Measurement: tokens/sec, first-token latency, batch scaling, context scaling
- Note: Nsight counters unavailable (ERR_NVGPUCTRPERM), so bottlenecks classified via scaling + cost model.

## Key Results
### Throughput vs Batch (context=4096, decode=128)
| batch | tok/s |
|------:|------:|
| 8     | 360   |
| 16    | 711   |
| 64    | 2480  |
| 1024  | 7030  |

Observations:
- 8→16 scaling efficiency ≈ 98.7% (near-linear).
- 64→1024 shows strong saturation (scaling efficiency drops sharply).

### Context Sensitivity (batch=8, decode=128)
| context | tok/s |
|--------:|------:|
| 512     | 360.96 |
| 2048    | 360.96 |
| 4096    | 360.76 |
| 8192    | 360.97 |

Observation: decode throughput ~flat across 16× context → not bandwidth-limited in this regime.

### Prefill vs Decode (batch=1)
(context=512, decode=128)
- first_token latency: 52.10 ms
- full_decode: 46.90 tok/s (≈21.3 ms/token)
- estimated prefill ≈ first_token - 1/decode_tok_s ≈ 52.1 - 21.3 ≈ 30.8 ms

(context=8192, decode=128)
- full_decode: 43.81 tok/s (≈22.8 ms/token)
- decode per-token increases ~7% from 512→8192.

## Bottleneck Classification (Scaling + Cost Model)
### KV-cache bandwidth estimate (decode)
Assume Llama-like config: layers=32, heads=32, head_dim=128, dtype=BF16(2B)

Bytes/token (KV reads) ≈ 2(K,V) * layers * heads * head_dim * context * 2 bytes
= 2*32*32*128*context*2B
≈ 524,288 * context bytes/token

At context=8192:
- bytes/token ≈ 4.29 GB
- required BW at ~44 tok/s ≈ 189 GB/s
- L40S peak BW ~864 GB/s → ~22% of peak

Conclusion: bandwidth headroom remains; decode mostly compute/overall-runtime limited at these settings.

## Saturation Point
- Near-linear scaling holds up to ~batch 16–64.
- Large batch (e.g., 1024) enters saturation regime: non-linear scaling indicates scheduler/runtime/compute ceiling effects.

## Next Steps
1) Repeat same sweep with TensorRT-LLM (same model, same prompts) and compare:
   - peak tok/s
   - scaling knee
   - first-token latency
2) Add controlled prompt tokens (real token count, not word repeats) and log input/output tokens explicitly.
3) Optional: run roofline-style estimate using refined FLOPs/token for Llama-3.1-8B.