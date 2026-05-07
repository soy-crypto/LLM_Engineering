import argparse
import time
import torch
from vllm import LLM, SamplingParams

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common import load_prompts, parse_batch_sizes, build_batch, get_gpu_mem_mb, CsvWriter


def get_args():
    parser = argparse.ArgumentParser(description="vLLM LLM Benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--batch_size", type=str, default="1,2,4,8")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--backend", type=str, default="vLLM")
    return parser.parse_args()


def load_model(args) -> LLM:
    return LLM(
        model=args.model,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
    )


def measure_ttft(llm: LLM, batch) -> float:
    params = SamplingParams(max_tokens=1, temperature=0.0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(batch, params)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def measure_throughput(llm: LLM, batch, max_tokens: int):
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = llm.generate(batch, params)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_tokens = sum(len(r.outputs[0].token_ids) for r in outputs)
    return elapsed, new_tokens


def benchmark_batch(llm, batch, args):
    for _ in range(args.warmup):
        measure_throughput(llm, batch, min(16, args.max_new_tokens))

    ttft_list, latency_list, tokps_list = [], [], []
    for _ in range(args.runs):
        ttft = measure_ttft(llm, batch)
        lat, new_tokens = measure_throughput(llm, batch, args.max_new_tokens)
        ttft_list.append(ttft)
        latency_list.append(lat)
        tokps_list.append(new_tokens / lat if lat > 0 else 0)

    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_lat = sum(latency_list) / len(latency_list)
    avg_tokps = sum(tokps_list) / len(tokps_list)
    return avg_ttft, avg_lat, avg_tokps, get_gpu_mem_mb()


def main():
    args = get_args()
    prompts = load_prompts(args.prompts)
    batch_sizes = parse_batch_sizes(args.batch_size)
    llm = load_model(args)

    print(f"vLLM | model={args.model} | dtype={args.dtype}")
    print("-" * 80)

    writer = CsvWriter(args.out_csv)

    for bs in batch_sizes:
        batch = build_batch(prompts, bs)
        avg_ttft, avg_lat, avg_tokps, gpu_mem = benchmark_batch(llm, batch, args)
        print(f"[bs={bs}] ttft={avg_ttft:.4f}s  lat={avg_lat:.4f}s  tok/s={avg_tokps:.1f}  gpu={gpu_mem:.0f}MB")
        writer.write(args.backend, args.model, args.dtype, bs, avg_ttft, avg_lat, avg_tokps, gpu_mem)

    writer.close()
    print("-" * 80)
    print(f"Results saved to: {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
