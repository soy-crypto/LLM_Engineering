import argparse
import time
import csv
from typing import List, Tuple, Optional

import torch
from vllm import LLM, SamplingParams


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM LLM Benchmark")

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompts", type=str, default="prompts.txt")
    parser.add_argument("--batch_size", type=str, default="1,2,4,8,16,32")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--out_csv", type=str, default="vllm_results.csv")
    parser.add_argument("--backend", type=str, default="vLLM")

    return parser.parse_args()


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)

    if not prompts:
        raise ValueError(f"No prompts found in {path}")

    return prompts


def get_sizes(batch_size: str) -> List[int]:
    return [int(x.strip()) for x in batch_size.split(",") if x.strip()]


def build_batch(prompts: List[str], size: int) -> List[str]:
    return (prompts * (size // len(prompts) + 1))[:size]


def measure(llm: LLM, batch_prompts: List[str], max_tokens: int,) -> Tuple[float, int]:
    params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    outputs = llm.generate(batch_prompts, params)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end = time.perf_counter()

    elapsed = end - start

    gen_tokens = 0
    for r in outputs:
        gen_tokens += len(r.outputs[0].token_ids)

    return elapsed, gen_tokens


def main():
    args = get_args()
    device = get_device()
    prompts = get_prompts(args.prompts)
    batch_sizes = get_sizes(args.batch_size)

    print(f"model: {args.model} | device: {device} | dtype: {args.dtype}")
    print(f"prompts: {args.prompts} ({len(prompts)} lines)")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print("-" * 90)

    # Initialize vLLM engine once
    llm = LLM(model=args.model, dtype=args.dtype, gpu_memory_utilization=args.gpu_mem_util, enforce_eager=args.enforce_eager,tensor_parallel_size=1)

    # CSV header
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "model", "dtype", "device", "batch_size", "avg_ttft", "avg_latency", "tokps_new"])

        # Loop over batch sizes
        for size in batch_sizes:
            batch_prompts = build_batch(prompts, size)

            # Warmup
            for _ in range(args.warmup):
                _ = measure(llm, batch_prompts, min(args.max_new_tokens, 16))

            ttfts: List[float] = []
            lats: List[float] = []
            tokps_list: List[float] = []

            for _ in range(args.runs):
                tt, _ = measure(llm, batch_prompts, max_tokens=1)
                ttfts.append(tt)

                lat, gen_tokens = measure(llm, batch_prompts, max_tokens=args.max_new_tokens)
                lats.append(lat)
                tokps = gen_tokens / lat if lat > 0 else 0.0
                tokps_list.append(tokps)

            # Averages
            avg_ttft = sum(ttfts) / len(ttfts)
            avg_lat = sum(lats) / len(lats)
            avg_tokps_new = sum(tokps_list) / len(tokps_list)

            print(f"[{size}] " f"avg_ttft: {avg_ttft:.6f} | " f"avg_latency: {avg_lat:.6f} | " f"tok/s(new): {avg_tokps_new:.3f}")

            writer.writerow([args.backend, args.model, args.dtype, device, size, f"{avg_ttft:.6f}", f"{avg_lat:.6f}", f"{avg_tokps_new:.6f}"])
        
        pass

    print("-" * 90)
    print(f"Wrote CSV to: {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
