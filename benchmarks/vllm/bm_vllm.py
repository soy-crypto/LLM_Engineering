import argparse
import time
import csv
from typing import List, Tuple

import torch
from vllm import LLM, SamplingParams


# ----------------------------
# Args
# ----------------------------
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM LLM Benchmark")
    
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--batch_size", type=str, default="1,2,4,8")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--backend", type=str, default="vLLM")

    return parser.parse_args()


# ----------------------------
# Device
# ----------------------------
def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------
# Prompts
# ----------------------------
def get_prompts(path: str) -> List[str]:

    prompts: List[str] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)

    if not prompts:
        raise ValueError("No prompts found")

    return prompts


# ----------------------------
# Batch sizes
# ----------------------------
def get_sizes(batch_size: str) -> List[int]:
    return [int(x.strip()) for x in batch_size.split(",") if x.strip()]


# ----------------------------
# Build batch
# ----------------------------
def build_batch(prompts: List[str], size: int) -> List[str]:
    return (prompts * (size // len(prompts) + 1))[:size]


# ----------------------------
# GPU memory
# ----------------------------
def get_gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0

    torch.cuda.synchronize()

    return torch.cuda.memory_allocated() / (1024**2)


# ----------------------------
# Measure TTFT
# ----------------------------
def measure_ttft(llm: LLM, batch: List[str]) -> float:
    params = SamplingParams(max_tokens=1, temperature=0.0)
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    llm.generate(batch, params)
    
    torch.cuda.synchronize()
    end = time.perf_counter()

    return end - start

# ----------------------------
# Measure throughput
# ----------------------------
def measure_throughput(llm: LLM, batch: List[str], max_tokens: int) -> Tuple[float, int]:

    params = SamplingParams( max_tokens=max_tokens, temperature=0.0)

    torch.cuda.synchronize()
    start = time.perf_counter()
    
    outputs = llm.generate(batch, params)
    
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    elapsed = end - start
    total_tokens = 0

    for r in outputs:
        total_tokens += len(r.outputs[0].token_ids)

    return elapsed, total_tokens

# ----------------------------
# Main
# ----------------------------
def main():

    args = get_args()
    device = get_device()
    prompts = get_prompts(args.prompts)
    batch_sizes = get_sizes(args.batch_size)

    print(f"model: {args.model}")
    print(f"device: {device}")
    print(f"dtype: {args.dtype}")
    print("-" * 90)

    # ----------------------------
    # Initialize engine ONCE
    # ----------------------------
    llm = LLM(model=args.model, dtype=args.dtype, gpu_memory_utilization=args.gpu_mem_util, enforce_eager=args.enforce_eager, tensor_parallel_size=1)

    # ----------------------------
    # CSV
    # ----------------------------
    file_exists = False
    try:
        open(args.out_csv)
        file_exists = True
    except:
        pass


    with open(args.out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["backend", "model", "dtype", "device", "batch_size", "avg_ttft", "avg_latency", "tokens_per_sec", "gpu_mem_mb"])

        # ----------------------------
        # Loop batch sizes
        # ----------------------------
        for size in batch_sizes:
            batch = build_batch(prompts, size)
            print(f"\nBatch size {size}")
            
            # Warmup
            for _ in range(args.warmup):
                measure_throughput( llm, batch, min(args.max_new_tokens, 16) )

            ttfts = []
            lats = []
            tokps = []

            for _ in range(args.runs):
                ttft = measure_ttft(llm, batch)
                lat, tokens = measure_throughput(llm, batch, args.max_new_tokens)
                ttfts.append(ttft)
                lats.append(lat)
                tokps.append(tokens / lat)

            avg_ttft = sum(ttfts) / len(ttfts)
            avg_lat = sum(lats) / len(lats)
            avg_tokps = sum(tokps) / len(tokps)
            gpu_mem = get_gpu_mem_mb()

            print(
                f"[{size}] "
                f"TTFT={avg_ttft:.4f}s "
                f"LAT={avg_lat:.4f}s "
                f"TOK/s={avg_tokps:.2f} "
                f"GPU_mem={gpu_mem:.0f}MB"
            )


            writer.writerow([args.backend, args.model, args.dtype, device, size, f"{avg_ttft:.6f}", f"{avg_lat:.6f}", f"{avg_tokps:.2f}", f"{gpu_mem:.0f}"])

    print("\nDone.")
    print("Results saved to:", args.out_csv)

# ----------------------------
# Entry
# ----------------------------

if __name__ == "__main__":

    try:
        main()
    except Exception as e:
        print("Error:", e)