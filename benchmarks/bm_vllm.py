import argparse
import time
import csv
from typing import List

import torch
from vllm import LLM, SamplingParams


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--prompts", type=str, default="prompts_mid.txt")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--out_csv", type=str, default="vllm_results.csv")
    # vLLM knobs (keep simple)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    p.add_argument("--gpu_mem_util", type=float, default=0.90)
    p.add_argument("--enforce_eager", action="store_true")  # usually keep False
    return p.parse_args()


def read_prompts(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
    if not out:
        raise ValueError(f"no prompts in {path}")
    return out


def main():
    #Get input parameters
    args = get_args()
    prompts = read_prompts(args.prompts)
    batch_prompts = (prompts * (args.batch // len(prompts) + 1))[: args.batch]
    llm = LLM(model=args.model, dtype=args.dtype, gpu_memory_utilization=args.gpu_mem_util, enforce_eager=args.enforce_eager, tensor_parallel_size=1)
    ttft_params = SamplingParams(max_tokens=1, temperature=0.0)
    full_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    #Warmup (full run)
    warm_params = SamplingParams(max_tokens=min(args.max_new_tokens, 16), temperature=0.0)
    for _ in range(args.warmup):
        _ = llm.generate(batch_prompts, warm_params)

    #Computation
    ttfts = []
    lats = []
    new_tokps = []
    for _ in range(args.runs):
        # TTFT run
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        t0 = time.perf_counter()
        
        _ = llm.generate(batch_prompts, ttft_params)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        t1 = time.perf_counter()
        
        ttfts.append(t1 - t0)

        # Full run
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        s0 = time.perf_counter()
        outs = llm.generate(batch_prompts, full_params)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        s1 = time.perf_counter()

        lat = s1 - s0
        lats.append(lat)

        # Count generated tokens (new tokens only)
        # Each RequestOutput has outputs[0].token_ids for generated tokens
        gen_tokens = 0
        for r in outs:
            gen_tokens += len(r.outputs[0].token_ids)

        tokps = gen_tokens / lat if lat > 0 else 0.0
        new_tokps.append(tokps)
        
    #Compute avg
    avg_ttft = sum(ttfts) / len(ttfts)
    avg_lat = sum(lats) / len(lats)
    avg_tokps_new = sum(new_tokps) / len(new_tokps)

    print(f"vLLM | model={args.model} | batch={args.batch} | new_tokens={args.max_new_tokens}")
    print(f"avg_ttft≈ {avg_ttft:.6f}s | avg_latency {avg_lat:.6f}s | tok/s(new) {avg_tokps_new:.3f}")

    #Write CSV file
    try:
        with open(args.out_csv, "r", encoding="utf-8"):
            pass
    except FileNotFoundError:
        write_header = True

    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["backend", "model", "dtype", "batch", "max_new_tokens", "avg_ttft", "avg_latency", "tokps_new"])
        w.writerow(["vLLM", args.model, args.dtype, args.batch, args.max_new_tokens, f"{avg_ttft:.6f}", f"{avg_lat:.6f}", f"{avg_tokps_new:.6f}"])

if __name__ == "__main__":
    main()