import argparse
import time
import torch
from typing import Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common import load_prompts, parse_batch_sizes, build_batch, get_gpu_mem_mb, CsvWriter


def get_args():
    parser = argparse.ArgumentParser(description="HuggingFace LLM Benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--batch_size", type=str, default="1,2,4,8")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--backend", type=str, default="HF")
    return parser.parse_args()


def load_model(model_name: str, dtype_str: str, device: str):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype_str]
    if device == "cpu":
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device).eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    return model, tokenizer, str(torch_dtype).split(".")[-1]


def build_inputs(prompts, batch_size, tokenizer, device) -> Dict[str, Any]:
    batch = build_batch(prompts, batch_size)
    tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    return {k: v.to(device) for k, v in tokens.items()}


@torch.inference_mode()
def generate(model, inputs: Dict[str, Any], max_new_tokens: int, tokenizer) -> Tuple[float, int]:
    device = next(model.parameters()).device.type
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_tokens = outputs.shape[0] * (outputs.shape[1] - inputs["input_ids"].shape[1])
    return elapsed, new_tokens


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prompts = load_prompts(args.prompts)
    batch_sizes = parse_batch_sizes(args.batch_size)

    model, tokenizer, dtype_label = load_model(args.model, args.dtype, device)
    print(f"HuggingFace | model={args.model} | dtype={dtype_label} | device={device}")
    print("-" * 80)

    writer = CsvWriter(args.out_csv)

    for bs in batch_sizes:
        inputs = build_inputs(prompts, bs, tokenizer, device)

        for _ in range(args.warmup):
            generate(model, inputs, min(8, args.max_new_tokens), tokenizer)

        ttft_list, latency_list, tokps_list = [], [], []

        for _ in range(args.runs):
            ttft, _ = generate(model, inputs, 1, tokenizer)
            lat, new_tokens = generate(model, inputs, args.max_new_tokens, tokenizer)
            ttft_list.append(ttft)
            latency_list.append(lat)
            tokps_list.append(new_tokens / lat if lat > 0 else 0)

        avg_ttft = sum(ttft_list) / len(ttft_list)
        avg_lat = sum(latency_list) / len(latency_list)
        avg_tokps = sum(tokps_list) / len(tokps_list)
        gpu_mem = get_gpu_mem_mb()

        print(f"[bs={bs}] ttft={avg_ttft:.4f}s  lat={avg_lat:.4f}s  tok/s={avg_tokps:.1f}  gpu={gpu_mem:.0f}MB")
        writer.write(args.backend, args.model, dtype_label, bs, avg_ttft, avg_lat, avg_tokps, gpu_mem)

    writer.close()
    print("-" * 80)
    print(f"Results saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
