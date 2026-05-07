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
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="results.csv")
    parser.add_argument("--backend", type=str, default="HF")
    return parser.parse_args()


def set_seed(seed: int, device: str):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, dtype_str: str, device: str):
    if dtype_str == "auto":
        if device == "cuda":
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            torch_dtype = torch.float32
    else:
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map[dtype_str]

    if device == "cpu":
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DO NOT use device_map="auto" — force full GPU placement
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True, low_cpu_mem_usage=True
    ).to(device).eval()

    if device == "cuda":
        torch.cuda.empty_cache()

    return model, tokenizer, torch_dtype


def build_params(inputs: Dict[str, Any], args, tokenizer) -> Dict[str, Any]:
    params = dict(inputs)
    params.update(dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    ))
    if args.do_sample:
        params.update(dict(temperature=args.temperature, top_p=args.top_p))
    return params


@torch.inference_mode()
def measure(
    model,
    params: Dict[str, Any],
    device: str,
    max_new_token_override: Optional[int] = None,
) -> Tuple[float, int, int]:
    p = dict(params)
    if max_new_token_override is not None:
        p["max_new_tokens"] = max_new_token_override

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = model.generate(**p)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    batch_size = outputs.size(0)
    total_seq_len = outputs.size(1)
    output_tokens = batch_size * total_seq_len
    # Use attention_mask to precisely count prompt tokens across padded batches
    prompt_tokens = int(p["attention_mask"].sum().item())
    new_tokens = max(output_tokens - prompt_tokens, 0)

    return elapsed, output_tokens, new_tokens


def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed, device)
    prompts = load_prompts(args.prompts)
    batch_sizes = parse_batch_sizes(args.batch_size)

    model, tokenizer, torch_dtype = load_model(args.model, args.dtype, device)
    dtype_label = str(torch_dtype).split(".")[-1]

    print(f"HuggingFace | model={args.model} | dtype={dtype_label} | device={device}")
    print(f"prompts={args.prompts} ({len(prompts)} lines) | max_new_tokens={args.max_new_tokens}")
    print("-" * 90)

    writer = CsvWriter(args.out_csv)

    for bs in batch_sizes:
        batch = build_batch(prompts, bs)
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in tokens.items()}
        params = build_params(inputs, args, tokenizer)

        for _ in range(args.warmup):
            measure(model, params, device)

        ttft_list, latency_list, new_tokens_list = [], [], []

        for _ in range(args.runs):
            ttft, _, _ = measure(model, params, device, max_new_token_override=1)
            lat, _, new_tokens = measure(model, params, device)
            ttft_list.append(ttft)
            latency_list.append(lat)
            new_tokens_list.append(new_tokens)

        avg_ttft = sum(ttft_list) / len(ttft_list)
        avg_lat = sum(latency_list) / len(latency_list)
        avg_new = sum(new_tokens_list) / len(new_tokens_list)
        tokps_new = avg_new / avg_lat if avg_lat > 0 else 0
        gpu_mem = get_gpu_mem_mb()

        print(
            f"[bs={bs}] ttft={avg_ttft:.4f}s  lat={avg_lat:.4f}s  "
            f"tok/s(new)={tokps_new:.2f}  gpu={gpu_mem:.0f}MB"
        )
        writer.write(args.backend, args.model, dtype_label, bs, avg_ttft, avg_lat, tokps_new, gpu_mem)

    writer.close()
    print("-" * 90)
    print(f"Results saved to: {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
