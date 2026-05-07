import argparse
import time
import torch
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner, SamplingConfig

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from common import load_prompts, parse_batch_sizes, build_batch, get_gpu_mem_mb, CsvWriter


def get_args():
    parser = argparse.ArgumentParser(description="TensorRT-LLM Benchmark")
    parser.add_argument("--engine_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--batch_size", type=str, default="1,2,4,8")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--backend", type=str, default="TensorRT-LLM")
    return parser.parse_args()


def load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_batch(tokenizer, prompts, batch_size: int):
    batch = build_batch(prompts, batch_size)
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].cuda()
    return [input_ids[i].contiguous() for i in range(input_ids.size(0))]


@torch.inference_mode()
def trt_generate(runner: ModelRunner, batch_input_ids, max_new_tokens: int, end_id: int, pad_id: int):
    sampling_config = SamplingConfig(
        end_id=end_id, pad_id=pad_id, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0
    )
    return runner.generate(batch_input_ids=batch_input_ids, sampling_config=sampling_config)


def measure_ttft(runner: ModelRunner, batch_input_ids, end_id: int, pad_id: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trt_generate(runner, batch_input_ids, 1, end_id, pad_id)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def measure_throughput(runner: ModelRunner, batch_input_ids, max_new_tokens: int, end_id: int, pad_id: int):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trt_generate(runner, batch_input_ids, max_new_tokens, end_id, pad_id)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    new_tokens = len(batch_input_ids) * max_new_tokens
    return elapsed, new_tokens


def benchmark_batch(runner, batch_input_ids, args, end_id: int, pad_id: int):
    for _ in range(args.warmup):
        measure_throughput(runner, batch_input_ids, min(8, args.max_new_tokens), end_id, pad_id)

    ttft_list, latency_list, tokps_list = [], [], []
    for _ in range(args.runs):
        ttft = measure_ttft(runner, batch_input_ids, end_id, pad_id)
        lat, new_tokens = measure_throughput(runner, batch_input_ids, args.max_new_tokens, end_id, pad_id)
        ttft_list.append(ttft)
        latency_list.append(lat)
        tokps_list.append(new_tokens / lat if lat > 0 else 0)

    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_lat = sum(latency_list) / len(latency_list)
    avg_tokps = sum(tokps_list) / len(tokps_list)
    return avg_ttft, avg_lat, avg_tokps, get_gpu_mem_mb()


def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT-LLM benchmark")

    prompts = load_prompts(args.prompts)
    batch_sizes = parse_batch_sizes(args.batch_size)
    tokenizer = load_tokenizer(args.model_id)
    runner = ModelRunner.from_dir(args.engine_dir)

    print(f"TensorRT-LLM | model={args.model_id} | dtype={args.dtype}")
    print(f"Engine: {args.engine_dir}")
    print("-" * 80)

    writer = CsvWriter(args.out_csv)
    end_id, pad_id = tokenizer.eos_token_id, tokenizer.pad_token_id

    for bs in batch_sizes:
        batch_input_ids = tokenize_batch(tokenizer, prompts, bs)
        avg_ttft, avg_lat, avg_tokps, gpu_mem = benchmark_batch(runner, batch_input_ids, args, end_id, pad_id)
        print(f"[bs={bs}] ttft={avg_ttft:.4f}s  lat={avg_lat:.4f}s  tok/s={avg_tokps:.1f}  gpu={gpu_mem:.0f}MB")
        writer.write(args.backend, args.model_id, args.dtype, bs, avg_ttft, avg_lat, avg_tokps, gpu_mem)

    writer.close()
    print("-" * 80)
    print(f"Results saved to: {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
