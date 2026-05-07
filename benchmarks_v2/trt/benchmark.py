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


@torch.inference_mode()
def trt_generate(runner: ModelRunner, batch_input_ids, max_new_tokens: int, end_id: int, pad_id: int):
    sampling_config = SamplingConfig(
        end_id=end_id, pad_id=pad_id, max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0
    )
    return runner.generate(batch_input_ids=batch_input_ids, sampling_config=sampling_config)


def main():
    args = get_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT-LLM benchmark")

    prompts = load_prompts(args.prompts)
    batch_sizes = parse_batch_sizes(args.batch_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    runner = ModelRunner.from_dir(args.engine_dir)
    print(f"TensorRT-LLM | model={args.model_id} | dtype={args.dtype}")
    print(f"Engine: {args.engine_dir}")
    print("-" * 80)

    writer = CsvWriter(args.out_csv)

    for bs in batch_sizes:
        batch = build_batch(prompts, bs)
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].cuda()
        batch_input_ids = [input_ids[i].contiguous() for i in range(input_ids.size(0))]

        for _ in range(args.warmup):
            trt_generate(runner, batch_input_ids, min(8, args.max_new_tokens), tokenizer.eos_token_id, tokenizer.pad_token_id)

        ttft_list, latency_list, tokps_list = [], [], []

        for _ in range(args.runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            trt_generate(runner, batch_input_ids, 1, tokenizer.eos_token_id, tokenizer.pad_token_id)
            torch.cuda.synchronize()
            ttft_list.append(time.perf_counter() - t0)

            torch.cuda.synchronize()
            s0 = time.perf_counter()
            trt_generate(runner, batch_input_ids, args.max_new_tokens, tokenizer.eos_token_id, tokenizer.pad_token_id)
            torch.cuda.synchronize()
            lat = time.perf_counter() - s0
            latency_list.append(lat)
            tokps_list.append((bs * args.max_new_tokens) / lat if lat > 0 else 0)

        avg_ttft = sum(ttft_list) / len(ttft_list)
        avg_lat = sum(latency_list) / len(latency_list)
        avg_tokps = sum(tokps_list) / len(tokps_list)
        gpu_mem = get_gpu_mem_mb()

        print(f"[bs={bs}] ttft={avg_ttft:.4f}s  lat={avg_lat:.4f}s  tok/s={avg_tokps:.1f}  gpu={gpu_mem:.0f}MB")
        writer.write(args.backend, args.model_id, args.dtype, bs, avg_ttft, avg_lat, avg_tokps, gpu_mem)

    writer.close()
    print("-" * 80)
    print(f"Results saved to: {args.out_csv}")


if __name__ == "__main__":
    main()
