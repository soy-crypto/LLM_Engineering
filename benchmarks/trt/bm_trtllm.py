# benchmarks/trt/bm_trtllm.py

import argparse
import time
import csv
import torch
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner, SamplingConfig


def get_args():
    parser = argparse.ArgumentParser(description="TensorRT-LLM Benchmark")
    parser.add_argument("--engine_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)
    parser.add_argument("--batch_size", type=str, default="1,2,4,8,16")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out_csv", type=str, default="trt_results.csv")
    parser.add_argument("--backend", type=str, default="TensorRT-LLM")
    return parser.parse_args()


def parse_batch_sizes(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_prompts(path: str):
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                prompts.append(s)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def build_batch(prompts, batch_size: int):
    return (prompts * (batch_size // len(prompts) + 1))[:batch_size]


@torch.no_grad()
def trt_generate(runner, batch_input_ids, max_new_tokens, end_id, pad_id):
    scfg = SamplingConfig(
        end_id=end_id,
        pad_id=pad_id,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return runner.generate(
        batch_input_ids=batch_input_ids,
        sampling_config=scfg,
    )


def main():
    args = get_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TRT benchmark.")

    batch_sizes = parse_batch_sizes(args.batch_size)
    prompts = load_prompts(args.prompts)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    end_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    runner = ModelRunner.from_dir(args.engine_dir)

    print(f"TRTLLM | model={args.model_id}")
    print("-" * 90)

    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "backend",
            "model",
            "batch_size",
            "avg_ttft",
            "avg_latency",
            "tokps_new"
        ])

        for BATCH in batch_sizes:

            batch_prompts = build_batch(prompts, BATCH)

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = inputs["input_ids"].to("cuda")
            batch_input_ids = [input_ids[i] for i in range(input_ids.size(0))]

            # Warmup
            for _ in range(args.warmup):
                _ = trt_generate(
                    runner,
                    batch_input_ids,
                    min(8, args.max_new_tokens),
                    end_id,
                    pad_id
                )

            ttft_list = []
            lat_list = []
            tokps_list = []

            for _ in range(args.runs):

                # TTFT
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = trt_generate(runner, batch_input_ids, 1, end_id, pad_id)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                ttft_list.append(t1 - t0)

                # Full decode
                torch.cuda.synchronize()
                s0 = time.perf_counter()
                _ = trt_generate(
                    runner,
                    batch_input_ids,
                    args.max_new_tokens,
                    end_id,
                    pad_id
                )
                torch.cuda.synchronize()
                s1 = time.perf_counter()

                elapsed = s1 - s0
                lat_list.append(elapsed)
                tokps_list.append((BATCH * args.max_new_tokens) / elapsed)

            avg_ttft = sum(ttft_list) / len(ttft_list)
            avg_lat = sum(lat_list) / len(lat_list)
            avg_tokps = sum(tokps_list) / len(tokps_list)

            print(
                f"[{BATCH}] "
                f"avg_ttft: {avg_ttft:.6f} | "
                f"avg_latency: {avg_lat:.6f} | "
                f"tok/s(new): {avg_tokps:.3f}"
            )

            writer.writerow([
                args.backend,
                args.model_id,
                BATCH,
                f"{avg_ttft:.6f}",
                f"{avg_lat:.6f}",
                f"{avg_tokps:.6f}"
            ])

    print("-" * 90)
    print(f"Wrote CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()
