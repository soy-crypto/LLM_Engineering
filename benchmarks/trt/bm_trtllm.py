import argparse
import time
import csv
import torch
from typing import List

from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner, SamplingConfig


# ------------------------------------------------
# Args
# ------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(description="TensorRT-LLM Benchmark")

    parser.add_argument("--engine_dir", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--prompts", type=str, required=True)

    parser.add_argument("--batch_size", type=str, default="1,2,4,8")
    parser.add_argument("--max_new_tokens", type=int, default=512)

    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)

    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--backend", type=str, default="TensorRT-LLM")

    return parser.parse_args()


# ------------------------------------------------
# Helpers
# ------------------------------------------------

def parse_batch_sizes(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_prompts(path: str) -> List[str]:

    prompts = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)

    if not prompts:
        raise ValueError(f"No prompts found in {path}")

    return prompts


def build_batch(prompts: List[str], batch_size: int) -> List[str]:
    return (prompts * (batch_size // len(prompts) + 1))[:batch_size]


# ------------------------------------------------
# TRT Generate
# ------------------------------------------------

@torch.inference_mode()
def trt_generate(
    runner: ModelRunner,
    batch_input_ids,
    max_new_tokens,
    end_id,
    pad_id,
):

    sampling_config = SamplingConfig(
        end_id=end_id,
        pad_id=pad_id,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    return runner.generate(
        batch_input_ids=batch_input_ids,
        sampling_config=sampling_config,
    )


# ------------------------------------------------
# Main
# ------------------------------------------------

def main():

    args = get_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT-LLM benchmark")

    batch_sizes = parse_batch_sizes(args.batch_size)

    prompts = load_prompts(args.prompts)

    # ------------------------------------------------
    # Load tokenizer
    # ------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=True,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    end_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    # ------------------------------------------------
    # Load TRT engine
    # ------------------------------------------------

    print("Loading TensorRT engine...")
    runner = ModelRunner.from_dir(args.engine_dir)

    print(f"TensorRT-LLM | model={args.model_id}")
    print(f"Engine dir: {args.engine_dir}")
    print("-" * 90)

    # ------------------------------------------------
    # CSV
    # ------------------------------------------------

    file_exists = False

    try:
        with open(args.out_csv, "r"):
            file_exists = True
    except:
        pass

    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        if not file_exists:

            writer.writerow(
                [
                    "backend",
                    "model",
                    "batch_size",
                    "avg_ttft",
                    "avg_latency",
                    "tokps_new",
                ]
            )

        # ------------------------------------------------
        # Benchmark loop
        # ------------------------------------------------

        for BATCH in batch_sizes:

            batch_prompts = build_batch(prompts, BATCH)

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            input_ids = inputs["input_ids"].cuda()

            batch_input_ids = [
                input_ids[i].contiguous()
                for i in range(input_ids.size(0))
            ]

            # -----------------------------
            # Warmup
            # -----------------------------

            for _ in range(args.warmup):

                trt_generate(
                    runner,
                    batch_input_ids,
                    min(8, args.max_new_tokens),
                    end_id,
                    pad_id,
                )

            ttft_list = []
            latency_list = []
            tokps_list = []

            # -----------------------------
            # Runs
            # -----------------------------

            for _ in range(args.runs):

                torch.cuda.synchronize()

                t0 = time.perf_counter()

                trt_generate(
                    runner,
                    batch_input_ids,
                    1,
                    end_id,
                    pad_id,
                )

                torch.cuda.synchronize()

                t1 = time.perf_counter()

                ttft = t1 - t0
                ttft_list.append(ttft)

                # full decode

                torch.cuda.synchronize()

                s0 = time.perf_counter()

                trt_generate(
                    runner,
                    batch_input_ids,
                    args.max_new_tokens,
                    end_id,
                    pad_id,
                )

                torch.cuda.synchronize()

                s1 = time.perf_counter()

                latency = s1 - s0

                latency_list.append(latency)

                tokps = (BATCH * args.max_new_tokens) / latency

                tokps_list.append(tokps)

            # -----------------------------
            # averages
            # -----------------------------

            avg_ttft = sum(ttft_list) / len(ttft_list)
            avg_latency = sum(latency_list) / len(latency_list)
            avg_tokps = sum(tokps_list) / len(tokps_list)

            print(
                f"[{BATCH}] "
                f"avg_ttft: {avg_ttft:.6f} | "
                f"avg_latency: {avg_latency:.6f} | "
                f"tok/s(new): {avg_tokps:.3f}"
            )

            writer.writerow(
                [
                    args.backend,
                    args.model_id,
                    BATCH,
                    f"{avg_ttft:.6f}",
                    f"{avg_latency:.6f}",
                    f"{avg_tokps:.6f}",
                ]
            )

    print("-" * 90)
    print(f"Wrote CSV to: {args.out_csv}")


# ------------------------------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e)