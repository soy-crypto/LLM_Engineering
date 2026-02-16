# benchmarks/bm_trtllm.py
import time
import torch
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner, SamplingConfig

ENGINE_DIR = "/workspace/trt_engine/qwen2p5_7b_fp16_b16_i2048_s2560"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
PROMPT_FILE = "/workspace/LLM_Engineering/prompts/prompts_mid.txt"

BATCH = 16
MAX_NEW_TOKENS = 512
WARMUP = 1
RUNS = 3


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
def trt_generate(runner: ModelRunner, batch_input_ids, max_new_tokens: int, end_id: int, pad_id: int):
    # Deterministic decoding (matches HF do_sample=False behavior)
    scfg = SamplingConfig(
        end_id=end_id,
        pad_id=pad_id,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return runner.generate(batch_input_ids=batch_input_ids, sampling_config=scfg)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    prompts = load_prompts(PROMPT_FILE)
    batch_prompts = build_batch(prompts, BATCH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    end_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    print(f"eos(end_id)={end_id} pad_id={pad_id}")

    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to("cuda")  # [B, S]

    # TRT-LLM expects a list of 1D tensors (one per request)
    batch_input_ids = [input_ids[i] for i in range(input_ids.size(0))]

    runner = ModelRunner.from_dir(ENGINE_DIR)

    # Warmup
    for _ in range(WARMUP):
        _ = trt_generate(runner, batch_input_ids, max_new_tokens=min(8, MAX_NEW_TOKENS), end_id=end_id, pad_id=pad_id)

    # TTFT≈ via 1-token generate
    ttft_list = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = trt_generate(runner, batch_input_ids, max_new_tokens=1, end_id=end_id, pad_id=pad_id)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        ttft_list.append(t1 - t0)

    # Full run throughput
    lat_list = []
    tokps_new_list = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        s0 = time.perf_counter()
        _ = trt_generate(runner, batch_input_ids, max_new_tokens=MAX_NEW_TOKENS, end_id=end_id, pad_id=pad_id)
        torch.cuda.synchronize()
        s1 = time.perf_counter()

        elapsed = s1 - s0
        lat_list.append(elapsed)
        tokps_new_list.append((BATCH * MAX_NEW_TOKENS) / elapsed)

    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_lat = sum(lat_list) / len(lat_list)
    avg_tokps_new = sum(tokps_new_list) / len(tokps_new_list)

    print(f"TRTLLM | model={MODEL_ID} | engine={ENGINE_DIR}")
    print(f"batch={BATCH} | max_new_tokens={MAX_NEW_TOKENS} | runs={RUNS}")
    print(f"avg_ttft≈ {avg_ttft:.6f}s | avg_latency {avg_lat:.6f}s | tok/s(new) {avg_tokps_new:.3f}")


if __name__ == "__main__":
    main()