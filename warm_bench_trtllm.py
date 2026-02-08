import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from tensorrt_llm.runtime import ModelRunner, SamplingConfig

ENGINE_DIR = "/workspace/trtllm_engine_tinyllama"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Explain GPUs in one sentence."},
]

def now():
    return time.perf_counter()

def main():
    assert Path(ENGINE_DIR).exists(), f"ENGINE_DIR not found: {ENGINE_DIR}"

    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)

    # Ensure special tokens exist
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise RuntimeError("tokenizer.eos_token_id is None. Can't run TRT-LLM decode without end_id.")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    t0 = now()
    runner = ModelRunner.from_dir(engine_dir=ENGINE_DIR, rank=0, debug_mode=False)
    t1 = now()
    print(f"Engine load (single-process): {t1 - t0:.3f}s")

    # deterministic benchmark
    max_new_tokens = 128
    temperature = 0.0   # greedy
    top_p = 1.0

    scfg = SamplingConfig(
        end_id=eos_id,
        pad_id=pad_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    input_ids = tok(prompt, return_tensors="pt").input_ids.cuda()

    # warm-up
    _ = runner.generate(input_ids, sampling_config=scfg)
    torch.cuda.synchronize()

    # timed runs
    N = 10
    total_times = []
    total_new_tokens = []

    for i in range(N):
        torch.cuda.synchronize()
        start = now()

        out = runner.generate(input_ids, sampling_config=scfg)

        torch.cuda.synchronize()
        end = now()

        new_tokens = out.shape[-1] - input_ids.shape[-1]
        total_new_tokens.append(new_tokens)
        total_times.append(end - start)

        print(f"Run {i+1:02d}: total={end-start:.3f}s, new_tokens={new_tokens}")

    avg_total = sum(total_times) / len(total_times)
    avg_new = sum(total_new_tokens) / len(total_new_tokens)

    print("\n==== SUMMARY (warm, single process) ====")
    print(f"Avg total latency: {avg_total:.3f}s")
    print(f"Avg new tokens:   {avg_new:.1f}")
    print(f"Tokens/s (avg):   {avg_new / avg_total:.2f}")
    print(f"eos_id={eos_id}, pad_id={pad_id}")

if __name__ == "__main__":
    main()
