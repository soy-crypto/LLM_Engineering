import time
import torch
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner, SamplingConfig

ENGINE_DIR = "/workspace/trtllm_engine_tinyllama_1024"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def now(): return time.perf_counter()

def run_one(runner, tok, eos_id, pad_id, prompt, max_new_tokens=128):
    input_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    in_len = input_ids.shape[-1]

    scfg = SamplingConfig(end_id=eos_id, pad_id=pad_id, max_new_tokens=max_new_tokens,
                          temperature=0.0, top_p=1.0)

    # warmup
    _ = runner.generate(input_ids, sampling_config=scfg)
    torch.cuda.synchronize()

    # TTFT (1 token)
    scfg1 = SamplingConfig(end_id=eos_id, pad_id=pad_id, max_new_tokens=1,
                           temperature=0.0, top_p=1.0)
    torch.cuda.synchronize()
    t0 = now()
    _ = runner.generate(input_ids, sampling_config=scfg1)
    torch.cuda.synchronize()
    t1 = now()
    ttft = t1 - t0

    # full
    torch.cuda.synchronize()
    t2 = now()
    out = runner.generate(input_ids, sampling_config=scfg)
    torch.cuda.synchronize()
    t3 = now()

    total = t3 - t2
    new_tokens = out.shape[-1] - in_len
    decode_time = max(1e-9, total - ttft)
    decode_tps = (new_tokens - 1) / decode_time if new_tokens > 1 else 0.0

    return in_len, new_tokens, ttft, total, decode_tps

def main():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    runner = ModelRunner.from_dir(engine_dir=ENGINE_DIR, rank=0, debug_mode=False)

    base_msgs = [
        {"role":"system","content":"You are helpful."},
        {"role":"user","content":"Explain GPUs in one sentence."},
    ]

    # repeat user content to grow prompt
    reps_list = [1, 4, 16, 64, 128]  # 你可以再加 256/512
    print("reps,input_len,ttft_s,total_s,decode_tps")

    for reps in reps_list:
        msgs = [
            {"role":"system","content":"You are helpful."},
            {"role":"user","content":("Explain GPUs in one sentence. " * reps).strip()},
        ]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        in_len, new_tokens, ttft, total, decode_tps = run_one(
            runner, tok, eos_id, pad_id, prompt, max_new_tokens=128
        )
        print(f"{reps},{in_len},{ttft:.6f},{total:.6f},{decode_tps:.2f}")

if __name__ == "__main__":
    main()
