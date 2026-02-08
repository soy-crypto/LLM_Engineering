import time
import torch
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner, SamplingConfig

ENGINE_DIR = "/workspace/trtllm_engine_tinyllama"
MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def now():
    return time.perf_counter()

def main():
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id
    assert eos_id is not None

    prompt = tok.apply_chat_template(
        [
            {"role":"system","content":"You are helpful."},
            {"role":"user","content":"Explain GPUs in one sentence."},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    runner = ModelRunner.from_dir(engine_dir=ENGINE_DIR, rank=0, debug_mode=False)

    input_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    in_len = input_ids.shape[-1]

    # choose one:
    GREEDY = True
    if GREEDY:
        scfg = SamplingConfig(end_id=eos_id, pad_id=pad_id, max_new_tokens=128, temperature=0.0, top_p=1.0)
    else:
        scfg = SamplingConfig(end_id=eos_id, pad_id=pad_id, max_new_tokens=128, temperature=0.8, top_p=0.95)

    # warmup
    _ = runner.generate(input_ids, sampling_config=scfg)
    torch.cuda.synchronize()

    # --- TTFT measurement ---
    # Trick: generate only 1 token => total time ≈ TTFT (prefill + first token)
    scfg_1 = SamplingConfig(
        end_id=eos_id, pad_id=pad_id,
        max_new_tokens=1,
        temperature=scfg.temperature,
        top_p=scfg.top_p,
    )

    torch.cuda.synchronize()
    t0 = now()
    out1 = runner.generate(input_ids, sampling_config=scfg_1)
    torch.cuda.synchronize()
    t1 = now()
    ttft = t1 - t0

    # --- decode throughput measurement ---
    # full generation
    torch.cuda.synchronize()
    t2 = now()
    out = runner.generate(input_ids, sampling_config=scfg)
    torch.cuda.synchronize()
    t3 = now()

    total = t3 - t2
    new_tokens = out.shape[-1] - in_len

    # approximate decode-only time = total - TTFT
    decode_time = max(1e-9, total - ttft)
    decode_tps = (new_tokens - 1) / decode_time if new_tokens > 1 else 0.0

    print("\n==== METRICS ====")
    print(f"Input len: {in_len}")
    print(f"New tokens: {new_tokens}")
    print(f"TTFT (1 token total): {ttft:.4f}s")
    print(f"Total latency (N tokens): {total:.4f}s")
    print(f"Approx decode tok/s: {decode_tps:.2f}")

if __name__ == "__main__":
    main()
