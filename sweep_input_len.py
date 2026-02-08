# sweep_input_len.py
import os, time, math
import torch
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner
from tensorrt_llm.runtime import SamplingConfig

ENGINE_DIR = os.environ.get("ENGINE_DIR", "/workspace/trtllm_engine_tinyllama")
MODEL_NAME = os.environ.get("MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# 你要扫的 reps（reps 越大，input 越长）
REPS_LIST = [1, 4, 16, 64, 128]

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "128"))
WARMUP = int(os.environ.get("WARMUP", "2"))      # 每个点先跑几次不计入
MEASURE = int(os.environ.get("MEASURE", "5"))    # 每个点记录几次取 median

BASE_TEXT = "Explain GPUs in one sentence.\n"

def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2 == 1:
        return xs[n//2]
    return 0.5 * (xs[n//2 - 1] + xs[n//2])

def build_input_ids(tok, reps: int, device="cuda"):
    # 用 chat template 更接近真实聊天
    msgs = [
        {"role":"system", "content":"You are a helpful assistant."},
        {"role":"user", "content": BASE_TEXT * reps}
    ]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    return ids

def run_once(runner, input_ids, eos_id, pad_id):
    scfg = SamplingConfig(
        end_id=eos_id,
        pad_id=pad_id,
        max_new_tokens=MAX_NEW_TOKENS,
        # 稳定：greedy
        temperature=0.0,
        top_p=1.0
    )

    # TTFT：让它只生成 1 token
    t0 = time.time()
    _ = runner.generate(input_ids, sampling_config=SamplingConfig(
        end_id=eos_id, pad_id=pad_id, max_new_tokens=1, temperature=0.0, top_p=1.0
    ))
    ttft = time.time() - t0

    # Total：生成 MAX_NEW_TOKENS
    t1 = time.time()
    out = runner.generate(input_ids, sampling_config=scfg)
    total = time.time() - t1

    # 估算 decode tok/s： (new_tokens-1)/(total-ttft) 近似
    # 这里用 total / MAX_NEW_TOKENS 也行，但 TTFT 单独测更直观
    decode_tps = (MAX_NEW_TOKENS - 1) / max(1e-9, (total - ttft))
    return ttft, total, decode_tps

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    runner = ModelRunner.from_dir(ENGINE_DIR, rank=0)

    # engine limit：从环境变量读（你之前已经在脚本里做过）
    # 尽量从 engine/config 自动拿上限，避免你设错
    engine_limit = None
    for attr in ["max_input_len", "max_seq_len", "max_sequence_length"]:
        if hasattr(runner, attr):
            engine_limit = int(getattr(runner, attr))
            break

    # 有的版本把 config 放在 runner.config / runner.model_config
    if engine_limit is None:
        cfg = getattr(runner, "config", None) or getattr(runner, "model_config", None)
        if cfg is not None:
            for k in ["max_input_len", "max_seq_len", "max_sequence_length"]:
                v = getattr(cfg, k, None)
                if v is not None:
                    engine_limit = int(v)
                    break

    if engine_limit is None:
        engine_limit = int(os.environ.get("ENGINE_LIMIT", "512"))

    print(f"# engine_limit={engine_limit}")

    
    print("reps,input_len,status,ttft_s,total_s,decode_tps")

    for reps in REPS_LIST:
        input_ids = build_input_ids(tok, reps)
        in_len = int(input_ids.shape[1])

        if in_len > engine_limit:
            print(f"{reps},{in_len},SKIP,nan,nan,nan")
            continue

        # warm-up
        for _ in range(WARMUP):
            _ = runner.generate(input_ids, sampling_config=SamplingConfig(
                end_id=eos_id, pad_id=pad_id, max_new_tokens=8, temperature=0.0, top_p=1.0
            ))

        # measure
        ttfts, totals, tps = [], [], []
        for _ in range(MEASURE):
            ttft, total, decode_tps = run_once(runner, input_ids, eos_id, pad_id)
            ttfts.append(ttft); totals.append(total); tps.append(decode_tps)

        print(f"{reps},{in_len},OK,{median(ttfts):.6f},{median(totals):.6f},{median(tps):.2f}")

if __name__ == "__main__":
    main()
