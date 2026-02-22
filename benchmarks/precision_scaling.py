import torch
import time
import csv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
MAX_NEW_TOKENS = 128
SEQUENCE_LENGTH = 4096
BATCH_SIZE = 1

RESULTS_PATH = "../results/raw_data/precision_scaling.csv"


def generate_prompt(target_length, tokenizer):
    base_text = "Hello world. "
    tokens = tokenizer(base_text)["input_ids"]
    repeated = tokens * (target_length // len(tokens) + 1)
    return tokenizer.decode(repeated[:target_length])


def run_experiment(dtype):
    print(f"\nRunning dtype = {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map="auto"
    )

    model.eval()

    prompt = generate_prompt(SEQUENCE_LENGTH, tokenizer)
    inputs = tokenizer([prompt] * BATCH_SIZE, return_tensors="pt", padding=True).to(DEVICE)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():

        # Prefill
        torch.cuda.synchronize()
        start_prefill = time.time()

        outputs = model(**inputs, use_cache=True)

        torch.cuda.synchronize()
        end_prefill = time.time()

        prefill_time = end_prefill - start_prefill

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        # Decode
        torch.cuda.synchronize()
        decode_start = time.time()

        for _ in range(MAX_NEW_TOKENS):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        torch.cuda.synchronize()
        decode_end = time.time()

        decode_time = decode_end - decode_start

    per_token_decode_latency_ms = (decode_time / MAX_NEW_TOKENS) * 1000
    throughput_tokens_per_sec = (BATCH_SIZE * MAX_NEW_TOKENS) / decode_time
    peak_memory_MB = torch.cuda.max_memory_allocated() / (1024 ** 2)

    result = {
        "dtype": str(dtype),
        "prefill_time_sec": prefill_time,
        "decode_time_sec": decode_time,
        "per_token_decode_latency_ms": per_token_decode_latency_ms,
        "throughput_tokens_per_sec": throughput_tokens_per_sec,
        "gpu_memory_MB": peak_memory_MB
    }

    print(result)
    return result


def save_result(result):
    file_exists = os.path.isfile(RESULTS_PATH)

    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def main():
    for dtype in [torch.float16, torch.float32]:
        result = run_experiment(dtype)
        save_result(result)


if __name__ == "__main__":
    main()
