import torch
import time
import csv
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16
MAX_NEW_TOKENS = 128

RESULTS_PATH = "../results/raw_data/sequence_scaling_prefill_decode.csv"


def generate_prompt(target_length, tokenizer):
    base_text = "Hello world. "
    tokens = tokenizer(base_text)["input_ids"]
    repeated = tokens * (target_length // len(tokens) + 1)
    return tokenizer.decode(repeated[:target_length])


def run_experiment(seq_len):
    print(f"\nRunning sequence length = {seq_len}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto"
    )

    model.eval()

    prompt = generate_prompt(seq_len, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    with torch.no_grad():

        # -----------------
        # Prefill Phase
        # -----------------
        torch.cuda.synchronize()
        start_prefill = time.time()

        outputs = model(**inputs, use_cache=True)

        torch.cuda.synchronize()
        end_prefill = time.time()

        prefill_time = end_prefill - start_prefill

        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)

        # -----------------
        # Decode Phase
        # -----------------
        torch.cuda.synchronize()
        decode_start = time.time()

        for _ in range(MAX_NEW_TOKENS):
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True
            )

            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)

        torch.cuda.synchronize()
        decode_end = time.time()

        decode_time = decode_end - decode_start

    per_token_decode_latency_ms = (decode_time / MAX_NEW_TOKENS) * 1000
    throughput_tokens_per_sec = MAX_NEW_TOKENS / decode_time

    peak_memory_MB = torch.cuda.max_memory_allocated() / (1024 ** 2)

    result = {
        "sequence_length": seq_len,
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
    sequence_length = 4096  # Change manually per run

    result = run_experiment(sequence_length)
    save_result(result)


if __name__ == "__main__":
    main()
