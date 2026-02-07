import time
import subprocess

prompt = "<|system|>You are helpful.</s><|user|>Explain GPUs in one sentence.</s><|assistant|>"

cmd = [
    "python3",
    "/workspace/TensorRT-LLM/examples/run.py",
    "--engine_dir", "/workspace/trtllm_engine_tinyllama",
    "--tokenizer_dir", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "--max_output_len", "128",
    "--input_text", prompt,
]

start = time.time()
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

first_token_time = None
tokens = 0

for line in p.stdout:
    # TensorRT-LLM 会逐行打印 token / 文本
    now = time.time()

    # 只在真正输出 token 时记录 TTFT
    if first_token_time is None and "Output" in line:
        first_token_time = now

    tokens += len(line.split())
    print(line, end="")

p.wait()
end = time.time()

print("\n==== METRICS ====")
print("TTFT:", first_token_time - start if first_token_time else "N/A")
print("Total latency:", end - start)
print("Tokens/s:", tokens / (end - first_token_time) if first_token_time else "N/A")
