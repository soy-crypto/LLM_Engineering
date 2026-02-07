import time, subprocess

prompt = "<|system|>You are helpful.</s><|user|>Explain GPUs in one sentence.</s><|assistant|>"

cmd = [
    "python3", "/workspace/TensorRT-LLM/examples/run.py",
    "--engine_dir", "/workspace/trtllm_engine_tinyllama",
    "--tokenizer_dir", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "--max_output_len", "128",
    "--input_text", prompt,
]

def run_once(tag):
    start = time.time()
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    end = time.time()
    print(f"\n===== {tag} =====")
    print(out)
    print("Wall time:", end - start)

run_once("COLD")
run_once("COLD again (still cold because new process)")
