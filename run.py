import time
import subprocess

prompt =  "<|system|>You are helpful.</s><|user|>Explain GPUs in one sentence.</s><|assistant|>"

start = time.time()
p = subprocess.Popen([...], stdout=subprocess.PIPE, text=True)

first_token_time = None
tokens = 0

for line in p.stdout:
    now = time.time()
    if first_token_time is None:
        first_token_time = now
    
    tokens += len(line.split())

end = time.time()

print("TTFT:", first_token_time - start)
print("Total latency:", end - start)
print("Tokens/s:", tokens / (end - first_token_time))