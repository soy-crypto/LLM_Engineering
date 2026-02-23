import requests
import time

URL = "http://localhost:8001/generate"

prompt = "Explain KV cache."

start = time.time()

r = requests.post(URL, json={"prompt": prompt})

print(r.json())
print("Latency:", time.time() - start)