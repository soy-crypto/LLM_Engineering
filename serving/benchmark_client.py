import requests
import time

URL = "http://localhost:8000/generate"
prompt = "Explain transformer inference"
start = time.time()
response = requests.post(URL, json={"prompt": prompt})
end = time.time()
print("Latency:", end - start)
print(response.json())