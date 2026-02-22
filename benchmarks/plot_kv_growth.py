import matplotlib.pyplot as plt
import numpy as np

layers = 28
hidden = 3584
bytes_per_elem = 2

def kv_mb(tokens, batch):
    return (2 * layers * hidden * bytes_per_elem * tokens * batch) / (1024**2)

tokens = np.arange(1, 1025, 64)
batch = 8

kv_vals = [kv_mb(t, batch) for t in tokens]

plt.figure()
plt.plot(tokens, kv_vals)
plt.xlabel("Generated Tokens")
plt.ylabel("KV Cache (MB)")
plt.title("KV Cache Growth Curve")
plt.grid(True)
plt.savefig("results/kv_growth.png")
plt.show()
