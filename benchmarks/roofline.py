import math

# --- Model parameters (Qwen2.5-7B example) ---
layers = 28
hidden_size = 3584
bytes_per_elem = 2  # bf16

# --- GPU theoretical specs (RTX 5090 approx placeholder) ---
peak_flops = 82e12      # 82 TFLOPs (example)
mem_bandwidth = 1000e9  # 1000 GB/s example

def kv_bytes_per_token(batch_size, seq_len):
    return 2 * layers * hidden_size * bytes_per_elem * batch_size * seq_len

def flops_per_token(batch_size, seq_len):
    # Rough attention + MLP estimate
    return 2 * layers * (hidden_size ** 2) * batch_size

print("=== Roofline Estimate ===")

batch = 16
seq = 512

kv_bytes = kv_bytes_per_token(batch, seq)
flops = flops_per_token(batch, seq)

bw_time = kv_bytes / mem_bandwidth
compute_time = flops / peak_flops

print(f"Estimated memory-bound time: {bw_time:.6f}s")
print(f"Estimated compute-bound time: {compute_time:.6f}s")

if bw_time > compute_time:
    print("Likely memory-bandwidth bound.")
else:
    print("Likely compute bound.")
