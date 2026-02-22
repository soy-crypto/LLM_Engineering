import argparse
from transformers import AutoConfig

def kv_mb_per_token(cfg, bytes_per_elem=2, batch=1):
    # KV bytes/token = 2(K+V) * L * B * hidden_size * bytes
    # because heads*head_dim = hidden_size
    L = int(cfg.num_hidden_layers)
    hidden = int(cfg.hidden_size)
    kv_bytes = 2 * L * batch * hidden * bytes_per_elem
    return kv_bytes / (1024 ** 2)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--bytes", type=int, default=2)  # bf16/fp16=2
    args = p.parse_args()

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    L = getattr(cfg, "num_hidden_layers", None)
    H = getattr(cfg, "num_attention_heads", None)
    hidden = getattr(cfg, "hidden_size", None)

    print(f"model: {args.model}")
    print(f"layers={L} heads={H} hidden_size={hidden}")
    print(f"theory_kv_mb_per_token(batch={args.batch}, bytes={args.bytes}) = {kv_mb_per_token(cfg, args.bytes, args.batch):.6f} MB")

if __name__ == "__main__":
    main()