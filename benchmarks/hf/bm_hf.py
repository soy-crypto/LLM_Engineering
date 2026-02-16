import argparse
import torch
import time
import csv
import os
from typing import List, Dict, Any, Tuple, Optional
from transformers import (AutoTokenizer, AutoModelForCausalLM, PreTrainedModel,PreTrainedTokenizerBase,)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF LLM Benchmark")
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--prompts", type=str, default="prompts.txt")
    parser.add_argument("--batch_size", type=str, default="1,2,4")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default="results.csv")
    parser.add_argument("--backend", type=str, default="HF")

    return parser.parse_args()

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int, device: str):
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    
    return


def get_prompts(path: str) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)

    if not prompts:
        raise ValueError(f"No prompts found in {path}")

    return prompts


def get_sizes(batch_size: str) -> List[int]:
    return [int(x.strip()) for x in batch_size.split(",") if x.strip()]


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model(model_name: str, device: str, dtype_str: str) -> Tuple[PreTrainedModel, torch.dtype]:
    if dtype_str == "auto":
        if device == "cuda":
            torch_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        else:
            torch_dtype = torch.float32
    else:
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        torch_dtype = dtype_map[dtype_str]

    if device == "cpu" and torch_dtype != torch.float32:
        print("CPU fallback to float32")
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype,low_cpu_mem_usage=True).to(device)
    model.eval()

    return model, torch_dtype


def build_parameters(size: int, prompts: List[str], device: str, tokenizer: PreTrainedTokenizerBase, args: argparse.Namespace,) -> Dict[str, Any]:
    text = (prompts * (size // len(prompts) + 1))[:size]
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    params: Dict[str, Any] = dict(tokens)
    params.update(dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, pad_token_id=tokenizer.pad_token_id, use_cache=True))

    if args.do_sample:
        params.update(dict(temperature=args.temperature,top_p=args.top_p))

    return params


@torch.inference_mode()
def measure(model: PreTrainedModel, parameters: Dict[str, Any], device: str, max_new_token_override: Optional[int] = None, measure_kv: bool = False) -> Tuple[float, int, int, Optional[float]]:
    params = dict(parameters)
    if max_new_token_override is not None:
        params["max_new_tokens"] = max_new_token_override

    if measure_kv:
        params["return_dict_in_generate"] = True
        params["output_attentions"] = False
        params["output_hidden_states"] = False

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    outputs = model.generate(**params)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()
    elapsed = end - start

    if measure_kv:
        sequences = outputs.sequences
    else:
        sequences = outputs

    batch_size = sequences.size(0)
    total_seq_len = sequences.size(1)

    output_tokens = batch_size * total_seq_len
    prompt_tokens = int(params["attention_mask"].sum().item())
    new_tokens = max(output_tokens - prompt_tokens, 0)

    kv_cache_mb = None
    if measure_kv:
        past = outputs.past_key_values
        kv_bytes = 0

        for layer in past:
            for tensor in layer:
                kv_bytes += tensor.numel() * tensor.element_size()

        kv_cache_mb = kv_bytes / (1024 ** 2)


    return elapsed, output_tokens, new_tokens, kv_cache_mb


def main():
    #Init
    args = get_args()
    device = get_device()
    set_seed(args.seed, device)
    prompts = get_prompts(args.prompts)
    batch_sizes = get_sizes(args.batch_size)
    tokenizer = get_tokenizer(args.model)
    model, torch_dtype = get_model(args.model, device, args.dtype)

    #output
    print(f"model: {args.model} | device: {device} | dtype: {torch_dtype}")
    print(f"prompts: {args.prompts} ({len(prompts)} lines)")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print("-" * 90)

    #Compute
    file_exists = os.path.exists(args.out_csv)
    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["backend", "model", "dtype", "device", "batch_size", "avg_ttft", "avg_latency", "tokps_total", "tokps_new","avg_kv_cache_mb"])

        for size in batch_sizes:
            params = build_parameters(size, prompts, device, tokenizer, args)

            #Warmup
            for _ in range(args.warmup):
                measure(model, params, device)

            ttft_list = []
            latency_list = []
            total_tokens_list = []
            new_tokens_list = []
            kv_list = []
            for _ in range(args.runs):
                # TTFT
                tt, _, _, _ = measure(model, params, device, max_new_token_override=1)
                ttft_list.append(tt)

                # Full run
                lat, tot, new, kv_mb = measure(model, params, device, measure_kv=True)

                latency_list.append(lat)
                total_tokens_list.append(tot)
                new_tokens_list.append(new)
                kv_list.append(kv_mb)
            
            pass

            #Avg
            avg_ttft = sum(ttft_list) / len(ttft_list)
            avg_lat  = sum(latency_list) / len(latency_list)
            avg_tot  = sum(total_tokens_list) / len(total_tokens_list)
            avg_new  = sum(new_tokens_list) / len(new_tokens_list)
            avg_kv   = sum(kv_list) / len(kv_list)
            
            tokps_total = avg_tot / avg_lat
            tokps_new = avg_new / avg_lat

            print(f"[{size}] " f"avg_ttft: {avg_ttft:.4f} | "f"avg_latency: {avg_lat:.4f} | " f"tokens/s(total): {tokps_total:.2f} | " f"tokens/s(new): {tokps_new:.2f} | "f"kv_cache(MB): {avg_kv:.1f}")
            writer.writerow([args.backend, args.model, str(torch_dtype), device, size, f"{avg_ttft:.6f}", f"{avg_lat:.6f}", f"{tokps_total:.6f}", f"{tokps_new:.6f}", f"{avg_kv:.2f}"])
        
        pass
    
    
    pass

    print("-" * 90)
    print(f"Wrote CSV to: {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")