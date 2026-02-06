import argparse
import torch
import time
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from typing import List, Dict, Any, Tuple, Optional


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF LLM")
    parser.add_argument("--model",          type=str, default="gpt2")
    parser.add_argument("--prompts",        type=str, default="prompts.txt")
    parser.add_argument("--batch_size",     type=str, default="1, 2, 4")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--warmup",         type=int, default=1)
    parser.add_argument("--runs",           type=int, default=5000)
    parser.add_argument("--do_sample",      action="store_true")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--out_csv",        type=str, default="results.csv")
    parser.add_argument("--backend",        type=str, default="HF")

    args = parser.parse_args()

    return args


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
        raise ValueError(f"No file found in {path}")

    return prompts


def get_sizes(batch_size: str) -> List[int]:
    return [int(x.strip()) for x in batch_size.split(",") if x.strip()]


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_model(model_name: str, device: str) -> Tuple[PreTrainedModel, torch.dtype]:
    if device == "cuda":
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    model.eval()

    #return
    return model, torch_dtype


def get_parameters(size: int, prompts: List[str], device: str, tokenizer: PreTrainedTokenizerBase, args: argparse.Namespace) -> Dict[str, Any]:
    text = (prompts * (size // len(prompts) + 1))[: size]
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(device)
    parameters: Dict[str, Any] = {k: v for k, v in tokens.items()}
    parameters.update(dict(max_new_tokens=args.max_new_tokens, do_sample=args.do_sample, pad_token_id=tokenizer.pad_token_id))
    if args.do_sample:
        parameters.update(dict(temperature=args.temperature, top_p=args.top_p))

    return parameters


@torch.no_grad()
def measure(model: PreTrainedModel, parameters: Dict[str, Any], device: str, max_new_token_override: Optional[int] = None) -> Tuple[float, int, int]:
    #update parameters
    params = dict(parameters)
    if max_new_token_override is not None:
        params["max_new_tokens"] = max_new_token_override

    #start and end time
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()

    outputs = model.generate(**params)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    #metric
    elapsed = end - start
    output_tokens = int(outputs.numel())
    prompt_tokens = int(params["attention_mask"].sum().item())
    new_tokens = max(output_tokens - prompt_tokens, 0)

    #return
    return elapsed, output_tokens, new_tokens


def main():
    args = get_args()
    device = get_device()
    prompts = get_prompts(args.prompts)
    batch_sizes = get_sizes(args.batch_size)
    tokenizer = get_tokenizer(args.model)
    set_seed(args.seed, device)
    inference_model, torch_dtype = get_model(args.model, device)

    print(f"model: {args.model} | device: {device} | dtype: {torch_dtype}")
    print(f"prompts: {args.prompts} ({len(prompts)}) lines")
    print(f"max_new_tokens: {args.max_new_tokens} | sampling: {args.do_sample}")
    print("-" * 90)

    with open(args.out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["backend", "model", "dtype", "device", "batch_size", "avg_ttft", "avg_latency", "tokps_total", "tokps_new"])
        for size in batch_sizes:
            #build parameters & warmup
            parameters = get_parameters(size, prompts, device, tokenizer, args)
            for _ in range(args.warmup):
                _ = measure(inference_model, parameters, device)

            #runs
            elapsed:       List[float] = []
            output_tokens: List[int] = []
            new_tokens:    List[int] = []
            ttft:          List[float] = []
            for _ in range(args.runs):
                #TTFT
                tt, _, _ = measure(inference_model, parameters, device, 1)
                ttft.append(tt)

                #Full throughput
                e, t, n = measure(inference_model, parameters, device)
                elapsed.append(e)
                output_tokens.append(t)
                new_tokens.append(n)

            #avg
            avg_ttft = sum(ttft) / len(ttft)
            avg_el = sum(elapsed) / len(elapsed)
            avg_ot = sum(output_tokens) / len(output_tokens)
            avg_nt = sum(new_tokens) / len(new_tokens)

            #rate
            ot_ps = avg_ot / avg_el
            nt_ps = avg_nt / avg_el

            #print
            print(f"[{size}] avg_ttft: {avg_ttft:.4f} | avg_latency: {avg_el:.4f} | tokens/s(total): {ot_ps:.4f} | tokens/s(new): {nt_ps:.4f}")

            #write csv
            writer.writerow([args.backend, args.model, str(torch_dtype), device, size, f"{avg_ttft:.6f}", f"{avg_el:.6f}", f"{ot_ps:.6f}", f"{nt_ps:.6f}"])

    pass

    #print
    print("-" * 90)
    print(f"wrote CSV to: {args.out_csv}")

    #return
    return


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"errors: {e}")