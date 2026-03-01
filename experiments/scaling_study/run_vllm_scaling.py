import time
import argparse
import subprocess
from vllm import LLM, SamplingParams


def get_gpu_memory():
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"]
    )
    return int(result.decode().strip().split("\n")[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--batches", required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--decode", type=int, required=True)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batches.split(",")]

    print("Loading model once...")
    llm = LLM(model=args.model)

    prompt = ("hello " * args.context).strip()

    print("\nbackend,phase,batch,latency_ms,tokens_per_sec,gpu_mem_mb")

    for batch in batch_sizes:
        prompts = [prompt] * batch

        # ------------------------
        # First token (prefill + 1 decode)
        # ------------------------
        one_token_params = SamplingParams(max_tokens=1)

        start = time.time()
        outputs = llm.generate(prompts, one_token_params)
        end = time.time()

        first_token_latency = end - start
        mem = get_gpu_memory()

        print(
            f"vllm,first_token,{batch},{first_token_latency*1000:.2f},0,{mem}"
        )

        # ------------------------
        # Full decode
        # ------------------------
        decode_params = SamplingParams(max_tokens=args.decode)

        start = time.time()
        outputs = llm.generate(prompts, decode_params)
        end = time.time()

        full_latency = end - start
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        tps = total_tokens / full_latency
        mem = get_gpu_memory()

        print(
            f"vllm,full_decode,{batch},{full_latency*1000:.2f},{tps:.2f},{mem}"
        )


if __name__ == "__main__":
    main()