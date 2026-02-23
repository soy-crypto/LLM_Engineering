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
    sampling_params = SamplingParams(max_tokens=args.decode)

    print("Warmup run...")
    llm.generate(["warmup prompt"], sampling_params)

    print("\nbackend,batch,total_latency_ms,tokens_per_sec,gpu_mem_mb")

    for batch in batch_sizes:
        prompts = ["Hello world " * 50] * batch

        start = time.time()
        outputs = llm.generate(prompts, sampling_params)
        end = time.time()

        total_latency = end - start
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        tps = total_tokens / total_latency
        mem = get_gpu_memory()

        print(
            f"vllm,{batch},{total_latency*1000:.2f},{tps:.2f},{mem}"
        )

if __name__ == "__main__":
    main()