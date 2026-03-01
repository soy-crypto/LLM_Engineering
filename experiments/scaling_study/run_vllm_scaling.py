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

    print("Warmup run...")
    llm.generate([prompt], SamplingParams(max_tokens=1))

    print("\nbackend,batch,prefill_ms,decode_ms_per_token,total_tps,gpu_mem_mb")

    for batch in batch_sizes:
        prompts = [prompt] * batch

        # ---- decode=1 (prefill + 1 token)
        params_1 = SamplingParams(max_tokens=1)

        start = time.time()
        llm.generate(prompts, params_1)
        end = time.time()

        latency_1 = end - start

        # ---- decode=N
        params_n = SamplingParams(max_tokens=args.decode)

        start = time.time()
        outputs = llm.generate(prompts, params_n)
        end = time.time()

        latency_n = end - start

        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_tps = total_tokens / latency_n

        # ---- isolate
        prefill_latency = latency_1
        decode_per_token = (latency_n - latency_1) / (args.decode - 1)

        mem = get_gpu_memory()

        print(
            f"vllm,{batch},{prefill_latency*1000:.2f},"
            f"{decode_per_token*1000:.4f},"
            f"{total_tps:.2f},{mem}"
        )


if __name__ == "__main__":
    main()