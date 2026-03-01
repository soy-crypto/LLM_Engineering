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

    # Construct prompt with controlled context length
    prompt = ("hello " * args.context).strip()

    print("Warmup run...")
    llm.generate([prompt], SamplingParams(max_tokens=1))

    print("\nbackend,phase,batch,latency_ms,tokens_per_sec,gpu_mem_mb")

    for batch in batch_sizes:
        prompts = [prompt] * batch

        # ------------------------
        # PREFILL ONLY (decode=0)
        # ------------------------
        prefill_params = SamplingParams(max_tokens=0)

        start = time.time()
        llm.generate(prompts, prefill_params)
        end = time.time()

        prefill_latency = end - start
        mem = get_gpu_memory()

        print(
            f"vllm,prefill,{batch},{prefill_latency*1000:.2f},0,{mem}"
        )

        # ------------------------
        # DECODE ONLY
        # ------------------------
        decode_params = SamplingParams(max_tokens=args.decode)

        start = time.time()
        outputs = llm.generate(prompts, decode_params)
        end = time.time()

        decode_latency = end - start
        total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

        tps = total_tokens / decode_latency
        mem = get_gpu_memory()

        print(
            f"vllm,decode,{batch},{decode_latency*1000:.2f},{tps:.2f},{mem}"
        )


if __name__ == "__main__":
    main()