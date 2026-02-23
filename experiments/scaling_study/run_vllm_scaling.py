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
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--context", type=int, required=True)
    parser.add_argument("--decode", type=int, required=True)
    args = parser.parse_args()

    prompts = ["Hello world " * 50] * args.batch

    llm = LLM(model=args.model)
    sampling_params = SamplingParams(max_tokens=args.decode)

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end = time.time()

    total_latency = end - start
    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)

    tps = total_tokens / total_latency
    mem = get_gpu_memory()

    print(
        f"vllm,{args.batch},{args.context},{args.decode},"
        f"{total_latency * 1000:.2f},{tps:.2f},{mem}"
    )

if __name__ == "__main__":
    main()