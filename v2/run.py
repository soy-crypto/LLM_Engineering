import argparse
import os
import subprocess


TRT_MODELS = [
    {"model_id": "meta-llama/Llama-3.1-8B",            "engine_dir": "/workspace/trt_engine/llama3_1_8b_bf16_b16_s4096"},
    {"model_id": "mistralai/Mistral-7B-Instruct-v0.3",  "engine_dir": "/workspace/trt_engine/mistral_7b_bf16_b16_s4096"},
    {"model_id": "google/gemma-7b-it",                  "engine_dir": "/workspace/trt_engine/gemma_7b_bf16_b16_s4096"},
    {"model_id": "Qwen/Qwen2.5-7B-Instruct",            "engine_dir": "/workspace/trt_engine/qwen2_5_7b_bf16_b16_s4096"},
    {"model_id": "microsoft/phi-3-mini-4k-instruct",    "engine_dir": "/workspace/trt_engine/phi_3_mini_bf16_b16_s4096"},
]


def get_args():
    parser = argparse.ArgumentParser(description="Full experiment pipeline")
    parser.add_argument("--skip_trt_build", action="store_true", help="Skip TRT engine build step")
    parser.add_argument("--skip_experiments", action="store_true", help="Skip benchmark runs")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip aggregation and plotting")
    return parser.parse_args()


def engine_exists(engine_dir: str) -> bool:
    return os.path.isfile(os.path.join(engine_dir, "rank0.engine"))


def build_trt_engines():
    print("=" * 80)
    print("Building TRT engines")
    print("=" * 80)
    for entry in TRT_MODELS:
        model_id = entry["model_id"]
        engine_dir = entry["engine_dir"]
        if engine_exists(engine_dir):
            print(f"[skip] engine already exists: {engine_dir}")
            continue
        print(f"\nBuilding: {model_id}")
        subprocess.run(
            ["bash", "v2/benchmarks/trt/build.sh",
             "--model_id", model_id,
             "--engine_dir", engine_dir],
            check=True,
        )


def run_experiments():
    print("=" * 80)
    print("Running experiments")
    print("=" * 80)
    subprocess.run(["bash", "v2/scripts/bootstrap/bootstrap_hf.sh"], check=True)
    subprocess.run(["bash", "v2/scripts/bootstrap/bootstrap_vllm.sh"], check=True)
    subprocess.run(["bash", "v2/scripts/run/run_all_backends.sh"], check=True)


def run_analysis():
    print("=" * 80)
    print("Aggregating results")
    print("=" * 80)
    subprocess.run(["python", "analysis/aggregate.py"], check=True)

    print("Plotting results")
    subprocess.run(["python", "analysis/plot_results.py"], check=True)


def main():
    args = get_args()

    if not args.skip_trt_build:
        build_trt_engines()

    if not args.skip_experiments:
        run_experiments()

    if not args.skip_analysis:
        run_analysis()

    print("Done.")


if __name__ == "__main__":
    main()
