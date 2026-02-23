import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "results"
OUT_DIR = os.path.join(BASE_DIR, "aggregate")

HF_DIR = os.path.join(BASE_DIR, "hf")
VLLM_DIR = os.path.join(BASE_DIR, "vllm")
TRT_DIR = os.path.join(BASE_DIR, "trt")

OUT_CSV = os.path.join(OUT_DIR, "aggregate_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)


def load_backend(path, backend):

    files = glob.glob(os.path.join(path, "*.csv"))

    dfs = []

    for f in files:

        df = pd.read_csv(f)

        df["backend"] = backend

        model = os.path.basename(f).replace(".csv", "")
        df["model"] = model

        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame()


def aggregate():

    print("Aggregating results...")

    hf = load_backend(HF_DIR, "HF")
    vllm = load_backend(VLLM_DIR, "vLLM")
    trt = load_backend(TRT_DIR, "TensorRT-LLM")

    df = pd.concat([hf, vllm, trt], ignore_index=True)

    df.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)

    return df


def plot(df):

    print("Generating plots...")

    # Throughput plot
    plt.figure()

    for backend in df["backend"].unique():

        sub = df[df["backend"] == backend]

        plt.plot(
            sub["batch_size"],
            sub["tokens_per_sec"],
            marker="o",
            label=backend
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Tokens/sec")
    plt.title("Throughput vs Batch Size")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(OUT_DIR, "throughput_vs_batch.png"))

    # Latency plot
    plt.figure()

    for backend in df["backend"].unique():

        sub = df[df["backend"] == backend]

        plt.plot(
            sub["batch_size"],
            sub["total_latency_ms"],
            marker="o",
            label=backend
        )

    plt.xlabel("Batch Size")
    plt.ylabel("Latency (ms)")
    plt.title("Latency vs Batch Size")
    plt.legend()
    plt.grid()

    plt.savefig(os.path.join(OUT_DIR, "latency_vs_batch.png"))

    print("Plots saved.")


def summary(df):

    print("\nSummary tokens/sec:")

    print(
        df.groupby("backend")["tokens_per_sec"].mean()
    )


def main():

    df = aggregate()

    plot(df)

    summary(df)


if __name__ == "__main__":
    main()