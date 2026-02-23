import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/workspace/LLM_Engineering/results/aggregate"

CSV_FILE = os.path.join(BASE_DIR, "aggregate_results.csv")

OUT_THROUGHPUT = os.path.join(BASE_DIR, "throughput_vs_batch.png")
OUT_LATENCY = os.path.join(BASE_DIR, "latency_vs_batch.png")

def plot_throughput(df):

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

    plt.savefig(OUT_THROUGHPUT)

    print(f"Saved: {OUT_THROUGHPUT}")

def plot_latency(df):

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

    plt.savefig(OUT_LATENCY)

    print(f"Saved: {OUT_LATENCY}")

def main():

    df = pd.read_csv(CSV_FILE)

    plot_throughput(df)

    plot_latency(df)

    plt.show()

if __name__ == "__main__":
    main()