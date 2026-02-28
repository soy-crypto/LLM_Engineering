import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


############################################################
# Paths based on your project structure
############################################################

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_ROOT = os.path.join(PROJECT_ROOT, "benchmarks", "results")

HF_DIR   = os.path.join(RESULTS_ROOT, "hf")
VLLM_DIR = os.path.join(RESULTS_ROOT, "vllm")
TRT_DIR  = os.path.join(RESULTS_ROOT, "trt")

OUT_DIR  = os.path.join(PROJECT_ROOT, "results", "aggregate")

OUT_CSV = os.path.join(OUT_DIR, "aggregate_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)


############################################################
# Load CSVs
############################################################

def load_backend(path, backend):

    files = glob.glob(os.path.join(path, "*.csv"))

    dfs = []

    for f in files:

        df = pd.read_csv(f)

        model = os.path.basename(f).replace(".csv", "")

        df["backend"] = backend
        df["model"] = model

        # Normalize throughput column
        if "tokens_per_sec" in df.columns:
            df["throughput"] = df["tokens_per_sec"]

        elif "tokps_new" in df.columns:
            df["throughput"] = df["tokps_new"]

        else:
            print("Skipping (no throughput column):", f)
            continue

        # Normalize latency column
        if "total_latency_ms" in df.columns:
            df["latency"] = df["total_latency_ms"]

        elif "avg_latency" in df.columns:
            df["latency"] = df["avg_latency"]

        else:
            print("Skipping (no latency column):", f)
            continue

        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame()


############################################################
# Aggregate
############################################################

def aggregate():

    print("Aggregating benchmark results...")

    hf   = load_backend(HF_DIR, "HF")
    vllm = load_backend(VLLM_DIR, "vLLM")
    trt  = load_backend(TRT_DIR, "TensorRT-LLM")

    df = pd.concat([hf, vllm, trt], ignore_index=True)

    df.to_csv(OUT_CSV, index=False)

    print("Saved aggregate CSV:")
    print(OUT_CSV)

    return df


############################################################
# Plot
############################################################

def plot(df):

    print("Generating plots...")

    models = df["model"].unique()

    for model in models:

        sub_model = df[df["model"] == model]

        ####################################################
        # Throughput plot
        ####################################################

        plt.figure()

        for backend in sub_model["backend"].unique():

            sub = sub_model[sub_model["backend"] == backend]

            plt.plot(
                sub["batch_size"],
                sub["throughput"],
                marker="o",
                label=backend
            )

        plt.xlabel("Batch Size")
        plt.ylabel("Tokens/sec")
        plt.title(f"{model} Throughput")
        plt.legend()
        plt.grid()

        out = os.path.join(OUT_DIR, f"{model}_throughput.png")
        plt.savefig(out)
        plt.close()

        ####################################################
        # Latency plot
        ####################################################

        plt.figure()

        for backend in sub_model["backend"].unique():

            sub = sub_model[sub_model["backend"] == backend]

            plt.plot(
                sub["batch_size"],
                sub["latency"],
                marker="o",
                label=backend
            )

        plt.xlabel("Batch Size")
        plt.ylabel("Latency (ms)")
        plt.title(f"{model} Latency")
        plt.legend()
        plt.grid()

        out = os.path.join(OUT_DIR, f"{model}_latency.png")
        plt.savefig(out)
        plt.close()

    print("Plots saved to:")
    print(OUT_DIR)


############################################################
# Summary
############################################################

def summary(df):

    print("\nAverage throughput by backend:\n")

    print(
        df.groupby("backend")["throughput"]
        .mean()
        .sort_values(ascending=False)
    )


############################################################
# Main
############################################################

def main():

    df = aggregate()

    if df.empty:
        print("No benchmark results found.")
        return

    plot(df)

    summary(df)


if __name__ == "__main__":
    main()