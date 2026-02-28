import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = "/workspace/LLM_Engineering"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "aggregate")

os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "aggregate_results.csv")


############################################################
# Load CSV files
############################################################

def load_results():

    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))

    dfs = []

    print("Found CSV files:")

    for file in files:

        name = os.path.basename(file)

        if name.startswith("aggregate"):
            continue

        print("  ", name)

        df = pd.read_csv(file)

        ####################################################
        # Detect backend and normalize column names
        ####################################################

        if name.startswith("trt_"):

            df["backend"] = "TensorRT-LLM"
            df["throughput"] = df["tokps_new"]
            df["latency"] = df["avg_latency"]

        elif name.startswith("vllm_"):

            df["backend"] = "vLLM"
            df["throughput"] = df["tokens_per_sec"]
            df["latency"] = df["total_latency_ms"]

        elif name.startswith("results_hf"):

            df["backend"] = "HF"
            df["throughput"] = df["tokens_per_sec"]
            df["latency"] = df["total_latency_ms"]

        else:
            continue

        df["model_file"] = name.replace(".csv", "")

        dfs.append(df)

    if not dfs:
        print("ERROR: No benchmark results found.")
        return None

    return pd.concat(dfs, ignore_index=True)


############################################################
# Plot throughput
############################################################

def plot_throughput(df):

    models = df["model_file"].unique()

    for model in models:

        sub = df[df["model_file"] == model]

        plt.figure()

        for backend in sub["backend"].unique():

            s = sub[sub["backend"] == backend]

            plt.plot(
                s["batch_size"],
                s["throughput"],
                marker="o",
                label=backend
            )

        plt.title(f"Throughput vs Batch Size\n{model}")
        plt.xlabel("Batch Size")
        plt.ylabel("Tokens/sec")
        plt.legend()
        plt.grid()

        out = os.path.join(
            OUT_DIR,
            f"{model}_throughput.png"
        )

        plt.savefig(out)
        plt.close()

        print("Saved:", out)


############################################################
# Plot latency
############################################################

def plot_latency(df):

    models = df["model_file"].unique()

    for model in models:

        sub = df[df["model_file"] == model]

        plt.figure()

        for backend in sub["backend"].unique():

            s = sub[sub["backend"] == backend]

            plt.plot(
                s["batch_size"],
                s["latency"],
                marker="o",
                label=backend
            )

        plt.title(f"Latency vs Batch Size\n{model}")
        plt.xlabel("Batch Size")
        plt.ylabel("Latency (ms)")
        plt.legend()
        plt.grid()

        out = os.path.join(
            OUT_DIR,
            f"{model}_latency.png"
        )

        plt.savefig(out)
        plt.close()

        print("Saved:", out)


############################################################
# Summary
############################################################

def print_summary(df):

    print("\nAverage throughput (tokens/sec):")

    print(
        df.groupby("backend")["throughput"]
        .mean()
        .sort_values(ascending=False)
    )


############################################################
# Main
############################################################

def main():

    print("Loading benchmark results...\n")

    df = load_results()

    if df is None:
        return

    print("\nSaving aggregate CSV:")
    print(OUT_CSV)

    df.to_csv(OUT_CSV, index=False)

    print("\nGenerating plots...\n")

    plot_throughput(df)
    plot_latency(df)

    print_summary(df)

    print("\nDone.")
    print("Output folder:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()