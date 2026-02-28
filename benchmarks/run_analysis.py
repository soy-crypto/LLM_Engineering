import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


############################################################
# Paths
############################################################

PROJECT_ROOT = "/workspace/LLM_Engineering"

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
OUT_DIR = os.path.join(RESULTS_DIR, "aggregate")

OUT_CSV = os.path.join(OUT_DIR, "aggregate_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)


############################################################
# Column detection helper
############################################################

def detect_column(df, candidates):

    for c in candidates:

        if c in df.columns:
            return c

    raise RuntimeError(
        f"None of expected columns found.\n"
        f"Expected one of: {candidates}\n"
        f"Actual columns: {list(df.columns)}"
    )


############################################################
# Load results
############################################################

def load_results():

    print("Loading benchmark results...\n")

    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))

    if not files:
        print("No CSV files found.")
        return None

    dfs = []

    print("Found CSV files:")

    for file in files:

        name = os.path.basename(file)

        if name.startswith("aggregate"):
            continue

        print(f"  {name}")

        df = pd.read_csv(file)

        print(f"     columns: {list(df.columns)}")

        ####################################################
        # Backend detection
        ####################################################

        if name.startswith("trt_"):

            backend = "TensorRT-LLM"

            throughput_col = detect_column(
                df,
                ["tokps_new", "tokens_per_sec"]
            )

            latency_col = detect_column(
                df,
                ["avg_latency", "total_latency_ms"]
            )

        elif name.startswith("vllm_"):

            backend = "vLLM"

            throughput_col = detect_column(
                df,
                ["tokens_per_sec", "tokps_new"]
            )

            latency_col = detect_column(
                df,
                ["total_latency_ms", "avg_latency"]
            )

        elif name.startswith("results_hf"):

            backend = "HF"

            throughput_col = detect_column(
                df,
                ["tokens_per_sec", "tokps_new"]
            )

            latency_col = detect_column(
                df,
                ["total_latency_ms", "avg_latency"]
            )

        else:
            print("     Skipping unknown file.")
            continue

        ####################################################
        # Normalize columns
        ####################################################

        df["backend"] = backend
        df["throughput"] = df[throughput_col]
        df["latency"] = df[latency_col]

        df["model_file"] = name.replace(".csv", "")

        dfs.append(df)

    if not dfs:
        print("No benchmark results found.")
        return None

    return pd.concat(dfs, ignore_index=True)


############################################################
# Plot throughput
############################################################

def plot_throughput(df):

    print("\nGenerating throughput plots...")

    models = df["model_file"].unique()

    for model in models:

        sub = df[df["model_file"] == model]

        plt.figure(figsize=(8,6))

        for backend in sub["backend"].unique():

            s = sub[sub["backend"] == backend]

            plt.plot(
                s["batch_size"],
                s["throughput"],
                marker="o",
                linewidth=2,
                label=backend
            )

        plt.title(f"Throughput vs Batch Size\n{model}")
        plt.xlabel("Batch Size")
        plt.ylabel("Tokens/sec")
        plt.grid(True)
        plt.legend()

        out = os.path.join(
            OUT_DIR,
            f"{model}_throughput.png"
        )

        plt.savefig(out, dpi=150)
        plt.close()

        print("Saved:", out)


############################################################
# Plot latency
############################################################

def plot_latency(df):

    print("\nGenerating latency plots...")

    models = df["model_file"].unique()

    for model in models:

        sub = df[df["model_file"] == model]

        plt.figure(figsize=(8,6))

        for backend in sub["backend"].unique():

            s = sub[sub["backend"] == backend]

            plt.plot(
                s["batch_size"],
                s["latency"],
                marker="o",
                linewidth=2,
                label=backend
            )

        plt.title(f"Latency vs Batch Size\n{model}")
        plt.xlabel("Batch Size")
        plt.ylabel("Latency (ms)")
        plt.grid(True)
        plt.legend()

        out = os.path.join(
            OUT_DIR,
            f"{model}_latency.png"
        )

        plt.savefig(out, dpi=150)
        plt.close()

        print("Saved:", out)


############################################################
# Summary
############################################################

def print_summary(df):

    print("\n=================================")
    print("Average Throughput (tokens/sec)")
    print("=================================\n")

    summary = (
        df.groupby("backend")["throughput"]
        .mean()
        .sort_values(ascending=False)
    )

    print(summary)


############################################################
# Main
############################################################

def main():

    df = load_results()

    if df is None:
        return

    print("\nSaving aggregate CSV:")
    print(OUT_CSV)

    df.to_csv(OUT_CSV, index=False)

    plot_throughput(df)
    plot_latency(df)

    print_summary(df)

    print("\nDone.")
    print("\nOutput directory:")
    print(OUT_DIR)


############################################################

if __name__ == "__main__":
    main()