import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

OUT_DIR = os.path.join(RESULTS_DIR, "aggregate")

OUT_CSV = os.path.join(OUT_DIR, "aggregate_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)


############################################################
# Load CSV files
############################################################

def load_all():

    files = glob.glob(os.path.join(RESULTS_DIR, "*.csv"))

    dfs = []

    for f in files:

        name = os.path.basename(f)

        if name.startswith("aggregate"):
            continue

        df = pd.read_csv(f)

        ####################################################
        # Detect backend
        ####################################################

        if name.startswith("trt_"):
            backend = "TensorRT-LLM"

        elif name.startswith("vllm_"):
            backend = "vLLM"

        elif name.startswith("results_hf"):
            backend = "HF"

        else:
            continue

        ####################################################
        # Normalize columns
        ####################################################

        if "tokps_new" in df.columns:
            df["throughput"] = df["tokps_new"]

        elif "tokens_per_sec" in df.columns:
            df["throughput"] = df["tokens_per_sec"]

        else:
            continue

        if "avg_latency" in df.columns:
            df["latency"] = df["avg_latency"]

        elif "total_latency_ms" in df.columns:
            df["latency"] = df["total_latency_ms"]

        else:
            continue

        ####################################################
        # Add metadata
        ####################################################

        model = name.replace(".csv", "")
        df["model"] = model
        df["backend"] = backend

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


############################################################
# Plot
############################################################

def plot(df):

    models = df["model"].unique()

    for model in models:

        sub = df[df["model"] == model]

        ###################################
        # Throughput
        ###################################

        plt.figure()

        for backend in sub["backend"].unique():

            s = sub[sub["backend"] == backend]

            plt.plot(
                s["batch_size"],
                s["throughput"],
                marker="o",
                label=backend
            )

        plt.title(model)
        plt.xlabel("Batch Size")
        plt.ylabel("Tokens/sec")
        plt.legend()
        plt.grid()

        plt.savefig(
            os.path.join(OUT_DIR, f"{model}_throughput.png")
        )

        plt.close()

        ###################################
        # Latency
        ###################################

        plt.figure()

        for backend in sub["backend"].unique():

            s = sub[sub["backend"] == backend]

            plt.plot(
                s["batch_size"],
                s["latency"],
                marker="o",
                label=backend
            )

        plt.title(model)
        plt.xlabel("Batch Size")
        plt.ylabel("Latency ms")
        plt.legend()
        plt.grid()

        plt.savefig(
            os.path.join(OUT_DIR, f"{model}_latency.png")
        )

        plt.close()


############################################################
# Main
############################################################

def main():

    print("Loading results...")

    df = load_all()

    print("Saving aggregate CSV...")

    df.to_csv(OUT_CSV, index=False)

    plot(df)

    print("Done.")
    print("Results:", OUT_DIR)


if __name__ == "__main__":
    main()