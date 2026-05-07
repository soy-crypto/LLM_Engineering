import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
OUT_DIR      = os.path.join(RESULTS_DIR, "aggregate")

os.makedirs(OUT_DIR, exist_ok=True)

SCHEMA_COLS = ["backend", "model", "dtype", "batch_size", "avg_ttft_s", "avg_latency_s", "tokps_new", "gpu_mem_mb"]


def load_results() -> pd.DataFrame:
    files = [f for f in glob.glob(os.path.join(RESULTS_DIR, "*.csv"))
             if not os.path.basename(f).startswith("aggregate")]

    if not files:
        raise FileNotFoundError(f"No CSV files found in {RESULTS_DIR}")

    dfs = []
    print("Loading results:")
    for path in files:
        df = pd.read_csv(path)
        missing = [c for c in SCHEMA_COLS if c not in df.columns]
        if missing:
            print(f"  SKIP {os.path.basename(path)} — missing columns: {missing}")
            continue
        print(f"  {os.path.basename(path)}  ({len(df)} rows)")
        dfs.append(df[SCHEMA_COLS])

    if not dfs:
        raise ValueError("No valid result files found. Run benchmarks first.")

    return pd.concat(dfs, ignore_index=True)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str):
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        plt.figure(figsize=(8, 5))
        for backend in sub["backend"].unique():
            s = sub[sub["backend"] == backend].sort_values("batch_size")
            plt.plot(s["batch_size"], s[metric], marker="o", linewidth=2, label=backend)
        short_model = model.split("/")[-1]
        plt.title(f"{ylabel} vs Batch Size\n{short_model}")
        plt.xlabel("Batch Size")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.legend()
        out = os.path.join(OUT_DIR, f"{short_model}_{metric}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


def print_summary(df: pd.DataFrame):
    print("\n=== Average tok/s by backend ===")
    print(df.groupby("backend")["tokps_new"].mean().sort_values(ascending=False).to_string())
    print("\n=== Average latency (s) by backend ===")
    print(df.groupby("backend")["avg_latency_s"].mean().sort_values().to_string())


def main():
    df = load_results()

    agg_csv = os.path.join(OUT_DIR, "aggregate_results.csv")
    df.to_csv(agg_csv, index=False)
    print(f"\nAggregate CSV: {agg_csv}")

    print("\nGenerating plots...")
    plot_metric(df, "tokps_new", "Tokens/sec")
    plot_metric(df, "avg_latency_s", "Latency (s)")
    plot_metric(df, "avg_ttft_s", "TTFT (s)")

    print_summary(df)
    print(f"\nDone. Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
