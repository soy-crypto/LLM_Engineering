import os
import glob
import pandas as pd

BASE_DIR = "/workspace/LLM_Engineering/results"

HF_DIR = os.path.join(BASE_DIR, "hf")
VLLM_DIR = os.path.join(BASE_DIR, "vllm")
TRT_DIR = os.path.join(BASE_DIR, "trt")

OUT_DIR = os.path.join(BASE_DIR, "aggregate")
OUT_FILE = os.path.join(OUT_DIR, "aggregate_results.csv")

os.makedirs(OUT_DIR, exist_ok=True)

def load_backend(dir_path, backend_name):

    files = glob.glob(os.path.join(dir_path, "*.csv"))

    dfs = []

    for f in files:

        df = pd.read_csv(f)

        df["backend"] = backend_name

        # infer model name from filename
        model_name = os.path.basename(f).replace(".csv", "")
        df["model"] = model_name

        dfs.append(df)

    if dfs:
        return pd.concat(dfs, ignore_index=True)

    return pd.DataFrame()

def main():

    print("Loading HF results...")
    hf = load_backend(HF_DIR, "HF")

    print("Loading vLLM results...")
    vllm = load_backend(VLLM_DIR, "vLLM")

    print("Loading TensorRT results...")
    trt = load_backend(TRT_DIR, "TensorRT-LLM")

    df = pd.concat([hf, vllm, trt], ignore_index=True)

    df.to_csv(OUT_FILE, index=False)

    print(f"Aggregate saved: {OUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    main()