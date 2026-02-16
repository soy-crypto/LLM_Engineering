import pandas as pd

hf = pd.read_csv("results/hf_results.csv")
vllm = pd.read_csv("results/vllm_results.csv")
trt = pd.read_csv("results/trt_results.csv")

df = pd.concat([hf, vllm, trt], ignore_index=True)

df.to_csv("results/all_results.csv", index=False)

print("Merged results written to results/all_results.csv")
print(df.groupby(["backend", "batch_size"])["tokps_new"].mean())
