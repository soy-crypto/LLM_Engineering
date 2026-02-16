import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/all_results.csv")

plt.figure()

for backend in df["backend"].unique():
    sub = df[df["backend"] == backend]
    plt.plot(sub["batch_size"], sub["avg_ttft"], marker="o", label=backend)

plt.xlabel("Batch Size")
plt.ylabel("TTFT (seconds)")
plt.title("TTFT vs Batch Size")
plt.legend()
plt.grid(True)

plt.savefig("results/ttft_vs_batch.png")
plt.show()

