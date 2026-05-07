import csv
import os
from typing import List

SCHEMA = ["backend", "model", "dtype", "batch_size", "avg_ttft_s", "avg_latency_s", "tokps_new", "gpu_mem_mb"]


def load_prompts(path: str) -> List[str]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def parse_batch_sizes(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def build_batch(prompts: List[str], size: int) -> List[str]:
    return (prompts * (size // len(prompts) + 1))[:size]


def get_gpu_mem_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except ImportError:
        pass
    return 0.0


class CsvWriter:
    def __init__(self, path: str):
        exists = os.path.exists(path)
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._f)
        if not exists:
            self._writer.writerow(SCHEMA)

    def write(self, backend: str, model: str, dtype: str, batch_size: int,
              avg_ttft_s: float, avg_latency_s: float, tokps_new: float, gpu_mem_mb: float):
        self._writer.writerow([
            backend, model, dtype, batch_size,
            f"{avg_ttft_s:.6f}", f"{avg_latency_s:.6f}",
            f"{tokps_new:.3f}", f"{gpu_mem_mb:.1f}",
        ])
        self._f.flush()

    def close(self):
        self._f.close()
