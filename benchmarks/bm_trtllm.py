import time
import torch
from transformers import AutoTokenizer
from tensorrt_llm.runtime import ModelRunner

ENGINE_DIR = "/workspace/trt_engine/qwen2p5_7b_fp16_b16_i2048_s2560"
HF_MODEL_DIR = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"  # adjust if needed

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

runner = ModelRunner.from_dir(ENGINE_DIR)

prompts = ["Explain KV cache in one paragraph."] * 16
inputs = tokenizer(prompts, return_tensors="pt", padding=True)
input_ids = inputs["input_ids"].cuda()

torch.cuda.synchronize()
start = time.perf_counter()

outputs = runner.generate(
    input_ids=input_ids,
    max_new_tokens=512,
)

torch.cuda.synchronize()
end = time.perf_counter()

elapsed = end - start

num_new_tokens = 512 * 16
tokps = num_new_tokens / elapsed

print(f"Latency: {elapsed:.3f}s")
print(f"tok/s(new): {tokps:.2f}")