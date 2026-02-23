import asyncio
import time
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/workspace/hf_models/llama3_1_8b"
DEVICE = "cuda"
DTYPE = torch.bfloat16

MAX_BATCH_SIZE = 8
MAX_WAIT_TIME = 0.02
MAX_NEW_TOKENS = 64

print("Loading HF model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=DTYPE,
    device_map="auto"
)

model.eval()

print("HF model loaded.")

class GenerateRequest(BaseModel):
    prompt: str

request_queue: List[dict] = []
queue_lock = asyncio.Lock()

app = FastAPI()

@app.post("/generate")
async def generate(req: GenerateRequest):

    loop = asyncio.get_event_loop()
    future = loop.create_future()

    async with queue_lock:
        request_queue.append({
            "prompt": req.prompt,
            "future": future,
            "arrival": time.time()
        })

    return await future

async def batching_worker():

    while True:

        await asyncio.sleep(MAX_WAIT_TIME)

        async with queue_lock:

            if not request_queue:
                continue

            batch = request_queue[:MAX_BATCH_SIZE]
            del request_queue[:MAX_BATCH_SIZE]

        prompts = [item["prompt"] for item in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for item, text in zip(batch, decoded):

            item["future"].set_result({
                "text": text,
                "latency": time.time() - item["arrival"]
            })

@app.on_event("startup")
async def startup():
    asyncio.create_task(batching_worker())