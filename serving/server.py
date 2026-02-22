import asyncio
import time
from typing import List

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==============================
# Config
# ==============================

MODEL_ID = "meta-llama/Llama-3.1-8B"
DEVICE = "cuda"
DTYPE = torch.float16

MAX_BATCH_SIZE = 8
MAX_WAIT_TIME = 0.02  # 20ms
MAX_NEW_TOKENS = 64

# ==============================
# Initialize Model Once
# ==============================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto"
)

model.eval()

print("Model loaded.")

# ==============================
# Request Schema
# ==============================

class GenerateRequest(BaseModel):
    prompt: str

# ==============================
# Global Queue
# ==============================

request_queue: List[dict] = []
queue_lock = asyncio.Lock()

# ==============================
# FastAPI App
# ==============================

app = FastAPI()

@app.post("/generate")
async def generate(req: GenerateRequest):
    loop = asyncio.get_event_loop()
    future = loop.create_future()

    request_item = {
        "prompt": req.prompt,
        "future": future,
        "arrival_time": time.time()
    }

    async with queue_lock:
        request_queue.append(request_item)

    return await future


# ==============================
# Batching Worker
# ==============================

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

        for item, output_text in zip(batch, decoded):
            item["future"].set_result({
                "response": output_text,
                "latency_sec": time.time() - item["arrival_time"]
            })


# ==============================
# Startup Event
# ==============================

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(batching_worker())