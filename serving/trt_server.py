import asyncio
import time
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from tensorrt_llm.runtime import ModelRunner

ENGINE_DIR = "/workspace/trt_engine/llama3_1_8b_bf16_b16_s4096"

MAX_BATCH_SIZE = 8
MAX_WAIT_TIME = 0.02
MAX_NEW_TOKENS = 64

print("Loading TensorRT engine...")

runner = ModelRunner.from_dir(ENGINE_DIR)

print("TensorRT engine loaded.")

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

        outputs = runner.generate(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS
        )

        for item, text in zip(batch, outputs):

            item["future"].set_result({
                "text": text,
                "latency": time.time() - item["arrival"]
            })


@app.on_event("startup")
async def startup():
    asyncio.create_task(batching_worker())