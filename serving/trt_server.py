from fastapi import FastAPI
from tensorrt_llm.runtime import ModelRunner

ENGINE_DIR = "/workspace/trt_engine/llama3_1_8b_bf16_b16_s4096"

runner = ModelRunner.from_dir(ENGINE_DIR)

app = FastAPI()

@app.post("/generate")
def generate(prompt: str):

    output = runner.generate(prompt, max_new_tokens=128)

    return {"text": output}