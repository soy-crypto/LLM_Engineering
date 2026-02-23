from vllm import LLM, SamplingParams
from fastapi import FastAPI
from pydantic import BaseModel

MODEL = "/workspace/hf_models/llama3_1_8b"

llm = LLM(model=MODEL, dtype="bfloat16")

app = FastAPI()

class Request(BaseModel):
    prompt: str
    max_tokens: int = 128

@app.post("/generate")
def generate(req: Request):
    outputs = llm.generate(
        [req.prompt],
        SamplingParams(max_tokens=req.max_tokens)
    )
    return {"text": outputs[0].outputs[0].text}