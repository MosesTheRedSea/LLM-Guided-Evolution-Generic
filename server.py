import time
import torch
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
from src.cfg.constants import *

app = FastAPI(title="LLM API", version="1.0")

# Path To Local Large Language Model
MODEL_PATH = "LLM_MODEL_PATH"

llm_model = None
llm_tokenizer = None
llm_pipeline = None

class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 764
    top_p: float = 0.8
    temperature: float = 0.7

@app.on_event("startup")
async def load_model_on_startup():
    global llm_model, llm_tokenizer, llm_pipeline
    print("Loading model and tokenizer at startup...")
    try:
        llm_model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()

        llm_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

        llm_pipeline = transformers.pipeline(
            model=llm_model,
            tokenizer=llm_tokenizer,
            return_full_text=False,
            task="text-generation",
            temperature=0.1,
            top_p=0.15,
            top_k=0,
            max_new_tokens=1000,
            repetition_penalty=1.1,
            do_sample=True,
        )
        print("Model and pipeline loaded successfully.")
    except Exception as e:
        print("ERROR loading model: {e}")

def submit_to_local_model(txt2llm, max_new_tokens=764, top_p=0.2, temperature=0.1):
    """
    Submits txt2llm to the local model. If the model has not been intialized before, this function will first have to intialize the model.

    Parameters:
    txt2llm (str): input to llm
    max_new_tokens (int): maximum number of tokens model should generate
    top_p (int): threshold, higher to consider wider range of words
    temperature (int): randomness, higher for more varied outputs

    Returns:
    dict: generated_text (output of LLM) and response_time (time after intializing model until generated text obtained)
    """
    if llm_pipeline is None:
        raise(RuntimeError("Model not loaded yet..."))

    start_time = time.time()
    result = llm_pipeline(
        txt2llm,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        reptition_penalty=1.1,
        do_sample=True
    )
    output_txt = result[0]["generated_text"]
    response_time = round(time.time() - start_time, 2)
    return {"generated_text": output_txt, "response_time_sec": response_time}


@app.post("/generate")
async def generate_text(request: LLMRequest):
    try:
        return submit_to_local_model(
            request.prompt,
            request.max_new_tokens,
            request.top_p,
            request.temperature,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    status = "running" if llm_pipeline else "loading"
    return {"message": f"LLM API is {status}."}

print('Server running!')