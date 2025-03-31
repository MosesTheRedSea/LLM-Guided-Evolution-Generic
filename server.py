import time
import torch
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import threading
from src.cfg.constants import *

app = FastAPI(title="LLM API", version="1.0")

# Model Path
MODEL_PATH = '/storage/ice-shared/vip-vvk/llm_storage/mixtral/Mixtral-8x7B-Instruct-v0.1/'

class LLMModel:
    _instance = None
    _lock = threading.Lock()

    # called before __init__
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                # in case another thread created an instance already
                if cls._instance is None:
                    print(f"Loading model at {MODEL_PATH} for the first time")
                    cls._instance = super(LLMModel, cls).__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        ).eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

        self.pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task="text-generation",
            temperature=0.1,
            top_p=0.15,
            top_k=0,
            max_new_tokens=1000,
            repetition_penalty=1.1,
            do_sample=True,
        )

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
    model = LLMModel()

    start_time = time.time()
    result = model.pipeline(
        txt2llm,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )

    output_txt = result[0]["generated_text"]
    response_time = round(time.time() - start_time, 2)
    
    return {"generated_text": output_txt, "response_time_sec": response_time}

class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 764
    top_p: float = 0.8
    temperature: float = 0.7

@app.post("/generate")
async def generate_text(request: LLMRequest):
    try:
        print(time.ctime(time.time()))
        return submit_to_local_model(request.prompt, request.max_new_tokens, request.top_p, request.temperature)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "LLM API is running."}

print('Server running!')