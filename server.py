import time
import torch
import transformers
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="LLM API", version="1.0")
llm_model = None
llm_tokenizer = None
llm_pipeline = None
MODEL_PATH = "/projects/frostbyte-1/Combat_Automation/LLM_guided_evolution/llms/llama3.3-70B-Instruct/"

class LLMRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 764
    top_p: float = 0.8
    temperature: float = 0.7

@app.on_event("startup")
async def load_model_on_startup():
    """
    Loads the LLM model and tokenizer when the FastAPI application starts.
    This prevents blocking the first request for model initialization.
    """
    global llm_model, llm_tokenizer, llm_pipeline

    print(f"[{time.ctime()}] Starting model initialization during application startup...")
    try:
        print(f"[{time.ctime()}] Loading model from: {MODEL_PATH}")
        # device_map="auto" will attempt to distribute the model across available GPUs,
        # or use CPU if no GPU is found or sufficient VRAM is not available.
        llm_model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for larger models and modern GPUs
            device_map="auto",
            attn_implementation="sdpa", # Scaled Dot Product Attention for efficiency
        ).eval() # Set model to evaluation mode

        print(f"[{time.ctime()}] Model loaded successfully. Device map: {llm_model.hf_device_map}")

        print(f"[{time.ctime()}] Loading tokenizer from: {MODEL_PATH}")
        llm_tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)
        print(f"[{time.ctime()}] Tokenizer loaded successfully.")

        print(f"[{time.ctime()}] Creating pipeline...")
        llm_pipeline = transformers.pipeline(
            model=llm_model,
            tokenizer=llm_tokenizer,
            return_full_text=False, # Only return the generated part of the text
            task="text-generation",
            temperature=0.1,    # Lower temperature for more deterministic output
            top_p=0.15,         # Lower top_p for more focused output
            top_k=0,            # top_k=0 means no top_k sampling is applied, which is fine if top_p is used
            max_new_tokens=1000,# Max number of tokens to generate
            repetition_penalty=1.1, # Penalty for repeating tokens
            do_sample=True,     # Enable sampling
        )
        print(f"[{time.ctime()}] Pipeline created successfully. LLM API is ready!")

    except Exception as e:
        print(f"[{time.ctime()}] FATAL ERROR: Failed to load model or tokenizer during startup. "
              f"Please check MODEL_PATH and resource availability (RAM/VRAM). Error: {e}")


def submit_to_local_model(prompt_text, max_new_tokens, top_p, temperature):
    if llm_pipeline is None:
        raise RuntimeError("LLM model and pipeline not loaded. Server might not have started correctly.")

    start_time = time.time()
    
    # Pass generation parameters directly to the pipeline call
    result = llm_pipeline(
        prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        # Ensure other pipeline parameters are consistent if changed in initialization
        repetition_penalty=1.1,
        do_sample=True
    )

    # The result from the pipeline is typically a list of dictionaries
    # with 'generated_text' key. We take the first result.
    output_txt = result[0]["generated_text"]
    response_time = round(time.time() - start_time, 2)
    
    print(f"[{time.ctime()}] Generated text in {response_time} seconds.")
    return {"generated_text": output_txt, "response_time_sec": response_time}

@app.post("/generate")
async def generate_text(request: LLMRequest):
    """
    API endpoint to generate text using the LLM.
    Expects a JSON body with 'prompt' and optional 'max_new_tokens', 'top_p', 'temperature'.
    """
    try:
        print(f"[{time.ctime()}] Received request for prompt: '{request.prompt[:50]}...'")
        response = submit_to_local_model(
            request.prompt,
            request.max_new_tokens,
            request.top_p,
            request.temperature
        )
        return response
    except RuntimeError as e:
        # Catch our specific RuntimeError if pipeline isn't loaded
        print(f"[{time.ctime()}] Error: {e}")
        raise HTTPException(status_code=503, detail=str(e) + " Model not ready.")
    except Exception as e:
        print(f"[{time.ctime()}] An error occurred during text generation: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

@app.get("/")
async def root():
    """
    Root endpoint to check if the LLM API is running.
    """
    status = "running" if llm_pipeline is not None else "loading model..."
    return {"message": f"LLM API is {status}."}

print(f"[{time.ctime()}] FastAPI server setup complete. Waiting for startup event to load model.")

