import time
import torch
import transformers
import asyncio
import threading
import zmq
import zmq.asyncio
import json
from src.cfg.constants import *

# Path To Local Large Language Model
MODEL_PATH = "LLM_MODEL_PATH"
BATCH_SIZE = 8
BATCH_WAIT_TIME = 2  # seconds

class LLMModel:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print(f"Loading model at {MODEL_PATH} for the first time")
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa" # faster inference
        ).eval()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.pipeline = transformers.pipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=False,
            task="text-generation",
            temperature=0.1,
            top_p=0.15,
            top_k=0,
            max_new_tokens=1648,
            repetition_penalty=1.1,
            do_sample=True,
            batch_size=BATCH_SIZE
        )

        self.request_queue = asyncio.Queue()
        self.batch_task = None
        self.batch_lock = asyncio.Lock()
        self.is_processing = False

    async def start_batch_processor(self):
        async with self.batch_lock:
            if not self.is_processing:
                self.is_processing = True
                self.batch_task = asyncio.create_task(self._batch_processor())

    async def _batch_processor(self):
        try:
            while True:
                batch = []
                futures = []
                try:
                    # check queue for requests
                    request, future = await self.request_queue.get()
                    batch.append(request)
                    futures.append(future)

                    batch_start_time = time.time()
                    while len(batch) < BATCH_SIZE and (time.time() - batch_start_time) < BATCH_WAIT_TIME:
                        try:
                            req, fut = await asyncio.wait_for(
                                self.request_queue.get(),
                                timeout=max(0, BATCH_WAIT_TIME - (time.time() - batch_start_time))
                            )
                            batch.append(req)
                            futures.append(fut)
                        except asyncio.TimeoutError:
                            break

                    batch_size = len(batch)
                    print(f"Processing batch of {batch_size} requests")

                    prompts = [req["prompt"] for req in batch]
                    max_new_tokens = max(req["max_new_tokens"] for req in batch)
                    temperature = batch[0]["temperature"]
                    top_p = batch[0]["top_p"]

                    start_time = time.time()

                    results = self.pipeline(
                        prompts,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p
                    )

                    response_time = round(time.time() - start_time, 2)

                    for result, future in zip(results, futures):
                        output_txt = result[0].get("generated_text", str(result))

                        future.set_result({
                            "generated_text": output_txt,
                            "response_time_sec": response_time,
                            "batch_size": len(batch),
                        })

                        # we're done with the task
                        self.request_queue.task_done()
                        
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)

                    # Mark all tasks as done
                    for _ in range(len(futures)):
                        self.request_queue.task_done()

                if self.request_queue.empty():
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            print("Batch processor cancelled")
        finally:
            async with self.batch_lock:
                self.is_processing = False

    async def generate(self, request_dict):
        future = asyncio.Future()
        await self.request_queue.put((request_dict, future))
        await self.start_batch_processor()
        return await future

async def zmq_server():
    ctx = zmq.asyncio.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind("tcp://*:5555") 
    model = LLMModel()

    print("ZMQ LLM Server running on port 5555...")

    while True:
        msg = await socket.recv()
        try:
            request = json.loads(msg.decode("utf-8"))
            start_time = time.time()
            result = await model.generate(request)
            result["request_latency_sec"] = round(time.time() - start_time, 2)
            await socket.send_string(json.dumps(result))
        except Exception as e:
            await socket.send_string(json.dumps({"error": str(e)}))

# if __name__ == "__main__":
#     asyncio.run(zmq_server())