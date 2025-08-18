import zmq
import json

# ZMQ Method For Generating a Response from the LLM
def zmq_generate(prompt, max_new_tokens=800, top_p=0.8, temperature=0.7, server_url="tcp://localhost:5555"):

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(server_url)

    request = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "top_p": top_p,
        "temperature": temperature,
    }

    socket.send_string(json.dumps(request))
    reply = socket.recv()

    response = json.loads(reply.decode("utf-8"))
    return response

if __name__ == "__main__":

    resp = zmq_generate("Write a Python function to reverse a string")
    print(resp)