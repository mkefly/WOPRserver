#!/usr/bin/env python3
import json
import os

import httpx

HOST = os.getenv("MLSERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("MLSERVER_HTTP_PORT", "8080"))
MODEL = os.getenv("MODEL_NAME", "llm")  # change if needed (e.g. "llm" or "summodel")
PROMPT = os.getenv("PROMPT", "Once upon a time")

URL = f"http://{HOST}:{PORT}/v2/models/{MODEL}/infer_stream"

payload = {
    "id": "req-1",
    "inputs": [
        {
            "name": "text",           # Toy LLM expects "text"
            "datatype": "BYTES",
            "shape": [1],
            "data": [PROMPT],
            "parameters": {"content_type": "str"},
        }
    ],
}

headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

def iter_sse(resp):
    buf = []
    for line in resp.iter_lines():
        if not line:
            if buf:
                data = "\n".join(buf).strip()
                buf.clear()
                if data == "[DONE]":
                    return
                yield data
            continue
        if line.startswith((":", "event:")):
            continue
        if line.startswith("data:"):
            buf.append(line[5:].lstrip())
        elif line[:1] in "{[":
            buf.append(line)

if __name__ == "__main__":
    with httpx.stream("POST", URL, headers=headers, json=payload, timeout=None) as r:
        r.raise_for_status()
        for data in iter_sse(r):
            msg = json.loads(data)
            out = msg["outputs"][0]["data"][0]
            print(out, end="", flush=True)  # server already sends chunked text
