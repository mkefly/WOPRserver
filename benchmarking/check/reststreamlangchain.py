#!/usr/bin/env python3
import json
import os

import httpx

# ---------- Config (overridable via env) ----------
HOST = os.getenv("MLSERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("MLSERVER_HTTP_PORT", "8080"))
MODEL = os.getenv("MODEL_NAME", "langchain")
PROMPT = os.getenv("PROMPT", "Once upon a time")
FALLBACK_INPUT_NAME = os.getenv("MODEL_INPUT_NAME", "text")

STREAM_URL = f"http://{HOST}:{PORT}/v2/models/{MODEL}/infer_stream"
META_URL = f"http://{HOST}:{PORT}/v2/models/{MODEL}"

HEADERS = {
    "Content-Type": "application/json",
}


# ---------- Helpers ----------
def fetch_model_input_signature() -> tuple[str, str]:
    try:
        r = httpx.get(META_URL, timeout=5.0)
        if r.status_code == 200:
            meta = r.json()
            inputs = (meta or {}).get("inputs") or []
            if inputs:
                for item in inputs:
                    dt = (item.get("datatype") or "").upper()
                    if dt == "BYTES":
                        return item.get("name") or FALLBACK_INPUT_NAME, "BYTES"
                first = inputs[0]
                return first.get("name") or FALLBACK_INPUT_NAME, (first.get("datatype") or "BYTES").upper()
    except Exception:
        pass
    return FALLBACK_INPUT_NAME, "BYTES"


def build_payload(input_name: str, datatype: str) -> dict:
    return {
        "id": "req-1",
        "inputs": [
            {
                "name": input_name,
                "datatype": "BYTES",
                "shape": [1],
                "data": [PROMPT],
                "parameters": {"content_type": "str"},
            }
        ],
    }


def extract_text_chunk(msg: dict) -> str | None:
    try:
        outs = msg.get("outputs") or []
        if outs:
            data = outs[0].get("data") or []
            if data:
                chunk = data[0]
                if isinstance(chunk, (str, bytes)):
                    return chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
    except Exception:
        pass

    try:
        choices = msg.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            if "content" in delta:
                return str(delta["content"])
            if "text" in delta:
                return str(delta["text"])
        if "text" in msg:
            return str(msg["text"])
        if "content" in msg:
            return str(msg["content"])
    except Exception:
        pass
    return None


# ---------- Main ----------
if __name__ == "__main__":
    in_name, dtype = fetch_model_input_signature()
    payload = build_payload(in_name, dtype)

    # Instead of SSE, stream raw response chunks
    with httpx.stream("POST", STREAM_URL, headers=HEADERS, json=payload, timeout=None) as r:
        r.raise_for_status()
        for chunk in r.iter_bytes():
            if not chunk:
                continue
            try:
                msg = json.loads(chunk.decode("utf-8"))
                out = extract_text_chunk(msg)
                if out is not None:
                    print(out, end="", flush=True)
            except Exception:
                # If it's plain text, just print directly
                print(chunk.decode("utf-8", errors="ignore"), end="", flush=True)
