#!/usr/bin/env python3
import json
import os
from collections.abc import Iterator

import httpx

# ---------- Config (overridable via env) ----------
HOST = os.getenv("MLSERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("MLSERVER_HTTP_PORT", "8080"))
# Default to the LangChain-served model name
MODEL = os.getenv("MODEL_NAME", "langchain")
PROMPT = os.getenv("PROMPT", "Once upon a time")
# Optional override if your LangChain runtime uses a specific input tensor name
FALLBACK_INPUT_NAME = os.getenv("MODEL_INPUT_NAME", "text")

STREAM_URL = f"http://{HOST}:{PORT}/v2/models/{MODEL}/infer_stream"
META_URL = f"http://{HOST}:{PORT}/v2/models/{MODEL}"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
}


# ---------- Helpers ----------
def fetch_model_input_signature() -> tuple[str, str]:
    """
    Ask MLServer for model metadata and pick the first suitable input head.
    Returns (input_name, datatype). Falls back to (FALLBACK_INPUT_NAME, 'BYTES').
    """
    try:
        r = httpx.get(META_URL, timeout=5.0)
        if r.status_code == 200:
            meta = r.json()
            inputs = (meta or {}).get("inputs") or []
            if inputs:
                # Prefer a BYTES input if present (typical for text prompts)
                for item in inputs:
                    dt = (item.get("datatype") or "").upper()
                    if dt == "BYTES":
                        return item.get("name") or FALLBACK_INPUT_NAME, "BYTES"
                # Otherwise just take the first declared input
                first = inputs[0]
                return first.get("name") or FALLBACK_INPUT_NAME, (first.get("datatype") or "BYTES").upper()
    except Exception:
        pass
    return FALLBACK_INPUT_NAME, "BYTES"


def build_payload(input_name: str, datatype: str) -> dict:
    """
    Build a V2 request for /infer_stream. For text models we send BYTES + content_type='str'.
    """
    if datatype == "BYTES":
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
    else:
        # Generic fallback: still try BYTES->str (many runtimes accept it via codecs)
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


def iter_sse(resp: httpx.Response) -> Iterator[str]:
    """
    Minimal SSE line parser. Yields each full 'data:' JSON blob as a string.
    """
    buf = []
    for raw in resp.iter_lines():
        line = raw.decode() if isinstance(raw, (bytes, bytearray)) else (raw or "")
        if not line:
            # empty line => dispatch accumulated event
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


def extract_text_chunk(msg: dict) -> str | None:
    """
    Try to extract a printable text chunk from a variety of streaming payload shapes.
    Primary (MLServer V2): msg['outputs'][0]['data'][0]
    Fallbacks: OpenAI-ish shapes ('choices', 'delta', etc.).
    """
    # MLServer V2 shape
    try:
        outs = msg.get("outputs") or []
        if outs:
            data = outs[0].get("data") or []
            if data:
                # most runtimes emit string bytes already decoded in JSON
                chunk = data[0]
                if isinstance(chunk, (str, bytes)):
                    return chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk
    except Exception:
        pass

    # OpenAI-like shapes (best effort)
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

    with httpx.stream("POST", STREAM_URL, headers=HEADERS, json=payload, timeout=None) as r:
        r.raise_for_status()
        for data in iter_sse(r):
            try:
                msg = json.loads(data)
            except Exception:
                # If the server sends raw text lines (unlikely for MLServer), just print them
                print(data, end="", flush=True)
                continue

            out = extract_text_chunk(msg)
            if out is not None:
                print(out, end="", flush=True)
