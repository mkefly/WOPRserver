
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

# Public constants shared by runtimes
STREAM_DONE = ""
INPUT_CANDIDATES: tuple[str, ...] = ("text", "prompt", "input", "messages")


def extract_text(res: Any) -> str | None:
    """
    Robustly pull a text chunk out of various shapes.
    Returns None when there's nothing new to emit (e.g., a STREAM_DONE sentinel).
    """
    if res is None:
        return None

    if isinstance(res, (bytes, bytearray)):
        return bytes(res).decode("utf-8", errors="replace")

    if isinstance(res, str):
        s = res.strip()
        return None if s == STREAM_DONE else res

    if isinstance(res, dict):
        # direct text-like keys
        for key in ("content", "text", "output_text", "output", "answer"):
            if key in res:
                val = res[key]
                return val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)

        # OpenAI-ish streaming delta
        choices = res.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") or {}
            for key in ("content", "text"):
                if key in delta:
                    return str(delta[key])
        return None

    try:
        return json.dumps(res, ensure_ascii=False)
    except Exception:
        return str(res)


def fallback_inputs_to_native(req) -> Any:
    """
    Heuristics to pull a single native value out of an MLServer V2 request:
      1) Prefer BYTES/str from 'text'/'prompt'/'input'/'messages'
      2) If a single tensor is present, use its first value
      3) Else return a name->list mapping
    """
    if not getattr(req, "inputs", None):
        return ""

    by_name = {inp.name: inp for inp in req.inputs}

    # (1) Well-known single-field cases
    for name in INPUT_CANDIDATES:
        inp = by_name.get(name)
        if not inp or inp.data is None:
            continue
        vals = list(inp.data)
        if not vals:
            return ""
        v0 = vals[0]
        if inp.datatype == "BYTES":
            return bytes(v0).decode("utf-8", errors="replace") if isinstance(v0, (bytes, bytearray)) else str(v0)
        return v0

    # (2) Single input tensor
    if len(req.inputs) == 1:
        inp = req.inputs[0]
        vals = list(inp.data) if inp.data is not None else []
        if not vals:
            return ""
        v0 = vals[0]
        if inp.datatype == "BYTES":
            return bytes(v0).decode("utf-8", errors="replace") if isinstance(v0, (bytes, bytearray)) else str(v0)
        return v0 if len(vals) == 1 else vals

    # (3) Multi-input mapping
    out: dict[str, Any] = {}
    for inp in req.inputs:
        vals = list(inp.data) if inp.data is not None else []
        if inp.datatype == "BYTES":
            out[inp.name] = [
                (bytes(v).decode("utf-8", errors="replace") if isinstance(v, (bytes, bytearray)) else str(v))
                for v in vals
            ]
        else:
            out[inp.name] = vals
    return out


def extract_params(req) -> dict[str, Any]:
    """
    Best-effort to read request parameters:
      - prefer req.parameters.model_extra if present
      - else try dict()/model_dump()/__dict__
    """
    p = getattr(req, "parameters", None)

    model_extra = getattr(p, "model_extra", None)
    if isinstance(model_extra, dict):
        return dict(model_extra)

    if isinstance(p, dict):
        return dict(p)

    for attr in ("dict", "model_dump"):
        fn = getattr(p, attr, None)
        if callable(fn):
            try:
                return dict(fn())
            except Exception:
                pass
    return dict(getattr(p, "__dict__", {}) or {})


async def unwrap_first_request(payloads: AsyncIterator | object):
    """Return the first request from an async iterator, or the object itself (or None if empty)."""
    if hasattr(payloads, "__aiter__"):
        async for req in payloads:  # type: ignore[attr-defined]
            return req
        return None
    return payloads  # type: ignore[return-value]
