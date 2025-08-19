"""
CLI to generate test benchmark data (REST + gRPC) with stream-ready payloads
for both:
  - summodel
  - llm (ToyLLM)
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import numpy as np
from google.protobuf import json_format
from google.protobuf.internal.encoder import _VarintBytes  # type: ignore
from mlserver import types
from mlserver.grpc import converters

ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(ROOT, "data")

SUMMODEL_NAME = "summodel"
SUMMODEL_VERSION = "v1.2.3"
SUM_DATA_PATH = os.path.join(DATA_ROOT, "summodel")

LLM_MODEL_NAME = "llm"
LLM_MODEL_VERSION = "v0"
LLM_DATA_PATH = os.path.join(DATA_ROOT, "llm")


# ------------- Generators -------------

def generate_sum_requests(rng: np.random.Generator) -> list[types.InferenceRequest]:
    """
    Produce several numeric vectors of increasing size.
    We keep the first one as the canonical single payload (compat), but we will also
    persist a *many* file with all generated requests.
    """
    contents_lens = np.power(2, np.arange(10, 16)).astype(int)  # 1k .. 64k
    max_value = 9999.0

    requests: list[types.InferenceRequest] = []
    for contents_len in contents_lens:
        inputs = max_value * rng.random(contents_len)
        requests.append(
            types.InferenceRequest(
                id=str(uuid.uuid4()),
                inputs=[
                    types.RequestInput(
                        name="input-0",
                        shape=[int(contents_len)],
                        datatype="FP32",
                        data=types.TensorData.model_validate(inputs.tolist()),
                    )
                ]
            )
        )
    return requests


def generate_llm_requests(num_per_prompt: int = 4) -> list[types.InferenceRequest]:
    """
    Create multiple BYTES prompts, with unique IDs per request so ToyLLM's seed varies.
    For REST we keep strings; for gRPC we convert to bytes during serialization.
    """
    prompts = [
        "Summarize streaming inference versus unary inference in one paragraph.",
        "Explain chunk coalescing and backpressure in a streaming LLM server.",
        "Write a short note about SSE vs gRPC server streaming.",
    ]

    requests: list[types.InferenceRequest] = []
    for p in prompts:
        for _ in range(max(1, int(num_per_prompt))):
            requests.append(
                types.InferenceRequest(
                    id=str(uuid.uuid4()),
                    inputs=[
                        types.RequestInput(
                            name="text",
                            shape=[1],
                            datatype="BYTES",
                            data=[p],  # keep as str here (good for REST)
                        )
                    ],
                )
            )
    return requests


# ------------- Serializers -------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _to_list(data: Any) -> list:
    """
    Turn TensorData or list-like into a plain list without caring about
    Pydantic version internals.
    """
    try:
        return list(data)
    except TypeError:
        root = getattr(data, "root", None)
        if root is not None:
            return list(root)
        root = getattr(data, "__root__", None)
        if root is not None:
            return list(root)
        raise


def _encode_bytes_inplace(req: types.InferenceRequest) -> types.InferenceRequest:
    """
    Return a shallow copy of the request where any BYTES input has its data
    elements encoded to bytes (utf-8). This is ONLY for gRPC serialization.
    """
    new_inputs: list[types.RequestInput] = []
    for inp in req.inputs or []:
        if inp.datatype == "BYTES":
            values = _to_list(inp.data)
            bvalues = [v.encode("utf-8") if isinstance(v, str) else v for v in values]
            new_inputs.append(
                types.RequestInput(
                    name=inp.name,
                    shape=list(inp.shape) if inp.shape is not None else None,
                    datatype=inp.datatype,
                    data=types.TensorData.model_validate(bvalues),
                    parameters=inp.parameters,
                )
            )
        else:
            new_inputs.append(inp)

    return types.InferenceRequest(
        id=req.id,
        inputs=new_inputs,
        outputs=req.outputs,
        parameters=req.parameters,
    )


def _save_grpc_single(req: types.InferenceRequest, model_name: str, model_version: str, out_dir: str):
    """Write single gRPC request to .pb and .json (compat)."""
    infer_req = converters.ModelInferRequestConverter.from_types(
        req, model_name=model_name, model_version=model_version
    )

    # Binary (length-prefixed)
    requests_file_path = os.path.join(out_dir, "grpc-requests.pb")
    with open(requests_file_path, "wb") as requests_file:
        size = infer_req.ByteSize()
        size_varint = _VarintBytes(size)
        requests_file.write(size_varint)
        requests_file.write(infer_req.SerializeToString())

    # JSON mirror of the same protobuf
    requests_json_path = os.path.join(out_dir, "grpc-requests.json")
    with open(requests_json_path, "w") as json_file:
        as_dict = json_format.MessageToDict(infer_req)
        json.dump(as_dict, json_file)


def save_grpc_requests_many(
    requests: list[types.InferenceRequest],
    model_name: str,
    model_version: str,
    out_dir: str,
):
    """
    Saves:
      - grpc-requests.pb / grpc-requests.json    (single; compat with existing code)
      - grpc-requests-many.json                  (NEW: array of protobuf-shaped JSON requests)
    """
    _ensure_dir(out_dir)

    # SINGLE (compat): use the first request
    req0 = _encode_bytes_inplace(requests[0])
    _save_grpc_single(req0, model_name, model_version, out_dir)

    # MANY: write an array of protobuf-JSON requests
    many_json_path = os.path.join(out_dir, "grpc-requests-many.json")
    as_dicts = []
    for r in requests:
        r_bytes = _encode_bytes_inplace(r)
        mr = converters.ModelInferRequestConverter.from_types(
            r_bytes, model_name=model_name, model_version=model_version
        )
        as_dicts.append(json_format.MessageToDict(mr))

    with open(many_json_path, "w") as f:
        json.dump(as_dicts, f)


def save_rest_requests_many(
    requests: list[types.InferenceRequest],
    out_dir: str,
):
    """
    Saves:
      - rest-requests.json         (single; compat with existing code)
      - rest-requests-many.json    (NEW: array of InferenceRequest dicts)
    """
    _ensure_dir(out_dir)

    # SINGLE (compat)
    req_dict = requests[0].model_dump()
    requests_file_path = os.path.join(out_dir, "rest-requests.json")
    with open(requests_file_path, "w") as f:
        json.dump(req_dict, f)

    # MANY
    many_path = os.path.join(out_dir, "rest-requests-many.json")
    with open(many_path, "w") as f:
        json.dump([r.model_dump() for r in requests], f)


# ------------- Main -------------

def main():
    _ensure_dir(SUM_DATA_PATH)
    _ensure_dir(LLM_DATA_PATH)

    rng = np.random.default_rng(seed=1337)

    # summodel
    sum_reqs = generate_sum_requests(rng)
    save_grpc_requests_many(sum_reqs, SUMMODEL_NAME, SUMMODEL_VERSION, SUM_DATA_PATH)
    save_rest_requests_many(sum_reqs, SUM_DATA_PATH)

    # llm (more variety)
    llm_reqs = generate_llm_requests(num_per_prompt=4)
    save_grpc_requests_many(llm_reqs, LLM_MODEL_NAME, LLM_MODEL_VERSION, LLM_DATA_PATH)
    save_rest_requests_many(llm_reqs, LLM_DATA_PATH)


if __name__ == "__main__":
    main()
