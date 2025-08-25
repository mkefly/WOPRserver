import json
from pathlib import Path


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def read_test_data(name: str):
    base = Path(__file__).resolve().parent.parent / "data" / name
    d = {
        "rest": _read_json(base / "rest-requests.json"),
        "grpc": _read_json(base / "grpc-requests.json"),
        "rest_many": _read_json(base / "rest-requests-many.json"),
        "grpc_many": _read_json(base / "grpc-requests-many.json"),
    }
    if not isinstance(d["rest_many"], list):
        d["rest_many"] = None
    if not isinstance(d["grpc_many"], list):
        d["grpc_many"] = None
    return d
