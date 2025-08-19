#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HOST   = os.getenv("MLSERVER_HOST", "127.0.0.1")
PORT   = os.getenv("MLSERVER_GRPC_PORT", "8081")
MODEL  = os.getenv("MODEL_NAME", "llm")
PROMPT = os.getenv("PROMPT", "Once upon a time")

HERE = Path(__file__).resolve().parent
# proto lives at benchmarking/proto/dataplane.proto (one back from this file)
PROTO_FILE = (HERE / "../proto/dataplane.proto").resolve()
PROTO_INC  = PROTO_FILE.parent.parent  # repo root that contains "proto/"

def gen_and_import_stubs(tmpdir: Path):
    """Generate gRPC stubs into tmpdir and import them by proto basename."""
    try:
        import grpc_tools.protoc  # noqa: F401
    except Exception:
        sys.stderr.write(
            "Missing grpcio-tools in this environment.\n"
            "Install it (dev dep): poetry add grpcio grpcio-tools -G dev\n"
        )
        sys.exit(2)

    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"-I{PROTO_INC}",
        f"--python_out={tmpdir}",
        f"--grpc_python_out={tmpdir}",
        str(PROTO_FILE),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write("protoc failed:\n" + proc.stdout + proc.stderr + "\n")
        sys.exit(proc.returncode)

    # Make the temp dir importable
    if str(tmpdir) not in sys.path:
        sys.path.insert(0, str(tmpdir))

    import importlib

    base = PROTO_FILE.stem  # e.g., "dataplane"
    mod_pb2_name = f"{base}_pb2"
    mod_grpc_name = f"{base}_pb2_grpc"

    try:
        pb = importlib.import_module(mod_pb2_name)
        pb_grpc = importlib.import_module(mod_grpc_name)
    except ModuleNotFoundError as e:
        # Help debug by listing what was generated
        sys.stderr.write(
            f"Could not import {mod_pb2_name}/{mod_grpc_name} from {tmpdir}\n"
            f"Contents: {sorted(p.name for p in tmpdir.glob('*.py'))}\n"
        )
        raise e
    return pb, pb_grpc

def main():
    tmpdir = Path(tempfile.mkdtemp(prefix="grpc_stubs_"))
    try:
        pb, pb_grpc = gen_and_import_stubs(tmpdir)

        import grpc  # import after ensuring grpcio is present

        # Preflight: ensure gRPC server is reachable and speaks HTTP/2
        channel = grpc.insecure_channel(f"{HOST}:{PORT}")
        grpc.channel_ready_future(channel).result(timeout=5)

        stub = pb_grpc.GRPCInferenceServiceStub(channel)

        # One streaming request with BYTES input named "text"
        req = pb.ModelInferRequest(
            model_name=MODEL,
            id="req-1",
            inputs=[
                pb.ModelInferRequest.InferInputTensor(
                    name="text",
                    datatype="BYTES",
                    shape=[1],
                    contents=pb.InferTensorContents(
                        bytes_contents=[PROMPT.encode("utf-8")]
                    ),
                )
            ],
        )

        # Server-streaming: send one request, read many responses
        for resp in stub.ModelStreamInfer(iter([req])):
            try:
                out = resp.outputs[0]
                if getattr(out, "data", None):
                    sys.stdout.write(str(out.data[0]))
                elif out.contents.bytes_contents:
                    sys.stdout.write(out.contents.bytes_contents[0].decode("utf-8", "replace"))
                else:
                    sys.stdout.write(str(resp) + "\n")
            except Exception:
                sys.stdout.write(str(resp) + "\n")
            sys.stdout.flush()
        print()
    finally:
        # Always clean up the temp dir
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

if __name__ == "__main__":
    main()
