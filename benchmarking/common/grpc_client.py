# common/grpc_client.py
import os

import grpc
from google.protobuf import json_format

from common import dataplane_pb2 as pb2
from common import dataplane_pb2_grpc as pb2_grpc


class GrpcClient:
    def __init__(self, host=None, port=None, options=None, insecure=True, creds=None):
        host = host or os.environ.get("MLSERVER_HOST", "0.0.0.0")
        port = port or os.environ.get("MLSERVER_GRPC_PORT", "8081")
        self.addr = f"{host}:{port}"
        opts = options or [
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
            ("grpc.max_send_message_length",    128 * 1024 * 1024),
        ]
        if insecure:
            self.channel = grpc.insecure_channel(self.addr, options=opts)
        else:
            self.channel = grpc.secure_channel(self.addr, creds or grpc.local_channel_credentials(), options=opts)
        self.stub = pb2_grpc.GRPCInferenceServiceStub(self.channel)

    def close(self):
        self.channel.close()

    def stream_infer(self, payload: dict):
        """
        payload must match the ModelInferRequest JSON mapping
        (we only send a single request on the stream).
        """
        req = pb2.ModelInferRequest()
        json_format.ParseDict(payload, req, ignore_unknown_fields=True)
        return self.stub.ModelStreamInfer(iter([req]))

    # (optional) quick health helpers
    def server_live(self) -> bool:
        return self.stub.ServerLive(pb2.ServerLiveRequest()).live

    def server_ready(self) -> bool:
        return self.stub.ServerReady(pb2.ServerReadyRequest()).ready
