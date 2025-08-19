from common.grpc_client import GrpcClient
from common.helpers import read_test_data

g = GrpcClient()
payload = read_test_data("llm")["grpc"]  # or pick from "grpc_many"
for resp in g.stream_infer(payload):
    print(resp)
g.close()