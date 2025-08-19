# Locust MLServer Bench (Python translation of k6 setup)

This is a Python/Locust translation of your k6 benchmarks (REST & gRPC, unary & streaming, plus a Prometheus poller).

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Required environment

Set these before running (matching your k6 setup):

```bash
export MLSERVER_HOST=127.0.0.1
export MLSERVER_HTTP_PORT=8080
# If metrics are served on another port:
export MLSERVER_METRICS_PORT=8080
# For gRPC:
export MLSERVER_GRPC_PORT=8081
# For proto resolution (absolute path is best):
export MLSERVER_PROTO=proto/dataplane.proto
```

You should also have the JSON payloads under `data/` mirroring your k6 repo:
```
data/
  iris/
    rest-requests.json
    grpc-requests.json
  llm/
    rest-requests.json
    grpc-requests.json
    rest-requests-many.json   # optional array
    grpc-requests-many.json   # optional array
  summodel/
    rest-requests.json
    grpc-requests.json
    rest-requests-many.json   # optional
    grpc-requests-many.json   # optional
```

> The gRPC client compiles the proto at runtime into a temp folder using `grpcio-tools`. Make sure `MLSERVER_PROTO` points to a readable file. If unset, it will attempt to find `benchmarking/proto/dataplane.proto` or `proto/dataplane.proto` **relative** to this project.

## Running scenarios

Each scenario is its own file under `scenarios/`. Run them independently to mirror your k6 setup.

### REST streaming (summodel + llm) + Prometheus poller
```bash
locust -f scenarios/streaming_rest.py --headless -u 61 -r 61 -t 60s
# spawns: 40 summodel streams, 20 llm streams, 1 Prom poller
```

### REST unary (iris)
```bash
locust -f scenarios/inference_rest.py --headless -u 300 -r 300 -t 60s
```
(Equivalent thresholds: expect `http_2xx` rate to be high; Locust doesn't enforce thresholds by default.)

### gRPC unary (iris)
```bash
locust -f scenarios/inference_grpc.py --headless -u 300 -r 300 -t 60s
```

### gRPC streaming (summodel + llm) + Prometheus poller
```bash
locust -f scenarios/streaming_grpc.py --headless -u 61 -r 61 -t 60s
# spawns: 40 summodel, 20 llm, 1 poller
```

### Multi-model (MMS) loader
```bash
locust -f scenarios/mms.py --headless -u 2 -r 2 --iterations 10
```

## Notes

- Custom metrics are recorded using Locust's `events.request.fire` with synthetic names (e.g., `metric/worker_proc_cpu_pct`). The "response time" field stores the metric value for aggregation (avg, p95, max).
- SSE uses `sseclient-py` on top of Locust's `requests` session.
- gRPC uses runtime `protoc` compilation of `dataplane.proto`. If compilation fails, the error will explain how to point to the proto path.
- The Prometheus poller parses a few metrics from MLServer's `/metrics`, similar to your k6 regex helpers.
