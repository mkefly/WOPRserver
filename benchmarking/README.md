Here’s an updated `README.md` that reflects the switch from `Makefile` targets to the new **Click-based Python CLI** (`cli.py`), while keeping all the original context:

````markdown
# Benchmarking

This folder contains a set of tools to benchmark the gRPC and REST APIs of
`mlserver`.  
These load tests are run locally against a local server.

## Current results

|      | Requests/sec | Average (ms) | Slowest (ms) | Fastest (ms) |
| ---- | ------------ | ------------ | ------------ | ------------ |
| gRPC | 2259         | 128.46       | 703.84       | 3.09         |
| REST | 1300         | 226.93       | 304.98       | 2.06         |

## Setup

The benchmark scenarios in this folder leverage [`k6`](https://k6.io/).  
To install `k6`, please check their [installation docs page](https://k6.io/docs/getting-started/installation/).

You also need **Click** (already in this repo).  

```bash
pip install click
````

## Data

You can find pre-generated requests under the [`/data`](./data) folder.

### Generate

You can re-generate the test requests by using the
[`generator.py`](./generator.py) script directly, or via the CLI:

```bash
# direct
python generator.py

# via cli.py
python cli.py generate
```

## Usage

We provide a Python CLI (`cli.py`) that replaces the old `make` targets.
It offers consistent commands for server lifecycle, data generation, and all benchmarks.

### Server lifecycle

```bash
# start MLServer with local test models
python cli.py start

# stop it
python cli.py stop

# restart
python cli.py restart
```

### Inference benchmarks

Run unary inference benchmarks (REST or gRPC) against a **Scikit-Learn model trained on the Iris dataset**:

```bash
python cli.py bench-rest
python cli.py bench-grpc
```

Each benchmark lasts 60s. The model is configured to use **adaptive batching**.

### Streaming benchmarks

We also provide benchmarks for both **server-sent events (REST)** and **gRPC streaming**
against two toy models (`summodel` and `llm`):

```bash
# REST streaming
python cli.py stream-rest

# gRPC streaming
python cli.py stream-grpc

# both
python cli.py stream-all
```

### Multi-model scenario

To simulate loading and querying multiple models:

```bash
python cli.py mms
```

---

### Environment overrides

All commands accept overrides via flags or environment variables:

```bash
# override via env
MLSERVER_HOST=localhost MLSERVER_HTTP_PORT=9090 python cli.py bench-rest

# override via flags
python cli.py --mlserver-host localhost --http-port 9090 --grpc-port 9091 bench-grpc
```

# Benchmarking — notes

### SSE with k6
k6 needs an extension to handle SSE:
```bash
go install go.k6.io/xk6/cmd/xk6@latest
xk6 build v0.51.0 --with github.com/grafana/xk6-sse@latest -o k6
