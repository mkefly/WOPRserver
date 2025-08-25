from __future__ import annotations

import os
import random
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional

from common.grpc_client import GrpcClient
from common.helpers import read_test_data
from locust import User, between, events, task

# ─────────────────────────────────────────────────────────────────────────────
# Config (override via env)
# ─────────────────────────────────────────────────────────────────────────────
DATASET_KEY = os.getenv("LLM_DATASET_KEY", "llm")
WAIT_MIN_S  = float(os.getenv("WAIT_MIN_S", "0.10"))
WAIT_MAX_S  = float(os.getenv("WAIT_MAX_S", "0.10"))

STREAM_NAME = os.getenv("STREAM_NAME", "ModelStreamInfer llm")

# Metric names
MSG_EVENT   = os.getenv("MSG_EVENT", "grpc_stream_msgs")
TTFT_EVENT  = os.getenv("TTFT_EVENT", "grpc_stream_first_msg_ms")
IAT_EVENT   = os.getenv("IAT_EVENT", "grpc_stream_intermsg_ms")
BYTES_EVENT = os.getenv("BYTES_EVENT", "grpc_stream_bytes")

# Optional: thread pool size per user (1 is enough for one concurrent stream)
POOL_WORKERS = int(os.getenv("GRPC_THREADPOOL_WORKERS", "1"))

# Preload dataset once per worker
TEST_DATA = {DATASET_KEY: read_test_data(DATASET_KEY)}


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
def pick_payload(d: dict) -> Any:
    """Pick a single payload. If 'grpc_many' is present, choose randomly; else use 'grpc'."""
    many = d.get("grpc_many")
    if many:
        return random.choice(many)
    return d["grpc"]


def approx_msg_size(msg: Any) -> int:
    """
    Estimate byte size of a protobuf message.
    Prefer ByteSize()/SerializeToString(); fallback to len(str(msg).encode()).
    """
    try:
        if hasattr(msg, "ByteSize"):
            return int(msg.ByteSize())  # type: ignore[attr-defined]
        if hasattr(msg, "SerializeToString"):
            return len(msg.SerializeToString())  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        return len(str(msg).encode("utf-8"))
    except Exception:
        return 0


def iterate_stream_blocking(
    iterator: Iterable,
    on_first: Callable[[Any, float], None],
    on_msg: Callable[[Any, float], None],
    on_done: Callable[[int, int, float], None],
    on_error: Callable[[BaseException, float, int, int], None],
) -> None:
    """
    Consume a blocking gRPC stream in a native thread and invoke callbacks.
    """
    start = time.time()
    last_ts: Optional[float] = None
    msg_count = 0
    total_bytes = 0

    try:
        for msg in iterator:
            now = time.time()
            msg_count += 1
            size = approx_msg_size(msg)
            total_bytes += size

            if msg_count == 1:
                on_first(msg, now - start)

            # per message
            on_msg(msg, 0.0)  # response_time unused for count metric

            # inter-arrival (emit as its own metric)
            if last_ts is not None:
                iat_ms = (now - last_ts) * 1000.0
                events.request.fire(
                    request_type="metric",
                    name=os.getenv("IAT_EVENT", IAT_EVENT),
                    response_time=iat_ms,
                    response_length=size,
                    exception=None,
                )
            last_ts = now

            # bytes metric (sample)
            events.request.fire(
                request_type="metric",
                name=os.getenv("BYTES_EVENT", BYTES_EVENT),
                response_time=size,
                response_length=size,
                exception=None,
            )

        dur_ms = (time.time() - start) * 1000.0
        on_done(msg_count, total_bytes, dur_ms)

    except BaseException as e:
        dur_ms = (time.time() - start) * 1000.0
        on_error(e, dur_ms, msg_count, total_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Locust user
# ─────────────────────────────────────────────────────────────────────────────
class GrpcStreamLLMUser(User):
    wait_time = between(WAIT_MIN_S, WAIT_MAX_S)

    def on_start(self) -> None:
        # Per-user gRPC client and tiny thread pool to avoid gevent blocking
        self.grpc = GrpcClient()
        self.pool = ThreadPoolExecutor(max_workers=POOL_WORKERS)

        # Optional readiness probe (doesn't affect stats)
        try:
            _ = self.grpc.server_ready()
        except Exception:
            # If server isn't ready, tasks will fail and be counted below
            pass

    def on_stop(self) -> None:
        try:
            self.pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            self.grpc.close()
        except Exception:
            pass

    @task
    def stream_llm(self) -> None:
        data = TEST_DATA[DATASET_KEY]
        payload = pick_payload(data)

        # Callbacks emitting Locust metrics (safe from threads)
        def on_first(msg: Any, t_first_s: float) -> None:
            size = approx_msg_size(msg)
            events.request.fire(
                request_type="metric",
                name=os.getenv("TTFT_EVENT", TTFT_EVENT),
                response_time=t_first_s * 1000.0,
                response_length=size,
                exception=None,
            )

        def on_msg(msg: Any, _: float) -> None:
            size = approx_msg_size(msg)
            events.request.fire(
                request_type="metric",
                name=os.getenv("MSG_EVENT", MSG_EVENT),
                response_time=1.0,  # each msg increments count
                response_length=size,
                exception=None,
            )

        def on_done(msg_count: int, total_bytes: int, dur_ms: float) -> None:
            events.request.fire(
                request_type="gRPC",
                name=os.getenv("STREAM_NAME", STREAM_NAME),
                response_time=dur_ms,
                response_length=total_bytes,
                exception=None,
            )

        def on_error(exc: BaseException, dur_ms: float, msg_count: int, total_bytes: int) -> None:
            events.request.fire(
                request_type="gRPC",
                name=os.getenv("STREAM_NAME", STREAM_NAME),
                response_time=dur_ms,
                response_length=total_bytes,
                exception=exc,
            )

        # Submit to native thread so gevent isn't blocked
        def run_blocking():
            iterator = self.grpc.stream_infer(payload)
            iterate_stream_blocking(iterator, on_first, on_msg, on_done, on_error)

        self.pool.submit(run_blocking)
