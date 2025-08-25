"""
AsyncResponses
==============

More info of this module in dispatcher.py
"""

import asyncio
from asyncio import Future
from collections import defaultdict
from typing import Any

from prometheus_client import REGISTRY as PROM_REGISTRY
from prometheus_client import Gauge, Histogram

from ..logging import get_logger
from .errors import WorkerStop
from .messages import (
    Message,
    ModelResponseMessage,
)
from .worker import Worker

QUEUE_METRIC_NAME = "parallel_request_queue_size"
logger = get_logger()


class AsyncResponses:
    """Central registry for in‑flight requests and their async plumbing.

    Responsibilities
    ----------------
    * Maintain a ``Future`` per request id so callers can await a model
      response or an exception.
    * Track which worker pid is processing which message id.
    * Maintain per‑request ``asyncio.Queue`` instances for streaming
      (server‑sent chunks) and expose helpers to enqueue chunk messages.
    * Keep a Prometheus ``Gauge`` in sync with the current number of in‑flight
      requests.

    Lifecycle
    ---------
    1. ``_schedule`` registers a new in‑flight request and returns a ``Future``.
    2. The dispatcher forwards the message to a worker.
    3. When a response arrives from the responses queue, ``resolve`` completes
       the matching ``Future`` (success or exception).
    4. ``_wait`` awaits the ``Future`` and cleans up indexes/metrics.

    Error handling
    --------------
    * If a worker dies unexpectedly, ``cancel`` converts all of its in‑flight
      requests into ``WorkerStop`` exceptions and updates metrics.

    Streaming
    ---------
    * ``create_stream_queue`` sets up an ``asyncio.Queue`` for the given request
      id; ``put_stream_message`` non‑blockingly enqueues chunks and applies a
      drop‑oldest policy on overflow.
    """

    def __init__(self) -> None:
        self._futures: dict[str, Future[ModelResponseMessage]] = {}
        self._workers_map: dict[int, set[str]] = defaultdict(set)
        self._futures_map: dict[str, int] = {}
        self._streams: dict[str, asyncio.Queue] = {}
        self._metrics_cache: dict[str, Histogram] = {}
        # Lazily create or fetch the Gauge for in‑flight tracking.
        self.parallel_request_queue_size = self._get_or_create_metric()
        # Note: this intentionally shadows the previous cache type annotation to
        # match usage patterns in the original code (Gauge vs Histogram).
        self._metrics_cache: dict[str, Gauge] = {}

    def _get_or_create_metric(self) -> Gauge:
        """Return a process‑wide Gauge for in‑flight request count.

        The function is idempotent and resilient to duplicate registration:
        it attempts to create a new ``Gauge`` and, on ``ValueError`` (already
        registered), it looks up the existing collector from the registry.
        """
        if QUEUE_METRIC_NAME in self._metrics_cache:
            return self._metrics_cache[QUEUE_METRIC_NAME]
        try:
            g = Gauge(
                QUEUE_METRIC_NAME,
                "current number of in-flight requests for workers",
                registry=PROM_REGISTRY,
            )
        except ValueError:
            # If already registered, get the existing one from registry internals.
            g = PROM_REGISTRY._names_to_collectors[QUEUE_METRIC_NAME]  # type: ignore[attr-defined]
        self._metrics_cache[QUEUE_METRIC_NAME] = g
        return g

    async def schedule_and_wait(
        self, message: Message, worker: "Worker"
    ) -> ModelResponseMessage:
        """Convenience helper: schedule a message then await its response.

        Equivalent to ``schedule_only`` followed by awaiting ``_wait``.
        """
        message_id = message.id
        self._schedule(message, worker)
        return await self._wait(message_id)

    def schedule_only(self, message: Message, worker: "Worker") -> Future:
        """Register a new in‑flight request and return its ``Future``.

        The caller is responsible for actually sending the message to the
        worker; this function only prepares the indexes and metrics.
        """
        return self._schedule(message, worker)

    def _schedule(self, message: Message, worker: "Worker") -> Future:
        """Internal: create the ``Future`` and index the message to the worker."""
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        message_id = message.id
        self._futures[message_id] = future
        self._track_message(message, worker)
        in_flight_count = len(self._futures)
        self.parallel_request_queue_size.set(in_flight_count)
        return future

    def _track_message(self, message: Message, worker: "Worker") -> None:
        """Associate the message id with the worker pid for quick lookup."""
        self._futures_map[message.id] = worker.pid  # type: ignore
        self._workers_map[worker.pid].add(message.id)  # type: ignore

    async def _wait(self, message_id: str) -> ModelResponseMessage:
        """Await completion of the ``Future`` for ``message_id`` and clean up."""
        future = self._futures[message_id]
        try:
            response_message = await future
            return response_message
        finally:
            self._clear_message(message_id)

    def _clear_message(self, message_id: str) -> None:
        """Remove all traces of an in‑flight message and update metrics."""
        if message_id in self._futures:
            del self._futures[message_id]
        worker_pid = self._futures_map.pop(message_id, None)
        if worker_pid is not None and worker_pid in self._workers_map:
            self._workers_map[worker_pid].discard(message_id)
        self._streams.pop(message_id, None)
        # keep the Gauge accurate
        self.parallel_request_queue_size.set(len(self._futures))

    def resolve(self, response: ModelResponseMessage | None):
        """
        Complete the ``Future`` for a response id, success or exception.
        This is called by the response processing loop when a matching
        ``ModelResponseMessage`` is dequeued from the shared responses queue.
        Always resilient to None or missing IDs.
        """
        if response is None:
            # Defensive: nothing to resolve
            logger.warning("[responses] resolve called with None response")
            return

        message_id = getattr(response, "id", None)
        if not message_id:
            logger.warning(f"[responses] response has no id: {response!r}")
            return

        future = self._futures.get(message_id)
        if future is None:
            logger.debug(f"[responses] no pending future for id={message_id}")
            return

        loop = future.get_loop()
        if getattr(response, "exception", None):
            loop.call_soon_threadsafe(future.set_exception, response.exception)
        else:
            loop.call_soon_threadsafe(future.set_result, response)

    def cancel(self, worker: "Worker", exit_code: int):
        """Fail all in‑flight requests belonging to a dead worker.

        Converts each outstanding request into a ``WorkerStop`` exception,
        clears stream state, and refreshes the in‑flight Gauge.
        """
        in_flight = self._workers_map.get(worker.pid, set())  # type: ignore
        if in_flight:
            logger.info(
                f"Cancelling {len(in_flight)} in-flight requests for "
                f"worker {worker.pid} which died unexpectedly with "
                f"exit code {exit_code}..."
            )
        for message_id in list(in_flight):
            err = WorkerStop(exit_code)
            future = self._futures.get(message_id)
            if future is not None:
                loop = future.get_loop()
                loop.call_soon_threadsafe(future.set_exception, err)
            self._streams.pop(message_id, None)
        # clear indexes
        self._workers_map[worker.pid].clear()
        for mid in list(self._futures_map.keys()):
            if self._futures_map[mid] == worker.pid:
                self._futures_map.pop(mid, None)
        # keep the Gauge accurate
        self.parallel_request_queue_size.set(len(self._futures))

    def create_stream_queue(self, message_id: str, maxsize: int = 0) -> asyncio.Queue:
        """Create and register an ``asyncio.Queue`` for streaming a response.

        ``maxsize`` defaults to 0 (unbounded), but callers (e.g., the dispatcher
        for streamed responses) often set a finite bound to apply backpressure.
        """
        # 0 = unbounded, avoids QueueFull in dispatcher thread
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._streams[message_id] = q
        return q

    def put_stream_message(self, msg: Any) -> None:
        """Non‑blocking enqueue of a streaming message for its request id.

        If the target queue is full, drops the oldest element and inserts the
        new one (drop‑oldest), favoring recency under pressure.
        """
        q = self._streams.get(msg.id)
        if q is not None:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # Optional: drop oldest instead of raising
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                q.put_nowait(msg)

    def pop_stream_queue(self, message_id: str) -> None:
        """Remove the stream queue for a request id (on completion/cancel)."""
        self._streams.pop(message_id, None)

    def get_worker_pid(self, message_id: str) -> int | None:
        """Return the worker pid handling ``message_id`` if known."""
        return self._futures_map.get(message_id)
