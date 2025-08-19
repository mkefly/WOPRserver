# worker.py
# Rewritten end-to-end: DRY helpers for request decoding & unary result coercion,
# preserved streaming, metrics, and lifecycle behavior.

from __future__ import annotations

import asyncio
import inspect
import os
import signal
import threading
import time
from asyncio import CancelledError, Task
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext, AbstractContextManager
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Any, AsyncIterator, Optional

import numpy as np
from prometheus_client import REGISTRY as PROM_REGISTRY
from prometheus_client import Counter, Gauge, Histogram

from mlserver.context import model_context
from mlserver.env import Environment

from mlserver.metrics import configure_metrics
from mlserver.registry import MultiModelRegistry
from mlserver.settings import Settings
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.utils import install_uvloop_event_loop, schedule_with_callback, generate_uuid
from mlserver.codecs.numpy import NumpyCodec

from .errors import WorkerError
from ..logging import configure_logger, get_logger 
logger = get_logger()

from .messages import (
    InputChannelRef,
    ModelRequestMessage,
    ModelResponseMessage,
    ModelStreamChunkMessage,
    ModelStreamEndMessage,
    ModelStreamInputChunk,
    ModelStreamInputEnd,
    ModelUpdateMessage,
    ModelUpdateType,
)
from .utils import END_OF_QUEUE, terminate_queue

# -----------------------------------------------------------------------------
# Config & Metrics
# -----------------------------------------------------------------------------

_ENABLE_METRICS = os.getenv("WORKER_METRICS", "1") not in ("0", "false", "False", "")
IGNORED_SIGNALS = [signal.SIGINT, signal.SIGTERM] + ([signal.SIGQUIT] if hasattr(signal, "SIGQUIT") else [])
_UNKNOWN = "unknown"  # Single literal for unknown identifiers


class _NoOpMetric:
    """No-op metric used when Prometheus metrics are disabled."""
    def labels(self, *_, **__): return self
    def inc(self, *_a, **_k): pass
    def dec(self, *_a, **_k): pass
    def observe(self, *_a, **_k): pass
    def set(self, *_a, **_k): pass


class _Metrics:
    """
    Lazily-initialized metrics container. Falls back to no-op metrics when
    WORKER_METRICS is disabled. Ensures a single set of metric objects per process.
    """
    _inited = False
    def __init__(self) -> None:
        if _Metrics._inited:
            return
        if not _ENABLE_METRICS:
            self.requests_total = _NoOpMetric()
            self.request_exceptions_total = _NoOpMetric()
            self.request_latency_seconds = _NoOpMetric()
            self.stream_chunks_total = _NoOpMetric()
            self.stream_errors_total = _NoOpMetric()
            self.active_streams = _NoOpMetric()
            self.model_updates_total = _NoOpMetric()
            _Metrics._inited = True
            return

        self.requests_total = Counter(
            "worker_requests_total",
            "Total requests processed by a worker.",
            ["model", "method"],
            registry=PROM_REGISTRY,
        )
        self.request_exceptions_total = Counter(
            "worker_request_exceptions_total",
            "Total exceptions raised while processing requests.",
            ["model", "method"],
            registry=PROM_REGISTRY,
        )
        self.request_latency_seconds = Histogram(
            "worker_request_latency_seconds",
            "End-to-end request latency per method.",
            ["model", "method"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=PROM_REGISTRY,
        )
        self.stream_chunks_total = Counter(
            "worker_stream_chunks_total",
            "Number of chunks emitted by streaming responses.",
            ["model", "method"],
            registry=PROM_REGISTRY,
        )
        self.stream_errors_total = Counter(
            "worker_stream_errors_total",
            "Number of streaming errors.",
            ["model", "method"],
            registry=PROM_REGISTRY,
        )
        self.active_streams = Gauge(
            "worker_active_streams",
            "Number of active server->client streams.",
            registry=PROM_REGISTRY,
        )
        self.model_updates_total = Counter(
            "worker_model_updates_total",
            "Model updates applied by the worker.",
            ["type", "outcome"],
            registry=PROM_REGISTRY,
        )
        _Metrics._inited = True


_METRICS = _Metrics()


def _noop() -> None:
    """No-op signal callback used to ignore termination signals cleanly."""
    pass


# -----------------------------------------------------------------------------
# Internal outcomes (single return point)
# -----------------------------------------------------------------------------

@dataclass
class _Outcome:
    """
    Normalized result for any async task inside the worker.

    Attributes:
        id: Correlation id of the message being processed.
        return_value: Result value (None for streaming ACKs).
        exception: WorkerError if an exception occurred.
    """
    id: str
    return_value: Any | None = None
    exception: WorkerError | None = None


# -----------------------------------------------------------------------------
# DRY helpers: discovery, decoding & post-processing
# -----------------------------------------------------------------------------

def _is_stream_like(value: Any) -> bool:
    """Return True if the value behaves like a stream or an awaitable."""
    return (
        inspect.isasyncgen(value)
        or hasattr(value, "__aiter__")
        or inspect.isgenerator(value)
        or isinstance(value, Iterator)
        or inspect.isawaitable(value)
    )


def _should_decode_request(method: Any, args: list[Any]) -> bool:
    """
    True when we're about to pass a single InferenceRequest to a method that
    *doesn't* accept InferenceRequest (i.e., expects typed parameters).
    """
    if not (len(args) == 1 and isinstance(args[0], InferenceRequest)):
        return False
    try:
        sig = inspect.signature(method)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        if not params:
            return False
        if len(params) == 1:
            # one param: only decode if that param is NOT an InferenceRequest
            ann = params[0].annotation
            return ann not in (InferenceRequest, inspect._empty)
        # multiple params -> likely a typed signature (foo, bar, ...)
        return True
    except Exception:
        # if we can't inspect, be conservative and don't decode
        return False


def _decode_request_args_for(method: Any, request: InferenceRequest, model: Any) -> tuple[list[Any], dict[str, Any]]:
    """
    Version-agnostic decoder:
    - Tries to map RequestInput by parameter name.
    - Falls back to positional order.
    - Always decodes via model.decode(input).
    """
    inputs = request.inputs or []
    by_name = {ri.name: ri for ri in inputs if getattr(ri, "name", None)}

    try:
        sig = inspect.signature(method)
        params = [p for p in sig.parameters.values() if p.name != "self"]
    except Exception:
        # Fallback: just decode in order
        return [model.decode(ri) for ri in inputs], {}

    decoded: list[Any] = []
    used_idxs: set[int] = set()

    # first pass: by name (in parameter order)
    for p in params:
        ri = by_name.get(p.name)
        if ri is not None:
            decoded.append(model.decode(ri))
            try:
                idx = inputs.index(ri)
                used_idxs.add(idx)
            except ValueError:
                pass
        else:
            decoded.append(None)  # placeholder to fill by index later

    # second pass: fill remaining by index order
    it = (i for i, _ in enumerate(inputs) if i not in used_idxs)
    for i, val in enumerate(decoded):
        if val is None:
            try:
                idx = next(it)
            except StopIteration:
                raise WorkerError(ValueError("Not enough inputs to satisfy method parameters"))
            decoded[i] = model.decode(inputs[idx])

    return decoded, {}


def _coerce_to_inference_response(value: Any, model_name: str, method_name: str) -> InferenceResponse | None:
    """
    For unary results, optionally coerce plain returns into an InferenceResponse.
    We only coerce for `predict`, to avoid surprising other methods.
    """
    if method_name != "predict":
        return None

    # Already correct
    if isinstance(value, InferenceResponse):
        return value

    def _enc(name: str, arr_like: Any):
        return NumpyCodec.encode_output(name, np.asarray(arr_like))

    # Single array-like -> one output
    if isinstance(value, (np.ndarray, list, tuple)):
        try:
            return InferenceResponse(
                id=generate_uuid(),
                model_name=model_name,
                outputs=[_enc("output_0", value)],
            )
        except Exception:
            return None

    # Dict[str, array-like] -> many outputs
    if isinstance(value, dict):
        try:
            outputs = [_enc(str(k), v) for k, v in value.items()]
            return InferenceResponse(
                id=generate_uuid(),
                model_name=model_name,
                outputs=outputs,
            )
        except Exception:
            return None

    return None


def _sync_iter_to_async(
    gen: Iterator[Any],
    loop: asyncio.AbstractEventLoop,
    prefetch: int = 4,
) -> AsyncIterator[Any]:
    """
    Bridge a synchronous iterator to an async generator using a background thread.

    Ensures items are delivered into an asyncio.Queue with bounded prefetch,
    and propagates any exception raised in the sync generator back to the
    async consumer.
    """
    END = object()
    q: asyncio.Queue = asyncio.Queue(maxsize=max(prefetch, 1))
    exc_holder: dict[str, BaseException] = {}

    def _feeder() -> None:
        try:
            for item in gen:
                fut = asyncio.run_coroutine_threadsafe(q.put(item), loop)
                try:
                    fut.result()
                except Exception:
                    return
        except Exception as e:
            exc_holder["e"] = e
        finally:
            loop.call_soon_threadsafe(q.put_nowait, END)

    threading.Thread(target=_feeder, name="sync-stream-feeder", daemon=True).start()

    async def _agen() -> AsyncIterator[Any]:
        while True:
            item = await q.get()
            if item is END:
                if "e" in exc_holder:
                    raise exc_holder["e"]
                return
            yield item

    return _agen()


def _as_async_iter(
    value: Any,
    loop: asyncio.AbstractEventLoop,
    prefetch: int = 4,
) -> AsyncIterator[Any]:
    """
    Normalize a value into an async iterator.

    Supports:
      - async generators and async iterables
      - awaitables (await, then normalize the result)
      - sync generators / iterators (bridged via a background thread)
      - plain values (yield once)
    """
    if inspect.isasyncgen(value) or hasattr(value, "__aiter__"):
        return value  # type: ignore[return-value]

    if inspect.isawaitable(value):
        async def _wrap() -> AsyncIterator[Any]:
            inner = await value
            async for item in _as_async_iter(inner, loop, prefetch=prefetch):
                yield item
        return _wrap()

    if inspect.isgenerator(value) or isinstance(value, Iterator):
        return _sync_iter_to_async(value, loop, prefetch=prefetch)

    async def _single() -> AsyncIterator[Any]:
        yield value
    return _single()


def _labels_from_request(request: ModelRequestMessage) -> tuple[str, str]:
    """Extract `(model_name, method_name)` with safe fallbacks for metrics."""
    return getattr(request, "model_name", "?"), getattr(request, "method_name", "?")


def _as_outcome(obj: Any) -> _Outcome:
    """Coerce any object into an `_Outcome`, wrapping unexpected types as errors."""
    if isinstance(obj, _Outcome):
        return obj
    msg = f"Unexpected outcome type: {type(obj).__name__}"
    return _Outcome(id=getattr(obj, "id", _UNKNOWN), exception=WorkerError(RuntimeError(msg)))


def _inc_model_update(outcome: str, update_type: Any) -> None:
    """
    Increment the model update metric with a best-effort type label.
    outcome: 'success' | 'error' | 'unknown'
    """
    try:
        name = getattr(update_type, "name", str(update_type))
        _METRICS.model_updates_total.labels(type=name, outcome=outcome).inc()
    except Exception:
        # Metrics failures should never disrupt control flow.
        pass


class _MetricTimer(AbstractContextManager):
    """
    Times a request and records latency on exit. Also provides a one-time
    exception marker to increment the exceptions counter exactly once.
    """
    def __init__(self, model: str, method: str) -> None:
        self.model = model
        self.method = method
        self.start = time.perf_counter()
        self._exc_marked = False

    def __exit__(self, exc_type, exc, tb):
        _METRICS.request_latency_seconds.labels(
            model=self.model, method=self.method
        ).observe(max(time.perf_counter() - self.start, 0.0))
        return False  # never suppress exceptions

    def mark_exception(self) -> None:
        if not self._exc_marked:
            _METRICS.request_exceptions_total.labels(
                model=self.model, method=self.method
            ).inc()
            self._exc_marked = True


def _is_stop_signal(msg: object) -> bool:
    """
    Return True if `msg` is the official END_OF_QUEUE sentinel, or a bare
    `object()` poison pill used by some older pools/tests during shutdown.
    """
    if msg is END_OF_QUEUE:
        return True
    # Compatibility: treat anonymous object() as a stop signal
    return (msg.__class__ is object)


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------

class Worker(Process):
    """Isolated model executor running in a separate OS process."""

    # ---------------------------- lifecycle ---------------------------------

    def __init__(self, settings: Settings, responses: Queue, env: Optional[Environment] = None):
        """
        Initialize the worker process with internal request/update queues.

        Args:
            settings: MLServer Settings for logging and runtime config.
            responses: Inter-process queue where responses are posted.
            env: Optional context manager that sets up environment state for the process.
        """
        super().__init__()
        self._settings = settings
        self._responses: Queue = responses
        self._requests: Queue = Queue()
        self._model_updates: Queue = Queue()
        self._env = env
        self.__executor: Optional[ThreadPoolExecutor] = None

        self._model_registry: Optional[MultiModelRegistry] = None
        self._active: bool = False
        # Map (request_id, channel_id) -> asyncio.Queue that feeds the model method.
        self._inbound_streams: dict[tuple[str, str], asyncio.Queue] = {}

    @property
    def _executor(self) -> ThreadPoolExecutor:
        """Lazily create the thread pool used for bridging and sync->async adapters."""
        if self.__executor is None:
            self.__executor = ThreadPoolExecutor()
        return self.__executor

    def run(self) -> None:
        """
        Process entrypoint:
          - Install uvloop
          - Configure logging and metrics
          - Enter the async main routine
        """
        ctx = self._env or nullcontext()
        with ctx:
            install_uvloop_event_loop()
            configure_logger(self._settings)
            configure_metrics(self._settings)
            asyncio.run(self._coro_run())

    async def _coro_run(self) -> None:
        """Async main: init runtime, wire bridges, then serve requests + updates."""
        self._init_runtime()
        loop = asyncio.get_running_loop()
        self._install_signal_ignores(loop)

        request_async_queue: asyncio.Queue = asyncio.Queue()
        update_async_queue: asyncio.Queue = asyncio.Queue()
        self._start_mp_bridge(self._requests, request_async_queue)
        self._start_mp_bridge(self._model_updates, update_async_queue)

        await asyncio.gather(
            self._requests_loop(request_async_queue),
            self._updates_loop(update_async_queue),
        )

    # ---------------------------- init helpers ------------------------------

    def _install_signal_ignores(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Register benign handlers for termination signals so we stay in control
        of shutdown sequence.
        """
        for sign in IGNORED_SIGNALS:
            try:
                loop.add_signal_handler(sign, _noop)
            except NotImplementedError:
                # Some platforms (e.g., Windows) don't support this.
                pass

    def _init_runtime(self) -> None:
        """Create the model registry, mark active, and clear any residual streams."""
        self._model_registry = MultiModelRegistry()
        self._active = True
        self._inbound_streams.clear()

    def _start_mp_bridge(self, mpq: Queue, aq: asyncio.Queue) -> None:
        """
        Continuously pump a multiprocessing.Queue into an asyncio.Queue.
        Preserves END_OF_QUEUE to signal graceful shutdown.
        """
        loop = asyncio.get_running_loop()

        def _pump() -> None:
            while True:
                try:
                    msg = mpq.get()
                except (EOFError, OSError):
                    break
                if msg is END_OF_QUEUE:
                    loop.call_soon_threadsafe(aq.put_nowait, END_OF_QUEUE)
                    break
                loop.call_soon_threadsafe(aq.put_nowait, msg)

        loop.run_in_executor(self._executor, _pump)

    # ------------------------------ loops -----------------------------------

    async def _requests_loop(self, req_async_q: asyncio.Queue) -> None:
        """
        Consume inbound request messages and dispatch to handlers until a stop
        signal is received. Accepts both END_OF_QUEUE and a bare object() poison pill.
        """
        while self._active:
            msg = await req_async_q.get()

            if _is_stop_signal(msg):
                self._active = False
                break

            if isinstance(msg, ModelRequestMessage):
                schedule_with_callback(self._process_request(msg), self._finalize_and_enqueue)
                continue

            if isinstance(msg, (ModelStreamInputChunk, ModelStreamInputEnd)):
                self._handle_inbound_stream(msg)
                continue

            # Quiet during shutdown races
            pass

    async def _updates_loop(self, upd_async_q: asyncio.Queue) -> None:
        """
        Consume inbound model update messages until a stop signal is received.
        Accepts both END_OF_QUEUE and a bare object() poison pill.
        """
        while self._active:
            upd = await upd_async_q.get()

            if _is_stop_signal(upd):
                self._active = False
                break

            schedule_with_callback(self._process_model_update(upd), self._finalize_and_enqueue)

    # --------------------------- inbound streams -----------------------------

    def _handle_inbound_stream(self, msg: ModelStreamInputChunk | ModelStreamInputEnd) -> None:
        """
        Route client->server streaming input into the appropriate per-request queue.
        """
        key = (msg.id, getattr(msg, "channel_id", ""))
        q = self._inbound_streams.get(key)
        if q is None:
            return
        q.put_nowait(END_OF_QUEUE if isinstance(msg, ModelStreamInputEnd) else msg.item)

    def _clean_request_streams(self, req_id: str) -> None:
        """Remove all inbound queues associated with a given request id."""
        for key in [k for k in list(self._inbound_streams.keys()) if k[0] == req_id]:
            self._inbound_streams.pop(key, None)

    # --------------------------- arg wrapping --------------------------------

    def _wrap_channels(
        self, request: ModelRequestMessage, args: list[Any], kwargs: dict[str, Any]
    ) -> tuple[list[Any], dict[str, Any]]:
        """
        Replace any InputChannelRef occurrences inside args/kwargs with async
        iterators that read from the request-specific inbound queues.
        """

        async def _aiter_from_async_q(q: asyncio.Queue) -> AsyncIterator[Any]:
            while True:
                item = await q.get()
                if item is END_OF_QUEUE:
                    return
                yield item

        def _maybe_wrap(x: Any) -> Any:
            if not isinstance(x, InputChannelRef) or x.id != request.id:
                return x
            key = (request.id, getattr(x, "channel_id", ""))
            q = self._inbound_streams.get(key)
            if q is None:
                q = asyncio.Queue()
                self._inbound_streams[key] = q
            return _aiter_from_async_q(q)

        for i, a in enumerate(args):
            args[i] = _maybe_wrap(a)
        for k in list(kwargs.keys()):
            kwargs[k] = _maybe_wrap(kwargs[k])
        return args, kwargs

    # ----------------------------- method IO ---------------------------------

    async def _invoke_method(self, request: ModelRequestMessage) -> _Outcome:
        """
        Resolve the target model and method, call it, and normalize the result.

        Returns:
            _Outcome with either:
              - return_value (already awaited if it was a simple awaitable), or
              - exception as WorkerError
        """
        assert self._model_registry is not None
        try:
            model = await self._model_registry.get_model(request.model_name, request.model_version)
            method = getattr(model, request.method_name)
            args, kwargs = self._wrap_channels(request, list(request.method_args), dict(request.method_kwargs))

            with model_context(model.settings):
                # Decode InferenceRequest -> typed args if needed
                if _should_decode_request(method, args):
                    try:
                        args, add_kwargs = _decode_request_args_for(method, args[0], model)  # type: ignore[arg-type]
                        kwargs.update(add_kwargs)
                    except Exception as e:
                        return _Outcome(id=request.id, exception=WorkerError(e))

                # Call
                try:
                    result = method(*args, **kwargs)
                except Exception as e:
                    return _Outcome(id=request.id, exception=WorkerError(e))

                # Await plain awaitables (not async generators)
                if inspect.isawaitable(result) and not inspect.isasyncgen(result):
                    try:
                        result = await result
                    except Exception as e:
                        return _Outcome(id=request.id, exception=WorkerError(e))

                return _Outcome(id=request.id, return_value=result)

        except Exception as e:
            # Hard failure resolving model/method or executing
            return _Outcome(id=request.id, exception=WorkerError(e))

    async def _emit_stream(self, request_id: str, stream: AsyncIterator[Any], model: str, method: str) -> None:
        """
        Emit streaming chunks followed by a terminal end message, or an end+error on failure.
        """
        try:
            async for chunk in stream:
                self._responses.put(ModelStreamChunkMessage(id=request_id, chunk=chunk))
                _METRICS.stream_chunks_total.labels(model=model, method=method).inc()
            self._responses.put(ModelStreamEndMessage(id=request_id))
        except (Exception, CancelledError) as e:
            logger.exception("Streaming error in '%s' from model '%s'.", method, model)
            _METRICS.stream_errors_total.labels(model=model, method=method).inc()
            self._responses.put(ModelStreamEndMessage(id=request_id, exception=WorkerError(e)))

    async def _handle_stream_or_value(
        self,
        request: ModelRequestMessage,
        result: Any,
        model_name: str,
        method_name: str,
    ) -> _Outcome:
        """
        If the method result is streaming-like, emit chunks and a terminal end;
        otherwise, return the value directly (coercing plain returns for `predict`).
        """
        if not _is_stream_like(result):
            coerced = _coerce_to_inference_response(result, model_name, method_name)
            return _Outcome(id=request.id, return_value=coerced if coerced is not None else result)

        loop = asyncio.get_running_loop()
        _METRICS.active_streams.inc()
        try:
            stream_iter = _as_async_iter(result, loop=loop, prefetch=4)
            await self._emit_stream(request.id, stream_iter, model_name, method_name)
            return _Outcome(id=request.id, return_value=None)
        finally:
            self._clean_request_streams(request.id)
            _METRICS.active_streams.dec()

    # ----------------------------- core ops ----------------------------------

    async def _process_request(self, request: ModelRequestMessage) -> _Outcome:
        """
        Main handler for model requests:
          1) increment request counter,
          2) time end-to-end latency,
          3) invoke the target method,
          4) stream or return the result accordingly,
          5) mark exceptions exactly once.
        """
        model_name, method_name = _labels_from_request(request)
        _METRICS.requests_total.labels(model=model_name, method=method_name).inc()

        with _MetricTimer(model_name, method_name) as mt:
            try:
                invocation = await self._invoke_method(request)
                if invocation.exception is not None:
                    return invocation
                return await self._handle_stream_or_value(
                    request, invocation.return_value, model_name, method_name
                )
            except Exception as e:
                mt.mark_exception()
                self._clean_request_streams(request.id)
                return _Outcome(id=request.id, exception=WorkerError(e))

    async def _process_model_update(self, update: ModelUpdateMessage | Any) -> _Outcome:
        """
        Apply a model update (load/unload). Always returns an _Outcome to settle
        the parent future, whether success or error.
        """
        assert self._model_registry is not None
        try:
            if not isinstance(update, ModelUpdateMessage):
                # keep quiet about content; just surface a single WorkerError
                _inc_model_update("error", getattr(update, "update_type", _UNKNOWN))
                return _Outcome(
                    id=getattr(update, "id", _UNKNOWN),
                    exception=WorkerError(AttributeError("Invalid update msg")),
                )

            ms = update.model_settings
            if update.update_type == ModelUpdateType.Load:
                await self._model_registry.load(ms)
                _inc_model_update("success", update.update_type)
            elif update.update_type == ModelUpdateType.Unload:
                await self._model_registry.unload_version(ms.name, ms.version)
                _inc_model_update("success", update.update_type)
            else:
                logger.warning("Unknown model update message with type %s", update.update_type)
                _inc_model_update("unknown", update.update_type)

            return _Outcome(id=update.id)

        except (Exception, CancelledError) as e:
            _inc_model_update("error", getattr(update, "update_type", _UNKNOWN))
            return _Outcome(id=getattr(update, "id", _UNKNOWN), exception=WorkerError(e))

    # ------------------------ single return/enqueue --------------------------

    def _finalize_and_enqueue(self, task: Task) -> None:
        """
        Turn an internal _Outcome into a ModelResponseMessage and enqueue it.
        This is the only place that communicates back to the parent process.
        """
        try:
            outcome = task.result()
        except Exception as e:
            logger.exception("Process task crashed")
            outcome = _Outcome(id=_UNKNOWN, exception=WorkerError(e))

        outcome = _as_outcome(outcome)
        response = ModelResponseMessage(
            id=outcome.id,
            return_value=outcome.return_value,
            exception=outcome.exception,
        )

        try:
            self._responses.put(response)
        except Exception:
            logger.exception("Worker failed to enqueue response | mid=%s", response.id)

    # ----------------------------- API --------------------------------------

    def send_request(self, request_message: ModelRequestMessage) -> None:
        """Send a model request to the worker via the internal request queue."""
        self._requests.put(request_message)

    def send_update(self, model_update: ModelUpdateMessage) -> None:
        """Send a model load/unload update to the worker."""
        self._model_updates.put(model_update)

    async def _shutdown_queues(self, *qs: Queue) -> None:
        """
        Signal END_OF_QUEUE to all given queues, drain them via terminate_queue,
        and close them. Resilient to individual queue failures.
        """
        for q in qs:
            try:
                q.put_nowait(END_OF_QUEUE)
            except Exception:
                pass

        for q in qs:
            try:
                await terminate_queue(q)
            except Exception:
                pass
            try:
                q.close()
            except Exception:
                pass

    async def stop(self) -> None:
        """
        Gracefully stop the worker:
          - Signal END_OF_QUEUE to request/update queues
          - Drain and close them
          - Shutdown the thread pool executor
        """
        await self._shutdown_queues(self._model_updates, self._requests)
        self._executor.shutdown(wait=False, cancel_futures=True)
