from __future__ import annotations

import asyncio
import contextlib
import inspect
import os
import re
import signal
import threading
import time
import uuid
from asyncio import CancelledError, Task
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import TYPE_CHECKING, Any, Optional

from mlserver import types

from ..logging import configure_logger, get_logger
from .errors import WorkerError
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
from .utils import END_OF_QUEUE, make_queue, terminate_queue

if TYPE_CHECKING:
    from mlserver.env import Environment
    from mlserver.settings import Settings

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logger = get_logger()

# -----------------------------------------------------------------------------
# Config & Metrics
# -----------------------------------------------------------------------------

_ENABLE_METRICS = os.getenv("WORKER_METRICS", "1") not in ("0", "false", "False", "")
IGNORED_SIGNALS = [signal.SIGINT, signal.SIGTERM] + (
    [signal.SIGQUIT] if hasattr(signal, "SIGQUIT") else []
)
_UNKNOWN = "unknown"  # Single literal for unknown identifiers


class _NoOpMetric:
    """No-op metric used when Prometheus metrics are disabled."""

    def labels(self, *_, **__):
        return self

    def inc(self, *_a, **_k):
        pass

    def dec(self, *_a, **_k):
        pass

    def observe(self, *_a, **_k):
        pass

    def set(self, *_a, **_k):
        pass


class _Metrics:
    """
    Lazily-initialized metrics container. Falls back to no-op metrics when
    WORKER_METRICS is disabled or Prometheus import fails.
    Ensures a single set of metric objects per process.
    """

    _inited = False

    def __init__(self) -> None:
        if _Metrics._inited:
            return

        if not _ENABLE_METRICS:
            self._init_noop()
            return

        try:  # tolerate envs where prometheus_client isn't available
            from prometheus_client import REGISTRY as PROM_REGISTRY  # type: ignore
            from prometheus_client import Counter, Gauge, Histogram  # type: ignore
        except Exception as e:  # pragma: no cover
            logger.warning("Prometheus disabled (import error): %s", e)
            self._init_noop()
            return

        try:
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
        except Exception as e:  # pragma: no cover
            logger.warning("Prometheus disabled (metric init error): %s", e)
            self._init_noop()
            return

        _Metrics._inited = True

    def _init_noop(self) -> None:
        self.requests_total = _NoOpMetric()
        self.request_exceptions_total = _NoOpMetric()
        self.request_latency_seconds = _NoOpMetric()
        self.stream_chunks_total = _NoOpMetric()
        self.stream_errors_total = _NoOpMetric()
        self.active_streams = _NoOpMetric()
        self.model_updates_total = _NoOpMetric()
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
# Small local helpers (to avoid importing mlserver at module load)
# -----------------------------------------------------------------------------


def _schedule_with_callback(coro: Any, cb: Any) -> asyncio.Task:
    task = asyncio.create_task(coro)
    task.add_done_callback(cb)
    return task


def _uuid() -> str:
    return uuid.uuid4().hex


def _sanitize_sys_version() -> None:
    """Best-effort: strip vendor tags like \"| packaged by conda-forge |\"."""
    try:
        import sys

        if "packaged by conda-forge" in sys.version:
            sys.version = re.sub(
                r"\s*\|\s*packaged by conda-forge\s*\|\s*", " ", sys.version
            )
    except Exception:
        pass


def _is_stream_like(value: Any) -> bool:
    """Return True if the value behaves like a stream or an awaitable."""
    return (
        inspect.isasyncgen(value)
        or hasattr(value, "__aiter__")
        or inspect.isgenerator(value)
        or isinstance(value, Iterator)
        or inspect.isawaitable(value)
    )

# Replace your current _should_decode_request with this version

def _should_decode_request(method: Any, args: list[Any]) -> bool:
    """
    Decide whether to decode a single InferenceRequest into typed args.
    Robust to postponed annotations (PEP 563 / __future__.annotations) and
    ForwardRefs. Be conservative: if we cannot determine the annotation,
    do NOT decode.
    """
    if len(args) != 1:
        return False

    try:
        from mlserver.types import InferenceRequest as _IR  # lazy
    except Exception:
        return False

    # Only consider decoding if the single arg we have IS an InferenceRequest
    if not isinstance(args[0], _IR):
        return False

    try:
        sig = inspect.signature(method)
        params = [p for p in sig.parameters.values() if p.name != "self"]
        if not params:
            return False

        # If method has multiple params (excl self), it's a typed signature -> decode
        if len(params) > 1:
            return True

        # Exactly one param: check whether it's annotated as InferenceRequest
        p = params[0]
        ann = p.annotation

        # Try to resolve string/ForwardRef annotations into real types
        try:
            from typing import get_type_hints
            hints = get_type_hints(method, globalns=getattr(method, "__globals__", {}))
            resolved = hints.get(p.name, ann)
        except Exception:
            resolved = ann

        def _ann_is_ir(a: Any) -> bool:
            if a is _IR:
                return True
            # ForwardRef or string
            name = None
            if isinstance(a, str):
                name = a
            else:
                name = getattr(a, "__forward_arg__", None) or getattr(a, "__name__", None)
            if isinstance(name, str):
                return name.split(".")[-1] == "InferenceRequest"
            return False

        # Decode only if the parameter is NOT an InferenceRequest
        return not _ann_is_ir(resolved)

    except Exception:
        # If anything is ambiguous, DON'T decode (pass the InferenceRequest through)
        return False

def _decode_request_args_for(
    method: Any, request: types.InferenceRequest, model: Any
) -> tuple[list[Any], dict[str, Any]]:
    """
    Version-agnostic decoder:
    - Tries to map RequestInput by parameter name.
    - Falls back to positional order.
    - Always decodes via model.decode(input).
    """
    inputs = request.inputs or []
    by_name = {getattr(ri, "name", None): ri for ri in inputs if getattr(ri, "name", None)}

    try:
        sig = inspect.signature(method)
        params = [p for p in sig.parameters.values() if p.name != "self"]
    except Exception:
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
            decoded.append(None)  # placeholder

    # second pass: fill remaining by index order
    it = (i for i, _ in enumerate(inputs) if i not in used_idxs)
    for i, val in enumerate(decoded):
        if val is None:
            try:
                idx = next(it)
            except StopIteration:
                raise WorkerError(
                    ValueError("Not enough inputs to satisfy method parameters")
                ) from None
            decoded[i] = model.decode(inputs[idx])

    return decoded, {}


def _sync_iter_to_async(
    gen: Iterator[Any],
    loop: asyncio.AbstractEventLoop,
    prefetch: int = 4,
) -> AsyncIterator[Any]:
    """
    Bridge a synchronous iterator to an async generator using a background thread.
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

    threading.Thread(
        target=_feeder, name="sync-stream-feeder", daemon=True
    ).start()

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
    return _Outcome(
        id=getattr(obj, "id", _UNKNOWN),
        exception=WorkerError(RuntimeError(msg)),
    )


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
    return msg.__class__ is object


# -----------------------------------------------------------------------------
# Worker
# -----------------------------------------------------------------------------


class Worker(Process):
    """Isolated model executor running in a separate OS process (strict mode)."""

    # ---------------------------- lifecycle ---------------------------------

    def __init__(
        self, settings: Settings, responses: Queue, env: Optional[Environment] = None
    ):
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
        self._requests: Queue = make_queue()
        self._model_updates: Queue = make_queue()
        self._env = env
        self.__executor: Optional[ThreadPoolExecutor] = None
        # Futures returned by run_in_executor for our MP->async bridges.
        self._bridge_futures: list = []
        self._model_registry = None  # lazy; created after env is active
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
        - Activate env context (if provided)
        - Sanitize sys.version for strict platform parsers
        - Install uvloop
        - Configure logging and metrics
        - Enter the async main routine
        """
        ctx = self._env or nullcontext()
        with ctx:
            # Belt-and-suspenders: make sys.version parseable before heavy imports
            _sanitize_sys_version()

            # Import mlserver bits *after* env is active
            try:
                from mlserver.utils import install_uvloop_event_loop
            except Exception:
                def install_uvloop_event_loop() -> None:
                    pass

            install_uvloop_event_loop()
            configure_logger(self._settings)
            try:
                if _ENABLE_METRICS:
                    from mlserver.metrics import configure_metrics
                    configure_metrics(self._settings)
            except Exception as e:
                logger.warning(
                    "Worker metrics disabled (configure_metrics failed): %s", e
                )
            asyncio.run(self._coro_run())

    async def _coro_run(self) -> None:
        self._init_runtime()
        loop = asyncio.get_running_loop()
        self._install_signal_ignores(loop)

        request_async_queue: asyncio.Queue = asyncio.Queue()
        update_async_queue: asyncio.Queue = asyncio.Queue()
        self._start_mp_bridge(self._requests, request_async_queue)
        self._start_mp_bridge(self._model_updates, update_async_queue)

        try:
            await asyncio.gather(
                self._requests_loop(request_async_queue),
                self._updates_loop(update_async_queue),
            )
        finally:
            # Ensure bridge threads finish while the loop is still alive
            for fut in list(self._bridge_futures):
                with contextlib.suppress(Exception):
                    await fut  # <- important
            self._bridge_futures.clear()
            # Tear down the dedicated executor used for the bridges
            self._executor.shutdown(wait=False, cancel_futures=True)

    # ---------------------------- init helpers ------------------------------

    def _install_signal_ignores(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register benign handlers for termination signals."""
        for sign in IGNORED_SIGNALS:
            try:
                loop.add_signal_handler(sign, _noop)
            except NotImplementedError:
                # Some platforms (e.g., Windows) don't support this.
                pass

    def _init_runtime(self) -> None:
        """Create the model registry, mark active, and clear any residual streams."""
        from mlserver.registry import MultiModelRegistry  # lazy

        self._model_registry = MultiModelRegistry()
        self._active = True
        self._inbound_streams.clear()

    def _start_mp_bridge(self, mpq: Queue, aq: asyncio.Queue) -> None:
        """
        Bridge a multiprocessing.Queue (blocking, cross-process) into an asyncio.Queue
        (non-blocking, loop-bound).
        """
        loop = asyncio.get_running_loop()

        def safe_put(item) -> bool:
            if loop.is_closed():
                return False
            try:
                loop.call_soon_threadsafe(aq.put_nowait, item)
                return True
            except RuntimeError:
                return False

        def pump() -> None:
            while True:
                try:
                    msg = mpq.get()
                except (EOFError, OSError):
                    return  # Queue torn down; exit quietly.

                if msg is END_OF_QUEUE:
                    safe_put(END_OF_QUEUE)  # propagate sentinel to async side
                    return

                if not safe_put(msg):
                    return

        fut = loop.run_in_executor(self._executor, pump)
        self._bridge_futures.append(fut)

    # ------------------------------ loops -----------------------------------

    async def _requests_loop(self, req_async_q: asyncio.Queue) -> None:
        """Consume inbound request messages and dispatch to handlers."""
        while self._active:
            msg = await req_async_q.get()

            if _is_stop_signal(msg):
                self._active = False
                break

            if isinstance(msg, ModelRequestMessage):
                _schedule_with_callback(
                    self._process_request(msg), self._finalize_and_enqueue
                )
                continue

            if isinstance(msg, (ModelStreamInputChunk, ModelStreamInputEnd)):
                self._handle_inbound_stream(msg)
                continue

    async def _updates_loop(self, upd_async_q: asyncio.Queue) -> None:
        """Consume inbound model update messages."""
        while self._active:
            upd = await upd_async_q.get()

            if _is_stop_signal(upd):
                self._active = False
                break

            _schedule_with_callback(
                self._process_model_update(upd), self._finalize_and_enqueue
            )

    # --------------------------- inbound streams -----------------------------

    def _handle_inbound_stream(
        self, msg: ModelStreamInputChunk | ModelStreamInputEnd
    ) -> None:
        """Route client->server streaming input into the appropriate per-request queue."""
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
        """Replace any InputChannelRef occurrences inside args/kwargs with async iterators."""

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
        """
        assert self._model_registry is not None
        try:
            model = await self._model_registry.get_model(
                request.model_name, request.model_version
            )
            method = getattr(model, request.method_name)
            args, kwargs = self._wrap_channels(
                request, list(request.method_args), dict(request.method_kwargs)
            )
            logger.debug("model.settings: %r", getattr(model, "settings", None))

            from mlserver.context import model_context  # lazy

            with model_context(model.settings):
                # Decode InferenceRequest -> typed args if needed
                if _should_decode_request(method, args):
                    try:
                        args, add_kwargs = _decode_request_args_for(
                            method, args[0], model  # type: ignore[arg-type]
                        )
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

    async def _emit_stream(
        self, request_id: str, stream: AsyncIterator[Any], model: str, method: str
    ) -> None:
        """Emit streaming chunks followed by a terminal end message, or an end+error on failure."""
        try:
            async for chunk in stream:
                self._responses.put(
                    ModelStreamChunkMessage(id=request_id, chunk=chunk)
                )
                _METRICS.stream_chunks_total.labels(model=model, method=method).inc()
            self._responses.put(ModelStreamEndMessage(id=request_id))
        except (Exception, CancelledError) as e:
            logger.exception("Streaming error in '%s' from model '%s'.", method, model)
            _METRICS.stream_errors_total.labels(model=model, method=method).inc()
            self._responses.put(
                ModelStreamEndMessage(id=request_id, exception=WorkerError(e))
            )

    async def _handle_stream_or_value(
        self,
        request: ModelRequestMessage,
        result: Any,
        model_name: str,
        method_name: str,
    ) -> _Outcome:
        """
        If the method result is streaming-like, emit chunks and a terminal end;
        otherwise, return the value directly (strict contract for predict()).
        """
        if not _is_stream_like(result):
            # STRICT: predict() must return an InferenceResponse (no silent coercion)
            if method_name == "predict":
                try:
                    from mlserver.types import InferenceResponse  # lazy import
                except Exception as e:
                    return _Outcome(id=request.id, exception=WorkerError(e))

                if not isinstance(result, InferenceResponse):
                    return _Outcome(
                        id=request.id,
                        exception=WorkerError(
                            TypeError(
                                f"{model_name}.predict() must return InferenceResponse, "
                                f"got {type(result).__name__}"
                            )
                        ),
                    )
            return _Outcome(id=request.id, return_value=result)

        loop = asyncio.get_running_loop()
        _METRICS.active_streams.inc()
        try:
            stream_iter = _as_async_iter(result, loop=loop, prefetch=4)
            await self._emit_stream(
                request.id, stream_iter, model_name, method_name
            )
            return _Outcome(id=request.id, return_value=None)
        finally:
            self._clean_request_streams(request.id)
            _METRICS.active_streams.dec()

    # ----------------------------- core ops ----------------------------------

    async def _process_request(self, request: ModelRequestMessage) -> _Outcome:
        """
        Main handler for model requests with metrics & error normalization.
        """
        model_name, method_name = _labels_from_request(request)
        _METRICS.requests_total.labels(model=model_name, method=method_name).inc()

        with _MetricTimer(model_name, method_name) as mt:
            try:
                invocation = await self._invoke_method(request)
                if invocation.exception is not None:
                    mt.mark_exception()
                    return invocation
                return await self._handle_stream_or_value(
                    request,
                    invocation.return_value,
                    model_name,
                    method_name,
                )
            except Exception as e:
                mt.mark_exception()
                self._clean_request_streams(request.id)
                return _Outcome(id=request.id, exception=WorkerError(e))

    async def _process_model_update(self, update: ModelUpdateMessage | Any) -> _Outcome:
        """
        Apply a model update (load/unload). Always returns an _Outcome.
        """
        assert self._model_registry is not None
        try:
            if not isinstance(update, ModelUpdateMessage):
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
                logger.warning(
                    "Unknown model update message with type %s", update.update_type
                )
                _inc_model_update("unknown", update.update_type)

            return _Outcome(id=update.id)

        except (Exception, CancelledError) as e:
            _inc_model_update("error", getattr(update, "update_type", _UNKNOWN))
            return _Outcome(
                id=getattr(update, "id", _UNKNOWN), exception=WorkerError(e)
            )

    # ------------------------ single return/enqueue --------------------------

    def _finalize_and_enqueue(self, task: Task) -> None:
        """Turn an internal _Outcome into a ModelResponseMessage and enqueue it."""
        try:
            outcome = task.result()
        except Exception as e:
            logger.exception("Process task crashed")
            outcome = _Outcome(id=_UNKNOWN, exception=WorkerError(e))

        outcome = _as_outcome(outcome)
        response = ModelResponseMessage(
            id=outcome.id, return_value=outcome.return_value, exception=outcome.exception
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
        # Ensure bridge threads have finished before the loop is torn down.
        try:
            for fut in list(self._bridge_futures):
                try:
                    # Convert to asyncio Future so we can await it.
                    await asyncio.wrap_future(fut)
                except Exception:
                    pass
        finally:
            self._bridge_futures.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)
