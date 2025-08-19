"""
Parallel model proxy — didactic docs & commentary
=================================================

This module defines **ParallelModel**, a proxy around an ``MLModel`` that routes
method calls through a shared ``Dispatcher`` to a pool of worker processes. It
supports both *unary* calls (request → single response) and *streaming* calls,
including **bi-directional streaming** where the client sends an async stream of
inputs and receives a stream of outputs.

What this layer does
--------------------
1. **Introspection helpers** determine whether a target model method:
   - returns an async iterator (e.g., ``async def ...`` that ``yield``s), or
   - expects an async-iterable input anywhere among its parameters.
2. **Automatic parallelisation** of custom handlers: we wrap each custom handler
   from the underlying model so that calling it invokes the dispatcher/worker
   machinery transparently.
3. **Channel mapping for streaming inputs**: when a method expects async input,
   the proxy replaces each async-iterable argument with an ``InputChannelRef``
   (``id``, ``channel_id``). The worker process maps each channel id to an
   ``asyncio.Queue`` and pumps client items into the model method.
4. **Unary fallback for streamed inputs**: if the method does *not* expect an
   async-iterable, but the caller passes one, we consume the **first** item and
   pass it as a normal argument (closing the source). This avoids deadlocks.

Behavioral notes
----------------
This pass adds docstrings and comments for clarity while preserving the original
semantics and public API.
"""

from __future__ import annotations

import asyncio
import inspect
import itertools
from collections.abc import AsyncIterable as CAsyncIterable
from collections.abc import AsyncIterator, Callable
from collections.abc import AsyncIterator as CAsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, get_origin

from mlserver.errors import InferenceError
from mlserver.handlers.custom import get_custom_handlers, register_custom_handler
from mlserver.model import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, MetadataModelResponse

from .dispatcher import Dispatcher
from .messages import InputChannelRef, ModelRequestMessage

# ---------- Public constants --------------------------------------------------

class ModelMethods(Enum):
    """Canonical method names used by the proxy for built-in calls."""
    Predict = "predict"
    Metadata = "metadata"
    PredictStream = "predict_stream"


# ---------- Introspection utilities ------------------------------------------

def _is_async_iterable_type(tp: Any) -> bool:
    """Return True if a *type annotation* represents an async-iterable."""
    if tp is None:
        return False
    try:
        origin = get_origin(tp)
    except Exception:
        origin = None
    return tp in (CAsyncIterable, CAsyncIterator) or origin in (CAsyncIterable, CAsyncIterator)


def _returns_async_iterator(fn: Callable[..., Any]) -> bool:
    """Detect if a function is declared to return an async iterator."""
    ann = getattr(fn, "__annotations__", {})
    ret = ann.get("return")
    if _is_async_iterable_type(ret):
        return True
    return inspect.isasyncgenfunction(fn)


def _expects_async_iterable_input(fn: Callable[..., Any]) -> bool:
    """
    Detect whether **any** non-``self`` parameter is annotated as
    ``AsyncIterable``/``AsyncIterator``.
    """
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        for p in params:
            ann = p.annotation
            if ann is inspect._empty:
                continue
            if _is_async_iterable_type(ann):
                return True
    except Exception:
        pass
    return False


def _is_async_iterable(obj: Any) -> bool:
    """Runtime check for async-iterables (robust even if unannotated)."""
    return hasattr(obj, "__aiter__")


async def _first_and_close(ait: AsyncIterator[Any]) -> Any:
    """
    Pull the first item from an async iterator and make a best effort to close it.
    Returns None if the iterator is empty.
    """
    try:
        return await ait.__anext__()
    except StopAsyncIteration:
        return None
    finally:
        aclose = getattr(ait, "aclose", None)
        if callable(aclose):
            try:
                await aclose()
            except Exception:
                pass


async def _safe_aclose(ait: Any) -> None:
    """Close an async iterator if supported (no-op otherwise)."""
    aclose = getattr(ait, "aclose", None)
    if callable(aclose):
        await aclose()


# ---------- Stream planning & argument rewriting -----------------------------

@dataclass
class StreamPlanner:
    """
    Discovers async-iterable inputs among args/kwargs and prepares one of two plans:

    - **Channel plan** (method expects async inputs): replace streams with
      ``InputChannelRef``s and later bridge client items into worker channels.
    - **Unary fallback plan** (method does NOT expect async inputs): consume the
      first item from each stream and pass that value instead.

    In both cases, this class mutates the given ``ModelRequestMessage`` in-place.
    """
    # positions of async streams within args
    arg_positions: list[int] = field(default_factory=list)
    # keys of async streams within kwargs
    kw_keys: list[str] = field(default_factory=list)
    # the discovered client streams in discovery order
    streams: list[AsyncIterator[Any]] = field(default_factory=list)
    # assigned channel ids (one per discovered stream) if we go async-channel route
    channel_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_request(cls, req: ModelRequestMessage) -> StreamPlanner:
        planner = cls()
        # Positional args
        for i, a in enumerate(req.method_args):
            if _is_async_iterable(a):
                planner.arg_positions.append(i)
                planner.streams.append(a)  # type: ignore[arg-type]
        # Keyword args
        for k, v in list(req.method_kwargs.items()):
            if _is_async_iterable(v):
                planner.kw_keys.append(k)
                planner.streams.append(v)  # type: ignore[arg-type]
        return planner

    def assign_channels(self) -> None:
        if not self.streams:
            return
        counter = itertools.count()
        self.channel_ids = [f"ch-{next(counter)}" for _ in self.streams]

    def apply_channel_refs(self, req: ModelRequestMessage) -> None:
        """
        Replace each discovered async-iterable argument with an InputChannelRef.
        Requires channels to be assigned first.
        """
        if not self.streams:
            return
        if not self.channel_ids or len(self.channel_ids) != len(self.streams):
            raise RuntimeError("Channels not assigned or mismatched for StreamPlanner.")

        # Positional args: map by index
        for idx, pos in enumerate(self.arg_positions):
            req.method_args[pos] = InputChannelRef(id=req.id, channel_id=self.channel_ids[idx])

        # Keyword args: continuation of the index
        offset = len(self.arg_positions)
        for j, key in enumerate(self.kw_keys, start=offset):
            req.method_kwargs[key] = InputChannelRef(id=req.id, channel_id=self.channel_ids[j])

    async def apply_unary_fallback(self, req: ModelRequestMessage) -> None:
        """
        Consume the first item from each stream and replace the async inputs
        with their first values. Error if any stream is empty or raises.
        Always closes every discovered stream.
        """
        if not self.streams:
            return

        first_items: list[Any | None] = []
        first_error: BaseException | None = None

        # Pull first from every stream (each call to _first_and_close closes that stream)
        for s in self.streams:
            try:
                first_items.append(await _first_and_close(s))
            except Exception as e:
                # Record the first error, still iterate to ensure all streams are closed
                if first_error is None:
                    first_error = e
                first_items.append(None)

        if first_error is not None:
            # Preserve the original message so tests matching "boom" pass
            raise InferenceError(str(first_error)) from first_error

        if any(item is None for item in first_items):
            raise InferenceError("Empty input stream for unary streaming method.")

        # Positional args
        for idx, pos in enumerate(self.arg_positions):
            req.method_args[pos] = first_items[idx]

        # Keyword args
        offset = len(self.arg_positions)
        for j, key in enumerate(self.kw_keys, start=offset):
            req.method_kwargs[key] = first_items[j]

    def start_bridges(
        self,
        dispatcher: Dispatcher,
        worker: Any,
        request_id: str,
    ) -> list[asyncio.Task]:
        """
        For the channel plan, concurrently bridge client streams into
        worker input channels. Returns the created tasks so the caller
        can manage their lifecycle (no dangling tasks).
        """
        tasks: list[asyncio.Task] = []
        if not self.streams or not self.channel_ids:
            return tasks
        for s, cid in zip(self.streams, self.channel_ids, strict=False):
            t = asyncio.create_task(
                dispatcher._bridge_async_iterable_arg(worker, request_id, s, channel_id=cid)
            )
            tasks.append(t)
        return tasks


# ---------- Parallel proxy ---------------------------------------------------

class ParallelModel(MLModel):
    """
    Proxy over an ``MLModel`` that sends method calls to workers via a
    ``Dispatcher``, handling both unary and streaming (bi/uni) calls.

    Caching & concurrency
    ---------------------
    * Metadata is fetched once and cached under an ``asyncio.Lock`` to prevent
      thundering herds.

    Custom handlers
    ---------------
    Any custom handler exposed by the underlying model is discovered and wrapped
    so that calling it transparently routes through the dispatcher.
    """
    def __init__(self, model: MLModel, dispatcher: Dispatcher):
        super().__init__(model.settings)
        self._model = model
        self._dispatcher = dispatcher
        self._metadata: MetadataModelResponse | None = None
        self._metadata_lock = asyncio.Lock()
        self._register_custom_handlers()

    # -- custom handlers registration

    def _register_custom_handlers(self) -> None:
        """Discover model-specific custom handlers and wrap them for parallel use."""
        for handler, method in get_custom_handlers(self._model):
            wrapped = self._parallelise(method)
            register_custom_handler(handler, wrapped)
            setattr(self, method.__name__, wrapped)

    def _parallelise(self, method: Callable[..., Any]) -> Callable[..., Any]:
        """Wrap a target method with a unary or streaming proxy based on introspection."""
        if _returns_async_iterator(method):
            @wraps(method)
            async def _inner_streaming(*args, **kwargs):
                async for chunk in self._send_stream(method.__name__, *args, **kwargs):
                    yield chunk
            return _inner_streaming

        @wraps(method)
        async def _inner_unary(*args, **kwargs):
            return await self._send(method.__name__, *args, **kwargs)
        return _inner_unary

    # -- built-ins

    async def metadata(self) -> MetadataModelResponse:
        """Return cached model metadata, fetching once if needed."""
        if self._metadata is not None:
            return self._metadata
        async with self._metadata_lock:
            if self._metadata is None:
                md = await self._send(ModelMethods.Metadata.value)
                if not isinstance(md, MetadataModelResponse):
                    raise InferenceError(f"Model '{self.name}' returned no metadata")
                self._metadata = md
        return self._metadata

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        """Unary inference call via the dispatcher."""
        resp = await self._send(ModelMethods.Predict.value, payload)
        if not isinstance(resp, InferenceResponse):
            raise InferenceError(f"Model '{self.name}' returned no predictions after inference")
        return resp

    async def predict_stream(
        self,
        payloads: AsyncIterator[InferenceRequest]
    ) -> AsyncIterator[InferenceResponse]:
        """Bi-di streaming prediction; yields server->client chunks as they arrive."""
        async for chunk in self._send_stream(ModelMethods.PredictStream.value, payloads):
            yield chunk

    # -- core send helpers

    async def _send(self, method_name: str, *args, **kwargs) -> Any | None:
        """Send a unary request and await the single response value."""
        req = ModelRequestMessage(
            model_name=self.name,
            model_version=self.version,
            method_name=method_name,
            method_args=list(args),
            method_kwargs=dict(kwargs),
        )
        resp = await self._dispatcher.dispatch_request(req)
        return resp.return_value

    async def _send_stream(self, method_name: str, *args, **kwargs) -> AsyncIterator[Any]:
        """
        Send a streaming request, bridging any client async-iterable inputs.

        Flow
        ----
        1. Inspect the **underlying** model for async-iterable inputs.
        2. Build a StreamPlanner from args/kwargs.
        3. Choose a plan:
           - Channel plan: replace inputs with InputChannelRef and later bridge.
           - Unary fallback plan: consume first items and pass plain values.
        4. Dispatch and yield server chunks as they arrive.
        """
        # 1) Inspect underlying target method for input streams
        target = getattr(self._model, method_name)
        expects_async_input = _expects_async_iterable_input(target)

        # 2) Create request + discover streams
        req = ModelRequestMessage(
            model_name=self.name,
            model_version=self.version,
            method_name=method_name,
            method_args=list(args),
            method_kwargs=dict(kwargs),
        )
        planner = StreamPlanner.from_request(req)

        # 3) Apply plan
        bridge_tasks: list[asyncio.Task] = []
        if planner.streams and expects_async_input:
            # Channel plan: replace inputs by channel refs and start bridges
            planner.assign_channels()
            planner.apply_channel_refs(req)
            stream_iter = await self._dispatcher.dispatch_request_stream(req)
            worker = self._dispatcher.get_worker_for(req.id)
            if worker is not None:
                bridge_tasks = planner.start_bridges(self._dispatcher, worker, req.id)

        elif planner.streams and not expects_async_input:
            # Unary fallback: consume first elements and pass plain args
            await planner.apply_unary_fallback(req)
            stream_iter = await self._dispatcher.dispatch_request_stream(req)

        else:
            # No async inputs at all
            stream_iter = await self._dispatcher.dispatch_request_stream(req)

        # 4) Yield server->client chunks, ensuring cleanup
        try:
            async for chunk in stream_iter:
                yield chunk
        finally:
            await _safe_aclose(stream_iter)
            # Cancel & await bridges so they don't hang waiting for more client items
            for t in bridge_tasks:
                t.cancel()
            if bridge_tasks:
                await asyncio.gather(*bridge_tasks, return_exceptions=True)