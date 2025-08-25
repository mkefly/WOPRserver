"""
Dispatcher and AsyncResponses
=============================

This module provides two tightly‑coupled components used to fan‑out requests to a
pool of model "workers" and to fan‑in their responses back to the caller:

* ``AsyncResponses`` — book‑keeping for in‑flight requests. It indexes each
  request ``id`` to the worker process handling it, exposes a ``Future`` that
  resolves when a response arrives, maintains per‑request stream queues for
  server‑sent chunks, and keeps a Prometheus ``Gauge`` of the number of
  concurrent requests.

* ``Dispatcher`` — the coordination layer. It implements a simple
  round‑robin scheduler over available workers, forwards request/update messages
  to workers, bridges async input streams into worker channels, and runs a
  background loop that consumes the shared inter‑process ``responses`` queue and
  routes each message to the appropriate waiter or stream.

Why this structure?
-------------------

**Concurrency model.** The design combines ``asyncio`` for cooperative
concurrency on the main thread (futures, async generators, queues) with a
``ThreadPoolExecutor`` to blockingly read from a multi‑process ``Queue`` without
blocking the event loop. The result is a responsive event loop even when the
underlying IPC is blocking.

**Scheduling policy.** A minimal round‑robin policy ensures fair distribution of
requests across workers. It is stateless and does not consider load or latency,
which keeps it predictable and easy to reason about.

**Backpressure.** For streaming responses the per‑request asyncio queue is
created with a bounded ``maxsize`` (default 16) to cap memory usage. If the
queue is full, the newest chunk replaces the oldest (drop‑oldest policy), which
keeps the stream moving under pressure while avoiding a hard failure.

**Metrics.** A Prometheus ``Gauge`` named ``parallel_request_queue_size`` tracks
how many requests are currently in flight. This is updated whenever requests are
scheduled and when they complete/cancel.

References (recommended background)
-----------------------------------
- Python ``asyncio``: concepts of Futures, Tasks, Queues, and event loop.
- ``concurrent.futures.ThreadPoolExecutor`` for off‑loop blocking work.
- Multiprocessing ``Queue`` for inter‑process messaging.
- Prometheus client Python library: ``Gauge`` metrics.
- Producer/consumer and fan‑out/fan‑in patterns in concurrent systems.

Notes on imports & structure
----------------------------
The code below keeps functionality as‑is and focuses on documentation and
explanatory comments. 
"""

import asyncio
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from itertools import cycle
from multiprocessing import Queue
from typing import Any

from mlserver.utils import generate_uuid, schedule_with_callback

from ..logging import get_logger
from .messages import (
    ModelRequestMessage,
    ModelResponseMessage,
    ModelStreamChunkMessage,
    ModelStreamEndMessage,
    ModelStreamInputChunk,
    ModelStreamInputEnd,
    ModelUpdateMessage,
)
from .responses import AsyncResponses
from .utils import END_OF_QUEUE, cancel_task
from .worker import Worker  # for runtime type

logger = get_logger()
class Dispatcher:
    """Fan‑out requests to workers; fan‑in responses back to callers.

    Overview
    --------
    ``Dispatcher`` owns the set of *currently available* ``Worker`` instances
    (each with a pid) and a shared inter‑process ``responses`` queue. It offers
    high‑level APIs to:

    * dispatch unary requests (request → single response),
    * dispatch streaming requests (request → async iterator of chunks),
    * broadcast model updates to all workers,
    * track worker lifecycle events (start, ready, stop), and
    * run a background loop that demultiplexes the responses queue and hands
      messages to ``AsyncResponses`` for completion/stream fan‑in.

    Key design choices
    ------------------
    * **Round‑robin selection** via ``itertools.cycle`` ensures even request
      distribution without inspecting worker load.
    * **IPC integration**: ``_process_responses`` calls ``Queue.get`` inside a
      ``ThreadPoolExecutor`` so the asyncio event loop stays unblocked.
    * **Failure propagation**: when a worker stops, all of its in‑flight
      requests are failed with a ``WorkerStop`` carrying the exit code.

    Threading & async boundaries
    ----------------------------
    * The dispatcher itself is used from the asyncio event loop.
    * The blocking queue read runs in a thread; completions are handed back to
      the loop using thread‑safe ``Future`` methods.

    Usage sketch
    ------------
    >>> response = await dispatcher.dispatch_request(request_msg)
    >>> async for chunk in await dispatcher.dispatch_request_stream(stream_msg):
    ...     process(chunk)

    Caution
    -------
    ``Dispatcher`` assumes that every request carries a unique ``id`` and that
    workers echo that id back in responses/chunks. Mis‑matched ids will result
    in dropped/ignored messages.
    """

    def __init__(self, workers: dict[int, Worker], responses: Queue):
        self._responses = responses
        self._workers = workers
        self._workers_round_robin = self._reset_round_robin()
        self._worker_starting_lock = asyncio.Lock()
        self._active = False
        self._process_responses_task = None
        self._executor = ThreadPoolExecutor()
        self._async_responses = AsyncResponses()
        self._bg_tasks: set[asyncio.Task] = set()

    def _reset_round_robin(self) -> Iterator[int]:
        """Rebuild the round‑robin iterator from current worker pids.

        Called whenever the set of workers changes (start/ready/stop).
        """
        worker_pids = list(self._workers.keys())
        self._workers_round_robin = cycle(worker_pids)
        return self._workers_round_robin

    async def on_worker_start(self, worker: Worker):
        """Register a worker that's in the process of starting.

        A lock ensures that concurrent starts don't race updates to the workers
        map. Once the worker is *ready* (able to accept requests), call
        ``on_worker_ready``.
        """
        async with self._worker_starting_lock:
            self._workers[worker.pid] = worker  # type: ignore

    def on_worker_ready(self, worker: Worker):
        """Mark a worker as ready to receive requests and refresh scheduling."""
        self._reset_round_robin()

    def on_worker_stop(self, worker: Worker, exit_code: int):
        """Handle an unexpected worker stop and fail its in‑flight requests."""
        pid = worker.pid
        if pid in self._workers:
            del self._workers[pid]
        self._reset_round_robin()
        self._async_responses.cancel(worker, exit_code)

    def start(self):
        """Start the background response processing loop.

        Uses ``schedule_with_callback`` so that if the loop crashes, the
        callback logs and restarts it for resilience.
        """
        self._active = True
        self._process_responses_task = schedule_with_callback(
            self._process_responses(), self._process_responses_cb
        )

    def _process_responses_cb(self, process_responses):
        """Callback for the response loop task: log and restart on crash."""
        try:
            process_responses.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception("Response processing loop crashed. Restarting the loop...")
            self.start()

    async def _process_responses(self):
        """Continuously consume the shared responses queue and route messages.

        * ``ModelStreamChunkMessage`` / ``ModelStreamEndMessage`` → delivered to
          the per‑request stream queue for consumers of ``dispatch_request_stream``.
        * ``ModelResponseMessage`` → completes the waiting ``Future`` for unary
          calls and for stream acknowledgements.
        """
        logger.debug("Starting response processing loop...")
        loop = asyncio.get_event_loop()
        while self._active:
            try:
                # If executor is gone, exit quietly
                if self._executor is None or getattr(self._executor, "_shutdown", False):
                    return

                response = await loop.run_in_executor(self._executor, self._responses.get)
            except RuntimeError as e:
                if "after shutdown" in str(e):
                    return  # normal shutdown
                raise
            if response is END_OF_QUEUE:
                return
            if isinstance(response, (ModelStreamChunkMessage, ModelStreamEndMessage)):
                self._async_responses.put_stream_message(response)
                continue
            self._async_responses.resolve(response)


    async def dispatch_request(self, request_message: ModelRequestMessage) -> ModelResponseMessage:
        """Send a unary request to the next worker and await its response.

        The request is first registered with ``AsyncResponses`` so that the
        response loop can resolve the appropriate ``Future`` when a matching
        message appears on the responses queue.
        """
        worker, _ = self._get_worker()
        # Register future BEFORE sending
        _ = self._async_responses.schedule_only(request_message, worker)
        worker.send_request(request_message)
        return await self._async_responses._wait(request_message.id)

    async def dispatch_request_stream(
        self, request_message: ModelRequestMessage
    ) -> AsyncIterator[Any]:
        """Send a streaming request and return an async iterator over chunks.

        The iterator yields chunks as ``ModelStreamChunkMessage.chunk`` values
        until a ``ModelStreamEndMessage`` is received. Any terminal exception is
        raised to the consumer.
        """
        worker, _ = self._get_worker()
        q = self._async_responses.create_stream_queue(request_message.id, maxsize=16)
        fut = self._async_responses.schedule_only(request_message, worker)
        worker.send_request(request_message)

        async def _observe_ack():
            """Background task to await the first ack/exception for the stream.

            Some protocols send an immediate ack as a ``ModelResponseMessage``
            even for streaming requests; awaiting ``fut`` here lets us surface
            early errors to the log if desired, without blocking the stream.
            """
            try:
                await fut
            except Exception:
                pass

        # RUF006-clean: keep a reference so linter is happy.
        # TODO: improve lifecycle mgmt (track errors, cancellation, etc.)
        task = asyncio.create_task(_observe_ack())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

        async def agen():
            try:
                while True:
                    msg = await q.get()
                    if isinstance(msg, ModelStreamChunkMessage):
                        yield msg.chunk
                    elif isinstance(msg, ModelStreamEndMessage):
                        if msg.exception:
                            raise msg.exception
                        break
            finally:
                self._async_responses.pop_stream_queue(request_message.id)

        return agen()

    async def _bridge_async_iterable_arg(
        self,
        worker: "Worker",
        req_id: str,
        payloads: AsyncIterator[Any],
        *,
        channel_id: str,
    ) -> None:
        """Bridge a caller‑provided async iterable into a worker input channel.

        For streaming *inputs* (e.g., incremental prompt parts, audio frames),
        this helper reads items from ``payloads`` and forwards them to the
        worker wrapped in ``ModelStreamInputChunk`` messages, followed by a
        terminal ``ModelStreamInputEnd``.
        """
        try:
            async for item in payloads:
                worker.send_request(
                    ModelStreamInputChunk(id=req_id, item=item, channel_id=channel_id)
                )
        finally:
            worker.send_request(ModelStreamInputEnd(id=req_id, channel_id=channel_id))

    def _get_worker(self) -> tuple[Worker, int]:
        """Select the next worker using round‑robin.

        Raises ``RuntimeError`` if there are currently no workers registered.
        """
        if not self._workers:
            raise RuntimeError("No available workers to dispatch request.")
        try:
            worker_pid = next(self._workers_round_robin)
        except StopIteration:
            self._reset_round_robin()
            worker_pid = next(self._workers_round_robin)
        return self._workers[worker_pid], worker_pid

    def get_worker_for(self, message_id: str) -> Worker | None:
        """Return the worker currently associated with a given message id.

        Uses ``AsyncResponses``' internal mapping from message id → worker pid.
        """
        pid = self._async_responses.get_worker_pid(message_id)
        if pid is None:
            return None
        return self._workers.get(pid)

    async def dispatch_update(
        self, model_update: ModelUpdateMessage
    ) -> list[ModelResponseMessage]:
        """Broadcast a model update to *all* workers and gather their replies."""
        async with self._worker_starting_lock:
            return await asyncio.gather(
                *[
                    self.dispatch_update_to_worker(worker, model_update)
                    for worker in self._workers.values()
                ]
            )

    async def dispatch_update_to_worker(self, worker: Worker, model_update: ModelUpdateMessage) -> ModelResponseMessage:
        """Send a cloned model update to a *specific* worker and await reply.

        The update is deep‑copied and its id regenerated to avoid id collisions
        across multiple workers.
        """
        worker_update = model_update.model_copy(deep=True)
        worker_update.id = generate_uuid()
        # Register future BEFORE sending
        _ = self._async_responses.schedule_only(worker_update, worker)
        worker.send_update(worker_update)
        return await self._async_responses._wait(worker_update.id)

    async def stop(self):
        """Shut down the background response loop and its executor.

        The executor is shut down with ``cancel_futures=True`` to interrupt any
        pending ``Queue.get`` calls, and the response loop task is cancelled via
        ``cancel_task`` helper.
        """
        self._active = False

        # Cancel the response task *before* shutting down executor
        if self._process_responses_task is not None:
            await cancel_task(self._process_responses_task)
            self._process_responses_task = None

        # Now it is safe to close executor
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None