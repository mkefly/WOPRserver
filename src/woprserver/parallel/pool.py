"""
Inference Pool — architecture & didactic documentation
======================================================

This module wires together **workers** (separate processes), a **dispatcher**
(event‑loop component that fans out requests and fans in responses), and a small
**registry** of currently loaded models. It exposes a high‑level
``InferencePool`` interface that:

* starts N workers (``Settings.parallel_workers``),
* manages lifecycle (auto‑restart on unexpected exit with backoff),
* propagates model load/unload updates to workers, and
* returns a ``ParallelModel`` wrapper that executes inference through the
  shared ``Dispatcher``.

Key concepts
------------
- **Worker process**: executes model methods in isolation and communicates via
  multiprocessing queues. See ``worker_with_in_depth_docs.py`` for details.
- **Dispatcher**: coordinates requests/responses across workers with a
  round‑robin policy and async plumbing.
- **WorkerRegistry**: tracks which models (name+version) are considered "loaded"
  from the pool’s perspective, so newly started workers can be brought up to
  speed by replaying ``Load`` updates.
- **Hooks**: optional async callbacks invoked when a worker stops (both
  unexpectedly and during shutdown). Errors in hooks are logged but never crash
  the pool.

Lifecycle overview
------------------
1. ``InferencePool.__init__`` spawns the initial workers and starts the
   ``Dispatcher``.
2. ``load_model`` / ``reload_model`` / ``unload_model`` broadcast model updates
   to all workers via the dispatcher and keep the ``WorkerRegistry`` in sync.
3. If a worker stops unexpectedly, ``on_worker_stop``:
   - logs and informs the dispatcher,
   - runs hooks in isolation, and
   - restarts a replacement worker with exponential backoff, replaying current
     models into it before marking it ready.
4. ``close`` performs an orderly shutdown: stop workers, terminate the shared
   responses queue, then stop the dispatcher.

Design choices
--------------
- **Process isolation** for models avoids GIL contention and provides crash
  containment.
- **Round‑robin load distribution** keeps scheduling fair and predictable.
- **Replay on join**: new workers receive the current model set to reach the
  same state as existing workers.
- **Backoff on restart**: reduces thrash during transient failures.

Notes on behavior
-----------------
This documentation adds comments and docstrings only; no functional changes are
made.
"""

from __future__ import annotations

import os

import signal
import asyncio
from collections.abc import Awaitable, Callable, Iterable
from contextlib import nullcontext
from multiprocessing import Queue
from typing import Optional

from mlserver.env import Environment
from mlserver.model import MLModel
from mlserver.settings import ModelSettings, Settings
from mlserver.types import InferenceRequest, InferenceResponse

from .dispatcher import Dispatcher
from ..logging import get_logger 
from .messages import (
    ModelResponseMessage,
    ModelUpdateMessage,
    ModelUpdateType,
)
from .model import ParallelModel
from .utils import configure_inference_pool, terminate_queue, make_queue
from .worker import Worker
from contextlib import suppress

logger = get_logger()

PredictMethod = Callable[[InferenceRequest], Awaitable[InferenceResponse]]
InferencePoolHook = Callable[[Worker], Awaitable[None]]


def _spawn_worker(
    settings: Settings,
    responses: Queue,
    env: Optional[Environment] = None,
) -> Worker:
    """Create and start a new ``Worker`` process.

    The optional ``Environment`` acts as a context manager, allowing the child
    process to inherit configured environment variables, mounts, etc.
    """
    with (env or nullcontext()):
        worker = Worker(settings, responses, env)
        worker.start()
    return worker


class WorkerRegistry:
    """Keep track of models logically loaded into the pool.

    Stored entries are keyed by ``f"{name}-{version or ''}"`` so that ``None``
    versions are handled consistently.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelSettings] = {}

    def _key(self, model_settings: ModelSettings) -> str:
        # Handle None-version consistently
        version = getattr(model_settings, "version", None)
        return f"{model_settings.name}-{version or ''}"

    def add(self, model_settings: ModelSettings) -> None:
        self._models[self._key(model_settings)] = model_settings

    def remove(self, model_settings: ModelSettings) -> None:
        self._models.pop(self._key(model_settings), None)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._models)

    @property
    def models(self) -> Iterable[ModelSettings]:  # pragma: no cover - trivial
        return self._models.values()


class InferencePool:
    """Own the fleet of workers and the dispatcher used for inference.

    Responsibilities
    ----------------
    * Start initial workers based on settings.
    * Maintain a shared responses queue read by the dispatcher.
    * Replace workers that die unexpectedly; replay current models into the
      new worker before marking it ready.
    * Provide model lifecycle operations that broadcast updates to all workers.
    * Create ``ParallelModel`` wrappers bound to the dispatcher for clients to
      call ``predict`` etc.
    """

    def __init__(
        self,
        settings: Settings,
        env: Optional[Environment] = None,
        on_worker_stop: Optional[list[InferencePoolHook]] = None,
    ):
        # Pool-wide configuration (e.g., process start method, timeouts)
        configure_inference_pool(settings)

        self._on_worker_stop = on_worker_stop or []
        self._env = env
        self._workers: dict[int, Worker] = {}
        self._worker_registry = WorkerRegistry()
        self._settings = settings
        self._responses: Queue[ModelResponseMessage] = make_queue()
        self._restart_lock = asyncio.Lock()
        self._closing = False

        # Start initial workers
        for _ in range(self._settings.parallel_workers):
            worker = _spawn_worker(self._settings, self._responses, self._env)
            self._workers[worker.pid] = worker  # type: ignore[arg-type]

        # One dispatcher per pool reading from the shared responses queue
        self._dispatcher = Dispatcher(self._workers, self._responses)
        self._dispatcher.start()

    # ------------------- properties -------------------
    @property
    def env_hash(self) -> Optional[str]:  # pragma: no cover - trivial
        """Expose a hash of the attached ``Environment`` (if provided)."""
        return getattr(self._env, "env_hash", None) if self._env else None

    @property
    def name(self) -> str:  # pragma: no cover - trivial
        """Human-friendly label for logs and diagnostics."""
        return (
            f"inference pool with hash '{self.env_hash}'"
            if self.env_hash
            else "default inference pool"
        )

    # ------------------- failure handling -------------------
    async def on_worker_stop(self, pid: int, exit_code: int):
        """Called when a worker terminates unexpectedly.

        * Removes the worker from the map and notifies the dispatcher so it can
          cancel any in-flight requests tied to that pid.
        * Invokes any registered hooks with errors isolated.
        * Triggers a replacement worker start with bounded exponential backoff.
        """
        # Idempotent if worker already purged
        worker = self._workers.pop(pid, None)
        if not worker:
            return

        logger.warning(
            "Worker with PID %s on %s stopped unexpectedly with exit code %s. "
            "Triggering worker restart...",
            pid,
            self.name,
            exit_code,
        )
        self._dispatcher.on_worker_stop(worker, exit_code)

        if self._closing:
            return  # we're shutting down; do not run hooks or restart

        # Run hooks; don't let exceptions crash the pool
        hook_results = await asyncio.gather(
            *[callback(worker) for callback in self._on_worker_stop],
            return_exceptions=True,
        )
        for res in hook_results:
            if isinstance(res, Exception):
                logger.exception("on_worker_stop hook raised", exc_info=res)

        # Restart safely with a lock + basic backoff if start fails transiently
        await self._start_worker_with_backoff()

    async def _start_worker_with_backoff(self) -> Worker:
        """Start a worker with exponential backoff and mutual exclusion.

        Prevents stampedes when multiple failures occur close together and keeps
        retries bounded (max 5 attempts, capped delay).
        """
        async with self._restart_lock:
            if self._closing:
                raise RuntimeError("Pool is closing; not starting new workers.")
            delay = 0.5
            attempts = 0
            last_exc: Exception | None = None
            while not self._closing and attempts < 5:
                attempts += 1
                try:
                    return await self._start_worker()
                except Exception as e:  # pragma: no cover - rare path
                    last_exc = e
                    logger.exception(
                        "Worker start failed (attempt %s). Retrying...", attempts
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 5.0)
            if last_exc:
                raise last_exc
            raise RuntimeError("Worker start aborted")

    async def _start_worker(self) -> Worker:
        """Spawn a new worker, register it, and replay current models into it."""
        worker = _spawn_worker(self._settings, self._responses, self._env)
        logger.info(
            "Starting new worker with PID %s on %s...", worker.pid, self.name
        )

        self._workers[worker.pid] = worker  # type: ignore[arg-type]
        await self._dispatcher.on_worker_start(worker)

        # Replay loaded models into the new worker
        await asyncio.gather(
            *[
                self._dispatcher.dispatch_update_to_worker(
                    worker,
                    ModelUpdateMessage(
                        update_type=ModelUpdateType.Load,
                        model_settings=model_settings,  # type: ignore[arg-type]
                    ),
                )
                for model_settings in self._worker_registry.models
            ]
        )

        self._dispatcher.on_worker_ready(worker)
        logger.info(
            "New worker with PID %s on %s is now ready.", worker.pid, self.name
        )
        return worker

    # ------------------- model lifecycle -------------------
    async def load_model(self, model: MLModel) -> ParallelModel:
        """Broadcast a ``Load`` update and return a dispatcher‑bound wrapper."""
        load_message = ModelUpdateMessage(
            update_type=ModelUpdateType.Load,
            model_settings=model.settings,  # type: ignore[arg-type]
        )
        await self._dispatcher.dispatch_update(load_message)
        self._worker_registry.add(model.settings)
        return ParallelModel(model, self._dispatcher)

    async def reload_model(self, old_model: MLModel, new_model: MLModel) -> ParallelModel:
        """Replace an existing model with a new one.

        If name+version are the same, issue an ``Unload`` for the old first to
        avoid duplicating the registry entry, then load the new model.
        """
        if (
            old_model.settings.name == new_model.settings.name
            and old_model.settings.version == new_model.settings.version
        ):
            unload_message = ModelUpdateMessage(
                update_type=ModelUpdateType.Unload,
                model_settings=old_model.settings,  # type: ignore[arg-type]
            )
            await self._dispatcher.dispatch_update(unload_message)
            self._worker_registry.remove(old_model.settings)

        self._worker_registry.add(new_model.settings)
        return await self.load_model(new_model)

    async def unload_model(self, model: MLModel) -> MLModel:
        """Broadcast an ``Unload`` update and drop the model from the registry."""
        unload_message = ModelUpdateMessage(
            update_type=ModelUpdateType.Unload,
            model_settings=model.settings,  # type: ignore[arg-type]
        )
        await self._dispatcher.dispatch_update(unload_message)
        self._worker_registry.remove(model.settings)
        # Return the underlying model (not a ParallelModel) to avoid implying it's still usable via the pool.
        return model

    def empty(self) -> bool:  # pragma: no cover - trivial
        """Return True when no models are currently tracked by the registry."""
        return len(self._worker_registry) == 0

    # ------------------- shutdown -------------------
    async def close(self) -> None:
        """Orderly shutdown of workers, queues, then dispatcher."""
        self._closing = True
        # 1) stop workers first so dispatcher won't receive more responses
        await self._close_workers()
        # 2) drain/terminate the responses queue then close it
        await terminate_queue(self._responses)
        with suppress(Exception):
            self._responses.close()
        # 3) stop dispatcher at the end
        await self._dispatcher.stop()


    async def _close_workers(self, grace: float = 5.0) -> None:
        workers = list(self._workers.values())
        if not workers:
            self._workers.clear()
            return

        # ask all workers to stop (they enqueue END_OF_QUEUE in .stop())
        await asyncio.gather(*(w.stop() for w in workers), return_exceptions=True)

        # first join pass
        for w in workers:
            with suppress(Exception):
                w.join(timeout=grace)

        # escalate
        for w in workers:
            if getattr(w, "is_alive", lambda: False)():
                with suppress(Exception):
                    w.terminate()
                with suppress(Exception):
                    w.join(timeout=grace)

        # final resort
        for w in workers:
            if getattr(w, "is_alive", lambda: False)():
                with suppress(Exception):
                    os.kill(w.pid, signal.SIGKILL)  # POSIX hard-kill
                    w.join(timeout=grace)

        self._workers.clear()
