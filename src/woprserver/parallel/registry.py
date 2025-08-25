from __future__ import annotations

import asyncio
import os
import signal
import time
from collections import Counter as RefCounter
from contextlib import suppress
from typing import Optional

from mlserver.env import Environment, compute_hash_of_file, compute_hash_of_string
from mlserver.model import MLModel
from mlserver.registry import model_initialiser
from mlserver.settings import ModelSettings, Settings
from mlserver.utils import to_absolute_path

from ..logging import get_logger
from .errors import EnvironmentNotFound
from .pool import InferencePool, InferencePoolHook

logger = get_logger()

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------

ENV_HASH_ATTR = "__env_hash__"


def _set_environment_hash(model: MLModel, env_hash: Optional[str]) -> None:
    """
    Attach (or remove) the environment hash used to load `model`.
    """
    if env_hash is None:
        if hasattr(model, ENV_HASH_ATTR):
            delattr(model, ENV_HASH_ATTR)
        return
    setattr(model, ENV_HASH_ATTR, env_hash)


def _get_environment_hash(model: MLModel) -> Optional[str]:
    return getattr(model, ENV_HASH_ATTR, None)


def _get_env_tarball(model: MLModel) -> Optional[str]:
    """
    Return absolute path to the configured tarball, if any.
    """
    model_settings = model.settings
    if model_settings.parameters is None:
        return None
    env_tarball = model_settings.parameters.environment_tarball
    if env_tarball is None:
        return None
    return to_absolute_path(model_settings, env_tarball)


def _append_gid_environment_hash(env_hash: str, inference_pool_gid: Optional[str] = None) -> str:
    """
    If a GID is supplied, combine it with the base env hash to produce an isolated key.
    """
    return f"{env_hash}-{inference_pool_gid}"


# -----------------------------------------------------------------------------
# Metrics (best-effort; no-ops if prometheus isn't available)
# -----------------------------------------------------------------------------

_ENABLE_METRICS = os.getenv("POOL_REGISTRY_METRICS", "1") not in ("0", "false", "False", "")


class _NoOpMetric:
    def labels(self, *_a, **_k):
        return self

    def inc(self, *_a, **_k):
        pass

    def dec(self, *_a, **_k):
        pass

    def set(self, *_a, **_k):
        pass

    def observe(self, *_a, **_k):
        pass


try:  # pragma: no cover - optional
    if _ENABLE_METRICS:
        from prometheus_client import REGISTRY as PROM_REGISTRY  # type: ignore
        from prometheus_client import Counter as PromCounter  # type: ignore
        from prometheus_client import Gauge as PromGauge  # type: ignore
        from prometheus_client import Histogram as PromHistogram  # type: ignore
    else:
        raise ImportError
except Exception:  # pragma: no cover
    PROM_REGISTRY = None  # type: ignore
    PromCounter = PromGauge = PromHistogram = None  # type: ignore


class _RegistryMetrics:
    def __init__(self) -> None:
        if not _ENABLE_METRICS or PromCounter is None:
            self.pools_created_total = _NoOpMetric()
            self.pools_active = _NoOpMetric()
            self.models_loaded_total = _NoOpMetric()
            self.models_unloaded_total = _NoOpMetric()
            self.sigchld_events_total = _NoOpMetric()
            self.worker_stops_total = _NoOpMetric()
            self.pool_close_latency_seconds = _NoOpMetric()
            return

        self.pools_created_total = PromCounter(
            "inference_pools_created_total",
            "Total number of inference pools created.",
            ["source"],  # tarball | existing_env | default_gid
            registry=PROM_REGISTRY,
        )
        self.pools_active = PromGauge(
            "inference_pools_active",
            "Current number of active (non-default) inference pools.",
            registry=PROM_REGISTRY,
        )
        self.models_loaded_total = PromCounter(
            "models_loaded_total",
            "Total models loaded via registry.",
            ["pool"],  # env key or "default"
            registry=PROM_REGISTRY,
        )
        self.models_unloaded_total = PromCounter(
            "models_unloaded_total",
            "Total models unloaded via registry.",
            ["pool"],
            registry=PROM_REGISTRY,
        )
        self.sigchld_events_total = PromCounter(
            "sigchld_events_total",
            "Number of SIGCHLD events handled; may include multiple children per event.",
            registry=PROM_REGISTRY,
        )
        self.worker_stops_total = PromCounter(
            "worker_stops_total",
            "Number of worker stop events processed.",
            registry=PROM_REGISTRY,
        )
        self.pool_close_latency_seconds = PromHistogram(
            "pool_close_latency_seconds",
            "Time to close an inference pool.",
            ["pool"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
            registry=PROM_REGISTRY,
        )


_METRICS = _RegistryMetrics()


# -----------------------------------------------------------------------------
# InferencePoolRegistry
# -----------------------------------------------------------------------------

class InferencePoolRegistry:
    """
    Tracks inference pools. Each pool generally represents an isolated Python environment.
    Pools are keyed by:
      - existing env: hash(path) [+ gid]
      - tarball env:  hash(tarball) [+ gid]
      - no custom env but grouped: gid
      - default: kept in _default_pool (not present in _pools)
    """

    def __init__(self, settings: Settings, on_worker_stop: list[InferencePoolHook] | None = None):
        self._settings = settings
        self._on_worker_stop = on_worker_stop or []
        self._default_pool = InferencePool(self._settings, on_worker_stop=self._on_worker_stop)

        # Mapping of key -> pool (key is env-hash[+gid] or gid)
        self._pools: dict[str, InferencePool] = {}

        # Reference count of models using a given keyed pool
        self._pool_refcount: RefCounter[str] = RefCounter()

        # Per-key creation locks to avoid races
        self._pool_locks: dict[str, asyncio.Lock] = {}

        # Registry-wide lock for fast mutations (pop on unload)
        self._lock = asyncio.Lock()

        # For SIGCHLD processing
        self._original_sigchld_handler = None

        os.makedirs(self._settings.environments_dir, exist_ok=True)

        # Register SIGCHLD via the event loop when available; fall back otherwise
        try:
            if hasattr(signal, "SIGCHLD"):
                loop = asyncio.get_running_loop()
                self._original_sigchld_handler = signal.getsignal(signal.SIGCHLD)
                loop.add_signal_handler(
                    signal.SIGCHLD, lambda: asyncio.create_task(self._handle_worker_stop())
                )
                logger.debug("SIGCHLD handler registered via event loop.")
        except (RuntimeError, AttributeError, NotImplementedError):
            try:
                if hasattr(signal, "SIGCHLD"):
                    self._original_sigchld_handler = signal.getsignal(signal.SIGCHLD)
                    signal.signal(
                        signal.SIGCHLD, lambda *_: asyncio.create_task(self._handle_worker_stop())
                    )
                    logger.debug("SIGCHLD handler registered via signal.signal().")
            except Exception:  # pragma: no cover
                logger.debug("SIGCHLD handler not installed; platform may not support it.")

        _METRICS.pools_active.set(0)

    # ------------------------------------------------------------------ utils

    def _lock_for(self, key: str) -> asyncio.Lock:
        lk = self._pool_locks.get(key)
        if lk is None:
            lk = self._pool_locks[key] = asyncio.Lock()
        return lk

    # ---------------------------------------------------------------- creators

    async def _get_or_create(self, model: MLModel) -> InferencePool:
        """
        Return the pool for this model, creating it if necessary, and increment refcount.
        """
        params = model.settings.parameters
        if params is not None and params.environment_path:
            pool, key = await self._get_or_create_with_existing_env(
                params.environment_path, params.inference_pool_gid
            )
            # Increment refcount for every successful *logical* attach
            async with self._lock:
                self._pool_refcount[key] += 1
            return pool

        return await self._get_or_create_with_tarball_or_gid(model)

    async def _get_or_create_with_existing_env(
        self,
        environment_path: str,
        inference_pool_gid: Optional[str],
    ) -> tuple[InferencePool, str]:
        """
        Create or return the pool keyed by an existing Python environment path (+ optional gid).
        Returns (pool, key).
        """
        expanded_path = os.path.abspath(os.path.expanduser(os.path.expandvars(environment_path)))
        logger.info("Using existing environment: %s", expanded_path)

        env_hash = await compute_hash_of_string(expanded_path)
        key = env_hash
        if inference_pool_gid:
            key = _append_gid_environment_hash(env_hash, inference_pool_gid)

        async with self._lock_for(key):
            pool = self._pools.get(key)
            if pool:
                return pool, key

            env = Environment(env_path=expanded_path, env_hash=env_hash, delete_env=False)
            pool = InferencePool(self._settings, env=env, on_worker_stop=self._on_worker_stop)
            self._pools[key] = pool

            _METRICS.pools_created_total.labels(source="existing_env").inc()
            _METRICS.pools_active.set(len(self._pools))
            logger.info("Created inference pool for env '%s'.", key)
            return pool, key

    async def _get_or_create_with_tarball_or_gid(self, model: MLModel) -> InferencePool:
        """
        Create or return the pool for models using a tarball as their Python environment.
        If no tarball is configured:
         - with GID: returns/creates a GID-keyed pool
         - without GID: returns default pool
        Also increments the refcount for the chosen pool (except the default pool).
        """
        env_tarball = _get_env_tarball(model)
        gid = model.settings.parameters.inference_pool_gid if model.settings.parameters else None

        # No custom env tarball:
        if not env_tarball:
            if not gid:
                return self._default_pool
            key = gid  # GID-isolated pool
            async with self._lock_for(key):
                pool = self._pools.get(key)
                if pool is None:
                    pool = InferencePool(self._settings, on_worker_stop=self._on_worker_stop)
                    self._pools[key] = pool
                    _METRICS.pools_created_total.labels(source="default_gid").inc()
                    _METRICS.pools_active.set(len(self._pools))
                    logger.info("Created inference pool for gid '%s'.", gid)
            async with self._lock:
                self._pool_refcount[key] += 1
            return pool

        # Custom env tarball:
        env_hash = await compute_hash_of_file(env_tarball)
        key = env_hash
        if gid:
            key = _append_gid_environment_hash(env_hash, gid)

        async with self._lock_for(key):
            pool = self._pools.get(key)
            if pool is None:
                env = await self._extract_tarball(env_hash, env_tarball)
                pool = InferencePool(self._settings, env=env, on_worker_stop=self._on_worker_stop)
                self._pools[key] = pool
                _METRICS.pools_created_total.labels(source="tarball").inc()
                _METRICS.pools_active.set(len(self._pools))
                logger.info("Created inference pool for env '%s'.", key)

        async with self._lock:
            self._pool_refcount[key] += 1

        return pool

    async def _extract_tarball(self, env_hash: str, env_tarball: str) -> Environment:
        """
        Extract the tarball into a deterministic directory. Idempotent:
        if the directory exists, reuse it.
        """
        env_path = self._get_env_path(env_hash)
        try:
            os.makedirs(env_path, exist_ok=False)
        except FileExistsError:
            # Already extracted by this or another process.
            return Environment(env_path, env_hash)

        # Optional simple lock file to reduce cross-process races; best-effort.
        lock_path = os.path.join(env_path, ".extract.lock")
        try:
            with open(lock_path, "w"):
                return await Environment.from_tarball(env_tarball, env_path, env_hash)
        finally:
            with suppress(FileNotFoundError):
                os.remove(lock_path)

    def _get_env_path(self, env_hash: str) -> str:
        return os.path.join(self._settings.environments_dir, env_hash)

    # ---------------------------------------------------------------- lookups

    async def _find(self, model: MLModel) -> InferencePool:
        """
        Resolve the pool where `model` is loaded.
        """
        env_hash = _get_environment_hash(model)
        gid = model.settings.parameters.inference_pool_gid if model.settings.parameters else None

        if not env_hash:
            if not gid:
                return self._default_pool
            pool = self._pools.get(gid)
            if not pool:
                raise EnvironmentNotFound(model, gid)
            return pool

        pool = self._pools.get(env_hash)
        if not pool:
            raise EnvironmentNotFound(model, env_hash)
        return pool

    # ---------------------------------------------------------------- policy

    def _should_load_model(self, model_settings: ModelSettings) -> bool:
        if model_settings.parallel_workers is not None:
            logger.warning(
                "DEPRECATED!! The `parallel_workers` setting at the model-level "
                "has been moved to server-level settings and will be removed in MLServer 1.2.0. "
                "Current server `parallel_workers`: '%s'.",
                self._settings.parallel_workers,
            )
            if model_settings.parallel_workers <= 0:
                return False

        return bool(self._settings.parallel_workers)

    # ---------------------------------------------------------------- factory

    def model_initialiser(self, model_settings: ModelSettings) -> MLModel:
        """
        Used by the ModelRegistry to construct MLModel instances.
        If parallel inference should not be used, instantiate the model as normal.
        If the model uses a custom environment, return a lightweight placeholder
        to avoid importing the model class in the main process.
        """
        if not self._should_load_model(model_settings):
            return model_initialiser(model_settings)

        parameters = model_settings.parameters
        if not parameters or not parameters.environment_tarball:
            return model_initialiser(model_settings)

        # Placeholder prevents importing user code in the parent process
        return MLModel(model_settings)

    # ---------------------------------------------------------------- actions

    async def load_model(self, model: MLModel) -> MLModel:
        """
        Load `model` into the appropriate pool (creating one if necessary).
        """
        if not self._should_load_model(model.settings):
            return model

        pool = await self._get_or_create(model)
        loaded = await pool.load_model(model)
        # Persist the pool's env hash (may be None for gid-only/default)
        _set_environment_hash(loaded, pool.env_hash)

        label = pool.env_hash or "default"
        _METRICS.models_loaded_total.labels(pool=label).inc()
        return loaded

    async def reload_model(self, old_model: MLModel, new_model: MLModel) -> MLModel:
        """
        Reload `old_model` with `new_model`. If the environment/pool changed, the
        old pool is decremented and possibly torn down.
        """
        if not self._should_load_model(new_model.settings):
            return new_model

        old_key = _get_environment_hash(old_model) or (
            old_model.settings.parameters.inference_pool_gid if old_model.settings.parameters else None
        )

        new_pool = await self._get_or_create(new_model)
        loaded = await new_pool.reload_model(old_model, new_model)
        _set_environment_hash(loaded, new_pool.env_hash)

        # If we moved pools, unload the old model (which will decrement & maybe close)
        new_key = new_pool.env_hash or (
            new_model.settings.parameters.inference_pool_gid if new_model.settings.parameters else None
        )
        if old_key != new_key:
            await self.unload_model(old_model)

        label = new_pool.env_hash or "default"
        _METRICS.models_loaded_total.labels(pool=label).inc()
        return loaded

    async def unload_model(self, model: MLModel) -> MLModel:
        """
        Unload `model` from its pool. If the pool refcount reaches zero, remove
        the key from `_pools` **immediately** and then close the pool (best-effort).
        This guarantees the key disappears quickly for tests/liveness checks.
        """
        if not self._should_load_model(model.settings):
            return model

        pool = await self._find(model)
        unloaded = await pool.unload_model(model)

        # Determine the registry key for this pool (env_hash preferred; else gid)
        key = pool.env_hash or (
            model.settings.parameters.inference_pool_gid if model.settings.parameters else None
        )

        label = pool.env_hash or "default"
        _METRICS.models_unloaded_total.labels(pool=label).inc()

        # Default pool is never keyed in _pools
        if key is None:
            return unloaded

        # Decide whether to tear the keyed pool down
        to_close: Optional[InferencePool] = None
        async with self._lock:
            self._pool_refcount[key] -= 1
            if self._pool_refcount[key] <= 0:
                self._pool_refcount.pop(key, None)
                # Pop the pool entry **immediately** so external checks see it gone
                to_close = self._pools.pop(key, None)
                _METRICS.pools_active.set(len(self._pools))
                # Remove the per-key creation lock as well (not strictly required)
                self._pool_locks.pop(key, None)

        if to_close is not None:
            # Close the pool best-effort, but the key is already gone
            try:
                await asyncio.wait_for(self._close_pool_instance(to_close), timeout=2.5)
            except Exception as e:  # pragma: no cover - non-fatal cleanup
                logger.warning("Pool '%s' close() failed or timed out: %s", key, e)

        return unloaded

    # ---------------------------------------------------------------- teardown

    async def _handle_worker_stop(self) -> None:
        """
        Handle SIGCHLD: drain all available dead children and notify pools.
        """
        if PromCounter is not None:
            _METRICS.sigchld_events_total.inc()

        processed = 0
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break  # no child processes
            if pid == 0:
                break

            exit_code = os.waitstatus_to_exitcode(status)
            if exit_code == 0:
                continue  # graceful exit

            processed += 1
            if PromCounter is not None:
                _METRICS.worker_stops_total.inc()
            await self._default_pool.on_worker_stop(pid, exit_code)
            # Notify all keyed pools
            await asyncio.gather(*(pool.on_worker_stop(pid, exit_code) for pool in list(self._pools.values())))

        if processed:
            logger.debug("Handled %d worker stop event(s).", processed)

    async def close(self) -> None:
        """
        Close all pools and restore signal handlers.
        """
        # Restore signal handler if we installed it
        if hasattr(signal, "SIGCHLD") and self._original_sigchld_handler is not None:
            with suppress(Exception):
                signal.signal(signal.SIGCHLD, self._original_sigchld_handler)

        # Close keyed pools first
        for key, pool in list(self._pools.items()):
            with suppress(Exception):
                await self._close_pool_instance(pool)
            self._pools.pop(key, None)

        _METRICS.pools_active.set(len(self._pools))

        # Close default pool
        with suppress(Exception):
            await self._close_pool_instance(self._default_pool)

        # Clear bookkeeping
        self._pool_refcount.clear()
        self._pool_locks.clear()

    async def _close_pool(self, key: Optional[str] = None) -> None:
        """
        Close a pool by key (or default pool if key is None).
        Note: Prefer using unload_model to manage lifecycle automatically.
        """
        pool = self._default_pool if key is None else self._pools.get(key)
        if pool is None:
            return
        await self._close_pool_instance(pool)
        if key is not None:
            self._pools.pop(key, None)
            _METRICS.pools_active.set(len(self._pools))

    async def _close_pool_instance(self, pool: InferencePool) -> None:
        """
        Close a given InferencePool instance and record latency.
        """
        label = pool.env_hash or "default"
        logger.info("Waiting for shutdown of %s...", pool.name)
        t0 = time.perf_counter()
        await pool.close()
        dt = max(time.perf_counter() - t0, 0.0)
        _METRICS.pool_close_latency_seconds.labels(pool=label).observe(dt)
        logger.info("Shutdown of %s complete in %.3fs", pool.name, dt)
        # Help GC any Environment
        with suppress(Exception):
            pool._env = None  # pylint: disable=protected-access
