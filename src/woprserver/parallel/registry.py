import asyncio
import os
import signal
import time

from mlserver.env import Environment, compute_hash_of_file, compute_hash_of_string
from mlserver.model import MLModel
from mlserver.registry import model_initialiser
from mlserver.settings import ModelSettings, Settings
from mlserver.utils import to_absolute_path

from .errors import EnvironmentNotFound
from ..logging import get_logger 
logger = get_logger()

from .pool import InferencePool, InferencePoolHook

# -----------------------------------------------------------------------------
# Constants & helpers
# -----------------------------------------------------------------------------

ENV_HASH_ATTR = "__env_hash__"


def _set_environment_hash(model: MLModel, env_hash: str | None) -> None:
    if env_hash is None:
        if hasattr(model, ENV_HASH_ATTR):
            delattr(model, ENV_HASH_ATTR)
        return
    setattr(model, ENV_HASH_ATTR, env_hash)

def _get_environment_hash(model: MLModel) -> str | None:
    return getattr(model, ENV_HASH_ATTR, None)


def _get_env_tarball(model: MLModel) -> str | None:
    model_settings = model.settings
    if model_settings.parameters is None:
        return None
    env_tarball = model_settings.parameters.environment_tarball
    if env_tarball is None:
        return None
    return to_absolute_path(model_settings, env_tarball)


def _append_gid_environment_hash(env_hash: str, inference_pool_gid: str | None = None) -> str:
    return f"{env_hash}-{inference_pool_gid}"

_ENABLE_METRICS = os.getenv("POOL_REGISTRY_METRICS", "1") not in ("0", "false", "False", "")

class _NoOpMetric:
    def labels(self, *_a, **_k): return self
    def inc(self, *_a, **_k): pass
    def dec(self, *_a, **_k): pass
    def set(self, *_a, **_k): pass
    def observe(self, *_a, **_k): pass

try:
    if _ENABLE_METRICS:
        from prometheus_client import REGISTRY as PROM_REGISTRY  # type: ignore
        from prometheus_client import Counter, Gauge, Histogram
    else:
        raise ImportError
except Exception:  # pragma: no cover
    Counter = Gauge = Histogram = None  # type: ignore
    PROM_REGISTRY = None  # type: ignore


class _RegistryMetrics:
    def __init__(self) -> None:
        if not _ENABLE_METRICS or Counter is None:
            self.pools_created_total = _NoOpMetric()
            self.pools_active = _NoOpMetric()
            self.models_loaded_total = _NoOpMetric()
            self.models_unloaded_total = _NoOpMetric()
            self.sigchld_events_total = _NoOpMetric()
            self.worker_stops_total = _NoOpMetric()
            self.pool_close_latency_seconds = _NoOpMetric()
            return

        self.pools_created_total = Counter(
            "inference_pools_created_total",
            "Total number of inference pools created.",
            ["source"],  # tarball | existing_env | default_gid
            registry=PROM_REGISTRY,
        )
        self.pools_active = Gauge(
            "inference_pools_active",
            "Current number of active (non-default) inference pools.",
            registry=PROM_REGISTRY,
        )
        self.models_loaded_total = Counter(
            "models_loaded_total",
            "Total models loaded via registry.",
            ["pool"],  # env hash or "default" / gid
            registry=PROM_REGISTRY,
        )
        self.models_unloaded_total = Counter(
            "models_unloaded_total",
            "Total models unloaded via registry.",
            ["pool"],
            registry=PROM_REGISTRY,
        )
        self.sigchld_events_total = Counter(
            "sigchld_events_total",
            "Number of SIGCHLD events handled; may include multiple children per event.",
            registry=PROM_REGISTRY,
        )
        self.worker_stops_total = Counter(
            "worker_stops_total",
            "Number of worker stop events processed.",
            registry=PROM_REGISTRY,
        )
        self.pool_close_latency_seconds = Histogram(
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
      - default: not stored in _pools (kept in _default_pool)
    """

    def __init__(self, settings: Settings, on_worker_stop: list[InferencePoolHook] | None = None):
        self._settings = settings
        self._on_worker_stop = on_worker_stop or []
        self._default_pool = InferencePool(self._settings, on_worker_stop=self._on_worker_stop)
        self._pools: dict[str, InferencePool] = {}
        self._pool_locks: dict[str, asyncio.Lock] = {}  # per-key creation locks
        self._original_sigchld_handler = None

        os.makedirs(self._settings.environments_dir, exist_ok=True)

        # Register SIGCHLD using the event loop when possible; fall back gracefully if unavailable.
        try:
            if hasattr(signal, "SIGCHLD"):
                loop = asyncio.get_running_loop()
                self._original_sigchld_handler = signal.getsignal(signal.SIGCHLD)
                loop.add_signal_handler(signal.SIGCHLD, lambda: asyncio.create_task(self._handle_worker_stop()))
                logger.debug("SIGCHLD handler registered via event loop.")
        except (RuntimeError, AttributeError, NotImplementedError):
            # No running loop or unsupported platform; try process-level handler (best effort)
            try:
                if hasattr(signal, "SIGCHLD"):
                    self._original_sigchld_handler = signal.getsignal(signal.SIGCHLD)
                    signal.signal(signal.SIGCHLD, lambda *_: asyncio.create_task(self._handle_worker_stop()))
                    logger.debug("SIGCHLD handler registered via signal.signal().")
            except Exception:
                logger.debug("SIGCHLD handler not installed; platform may not support it.")

        # Initialize pools_active gauge
        _METRICS.pools_active.set(0)

    # ------------------------------------------------------------------ utils

    def _lock_for(self, key: str) -> asyncio.Lock:
        lk = self._pool_locks.get(key)
        if lk is None:
            lk = self._pool_locks[key] = asyncio.Lock()
        return lk

    # ---------------------------------------------------------------- creators

    async def _get_or_create(self, model: MLModel) -> InferencePool:
        params = model.settings.parameters
        if params is not None and params.environment_path:
            return await self._get_or_create_with_existing_env(
                params.environment_path, params.inference_pool_gid
            )
        return await self._get_or_create_with_tarball(model)

    async def _get_or_create_with_existing_env(
        self,
        environment_path: str,
        inference_pool_gid: str | None,
    ) -> InferencePool:
        """
        Create or return the pool keyed by an existing Python environment path (+ optional gid).
        """
        expanded_path = os.path.abspath(os.path.expanduser(os.path.expandvars(environment_path)))
        logger.info(f"Using existing environment: {expanded_path}")
        env_hash = await compute_hash_of_string(expanded_path)
        if inference_pool_gid:
            env_hash = _append_gid_environment_hash(env_hash, inference_pool_gid)

        async with self._lock_for(env_hash):
            pool = self._pools.get(env_hash)
            if pool:
                return pool

            env = Environment(env_path=expanded_path, env_hash=env_hash, delete_env=False)
            pool = InferencePool(self._settings, env=env, on_worker_stop=self._on_worker_stop)
            self._pools[env_hash] = pool

            _METRICS.pools_created_total.labels(source="existing_env").inc()
            _METRICS.pools_active.set(len(self._pools))
            logger.info(f"Created inference pool for env '{env_hash}'.")
            return pool

    async def _get_or_create_with_tarball(self, model: MLModel) -> InferencePool:
        """
        Create or return the pool for models using a tarball as their Python environment.
        If no tarball is configured:
         - with GID: returns/creates a GID-keyed pool
         - without GID: returns default pool
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
                if pool:
                    return pool
                pool = InferencePool(self._settings, on_worker_stop=self._on_worker_stop)
                self._pools[key] = pool
                _METRICS.pools_created_total.labels(source="default_gid").inc()
                _METRICS.pools_active.set(len(self._pools))
                logger.info(f"Created inference pool for gid '{gid}'.")
                return pool

        # Custom env tarball:
        env_hash = await compute_hash_of_file(env_tarball)
        if gid:
            env_hash = _append_gid_environment_hash(env_hash, gid)

        async with self._lock_for(env_hash):
            pool = self._pools.get(env_hash)
            if pool:
                return pool

            env = await self._extract_tarball(env_hash, env_tarball)
            pool = InferencePool(self._settings, env=env, on_worker_stop=self._on_worker_stop)
            self._pools[env_hash] = pool

            _METRICS.pools_created_total.labels(source="tarball").inc()
            _METRICS.pools_active.set(len(self._pools))
            logger.info(f"Created inference pool for env '{env_hash}'.")
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
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

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

        return MLModel(model_settings)

    # ---------------------------------------------------------------- actions

    async def load_model(self, model: MLModel) -> MLModel:
        if not self._should_load_model(model.settings):
            return model

        pool = await self._get_or_create(model)
        loaded = await pool.load_model(model)
        _set_environment_hash(loaded, pool.env_hash)

        label = pool.env_hash or "default"
        _METRICS.models_loaded_total.labels(pool=label).inc()
        return loaded

    async def reload_model(self, old_model: MLModel, new_model: MLModel) -> MLModel:
        if not self._should_load_model(new_model.settings):
            return new_model

        old_hash = _get_environment_hash(old_model)
        new_pool = await self._get_or_create(new_model)

        loaded = await new_pool.reload_model(old_model, new_model)
        _set_environment_hash(loaded, new_pool.env_hash)

        if old_hash != new_pool.env_hash:
            # Environment changed; unload old one from its pool
            await self.unload_model(old_model)

        label = new_pool.env_hash or "default"
        _METRICS.models_loaded_total.labels(pool=label).inc()
        return loaded

    async def unload_model(self, model: MLModel) -> MLModel:
        if not self._should_load_model(model.settings):
            return model

        pool = await self._find(model)
        unloaded = await pool.unload_model(model)

        label = pool.env_hash or "default"
        _METRICS.models_unloaded_total.labels(pool=label).inc()

        if pool is not self._default_pool and pool.empty():
            logger.info(f"Inference pool with hash '{pool.env_hash}' is now empty")
            await self._close_pool(pool.env_hash)

        return unloaded

    # ---------------------------------------------------------------- teardown

    async def _handle_worker_stop(self) -> None:
        """
        Handle SIGCHLD: drain all available dead children and notify pools.
        """
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
            _METRICS.worker_stops_total.inc()
            await self._default_pool.on_worker_stop(pid, exit_code)
            await asyncio.gather(*(pool.on_worker_stop(pid, exit_code) for pool in self._pools.values()))

        if processed:
            logger.debug("Handled %d worker stop event(s).", processed)

    async def close(self) -> None:
        # Restore signal handler if we installed it
        if hasattr(signal, "SIGCHLD") and self._original_sigchld_handler is not None:
            try:
                signal.signal(signal.SIGCHLD, self._original_sigchld_handler)
            except Exception:
                pass

        # Close default and all keyed pools
        await asyncio.gather(
            self._close_pool(None),
            *(self._close_pool(env_hash) for env_hash in list(self._pools)),
        )
        self._pools.clear()
        _METRICS.pools_active.set(len(self._pools))

    async def _close_pool(self, env_hash: str | None = None) -> None:
        pool = self._default_pool if env_hash is None else self._pools.get(env_hash)
        if pool is None:
            return

        label = pool.env_hash or "default"
        logger.info(f"Waiting for shutdown of {pool.name}...")
        t0 = time.perf_counter()
        await pool.close()
        dt = max(time.perf_counter() - t0, 0.0)
        _METRICS.pool_close_latency_seconds.labels(pool=label).observe(dt)
        logger.info(f"Shutdown of {pool.name} complete in {dt:.3f}s")

        if env_hash:
            # Force calling __del__ on `Environment` to clean up
            try:
                self._pools[env_hash]._env = None  # pylint: disable=protected-access
            except Exception:
                pass
            self._pools.pop(env_hash, None)
            _METRICS.pools_active.set(len(self._pools))
