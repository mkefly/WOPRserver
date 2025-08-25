# in tests/parallel/conftest.py
from __future__ import annotations

import asyncio
import contextlib
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys
import tarfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import uvloop
from filelock import FileLock, Timeout

from woprserver.server import WOPRserver

# Set multiprocessing *before* importing users of multiprocessing.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import subprocess

import prometheus_client
import pytest_asyncio
import yaml
from mlserver import types
from mlserver.codecs.string import StringCodec
from mlserver.env import Environment
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.metrics.registry import REGISTRY as METRICS_REGISTRY
from mlserver.metrics.registry import MetricsRegistry
from mlserver.model import MLModel
from mlserver.registry import MultiModelRegistry
from mlserver.repository import (
    DEFAULT_MODEL_SETTINGS_FILENAME,
    ModelRepository,
    SchemalessModelRepository,
)
from mlserver.rest import RESTServer
from mlserver.settings import ModelParameters, ModelSettings, Settings
from mlserver.types import InferenceRequest, InferenceResponse, MetadataModelResponse
from prometheus_client.registry import REGISTRY as PROM_DEFAULT_REGISTRY
from prometheus_client.registry import CollectorRegistry
from starlette_exporter import PrometheusMiddleware

from woprserver.logging import get_logger
from woprserver.parallel.dispatcher import Dispatcher
from woprserver.parallel.messages import (
    ModelRequestMessage,
    ModelUpdateMessage,
    ModelUpdateType,
)
from woprserver.parallel.model import ModelMethods, ParallelModel
from woprserver.parallel.pool import InferencePool, _spawn_worker
from woprserver.parallel.registry import InferencePoolRegistry
from woprserver.parallel.utils import cancel_task, configure_inference_pool
from woprserver.parallel.worker import Worker

# Import only model classes from fixtures; DO NOT re-declare root fixtures here.
from .fixtures import (
    EnvModel,
    ErrorModel,
    FakeModel,
    PidStreamModel,
    PidUnaryModel,
    SimpleModel,
    SumModel,
    TextModel,
    TextStreamModel,
)
from .utils import _get_tarball_name, _pack, get_available_ports

logger = get_logger()

# --------------------------------------------------------------------------------------
# Constants / paths
# --------------------------------------------------------------------------------------
MIN_PYTHON_VERSION = (3, 9)
MAX_PYTHON_VERSION = (3, 12)
TESTS_PATH = Path(__file__).parent
TESTDATA_PATH = TESTS_PATH / "testdata"
TESTDATA_CACHE_PATH = TESTDATA_PATH / ".cache"

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def _load_json_model(Model, path: Path):
    """
    Pydantic v1/v2 safe JSON loader for small models used in tests.
    """
    raw = path.read_text()
    if hasattr(Model, "model_validate_json"):  # Pydantic v2+
        return Model.model_validate_json(raw)
    try:
        # v1: prefer parse_raw (fast), fall back to parse_file for older models
        return Model.parse_raw(raw)  # type: ignore[attr-defined]
    except Exception:
        return Model.parse_file(str(path))  # type: ignore[attr-defined]


def make_dispatcher():
    """
    Tiny dispatcher substitute for unit tests which don't exercise real routing.
    """
    async def _noop_bridge(*_a, **_k):
        return None

    from types import SimpleNamespace

    return SimpleNamespace(
        dispatch_request=None,
        dispatch_request_stream=None,
        get_worker_for=lambda _rid: object(),
        _bridge_async_iterable_arg=_noop_bridge,
    )


def _default_py_matrix() -> list[tuple[int, int]]:
    """
    Fast default: test against the *current* interpreter only unless FULL_PY_MATRIX is set.
    """
    if os.getenv("FULL_PY_MATRIX") in {"1", "true", "yes"}:
        return [
            (major, minor)
            for major in range(MIN_PYTHON_VERSION[0], MAX_PYTHON_VERSION[0] + 1)
            for minor in range(MIN_PYTHON_VERSION[1], MAX_PYTHON_VERSION[1] + 1)
        ]
    return [(sys.version_info.major, sys.version_info.minor)]


PYTHON_VERSIONS = _default_py_matrix()


def unregister_metrics(registry: CollectorRegistry) -> None:
    """
    Best-effort unregister of all collectors to avoid cross-test name clashes.
    """
    collectors = list(getattr(registry, "_collector_to_names", {}).keys())
    for collector in collectors:
        with contextlib.suppress(Exception):
            registry.unregister(collector)


def assert_not_called_with(self: Mock, *args, **kwargs) -> None:
    """
    Convenience negative assertion for unittest.mock.Mock.
    """
    try:
        self.assert_called_with(*args, **kwargs)
    except AssertionError:
        return
    raise AssertionError(
        f"Expected {self._format_mock_call_signature(args, kwargs)} to not have been called."
    )


Mock.assert_not_called_with = assert_not_called_with  # monkey-patch for tests

# --------------------------------------------------------------------------------------
# Async loop & logging
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def event_loop():
    loop = uvloop.new_event_loop()
    yield loop

# --------------------------------------------------------------------------------------
# Caching / env tarballs
# --------------------------------------------------------------------------------------
def _is_stale(lock_file: Path, max_age_s: float = 120.0) -> bool:
    with contextlib.suppress(FileNotFoundError):
        age = time.time() - lock_file.stat().st_mtime
        return age > max_age_s
    return False


def _safe_remove(path: Path) -> None:
    with contextlib.suppress(Exception):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)


def _is_valid_tar(path: str) -> bool:
    """
    Quick sanity check: exists, not tiny, and tar.gz opens.
    """
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 1024:
            return False
        with tarfile.open(path, "r:gz"):
            return True
    except Exception:
        return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False

def _read_lock_pid(lock_path: Path) -> int | None:
    with contextlib.suppress(Exception):
        txt = lock_path.read_text().strip()
        return int(txt) if txt else None
    return None

def _file_size(path: Path) -> int:
    with contextlib.suppress(FileNotFoundError):
        return path.stat().st_size
    return 0

def _acquire_tarball_lock_with_ttl(
    path: str,
    ttl: float = float(os.getenv("WOPR_TEST_ENV_LOCK_TTL", "180")),
    ping: float = 2.0,
    max_wait: float = float(os.getenv("WOPR_TEST_ENV_LOCK_MAX_WAIT", "600")),  # 10 min default
) -> FileLock:
    lock_path = Path(f"{path}.lock")
    lock = FileLock(str(lock_path))
    start = time.time()

    while True:
        try:
            lock.acquire(timeout=ping)
            logger.info("Acquired lock %s", lock_path)
            return lock
        except Timeout:
            pid = _read_lock_pid(lock_path)
            size = _file_size(lock_path)
            age = time.time() - lock_path.stat().st_mtime if lock_path.exists() else -1

            # 1) Empty/corrupt lock file — safe to remove immediately
            if size == 0:
                logger.warning("Empty lock file detected; removing %s", lock_path)
                _safe_remove(lock_path)
                continue

            # 2) Lock written by a dead process — remove immediately
            if pid and not _pid_alive(pid):
                logger.warning("Lock %s held by dead PID %s; removing.", lock_path, pid)
                _safe_remove(lock_path)
                continue

            # 3) Age-based TTL fallback
            if _is_stale(lock_path, max_age_s=ttl):
                logger.warning("Stale lock detected (age=%.1fs ttl=%.1fs); removing %s", age, ttl, lock_path)
                _safe_remove(lock_path)
                continue

            # 4) Hard cap
            waited = time.time() - start
            if waited > max_wait:
                raise TimeoutError(
                    f"Timed out waiting for {lock_path} (waited {waited:.0f}s, age {age:.0f}s, size {size}B, pid {pid}). "
                    "If no packer is running, remove the .lock and any partial tarball."
                ) from None

            logger.info("Waiting on lock %s ... age=%.1fs ttl=%.1fs (held by PID %s?)", lock_path, age, ttl, pid)

def _export_poetry_requirements(cache_dir: str, with_dev: bool = True) -> Path:
    """
    Export Poetry dependencies into a pip requirements.txt.
    Includes dev/test dependencies if with_dev=True.
    """
    req_file = Path(cache_dir) / "requirements.txt"
    cmd = [
        "poetry", "export",
        "-f", "requirements.txt",
        "--without-hashes",
        "-o", str(req_file)
    ]
    if with_dev:
        cmd.extend(["--with", "dev", "--with", "test"])
    subprocess.run(cmd, check=True)
    return req_file

def _env_yml_abs_editable(cache_dir: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    _ = _export_poetry_requirements(cache_dir, with_dev=True)
    env_spec = {
        "name": "custom-runtime-environment",
        "channels": ["defaults"],
        "dependencies": [
            "python",  # patched to 3.10.* by _inject_python_version
            "pip",
            {"pip": [
                "numpy>=2.0",
                "scipy>=1.14",
                "scikit-learn==1.6.1",
                "mlflow>=2.16",
                str(repo_root),
            ]},
        ],
    }
    out = Path(cache_dir) / "environment-template.yml"
    out.write_text(yaml.safe_dump(env_spec, sort_keys=False))
    return str(out)


@pytest.fixture(scope="session")
def testdata_cache_path() -> str:
    """`tests/testdata/.cache` folder path."""
    TESTDATA_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return str(TESTDATA_CACHE_PATH)


@pytest.fixture(
    scope="session",
    params=PYTHON_VERSIONS,
    ids=[f"py{major}{minor}" for (major, minor) in PYTHON_VERSIONS],
)
def env_python_version(request: pytest.FixtureRequest) -> tuple[int, int]:
    """Parameterize tests by Python version (see `_default_py_matrix`)."""
    return request.param


@pytest.fixture
async def env_tarball(env_python_version: tuple[int, int], testdata_cache_path: str) -> str:
    tarball_name = _get_tarball_name(env_python_version)
    tarball_path = os.path.join(testdata_cache_path, tarball_name)

    # ✅ Fast path: if a valid tarball already exists, just use it.
    if os.path.isfile(tarball_path) and _is_valid_tar(tarball_path):
        return tarball_path

    Path(testdata_cache_path).mkdir(parents=True, exist_ok=True)
    lock = _acquire_tarball_lock_with_ttl(tarball_path, ttl=180.0, ping=2.0)
    with lock:
        # Re-check under the lock
        if os.path.isfile(tarball_path) and _is_valid_tar(tarball_path):
            return tarball_path
        if os.path.exists(tarball_path) and not _is_valid_tar(tarball_path):
            logger.warning("Removing invalid tarball at %s", tarball_path)
            _safe_remove(Path(tarball_path))

        env_yml_dynamic = _env_yml_abs_editable(testdata_cache_path)
        await _pack(env_python_version, env_yml_dynamic, tarball_path)

    return tarball_path


@pytest_asyncio.fixture(scope="module")
async def env(env_tarball: str, tmp_path_factory) -> Environment:
    """Extract env once per module."""
    env_root = tmp_path_factory.mktemp("env_mod")
    env_obj = await Environment.from_tarball(env_tarball, str(env_root))
    yield env_obj

# --------------------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------------------
@pytest.fixture
def settings() -> Settings:
    """Base settings with small pools for speed."""
    s = _load_json_model(Settings, TESTDATA_PATH / "settings.json")
    s.parallel_workers = 2
    return s

# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def metrics_registry() -> MetricsRegistry:
    """Module-scoped mlserver metrics registry."""
    try:
        yield METRICS_REGISTRY
    finally:
        unregister_metrics(METRICS_REGISTRY)


@pytest.fixture(scope="module")
def prometheus_registry(metrics_registry: MetricsRegistry) -> CollectorRegistry:
    """
    Ensure the default Prometheus registry is initialized; clean it up afterwards.
    """
    try:
        yield PROM_DEFAULT_REGISTRY
    finally:
        unregister_metrics(PROM_DEFAULT_REGISTRY)
        with contextlib.suppress(Exception):
            PrometheusMiddleware._metrics.clear()

# --------------------------------------------------------------------------------------
# Model settings & payloads
# --------------------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="module")
def sum_model_settings() -> ModelSettings:
    """SumModel from testdata/model-settings.json."""
    return _load_json_model(ModelSettings, TESTDATA_PATH / DEFAULT_MODEL_SETTINGS_FILENAME)


@pytest_asyncio.fixture(scope="module")
def simple_model_settings() -> ModelSettings:
    """SimpleModel overriding name/implementation from JSON base."""
    base = ModelSettings.parse_file(os.path.join(TESTDATA_PATH, DEFAULT_MODEL_SETTINGS_FILENAME))
    base.name = "simple-model"
    base.implementation = SimpleModel
    return base


@pytest_asyncio.fixture(scope="module")
def error_model_settings() -> ModelSettings:
    """ErrorModel overriding name/implementation from JSON base."""
    base = ModelSettings.parse_file(os.path.join(TESTDATA_PATH, DEFAULT_MODEL_SETTINGS_FILENAME))
    base.name = "error-model"
    base.implementation = ErrorModel
    return base


@pytest_asyncio.fixture(scope="module")
async def model_registry(sum_model_settings: ModelSettings) -> MultiModelRegistry:
    """MultiModelRegistry with SumModel preloaded."""
    registry = MultiModelRegistry()
    await registry.load(sum_model_settings)
    return registry


@pytest_asyncio.fixture(scope="module")
async def error_model(
    model_registry: MultiModelRegistry, error_model_settings: ModelSettings
) -> ErrorModel:
    """Loaded ErrorModel."""
    await model_registry.load(error_model_settings)
    return await model_registry.get_model(error_model_settings.name)


@pytest_asyncio.fixture(scope="module")
async def simple_model(
    model_registry: MultiModelRegistry, simple_model_settings: ModelSettings
) -> SimpleModel:
    """Loaded SimpleModel."""
    await model_registry.load(simple_model_settings)
    return await model_registry.get_model(simple_model_settings.name)


@pytest_asyncio.fixture(scope="module")
async def sum_model(
    model_registry: MultiModelRegistry, sum_model_settings: ModelSettings
) -> SumModel:
    """Return preloaded SumModel."""
    return await model_registry.get_model(sum_model_settings.name)


@pytest.fixture
def text_model_settings() -> ModelSettings:
    return ModelSettings(name="text-model", implementation=TextModel, parameters={"version": "v1.2.3"})


@pytest_asyncio.fixture
async def text_model(model_registry: MultiModelRegistry, text_model_settings: ModelSettings) -> TextModel:
    await model_registry.load(text_model_settings)
    return await model_registry.get_model(text_model_settings.name)


@pytest.fixture
def text_stream_model_settings() -> ModelSettings:
    return ModelSettings(name="text-stream-model", implementation=TextStreamModel, parameters={"version": "v1.2.3"})


@pytest_asyncio.fixture
async def text_stream_model(
    model_registry: MultiModelRegistry, text_stream_model_settings: ModelSettings
) -> TextModel:
    await model_registry.load(text_stream_model_settings)
    return await model_registry.get_model(text_stream_model_settings.name)

# --------------------------------------------------------------------------------------
# Static payload fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture
def metadata_server_response() -> types.MetadataServerResponse:
    return _load_json_model(types.MetadataServerResponse, TESTDATA_PATH / "metadata-server-response.json")


@pytest.fixture
def metadata_model_response() -> types.MetadataModelResponse:
    return _load_json_model(types.MetadataModelResponse, TESTDATA_PATH / "metadata-model-response.json")


@pytest.fixture
def inference_request() -> types.InferenceRequest:
    return _load_json_model(types.InferenceRequest, TESTDATA_PATH / "inference-request.json")


@pytest.fixture
def generate_request() -> types.InferenceRequest:
    return _load_json_model(types.InferenceRequest, TESTDATA_PATH / "generate-request.json")


@pytest.fixture
def inference_request_invalid_datatype() -> dict[str, Any]:
    with (TESTDATA_PATH / "inference-request-invalid-datatype.json").open("r") as f:
        return json.load(f)


@pytest.fixture
def inference_response() -> types.InferenceResponse:
    return _load_json_model(types.InferenceResponse, TESTDATA_PATH / "inference-response.json")

# --------------------------------------------------------------------------------------
# Server / repository
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _woprserver_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Module-scoped mlserver Settings with unique ports & metrics dir."""
    s = _load_json_model(Settings, TESTDATA_PATH / "settings.json")
    http_port, grpc_port, metrics_port = get_available_ports(3)
    s.http_port = http_port
    s.grpc_port = grpc_port
    s.metrics_port = metrics_port
    s.metrics_dir = str(tmp_path_factory.mktemp("metrics_mod"))
    s.parallel_workers = 2
    return s


@pytest_asyncio.fixture(scope="module")
async def mlserver(
    _woprserver_settings: Settings,
    sum_model_settings: ModelSettings,
    prometheus_registry: CollectorRegistry,  # force init
):
    """Boot an MLServer instance for integration tests."""

    server = WOPRserver(_woprserver_settings)
    task = asyncio.create_task(server.start())
    await server._model_registry.load(sum_model_settings)
    try:
        yield server
    finally:
        await server.stop()
        await task
        # ensure starlette-exporter files are cleaned
        prom_files = glob.glob(os.path.join(_woprserver_settings.metrics_dir, "*.db"))
        assert not prom_files


@pytest.fixture
def data_plane(model_registry: MultiModelRegistry, prometheus_registry: CollectorRegistry) -> DataPlane:
    """DataPlane backed by the shared model registry."""
    return DataPlane(settings=_load_json_model(Settings, TESTDATA_PATH / "settings.json"), model_registry=model_registry)


@pytest.fixture
def model_folder(tmp_path: str) -> str:
    """Temporary model folder containing a single model-settings file."""
    src = TESTDATA_PATH / DEFAULT_MODEL_SETTINGS_FILENAME
    dst = Path(tmp_path) / DEFAULT_MODEL_SETTINGS_FILENAME
    shutil.copyfile(src, dst)
    return str(Path(tmp_path))


@pytest.fixture
def model_repository(model_folder: str) -> ModelRepository:
    """Schemaless repository pointing at `model_folder`."""
    return SchemalessModelRepository(model_folder)


@pytest.fixture
def model_repository_handlers(
    model_repository: ModelRepository, model_registry: MultiModelRegistry
) -> ModelRepositoryHandlers:
    return ModelRepositoryHandlers(repository=model_repository, model_registry=model_registry)


@pytest.fixture
def repository_index_request() -> types.RepositoryIndexRequest:
    return types.RepositoryIndexRequest(ready=None)


@pytest.fixture
def repository_index_response(sum_model_settings: ModelSettings) -> types.RepositoryIndexResponse:
    return types.RepositoryIndexResponse(
        root=[
            types.RepositoryIndexResponseItem(
                name=sum_model_settings.name,
                version=sum_model_settings.parameters.version,
                state=types.State.READY,
                reason="",
            ),
        ]
    )

# --------------------------------------------------------------------------------------
# Inference pool / dispatcher (module-scoped)
# --------------------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="module")
async def inference_pool_registry(
    _woprserver_settings: Settings, prometheus_registry: CollectorRegistry
) -> InferencePoolRegistry:
    """Shared InferencePoolRegistry; ensure close on teardown."""
    reg = InferencePoolRegistry(_woprserver_settings)
    try:
        yield reg
    finally:
        await reg.close()


@pytest_asyncio.fixture(scope="module")
async def inference_pool(_woprserver_settings: Settings) -> InferencePool:
    """Shared InferencePool configured with tiny worker pool for speed."""
    configure_inference_pool(_woprserver_settings)
    pool = InferencePool(_woprserver_settings)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture(scope="module")
def dispatcher(inference_pool: InferencePool) -> Dispatcher:
    """Dispatcher grabbed from the shared pool."""
    return inference_pool._dispatcher

# --------------------------------------------------------------------------------------
# Models routed via the pool
# --------------------------------------------------------------------------------------
@pytest_asyncio.fixture
async def loaded_error_model(inference_pool: InferencePool) -> MLModel:
    """Load & unload ErrorModel through the pool for routing tests."""
    error_settings = ModelSettings(name="error-model", implementation=ErrorModel, parameters=ModelParameters())
    model = ErrorModel(error_settings)
    _ = await inference_pool.load_model(model)
    try:
        yield model
    finally:
        await inference_pool.unload_model(model)

# --------------------------------------------------------------------------------------
# Worker-process helpers (module scoped)
# --------------------------------------------------------------------------------------
@pytest_asyncio.fixture(scope="module")
async def responses():
    """Multiprocessing queue used to receive worker responses."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    try:
        yield q
    finally:
        q.close()


@pytest_asyncio.fixture(scope="module")
async def worker(
    _woprserver_settings: Settings,
    responses,
    load_message: ModelUpdateMessage,
) -> Worker:
    """Spawn a Worker, load a model, and ensure clean shutdown."""
    worker = Worker(_woprserver_settings, responses)
    task = asyncio.create_task(worker.coro_run())

    # Signal load; wait until worker reports back
    loop = asyncio.get_running_loop()
    worker.send_update(load_message)
    await loop.run_in_executor(None, lambda: responses.get(timeout=10))

    try:
        yield worker
    finally:
        await worker.stop()
        await cancel_task(task)

# --------------------------------------------------------------------------------------
# Messages
# --------------------------------------------------------------------------------------
@pytest.fixture
def load_message(sum_model_settings: ModelSettings) -> ModelUpdateMessage:
    return ModelUpdateMessage(update_type=ModelUpdateType.Load, model_settings=sum_model_settings)


@pytest.fixture
def unload_message(sum_model_settings: ModelSettings) -> ModelUpdateMessage:
    return ModelUpdateMessage(update_type=ModelUpdateType.Unload, model_settings=sum_model_settings)


@pytest.fixture
def inference_request_message(sum_model_settings: ModelSettings, inference_request: InferenceRequest) -> ModelRequestMessage:
    return ModelRequestMessage(
        model_name=sum_model_settings.name,
        model_version=sum_model_settings.parameters.version,
        method_name=ModelMethods.Predict.value,
        method_args=[inference_request],
    )


@pytest.fixture
def metadata_request_message(sum_model_settings: ModelSettings) -> ModelRequestMessage:
    return ModelRequestMessage(
        model_name=sum_model_settings.name,
        model_version=sum_model_settings.parameters.version,
        method_name=ModelMethods.Metadata.value,
    )


@pytest.fixture
def custom_request_message(sum_model_settings: ModelSettings) -> ModelRequestMessage:
    return ModelRequestMessage(
        model_name=sum_model_settings.name,
        model_version=sum_model_settings.parameters.version,
        method_name="my_payload",
        method_kwargs={"payload": [1, 2, 3]},
    )

# --------------------------------------------------------------------------------------
# Parallel model unit helpers
# --------------------------------------------------------------------------------------
@pytest.fixture
def parallel_model_settings() -> ModelSettings:
    return ModelSettings(
        name="dummy-parallel-model",
        implementation="tests.parallel.fixtures.SumModel",
        parameters={"version": "v0"},
    )


@pytest.fixture
def inner_model(parallel_model_settings: ModelSettings) -> MLModel:
    """Minimal MLModel to exercise ParallelModel behavior in isolation."""

    class Dummy(MLModel):
        async def load(self) -> None:
            return

        async def predict_stream(self, payloads: AsyncIterator[InferenceRequest]) -> AsyncIterator[InferenceResponse]:
            async for _ in payloads:
                break
            yield InferenceResponse(model_name=self.settings.name, outputs=[])
            yield InferenceResponse(model_name=self.settings.name, outputs=[])

        async def predict(self, request: InferenceRequest) -> InferenceResponse:
            return InferenceResponse(model_name=self.settings.name, outputs=[])

        async def metadata(self) -> MetadataModelResponse:
            return MetadataModelResponse(name=self.settings.name, platform="dummy")

        async def tokens(self) -> AsyncIterator[int]:
            yield 1
            yield 2

    return Dummy(parallel_model_settings)


@pytest.fixture
def parallel(inner_model: MLModel, dispatcher: Dispatcher) -> ParallelModel:
    return ParallelModel(inner_model, dispatcher)

# --------------------------------------------------------------------------------------
# Environment-model helpers (used by env tests)
# --------------------------------------------------------------------------------------
@pytest.fixture
def env_model_settings(env_tarball: str) -> ModelSettings:
    return ModelSettings(
        name="env-model",
        implementation=EnvModel,
        parameters=ModelParameters(environment_tarball=env_tarball),
    )


@pytest.fixture
def existing_env_model_settings(env_tarball: str, tmp_path) -> ModelSettings:
    """ModelSettings pointing at a pre-extracted env path."""
    from mlserver.env import _extract_env

    env_path = str(tmp_path)
    _extract_env(env_tarball, env_path)
    return ModelSettings(
        name="existing_env_model",
        implementation=EnvModel,
        parameters=ModelParameters(environment_path=env_path),
    )


@pytest_asyncio.fixture(scope="module")
async def worker_with_env(
    _woprserver_settings: Settings,
    responses,
    env: Environment,
    env_model_settings: ModelSettings,
):
    """Spawn a worker with a pre-provisioned Environment attached."""
    worker = _spawn_worker(_woprserver_settings, responses, env)
    worker.send_update(ModelUpdateMessage(update_type=ModelUpdateType.Load, model_settings=env_model_settings))

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: responses.get(timeout=10))
    try:
        yield worker
    finally:
        await worker.stop()

# --------------------------------------------------------------------------------------
# Misc small fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture
def datatype_error_message() -> str:
    return (
        "Input should be"
        " 'BOOL', 'UINT8', 'UINT16', 'UINT32',"
        " 'UINT64', 'INT8', 'INT16', 'INT32', 'INT64',"
        " 'FP16', 'FP32', 'FP64' or 'BYTES'"
    )


@pytest_asyncio.fixture
async def env_model(
    inference_pool_registry: InferencePoolRegistry, env_model_settings: ModelSettings
) -> MLModel:
    env_model = EnvModel(env_model_settings)
    model = await inference_pool_registry.load_model(env_model)

    yield model

    await inference_pool_registry.unload_model(model)



@pytest_asyncio.fixture
async def existing_env_model(
    inference_pool_registry: InferencePoolRegistry, existing_env_model_settings: ModelSettings
) -> MLModel:
    """EnvModel with a pre-extracted env."""
    model = await inference_pool_registry.load_model(EnvModel(existing_env_model_settings))
    try:
        yield model
    finally:
        await inference_pool_registry.unload_model(model)


@pytest.fixture
def pid_unary_settings() -> ModelSettings:
    return ModelSettings(name="pid-unary", implementation=PidUnaryModel, parallel_workers=2, parameters={"version": "v-test"})


@pytest.fixture
def pid_stream_settings() -> ModelSettings:
    return ModelSettings(name="pid-stream", implementation=PidStreamModel, parallel_workers=2, parameters={"version": "v-test"})


@pytest.fixture
def trivial_req() -> InferenceRequest:
    return InferenceRequest(inputs=[StringCodec.encode_input("input", payload="x", use_bytes=True)])


@pytest_asyncio.fixture
async def load_error_model() -> MLModel:
    """An ErrorModel configured to raise during load, for negative tests."""
    ms = ModelSettings(name="error-model-temp", implementation=ErrorModel, parameters=ModelParameters(load_error=True))
    yield ErrorModel(ms)


@pytest_asyncio.fixture
async def inference_pool_pid(settings: Settings):
    """Dedicated pool for the PID routing tests."""
    configure_inference_pool(settings)
    pool = InferencePool(settings)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture
async def rest_server(
    settings: Settings,
    data_plane: DataPlane,
    model_repository_handlers: ModelRepositoryHandlers,
    sum_model: SumModel,
    prometheus_registry: CollectorRegistry,
) -> RESTServer:
    """RESTServer with SumModel custom handlers mounted, auto-cleanup."""
    server = RESTServer(settings, data_plane=data_plane, model_repository_handlers=model_repository_handlers)
    sum_model = await server.add_custom_handlers(sum_model)
    try:
        yield server
    finally:
        await server.delete_custom_handlers(sum_model)

# --------------------------------------------------------------------------------------
# Global Prometheus hygiene
# --------------------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _prometheus_isolated_between_tests():
    """
    After each test, unregister all collectors from the *default* registry so
    subsequent tests can re-create DataPlane/RESTServer without metric name
    clashes. Also clear starlette-exporter's internal metric cache.
    """
    yield
    reg = prometheus_client.REGISTRY  # default global registry
    with contextlib.suppress(Exception):
        collectors = list(getattr(reg, "_collector_to_names", {}).keys())
        for c in collectors:
            with contextlib.suppress(Exception):
                reg.unregister(c)
    with contextlib.suppress(Exception):
        PrometheusMiddleware._metrics.clear()

# --------------------------------------------------------------------------------------
# ParallelModel baseline
# --------------------------------------------------------------------------------------
@pytest.fixture
def pm_and_dispatcher(mocker):
    """ParallelModel + dummy dispatcher, with custom handlers explicitly disabled."""
    model = FakeModel()
    dispatcher = make_dispatcher()
    pm = ParallelModel(model, dispatcher)
    mocker.patch("woprserver.parallel.model.get_custom_handlers", return_value=[])
    pm._register_custom_handlers()
    return pm, dispatcher, model
