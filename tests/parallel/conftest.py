#tests/parallel/conftest.py

import multiprocessing as mp

# Make sure 'spawn' is set BEFORE anything imports multiprocessing users
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import asyncio
import contextlib
import glob
import json
import os
import platform
import shutil

# tests/parallel/conftest.py
import sys
import tarfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import prometheus_client
import pytest
import pytest_asyncio
from filelock import FileLock, Timeout
from mlserver import types
from mlserver.codecs.string import StringCodec
from mlserver.env import Environment
from mlserver.handlers import DataPlane, ModelRepositoryHandlers
from mlserver.logging import configure_logger
from mlserver.metrics.registry import REGISTRY as METRICS_REGISTRY
from mlserver.metrics.registry import MetricsRegistry
from mlserver.model import MLModel
from mlserver.parallel import InferencePoolRegistry
from mlserver.registry import MultiModelRegistry
from mlserver.repository import (
    DEFAULT_MODEL_SETTINGS_FILENAME,
    ModelRepository,
    SchemalessModelRepository,
)
from mlserver.rest import RESTServer
from mlserver.settings import ModelParameters, ModelSettings, Settings
from mlserver.types import InferenceRequest, InferenceResponse, MetadataModelResponse
from prometheus_client.registry import REGISTRY, CollectorRegistry
from starlette_exporter import PrometheusMiddleware

from woprserver.parallel.dispatcher import Dispatcher
from woprserver.parallel.messages import (
    ModelRequestMessage,
    ModelUpdateMessage,
    ModelUpdateType,
)
from woprserver.parallel.model import ModelMethods, ParallelModel
from woprserver.parallel.pool import InferencePool, _spawn_worker
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
from .utils import RESTClient, get_available_ports

# --------------------------
# Constants / paths
# --------------------------
MIN_PYTHON_VERSION = (3, 9)
MAX_PYTHON_VERSION = (3, 12)
TESTS_PATH = Path(__file__).parent
TESTDATA_PATH = TESTS_PATH / "testdata"
TESTDATA_CACHE_PATH = TESTDATA_PATH / ".cache"


# --------------------------
# Pydantic v2-safe loader
# --------------------------
def _load_json_model(Model, path: Path):
    raw = path.read_text()
    if hasattr(Model, "model_validate_json"):  # Pydantic v2
        return Model.model_validate_json(raw)
    try:
        return Model.parse_raw(raw)  # type: ignore[attr-defined]
    except Exception:
        return Model.parse_file(str(path))  # type: ignore[attr-defined]


def make_dispatcher():
    async def _noop_bridge(*_a, **_k):
        return None

    from types import SimpleNamespace
    return SimpleNamespace(
        dispatch_request=None,
        dispatch_request_stream=None,
        get_worker_for=lambda _rid: object(),
        _bridge_async_iterable_arg=_noop_bridge,
    )


# --------------------------
# Speed helpers
# --------------------------
def _default_py_matrix() -> list[tuple[int, int]]:
    if os.getenv("FULL_PY_MATRIX") in {"1", "true", "yes"}:
        return [
            (major, minor)
            for major in range(MIN_PYTHON_VERSION[0], MAX_PYTHON_VERSION[0] + 1)
            for minor in range(MIN_PYTHON_VERSION[1], MAX_PYTHON_VERSION[1] + 1)
        ]
    import sys
    return [(sys.version_info.major, sys.version_info.minor)]


PYTHON_VERSIONS = _default_py_matrix()


def unregister_metrics(registry: CollectorRegistry):
    collectors = list(getattr(registry, "_collector_to_names", {}).keys())
    for collector in collectors:
        try:
            registry.unregister(collector)
        except Exception:
            pass


def assert_not_called_with(self, *args, **kwargs):
    try:
        self.assert_called_with(*args, **kwargs)
    except AssertionError:
        return
    raise AssertionError(
        f"Expected {self._format_mock_call_signature(args, kwargs)} to not have been called."
    )


Mock.assert_not_called_with = assert_not_called_with


# --------------------------
# Async loop & logging
# --------------------------
@pytest.fixture(scope="session")
def event_loop():
    """
    Session-scoped loop so module-/session-scoped async fixtures can depend on it.
    Compatible with pytest-asyncio 0.21.x strict/auto.
    """
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except Exception:
        pass

    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.run_until_complete(asyncio.sleep(0))
        loop.close()


@pytest.fixture(scope="session", autouse=True)
def _logger_once():
    try:
        _settings = _load_json_model(Settings, TESTDATA_PATH / "settings.json")
        configure_logger(_settings)
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def logger():
    return configure_logger(_load_json_model(Settings, TESTDATA_PATH / "settings.json"))


# --------------------------
# Caching / session-level heavy bits
# --------------------------
@pytest.fixture(scope="session")
def testdata_cache_path() -> str:
    TESTDATA_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return str(TESTDATA_CACHE_PATH)


@pytest.fixture(
    scope="session",
    params=PYTHON_VERSIONS,
    ids=[f"py{major}{minor}" for (major, minor) in PYTHON_VERSIONS],
)
def env_python_version(request: pytest.FixtureRequest) -> tuple[int, int]:
    return request.param


# --- helpers ----------------------------------------------------------------

def _is_stale(lock_file: Path, max_age_s: float = 120.0) -> bool:
    try:
        age = time.time() - lock_file.stat().st_mtime
        return age > max_age_s
    except FileNotFoundError:
        return False

def _safe_remove(path: Path) -> None:
    with contextlib.suppress(Exception):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)

# --- fixture ----------------------------------------------------------------

@pytest.fixture(scope="session")
def env_tarball(tmp_path_factory, request) -> str:
    """
    Build/provide the reusable environment tarball once per session.
    Uses a file lock with timeout and stale-lock recovery to avoid deadlocks.
    """
    base_tmp = Path(tmp_path_factory.getbasetemp())  # unique per session/worker
    # Make the tarball/lock path deterministic but namespaced by Python+arch
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    arch_tag = platform.machine()
    target_dir = base_tmp / f"env-tarball-{py_tag}-{arch_tag}"
    target_dir.mkdir(parents=True, exist_ok=True)

    tarball_path = target_dir / "env.tgz"
    lock_path = target_dir / ".env.tgz.lock"

    # Fast path: already built
    if tarball_path.exists() and tarball_path.stat().st_size > 0:
        print(f"[env_tarball] Reusing existing tarball at {tarball_path}", flush=True)
        return str(tarball_path)

    # Guard with a lock, but never wait forever
    lock = FileLock(str(lock_path), timeout=10.0)

    try:
        try:
            print(f"[env_tarball] Acquiring lock {lock_path} ...", flush=True)
            lock.acquire()  # explicit acquire so we can log around it
        except Timeout:
            # If the lock is stale, remove it and try once more
            if _is_stale(lock_path, max_age_s=180.0):
                print(f"[env_tarball] Stale lock detected; removing {lock_path}", flush=True)
                _safe_remove(lock_path)
                # Try one last time with a short timeout
                lock = FileLock(str(lock_path), timeout=5.0)
                lock.acquire()
            else:
                pytest.skip(f"[env_tarball] Lock busy: {lock_path}")

        # Double-check another proc didn't build while we waited
        if tarball_path.exists() and tarball_path.stat().st_size > 0:
            print(f"[env_tarball] Tarball appeared while waiting; using {tarball_path}", flush=True)
            return str(tarball_path)

        # Build the tarball (create a tiny, deterministic env payload for tests).
        # Replace this block with your real environment bundle generation if needed.
        build_dir = target_dir / "env-root"
        _safe_remove(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

        # Minimal file to make the tarball non-empty & deterministic
        (build_dir / "environment.yml").write_text(
            "name: test-env\nchannels: []\ndependencies: []\n",
            encoding="utf-8",
        )

        print(f"[env_tarball] Creating tarball at {tarball_path}", flush=True)
        with tarfile.open(tarball_path, "w:gz") as tf:
            tf.add(build_dir, arcname=".")

        assert tarball_path.exists() and tarball_path.stat().st_size > 0
        print(f"[env_tarball] Created {tarball_path} ({tarball_path.stat().st_size} bytes)", flush=True)
        return str(tarball_path)

    finally:
        with contextlib.suppress(Exception):
            lock.release()
        # Make sure the lock file is gone (filelock may keep a tiny file around)
        with contextlib.suppress(Exception):
            lock_path.unlink()


@pytest_asyncio.fixture(scope="module")
async def env(env_tarball: str, tmp_path_factory) -> Environment:
    """Extract env once per module."""
    env_root = tmp_path_factory.mktemp("env_mod")
    env_obj = await Environment.from_tarball(env_tarball, str(env_root))
    yield env_obj


# --------------------------
# Settings
# --------------------------
@pytest.fixture
def settings() -> Settings:
    s = _load_json_model(Settings, TESTDATA_PATH / "settings.json")
    # keep pools small for speed
    if not getattr(s, "parallel_workers", None):
        s.parallel_workers = 2
    else:
        s.parallel_workers = min(2, s.parallel_workers)
    return s


# --------------------------
# Metrics
# --------------------------
@pytest.fixture(scope="module")
def metrics_registry() -> MetricsRegistry:
    yield METRICS_REGISTRY
    unregister_metrics(METRICS_REGISTRY)


@pytest.fixture(scope="module")
def prometheus_registry(metrics_registry: MetricsRegistry) -> CollectorRegistry:
    try:
        yield REGISTRY
    finally:
        unregister_metrics(REGISTRY)
        PrometheusMiddleware._metrics.clear()


# --------------------------
# Model settings & payloads
# --------------------------
@pytest.fixture(scope="module")
def sum_model_settings() -> ModelSettings:
    return _load_json_model(ModelSettings, TESTDATA_PATH / DEFAULT_MODEL_SETTINGS_FILENAME)


@pytest.fixture(scope="module")
def simple_model_settings() -> ModelSettings:
    ms = _load_json_model(ModelSettings, TESTDATA_PATH / DEFAULT_MODEL_SETTINGS_FILENAME)
    ms.name = "simple-model"
    ms.implementation = SimpleModel
    return ms


@pytest.fixture(scope="module")
def error_model_settings() -> ModelSettings:
    ms = _load_json_model(ModelSettings, TESTDATA_PATH / DEFAULT_MODEL_SETTINGS_FILENAME)
    ms.name = "error-model"
    ms.implementation = ErrorModel
    return ms


@pytest_asyncio.fixture(scope="module")
async def model_registry(sum_model_settings: ModelSettings) -> MultiModelRegistry:
    registry = MultiModelRegistry()
    await registry.load(sum_model_settings)
    try:
        yield registry
    finally:
        for name in list(registry._models.keys()):
            try:
                await registry.unload(name)
            except Exception:
                pass


@pytest_asyncio.fixture
async def error_model(
    model_registry: MultiModelRegistry, error_model_settings: ModelSettings
) -> ErrorModel:
    await model_registry.load(error_model_settings)
    return await model_registry.get_model(error_model_settings.name)


@pytest_asyncio.fixture
async def simple_model(
    model_registry: MultiModelRegistry, simple_model_settings: ModelSettings
) -> SimpleModel:
    await model_registry.load(simple_model_settings)
    return await model_registry.get_model(simple_model_settings.name)


@pytest_asyncio.fixture
async def sum_model(
    model_registry: MultiModelRegistry, sum_model_settings: ModelSettings
) -> SumModel:
    return await model_registry.get_model(sum_model_settings.name)


@pytest.fixture
def text_model_settings() -> ModelSettings:
    return ModelSettings(
        name="text-model",
        implementation=TextModel,
        parallel_workers=2,
        parameters={"version": "v1.2.3"},
    )


@pytest_asyncio.fixture
async def text_model(
    model_registry: MultiModelRegistry, text_model_settings: ModelSettings
) -> TextModel:
    await model_registry.load(text_model_settings)
    return await model_registry.get_model(text_model_settings.name)


@pytest.fixture
def text_stream_model_settings() -> ModelSettings:
    return ModelSettings(
        name="text-stream-model",
        implementation=TextStreamModel,
        parallel_workers=2,
        parameters={"version": "v1.2.3"},
    )


@pytest_asyncio.fixture
async def text_stream_model(
    model_registry: MultiModelRegistry, text_stream_model_settings: ModelSettings
) -> TextModel:
    await model_registry.load(text_stream_model_settings)
    return await model_registry.get_model(text_stream_model_settings.name)


# --------------------------
# Static payload fixtures
# --------------------------
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


# --------------------------
# Server / repository
# --------------------------
@pytest.fixture(scope="module")
def _mlserver_settings(tmp_path_factory: pytest.TempPathFactory):
    s = _load_json_model(Settings, TESTDATA_PATH / "settings.json")
    http_port, grpc_port, metrics_port = get_available_ports(3)
    s.http_port = http_port
    s.grpc_port = grpc_port
    s.metrics_port = metrics_port
    s.metrics_dir = str(tmp_path_factory.mktemp("metrics_mod"))
    # keep small for speed
    if not getattr(s, "parallel_workers", None):
        s.parallel_workers = 2
    else:
        s.parallel_workers = min(2, s.parallel_workers)
    return s


@pytest_asyncio.fixture(scope="module")
async def mlserver(
    _mlserver_settings: Settings,
    sum_model_settings: ModelSettings,
    prometheus_registry: CollectorRegistry,  # force init
):
    from mlserver import MLServer
    server = MLServer(_mlserver_settings)
    task = asyncio.create_task(server.start())
    await server._model_registry.load(sum_model_settings)
    try:
        yield server
    finally:
        await server.stop()
        await task
        prom_files = glob.glob(os.path.join(_mlserver_settings.metrics_dir, "*.db"))
        assert not prom_files


@pytest_asyncio.fixture
async def rest_client(mlserver, _mlserver_settings: Settings):
    http_server = f"{_mlserver_settings.host}:{_mlserver_settings.http_port}"
    client = RESTClient(http_server)
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
def data_plane(
    model_registry: MultiModelRegistry,
    prometheus_registry: CollectorRegistry,
) -> DataPlane:
    return DataPlane(
        settings=_load_json_model(Settings, TESTDATA_PATH / "settings.json"),
        model_registry=model_registry,
    )


@pytest.fixture
def model_repository_handlers(
    model_repository: ModelRepository, model_registry: MultiModelRegistry
) -> ModelRepositoryHandlers:
    return ModelRepositoryHandlers(repository=model_repository, model_registry=model_registry)


@pytest.fixture
def model_folder(tmp_path: str) -> str:
    src = TESTDATA_PATH / DEFAULT_MODEL_SETTINGS_FILENAME
    dst = Path(tmp_path) / DEFAULT_MODEL_SETTINGS_FILENAME
    shutil.copyfile(src, dst)
    return str(Path(tmp_path))


@pytest.fixture
def model_repository(model_folder: str) -> ModelRepository:
    return SchemalessModelRepository(model_folder)


@pytest.fixture
def repository_index_request() -> types.RepositoryIndexRequest:
    return types.RepositoryIndexRequest(ready=None)


@pytest.fixture
def repository_index_response(sum_model_settings) -> types.RepositoryIndexResponse:
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


# --------------------------
# Inference pool / dispatcher (module-scoped)
# --------------------------
@pytest_asyncio.fixture(scope="module")
async def inference_pool_registry(
    _mlserver_settings: Settings,
    prometheus_registry: CollectorRegistry,
) -> InferencePoolRegistry:
    reg = InferencePoolRegistry(_mlserver_settings)
    try:
        yield reg
    finally:
        await reg.close()


@pytest_asyncio.fixture(scope="module")
async def inference_pool(_mlserver_settings: Settings) -> InferencePool:
    configure_inference_pool(_mlserver_settings)
    pool = InferencePool(_mlserver_settings)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture(scope="module")
def dispatcher(inference_pool: InferencePool) -> Dispatcher:
    return inference_pool._dispatcher


# --------------------------
# Models routed via the pool
# --------------------------
@pytest_asyncio.fixture
async def loaded_error_model(inference_pool: InferencePool) -> MLModel:
    error_settings = ModelSettings(name="error-model", implementation=ErrorModel, parameters=ModelParameters())
    model = ErrorModel(error_settings)
    _ = await inference_pool.load_model(model)
    try:
        yield model
    finally:
        await inference_pool.unload_model(model)


# --------------------------
# Worker-process helpers (module scoped)
# --------------------------
@pytest_asyncio.fixture(scope="module")
async def responses():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    try:
        yield q
    finally:
        q.close()


@pytest_asyncio.fixture(scope="module")
async def worker(
    _mlserver_settings: Settings,
    responses,
    load_message: ModelUpdateMessage,
) -> Worker:
    worker = Worker(_mlserver_settings, responses)
    task = asyncio.create_task(worker.coro_run())

    loop = asyncio.get_running_loop()
    worker.send_update(load_message)
    await loop.run_in_executor(None, lambda: responses.get(timeout=10))

    try:
        yield worker
    finally:
        await worker.stop()
        await cancel_task(task)


# --------------------------
# Messages
# --------------------------
@pytest.fixture
def load_message(sum_model_settings: ModelSettings) -> ModelUpdateMessage:
    return ModelUpdateMessage(update_type=ModelUpdateType.Load, model_settings=sum_model_settings)


@pytest.fixture
def unload_message(sum_model_settings: ModelSettings) -> ModelUpdateMessage:
    return ModelUpdateMessage(update_type=ModelUpdateType.Unload, model_settings=sum_model_settings)


@pytest.fixture
def inference_request_message(
    sum_model_settings: ModelSettings, inference_request: InferenceRequest
) -> ModelRequestMessage:
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


# --------------------------
# Parallel model unit helpers
# --------------------------
@pytest.fixture
def parallel_model_settings() -> ModelSettings:
    return ModelSettings(name="dummy-parallel-model", implementation="tests.parallel.fixtures.SumModel", parameters={"version": "v0"})


@pytest.fixture
def inner_model(parallel_model_settings: ModelSettings) -> MLModel:
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


# --------------------------
# Environment-model helpers (used by env tests)
# --------------------------
@pytest.fixture
def env_model_settings(env_tarball: str) -> ModelSettings:
    return ModelSettings(
        name="env-model",
        implementation=EnvModel,
        parameters=ModelParameters(environment_tarball=env_tarball),
    )


@pytest.fixture
def existing_env_model_settings(env_tarball: str, tmp_path) -> ModelSettings:
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
    _mlserver_settings: Settings,
    responses,
    env: Environment,
    env_model_settings: ModelSettings,
):
    worker = _spawn_worker(_mlserver_settings, responses, env)
    load_msg = ModelUpdateMessage(update_type=ModelUpdateType.Load, model_settings=env_model_settings)
    worker.send_update(load_msg)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: responses.get(timeout=10))
    try:
        yield worker
    finally:
        await worker.stop()


# --------------------------
# Misc
# --------------------------
@pytest.fixture
def datatype_error_message():
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
    try:
        yield model
    finally:
        await inference_pool_registry.unload_model(model)


@pytest_asyncio.fixture
async def existing_env_model(
    inference_pool_registry: InferencePoolRegistry,
    existing_env_model_settings: ModelSettings,
) -> MLModel:
    env_model = EnvModel(existing_env_model_settings)
    model = await inference_pool_registry.load_model(env_model)
    try:
        yield model
    finally:
        await inference_pool_registry.unload_model(model)


@pytest.fixture
def pid_unary_settings() -> ModelSettings:
    return ModelSettings(
        name="pid-unary",
        implementation=PidUnaryModel,
        parallel_workers=2,
        parameters={"version": "v-test"},
    )


@pytest.fixture
def pid_stream_settings() -> ModelSettings:
    return ModelSettings(
        name="pid-stream",
        implementation=PidStreamModel,
        parallel_workers=2,
        parameters={"version": "v-test"},
    )


@pytest.fixture
def trivial_req() -> InferenceRequest:
    return InferenceRequest(inputs=[StringCodec.encode_input("input", payload="x", use_bytes=True)])


@pytest_asyncio.fixture
async def load_error_model() -> MLModel:
    ms = ModelSettings(
        name="error-model-temp",
        implementation=ErrorModel,
        parameters=ModelParameters(load_error=True),
    )
    yield ErrorModel(ms)

@pytest_asyncio.fixture
async def inference_pool_pid(settings: Settings):
    # Dedicated pool for the PID routing tests
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
    server = RESTServer(
        settings,
        data_plane=data_plane,
        model_repository_handlers=model_repository_handlers,
    )

    sum_model = await server.add_custom_handlers(sum_model)

    yield server

    await server.delete_custom_handlers(sum_model)

@pytest.fixture(autouse=True)
def _prometheus_isolated_between_tests():
    """
    After each test, unregister all collectors from the *default* registry so
    subsequent tests can re-create DataPlane/RESTServer without metric name
    clashes. Also clear starlette-exporter's internal metric cache.
    """
    yield
    reg = prometheus_client.REGISTRY  # default global registry
    try:
        collectors = list(getattr(reg, "_collector_to_names", {}).keys())
        for c in collectors:
            try:
                reg.unregister(c)
            except Exception:
                pass
    except Exception:
        pass
    try:
        PrometheusMiddleware._metrics.clear()
    except Exception:
        pass


@pytest.fixture
def pm_and_dispatcher(mocker):
    model = FakeModel()
    dispatcher = make_dispatcher()
    pm = ParallelModel(model, dispatcher)
    # Ensure no custom handlers are added for this baseline fixture.
    mocker.patch("woprserver.parallel.model.get_custom_handlers", return_value=[])
    pm._register_custom_handlers()
    return pm, dispatcher, model

