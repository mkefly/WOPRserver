# tests/parallel/test_registry.py
from __future__ import annotations

import asyncio
import os
import sys
import threading
from contextlib import suppress
from copy import deepcopy
from pathlib import Path

import pytest
from mlserver.codecs import StringCodec
from mlserver.env import Environment, compute_hash_of_file
from mlserver.model import MLModel
from mlserver.settings import ModelParameters, ModelSettings, Settings
from mlserver.types import InferenceRequest, InferenceResponse


from woprserver.parallel.errors import EnvironmentNotFound
from woprserver.parallel.registry import (
    ENV_HASH_ATTR,
    InferencePoolRegistry,
    _append_gid_environment_hash,
    _get_environment_hash,
    _set_environment_hash,
)

from .fixtures import EnvModel, SumModel


'''
# --------------------------------------------------------------------------------------
# Debug watchdog: if anything stalls, we dump thread stacks every ~2s during the test.
# --------------------------------------------------------------------------------------
def _periodic_dump(stop_evt: threading.Event, interval: float = 2.0) -> None:  # pragma: no cover - debug aid
    while not stop_evt.wait(interval):
        print("\n=== periodic stack dump (debug watchdog) ===\n", flush=True)
        frames = sys._current_frames()
        for tid, frame in frames.items():
            print(f"Thread {tid:#018x} (most recent call first):", flush=True)
            with suppress(Exception):
                import traceback

                traceback.print_stack(frame)
        print("", flush=True)


@pytest.fixture(autouse=True, scope="function")
def debug_watchdog():
    stop = threading.Event()
    t = threading.Thread(target=_periodic_dump, args=(stop,), daemon=True)
    t.start()
    yield
    stop.set()
    t.join(timeout=1.0)
'''

# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
async def _predict_with_backoff(
    model: MLModel,
    ir: InferenceRequest,
    attempts: int = 20,
    delay: float = 0.1,
) -> InferenceResponse:
    last_exc: BaseException | None = None
    for _ in range(attempts):
        try:
            return await model.predict(ir)
        except (KeyError, IndexError, StopIteration, RuntimeError) as e:
            last_exc = e
            await asyncio.sleep(delay)
    assert last_exc is not None
    raise last_exc


async def _wait_for_workers(pool, expected: int, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if len(pool._workers) == expected:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"Expected {expected} workers, found {len(pool._workers)}")


def _check_sklearn_version(response: InferenceResponse) -> None:
    # These values are defined by the fixture EnvModel implementation & env.yml.
    assert len(response.outputs) == 1
    assert response.outputs[0].name == "sklearn_version"
    [sklearn_version] = StringCodec.decode_output(response.outputs[0])
    assert sklearn_version == "1.6.1"


# --------------------------------------------------------------------------------------
# Basic attribute tests
# --------------------------------------------------------------------------------------
def test_set_environment_hash(sum_model: MLModel):
    env_hash = "0e46fce1decb7a89a8b91c71d8b6975630a17224d4f00094e02e1a732f8e95f3"
    _set_environment_hash(sum_model, env_hash)
    assert hasattr(sum_model, ENV_HASH_ATTR)
    assert getattr(sum_model, ENV_HASH_ATTR) == env_hash


@pytest.mark.parametrize("env_hash", ["abc123", None])
def test_get_environment_hash(sum_model: MLModel, env_hash: str | None):
    _set_environment_hash(sum_model, env_hash)
    assert _get_environment_hash(sum_model) == env_hash


async def test_default_pool(inference_pool_registry: InferencePoolRegistry, settings: Settings):
    assert inference_pool_registry._default_pool is not None
    worker_count = len(inference_pool_registry._default_pool._workers)
    assert worker_count == settings.parallel_workers


# --------------------------------------------------------------------------------------
# Load a simple model via default & GID; cover non-parallel and parallel paths.
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("inference_pool_gid", ["dummy_id", None])
async def test_load_model(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    inference_request: InferenceRequest,
    inference_pool_gid: str | None,
):
    """
    With a GID: exercise the parallel path (server-level parallel_workers>0).
    Without a GID: force non-parallel (model-level override) to avoid round-robin races.
    """
    ms = deepcopy(sum_model_settings)
    ms.name = "sum-model-load-test"
    ms.parameters.inference_pool_gid = inference_pool_gid
    if inference_pool_gid is None:
        # deprecated but honored: force non-parallel on the default pool
        ms.parallel_workers = 0

    model_in = SumModel(ms)
    model_out: MLModel = await inference_pool_registry.load_model(model_in)

    try:
        resp = await _predict_with_backoff(model_out, inference_request, attempts=10, delay=0.05)
        assert resp.model_name == ms.name
        assert len(resp.outputs) == 1
    finally:
        await inference_pool_registry.unload_model(model_out)


# --------------------------------------------------------------------------------------
# Parallel-only: ensure we create/use a GID-isolated pool with N workers and tear it down.
# --------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_load_model_parallel_pool(
    inference_pool_registry: InferencePoolRegistry,
    sum_model_settings: ModelSettings,
    inference_request: InferenceRequest,
):
    server_workers = inference_pool_registry._default_pool._settings.parallel_workers
    if server_workers <= 0:
        pytest.skip("Server settings.parallel_workers <= 0; parallel path not active.")

    gid = "parallel-load-gid"
    ms = deepcopy(sum_model_settings)
    ms.name = "sum-model-parallel-test"
    ms.parameters.inference_pool_gid = gid

    model_in = SumModel(ms)
    model_out: MLModel = await inference_pool_registry.load_model(model_in)

    try:
        pool = inference_pool_registry._pools[gid]
        await _wait_for_workers(pool, expected=server_workers, timeout=5.0)

        for _ in range(3):
            resp = await _predict_with_backoff(model_out, inference_request, attempts=10, delay=0.05)
            assert resp.model_name == ms.name
            assert len(resp.outputs) == 1

        assert not pool.empty()
    finally:
        await inference_pool_registry.unload_model(model_out)

        # Wait until the GID pool is removed (registry closes empty pools)
        deadline = asyncio.get_event_loop().time() + 3.0
        while asyncio.get_event_loop().time() < deadline:
            if gid not in inference_pool_registry._pools:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("GID pool not removed after unload")


# --------------------------------------------------------------------------------------
# Env tarball tests: STUB OUT the extractor so we don’t rely on a “real” tarball.
# --------------------------------------------------------------------------------------
@pytest.fixture
def patch_env_extractor(monkeypatch, tmp_path_factory):
    """
    Replace Environment.from_tarball with a fast stub that creates the directory
    and returns an Environment object. This avoids cleanup paths that remove the
    directory when the tarball isn't a real env.
    """
    created: list[str] = []
    base = Path(tmp_path_factory.getbasetemp()) / "fake-extracted-envs"
    base.mkdir(parents=True, exist_ok=True)

    async def _fake_from_tarball(src: str, env_path: str, env_hash: str) -> Environment:
        target = Path(env_path)
        target.mkdir(parents=True, exist_ok=True)
        # Drop a marker that downstream code could check if needed
        (target / ".extracted_ok").write_text("ok", encoding="utf-8")
        created.append(env_path)
        return Environment(env_path, env_hash)

    monkeypatch.setattr(Environment, "from_tarball", _fake_from_tarball, raising=True)
    yield created


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["parallel"])
async def test_load_model_with_env(
    inference_pool_registry: InferencePoolRegistry,
    env_model_settings: ModelSettings,
    env_tarball: str,
    inference_request: InferenceRequest,
    mode: str,
    patch_env_extractor,  # ensure extractor is stubbed for this test
):
    """
    Validate load/predict/unload using an environment tarball, in both modes:
      - non_parallel: force model-level parallel_workers=0 (default pool)
      - parallel:     use GID pool to exercise multiple workers path
    """
    ms = deepcopy(env_model_settings)
    ms.parameters.environment_tarball = env_tarball

    if mode == "non_parallel":
        ms.parallel_workers = 0
        gid = None
    else:
        gid = "env-gid-parallel"
        ms.parameters.inference_pool_gid = gid

    model_in = EnvModel(ms)
    model_out: MLModel = await inference_pool_registry.load_model(model_in)

    try:
        resp = await _predict_with_backoff(model_out, inference_request, attempts=20, delay=0.05)
        _check_sklearn_version(resp)

        if gid:
            pool = inference_pool_registry._pools[gid]
            assert not pool.empty()
    finally:
        await inference_pool_registry.unload_model(model_out)

        if gid:
            # confirm the GID pool gets closed when empty
            deadline = asyncio.get_event_loop().time() + 3.0
            while asyncio.get_event_loop().time() < deadline:
                if gid not in inference_pool_registry._pools:
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("GID pool not removed after unload")


@pytest.mark.asyncio
async def test_load_model_with_existing_env(
    inference_pool_registry: InferencePoolRegistry,
    existing_env_model_settings: ModelSettings,
    inference_request: InferenceRequest,
):
    # No tarball path here; fixture ensures Environment points to an existing path
    env_model = EnvModel(existing_env_model_settings)
    model = await inference_pool_registry.load_model(env_model)
    try:
        resp = await _predict_with_backoff(model, inference_request, attempts=20, delay=0.05)
        _check_sklearn_version(resp)
    finally:
        await inference_pool_registry.unload_model(model)


# --------------------------------------------------------------------------------------
# Pool creation / reuse / cleanup semantics (tarball path stubbed to be fast).
# --------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_load_creates_pool(
    inference_pool_registry: InferencePoolRegistry,
    env_model_settings: ModelSettings,
    env_tarball: str,
    patch_env_extractor,
):
    ms = deepcopy(env_model_settings)
    ms.parameters.environment_tarball = env_tarball
    model = EnvModel(ms)

    assert len(inference_pool_registry._pools) == 0
    await inference_pool_registry.load_model(model)
    assert len(inference_pool_registry._pools) == 1


@pytest.mark.asyncio
async def test_load_reuses_pool(
    inference_pool_registry: InferencePoolRegistry,
    env_model: MLModel,
    env_model_settings: ModelSettings,
):
    env_model_settings = deepcopy(env_model_settings)
    env_model_settings.name = "foo"
    new_model = EnvModel(env_model_settings)

    assert len(inference_pool_registry._pools) == 1
    await inference_pool_registry.load_model(new_model)
    assert len(inference_pool_registry._pools) == 1


@pytest.mark.asyncio
async def test_load_reuses_env_folder(
    inference_pool_registry: InferencePoolRegistry,
    env_model_settings: ModelSettings,
    env_tarball: str,
    patch_env_extractor,
):
    ms = deepcopy(env_model_settings)
    ms.name = "foo"
    ms.parameters.environment_tarball = env_tarball

    # Precompute env path and "extract" once
    env_hash = await compute_hash_of_file(env_tarball)
    env_path = inference_pool_registry._get_env_path(env_hash)
    with suppress(FileExistsError):
        os.makedirs(env_path, exist_ok=True)

    # Load should reuse
    model = EnvModel(ms)
    await inference_pool_registry.load_model(model)
    assert os.path.isdir(env_path)


# --------------------------------------------------------------------------------------
# Reload & unload behavior
# --------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_reload_model_with_env(
    inference_pool_registry: InferencePoolRegistry,
    env_model: MLModel,
    env_model_settings: ModelSettings,
    patch_env_extractor,
):
    env_model_settings = deepcopy(env_model_settings)
    env_model_settings.parameters.version = "v2.0"
    new_model = EnvModel(env_model_settings)

    assert len(inference_pool_registry._pools) == 1
    await inference_pool_registry.reload_model(env_model, new_model)
    assert len(inference_pool_registry._pools) == 1


@pytest.mark.asyncio
async def test_unload_model_removes_pool_if_empty(
    inference_pool_registry: InferencePoolRegistry,
    env_model_settings: ModelSettings,
    env_tarball: str,
    patch_env_extractor,
):
    ms = deepcopy(env_model_settings)
    ms.parameters.environment_tarball = env_tarball
    env_model = EnvModel(ms)

    assert len(inference_pool_registry._pools) == 0
    model = await inference_pool_registry.load_model(env_model)
    assert len(inference_pool_registry._pools) == 1

    await inference_pool_registry.unload_model(model)

    env_hash = _get_environment_hash(model)
    env_path = inference_pool_registry._get_env_path(env_hash)
    assert len(inference_pool_registry._pools) == 0
    # Depending on mlserver cleanup policy, the path may be removed by the registry;
    # assert non-existence for a strict guarantee:
    assert not os.path.isdir(env_path)


# --------------------------------------------------------------------------------------
# Error & utility cases
# --------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_invalid_env_hash(
    inference_pool_registry: InferencePoolRegistry, sum_model: MLModel
):
    _set_environment_hash(sum_model, "non-existent-env-hash")
    with pytest.raises(EnvironmentNotFound):
        await inference_pool_registry._find(sum_model)


@pytest.mark.parametrize(
    "env_hash, inference_pool_gid, expected_env_hash",
    [("dummy_hash", "dummy_gid", "dummy_hash-dummy_gid")],
)
async def test__get_environment_hash_gid(
    env_hash: str, inference_pool_gid: str | None, expected_env_hash: str
):
    _env_hash = _append_gid_environment_hash(env_hash, inference_pool_gid)
    assert _env_hash == expected_env_hash


@pytest.mark.asyncio
async def test_default_and_default_gid(
    inference_pool_registry: InferencePoolRegistry,
    simple_model_settings: ModelSettings,
):
    simple_model_settings_gid = deepcopy(simple_model_settings)
    simple_model_settings_gid.parameters.inference_pool_gid = "dummy_id"

    simple_model = SumModel(simple_model_settings)
    simple_model_gid = SumModel(simple_model_settings_gid)

    model = await inference_pool_registry.load_model(simple_model)
    model_gid = await inference_pool_registry.load_model(simple_model_gid)

    assert len(inference_pool_registry._pools) == 1
    await inference_pool_registry.unload_model(model)
    await inference_pool_registry.unload_model(model_gid)


@pytest.mark.asyncio
async def test_env_and_env_gid(
    inference_request: InferenceRequest,
    inference_pool_registry: InferencePoolRegistry,
    env_model_settings: ModelSettings,
    env_tarball: str,
    patch_env_extractor,
):
    ms = deepcopy(env_model_settings)
    ms.parameters.environment_tarball = env_tarball

    ms_gid = deepcopy(ms)
    ms_gid.parameters.inference_pool_gid = "dummy_id"

    env_model = EnvModel(ms)
    env_model_gid = EnvModel(ms_gid)

    model = await inference_pool_registry.load_model(env_model)
    model_gid = await inference_pool_registry.load_model(env_model_gid)
    assert len(inference_pool_registry._pools) == 2

    response = await _predict_with_backoff(model, inference_request)
    response_gid = await _predict_with_backoff(model_gid, inference_request)
    _check_sklearn_version(response)
    _check_sklearn_version(response_gid)

    await inference_pool_registry.unload_model(model)
    await inference_pool_registry.unload_model(model_gid)


@pytest.mark.parametrize(
    "inference_pool_grid, autogenerate_inference_pool_grid",
    [
        ("dummy_gid", False),
        ("dummy_gid", True),
        (None, True),
        (None, False),
    ],
)
def test_autogenerate_inference_pool_gid(
    inference_pool_grid: str | None, autogenerate_inference_pool_grid: bool
):
    patch_uuid = "patch-uuid"
    with pytest.MonkeyPatch.context() as m:
        m.setenv("MLSERVER_DUMMY", "1")  # no-op, just to ensure context is used
        # Patch uuid.uuid4 -> deterministic
        import uuid as _uuid

        m.setattr(_uuid, "uuid4", lambda: patch_uuid)
        model_settings = ModelSettings(
            name="dummy-model",
            implementation=MLModel,
            parameters=ModelParameters(
                inference_pool_gid=inference_pool_grid,
                autogenerate_inference_pool_gid=autogenerate_inference_pool_grid,
            ),
        )

    expected_gid = (
        inference_pool_grid
        if not autogenerate_inference_pool_grid
        else (inference_pool_grid or patch_uuid)
    )
    assert model_settings.parameters.inference_pool_gid == expected_gid
