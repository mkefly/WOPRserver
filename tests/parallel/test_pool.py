import asyncio
import os
import time
from collections.abc import Awaitable, Callable
from copy import deepcopy
from typing import Any

import numpy as np
import pytest
from mlserver.codecs import NumpyCodec, StringCodec
from mlserver.errors import MLServerError
from mlserver.model import MLModel
from mlserver.settings import ModelSettings, Settings
from mlserver.types import InferenceRequest, InferenceResponse

from woprserver.parallel.pool import InferencePool
from woprserver.parallel.utils import configure_inference_pool
from woprserver.parallel.errors import WorkerError

from .fixtures import ErrorModel, PidStreamModel, PidUnaryModel, SumModel

# Tight overall deadlines, but avoid wrapping every single await with wait_for.
# We retry quickly on transient CancelledError/None-response while the dispatcher restarts.
PER_CALL_DEADLINE = 8.0            # max time to get a single successful op
SLEEP_BETWEEN_RETRIES = 0.05       # tiny pause between retries


# ---------------- small resiliency helpers ----------------

async def eventually(fn: Callable[[], Awaitable[Any]], *, deadline: float = PER_CALL_DEADLINE) -> Any:
    """
    Keep calling `await fn()` until it succeeds or overall deadline expires.
    Retries on transient asyncio.CancelledError and TimeoutError.
    Never blocks the loop; inserts tiny sleeps between retries.
    """
    start = time.monotonic()
    last_exc: BaseException | None = None
    while True:
        try:
            return await fn()
        except (asyncio.CancelledError, asyncio.TimeoutError):
            last_exc = None  # transient; don't keep noisy tracebacks
        except Exception as e:
            # If the dispatcher momentarily restarts and gives us odd states, retry briefly.
            last_exc = e
        if time.monotonic() - start > deadline:
            if last_exc:
                raise last_exc
            raise asyncio.TimeoutError("eventually() overall deadline exceeded")
        await asyncio.sleep(SLEEP_BETWEEN_RETRIES)


# ---------------- helpers ----------------

def check_pid(pid: int) -> bool:
    """
    Check for the existence of a unix pid.
    From https://stackoverflow.com/a/568285/5015573
    """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

# tiny async iterator that yields exactly one request (never pass None)
async def _one_req(req: InferenceRequest):
    yield req


async def _run_one_stream_once(pm, req: InferenceRequest) -> str:
    """
    Consume a single-chunk stream and return the PID.
    Resilient to brief dispatcher restarts.
    """
    async def _consume() -> str:
        gen = pm.predict_stream(_one_req(req))
        try:
            async for chunk in gen:
                return StringCodec.decode_output(chunk.outputs[0])[0]
        finally:
            try:
                await gen.aclose()
            except Exception:
                pass
        raise RuntimeError("Stream ended without a chunk")

    return await eventually(_consume)



# ---------------- fixtures ----------------

@pytest.fixture
def trivial_req() -> InferenceRequest:
    # Minimal BYTES input (use_bytes=True ensures BYTES for the simple pid models)
    from mlserver.codecs.string import StringCodec as _SC
    return InferenceRequest(inputs=[_SC.encode_input("input", payload="x", use_bytes=True)])


@pytest.mark.asyncio
@pytest.fixture
async def fresh_pool(settings: Settings):
    """
    Function-scoped clean pool for tests that need a pristine state and quick close.
    """
    configure_inference_pool(settings)
    pool = InferencePool(settings)
    try:
        yield pool
    finally:
        # Close without extra wait_for; tolerate brief queue contention
        try:
            await eventually(lambda: pool.close())
        except Exception:
            # Best-effort teardown; don't make the suite fail here
            pass


# ---------------- Worker-side PID models ----------------

@pytest.mark.asyncio
async def test_reload_model_same_name_version(
    fresh_pool: InferencePool,
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    # Load an initial instance
    pm1 = await eventually(lambda: fresh_pool.load_model(sum_model))
    out1 = await eventually(lambda: pm1.predict(inference_request))
    assert len(out1.outputs) == 1

    # Build a *new* instance with the same name+version (simulates in-place upgrade)
    new_settings = deepcopy(sum_model.settings)
    new_model = type(sum_model)(new_settings)  # same class, fresh instance

    # Reload should issue Unload(old) + Load(new) without growing the registry
    start_count = len(fresh_pool._worker_registry)
    pm2 = await eventually(lambda: fresh_pool.reload_model(sum_model, new_model))
    assert len(fresh_pool._worker_registry) == start_count

    out2 = await eventually(lambda: pm2.predict(inference_request))
    assert len(out2.outputs) == 1


@pytest.mark.asyncio
async def test_unload_model_drops_registry_and_keeps_pool_healthy(
    fresh_pool: InferencePool,
    sum_model: MLModel,
):
    start_count = len(fresh_pool._worker_registry)
    _ = await eventually(lambda: fresh_pool.load_model(sum_model))
    assert len(fresh_pool._worker_registry) == start_count + 1

    # Unload returns the underlying model (not ParallelModel)
    returned = await eventually(lambda: fresh_pool.unload_model(sum_model))
    assert isinstance(returned, MLModel)
    assert len(fresh_pool._worker_registry) == start_count

    # The pool should still be alive and usable (no workers leaked)
    assert len(fresh_pool._workers) >= 1


# A simple model that uses typed args and returns a raw ndarray (to exercise worker decoding + coercion)
class _RawArrayModel(MLModel):
    async def predict(self, foo: np.ndarray, bar: list[str]) -> np.ndarray:
        # returns raw array; worker should coerce to InferenceResponse for 'predict'
        return foo.sum(axis=1, keepdims=True)

@pytest.mark.asyncio
async def test_worker_restart_on_kill(
    fresh_pool: InferencePool,
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    pm = await eventually(lambda: fresh_pool.load_model(sum_model))

    # Ensure workers are hot
    _ = await eventually(lambda: pm.predict(inference_request))

    # Pick one worker PID and simulate a crash; then trigger restart flow
    assert fresh_pool._workers, "expected at least one worker"
    victim_pid = next(iter(fresh_pool._workers.keys()))

    # Simulate the dispatcher notifying the pool about a crashed worker
    await fresh_pool.on_worker_stop(victim_pid, exit_code=1)

    # We should be able to continue serving traffic while the pool replaces it
    out = await eventually(lambda: pm.predict(inference_request))
    assert len(out.outputs) == 1

    # And the new worker set should contain at least one live PID
    assert len(fresh_pool._workers) >= 1
    for pid in list(fresh_pool._workers.keys()):
        assert check_pid(pid)


@pytest.mark.asyncio
async def test_close_is_idempotent(fresh_pool: InferencePool):
    # First close
    await eventually(lambda: fresh_pool.close())
    assert len(fresh_pool._workers) == 0

    # Second close should be a no-op and not raise
    await eventually(lambda: fresh_pool.close())
    assert len(fresh_pool._workers) == 0


@pytest.mark.asyncio
async def test_streaming_multiple_chunks_and_cleanup(
    fresh_pool: InferencePool,
    pid_stream_settings: ModelSettings,
):
    """
    Validate stream consumption and ensure the generator is closable without leaks.
    """
    model = PidStreamModel(deepcopy(pid_stream_settings))
    pm = await eventually(lambda: fresh_pool.load_model(model))

    reqs = [
        InferenceRequest(inputs=[StringCodec.encode_input("input", payload=f"msg-{i}", use_bytes=True)])
        for i in range(3)
    ]

    # Consume three single-chunk streams consecutively
    for rq in reqs:
        gen = pm.predict_stream(_one_req(rq))
        try:
            got_one = False
            async for chunk in gen:
                assert chunk.outputs and chunk.outputs[0]
                _ = StringCodec.decode_output(chunk.outputs[0])[0]
                got_one = True
                break  # single-chunk stream
            assert got_one
        finally:
            # aclose should be safe / idempotent
            try:
                await gen.aclose()
            except Exception:
                pass


@pytest.mark.asyncio
async def test_pool_uses_two_workers_unary(
    fresh_pool: InferencePool,
    pid_unary_settings: ModelSettings,
    trivial_req: InferenceRequest,
):
    model = PidUnaryModel(deepcopy(pid_unary_settings))
    pm = await eventually(lambda: fresh_pool.load_model(model))

    # Warm up to ensure workers are ready
    _ = await eventually(lambda: pm.predict(trivial_req))

    # Minimal number of sequential requests to observe 2 distinct workers
    seen: set[str] = set()
    for _ in range(4):
        resp = await eventually(lambda: pm.predict(trivial_req))
        pid = StringCodec.decode_output(resp.outputs[0])[0]
        seen.add(pid)
        if len(seen) >= 2:
            break

    await eventually(lambda: fresh_pool.unload_model(model))
    assert len(seen) >= 2, f"Expected >=2 worker PIDs, saw: {seen}"


@pytest.mark.asyncio
async def test_pool_uses_two_workers_streaming(
    fresh_pool: InferencePool,
    pid_stream_settings: ModelSettings,
    trivial_req: InferenceRequest,
):
    model = PidStreamModel(deepcopy(pid_stream_settings))
    pm = await eventually(lambda: fresh_pool.load_model(model))

    # Warm up one stream so workers are hot
    _ = await _run_one_stream_once(pm, trivial_req)

    # Two quick sequential streams to catch two PIDs
    seen_pids: set[str] = set()
    for _ in range(4):
        pid = await _run_one_stream_once(pm, trivial_req)
        seen_pids.add(pid)
        if len(seen_pids) >= 2:
            break

    await eventually(lambda: fresh_pool.unload_model(model))
    assert len(seen_pids) >= 2, f"Expected streams to hit 2 workers, got PIDs: {seen_pids}"


# ---------------- pool lifecycle & behavior ----------------

@pytest.mark.asyncio
async def test_start_worker(
    settings: Settings,
    fresh_pool: InferencePool,
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    model = await eventually(lambda: fresh_pool.load_model(sum_model))

    # Start a new worker and keep serving traffic while it spins up
    start_worker_task = asyncio.create_task(fresh_pool._start_worker())

    async def _predict_once():
        out = await model.predict(inference_request)
        assert len(out.outputs) == 1

    # Keep traffic flowing until the worker task completes (with resiliency)
    while not start_worker_task.done():
        await eventually(_predict_once)
        await asyncio.sleep(0)

    await eventually(lambda: start_worker_task)


@pytest.mark.asyncio
async def test_start_worker_new_model(
    settings: Settings,
    fresh_pool: InferencePool,
    sum_model: MLModel,
    simple_model: MLModel,
):
    await eventually(lambda: fresh_pool.load_model(sum_model))

    start_worker_task = asyncio.create_task(fresh_pool._start_worker())
    new_model = await eventually(lambda: fresh_pool.load_model(simple_model))

    req = InferenceRequest(
        inputs=[
            NumpyCodec.encode_input("foo", np.array([[1, 2]], dtype=np.int32)),
            StringCodec.encode_input("bar", ["asd", "qwe"]),
        ]
    )

    async def _predict_once():
        out = await new_model.predict(req)
        assert len(out.outputs) == 1

    while not start_worker_task.done():
        await eventually(_predict_once)
        await asyncio.sleep(0)

    await eventually(lambda: start_worker_task)

    # Final quick pass
    for _ in range(settings.parallel_workers + 1):
        await eventually(_predict_once)


@pytest.mark.asyncio
async def test_close(fresh_pool: InferencePool):
    worker_pids = list(fresh_pool._workers.keys())
    await eventually(lambda: fresh_pool.close())

    assert len(fresh_pool._workers) == 0
    for pid in worker_pids:
        assert not check_pid(pid)


@pytest.mark.asyncio
async def test_load(
    fresh_pool: InferencePool,
    sum_model: MLModel,
    inference_request: InferenceRequest,
):
    # don’t mutate the shared instance’s name — build a fresh one
    settings = deepcopy(sum_model.settings)
    settings.name = "foo"
    model = SumModel(settings)

    start = len(fresh_pool._worker_registry)
    pm = await eventually(lambda: fresh_pool.load_model(model))
    assert len(fresh_pool._worker_registry) == start + 1

    out = await eventually(lambda: pm.predict(inference_request))
    assert out.model_name == settings.name
    assert len(out.outputs) == 1

    await eventually(lambda: fresh_pool.unload_model(model))
    assert len(fresh_pool._worker_registry) == start

@pytest.mark.asyncio
async def test_load_error(fresh_pool: InferencePool, load_error_model: MLModel):
    start = len(fresh_pool._worker_registry)
    with pytest.raises(WorkerError) as excinfo:
        await eventually(lambda: fresh_pool.load_model(load_error_model))

    assert len(fresh_pool._worker_registry) == start
    msg = str(excinfo.value)
    assert "MLServerError" in msg
    assert "something really bad happened" in msg

def test_workers_start(fresh_pool: InferencePool, settings: Settings):
    # after init, we should have at least the configured number of workers
    assert len(fresh_pool._workers) >= settings.parallel_workers
    for pid in list(fresh_pool._workers.keys()):
        assert check_pid(pid)
