import asyncio
import pickle
from types import SimpleNamespace

import pytest
from mlserver.errors import InferenceError
from mlserver.settings import ModelSettings
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.utils import generate_uuid

from woprserver.parallel.dispatcher import Dispatcher
from woprserver.parallel.messages import (
    InputChannelRef,
    ModelRequestMessage,
    ModelResponseMessage,
    ModelStreamChunkMessage,
    ModelStreamEndMessage,
)
from woprserver.parallel.model import ModelMethods
from woprserver.parallel.pool import InferencePool


async def _start_stream(dispatcher: Dispatcher, req: ModelRequestMessage):
    # Kick off the call that returns the async generator
    task = asyncio.create_task(dispatcher.dispatch_request_stream(req))
    await asyncio.sleep(0)  # let Dispatcher schedule its internals

    # NEW: immediately send the response that unblocks dispatch_request_stream
    dispatcher._responses.put(ModelResponseMessage(id=req.id, return_value=None))

    # Now the async generator will be ready
    return await task

def mk_metadata_msg(ms: ModelSettings) -> ModelRequestMessage:
    return ModelRequestMessage(
        id=generate_uuid(),
        model_name=ms.name,
        method_name=ModelMethods.Metadata.value,
        method_args=[],
        method_kwargs={},
    )

def mk_predict_msg(ms: ModelSettings, req: InferenceRequest) -> ModelRequestMessage:
    return ModelRequestMessage(
        id=generate_uuid(),
        model_name=ms.name,
        model_version=ms.parameters.version,
        method_name=ModelMethods.Predict.value,
        method_args=[req],
        method_kwargs={},
    )

def mk_stream_msg(ms: ModelSettings, method_name: str = "stream_tokens") -> ModelRequestMessage:
    return ModelRequestMessage(
        id=generate_uuid(),
        model_name=ms.name,
        model_version=ms.parameters.version,
        method_name=method_name,
        method_args=[],
        method_kwargs={},
    )

@pytest.mark.asyncio
async def test_worker_count(inference_pool: InferencePool):
    assert len(inference_pool._workers) == inference_pool._settings.parallel_workers == 2
    pids = [w.pid for w in inference_pool._workers.values()]
    assert len(pids) == len(set(pids))
    assert all(inference_pool._workers[pid].is_alive() for pid in pids)  # type: ignore[index]


@pytest.mark.asyncio
async def test_stream_demux_and_cleanup(inference_pool: InferencePool, parallel, sum_model_settings: ModelSettings):
    dispatcher: Dispatcher = inference_pool._dispatcher
    req = mk_stream_msg(sum_model_settings, "stream_tokens")
    agen = await _start_stream(dispatcher, req)

    dispatcher._responses.put(ModelStreamChunkMessage(id=req.id, chunk=b"a"))
    dispatcher._responses.put(ModelStreamChunkMessage(id=req.id, chunk=b"b"))
    dispatcher._responses.put(ModelStreamEndMessage(id=req.id))
    dispatcher._responses.put(ModelResponseMessage(id=req.id, return_value=None))

    seen: list[bytes] = []
    async for ch in agen:
        seen.append(ch)

    assert set(seen) == {b"a", b"b"}
    assert req.id not in dispatcher._async_responses._streams


@pytest.mark.asyncio
async def test_stream_error_propagation(inference_pool: InferencePool, parallel, sum_model_settings: ModelSettings):
    dispatcher: Dispatcher = inference_pool._dispatcher
    req = mk_stream_msg(sum_model_settings, "stream_err")
    agen = await _start_stream(dispatcher, req)
    dispatcher._responses.put(ModelStreamChunkMessage(id=req.id, chunk=b"x"))
    dispatcher._responses.put(ModelStreamEndMessage(id=req.id, exception=RuntimeError("boom")))
    dispatcher._responses.put(ModelResponseMessage(id=req.id, return_value=None))

    first = await agen.__anext__()
    assert first == b"x"
    with pytest.raises(RuntimeError) as err:
        await agen.__anext__()
    assert "boom" in str(err.value)
    assert req.id not in dispatcher._async_responses._streams


@pytest.mark.asyncio
async def test_stream_interleaving(inference_pool: InferencePool, parallel, sum_model_settings: ModelSettings):
    dispatcher: Dispatcher = inference_pool._dispatcher
    req_a = mk_stream_msg(sum_model_settings, "stream_A")
    req_b = mk_stream_msg(sum_model_settings, "stream_B")

    agen_a = await _start_stream(dispatcher, req_a)
    agen_b = await _start_stream(dispatcher, req_b)

    dispatcher._responses.put(ModelStreamChunkMessage(id=req_a.id, chunk=b"a1"))
    dispatcher._responses.put(ModelStreamChunkMessage(id=req_b.id, chunk=b"b1"))
    dispatcher._responses.put(ModelStreamChunkMessage(id=req_a.id, chunk=b"a2"))
    dispatcher._responses.put(ModelStreamChunkMessage(id=req_b.id, chunk=b"b2"))
    dispatcher._responses.put(ModelStreamEndMessage(id=req_a.id))
    dispatcher._responses.put(ModelStreamEndMessage(id=req_b.id))
    dispatcher._responses.put(ModelResponseMessage(id=req_a.id, return_value=None))
    dispatcher._responses.put(ModelResponseMessage(id=req_b.id, return_value=None))

    seen_a: list[bytes] = []
    seen_b: list[bytes] = []

    async def drain(gen, bucket: list[bytes]):
        async for ch in gen:
            bucket.append(ch)

    await asyncio.gather(drain(agen_a, seen_a), drain(agen_b, seen_b))
    assert seen_a == [b"a1", b"a2"]
    assert seen_b == [b"b1", b"b2"]
    assert req_a.id not in dispatcher._async_responses._streams
    assert req_b.id not in dispatcher._async_responses._streams


@pytest.mark.asyncio
async def test_stream_backpressure(inference_pool: InferencePool, parallel, sum_model_settings: ModelSettings, monkeypatch):
    dispatcher: Dispatcher = inference_pool._dispatcher
    real_create = dispatcher._async_responses.create_stream_queue
    monkeypatch.setattr(
        dispatcher._async_responses,
        "create_stream_queue",
        lambda msg_id, maxsize=16: real_create(msg_id, maxsize=256),
    )

    req = mk_stream_msg(sum_model_settings, "stream_many")
    agen = await _start_stream(dispatcher, req)

    collected: list[bytes] = []

    async def consume():
        async for ch in agen:
            collected.append(ch)
            await asyncio.sleep(0)

    consumer_task = asyncio.create_task(consume())

    for i in range(50):
        dispatcher._responses.put(ModelStreamChunkMessage(id=req.id, chunk=f"c{i}".encode()))
        await asyncio.sleep(0)

    dispatcher._responses.put(ModelStreamEndMessage(id=req.id))
    dispatcher._responses.put(ModelResponseMessage(id=req.id, return_value=None))

    await consumer_task
    assert set(collected) == {f"c{i}".encode() for i in range(50)}
    assert req.id not in dispatcher._async_responses._streams


@pytest.mark.asyncio
async def test_stream_consumer_cancel(inference_pool: InferencePool, parallel, sum_model_settings: ModelSettings):
    dispatcher: Dispatcher = inference_pool._dispatcher
    req = mk_stream_msg(sum_model_settings, "stream_cancel")
    agen = await _start_stream(dispatcher, req)

    dispatcher._responses.put(ModelStreamChunkMessage(id=req.id, chunk=b"head"))
    dispatcher._responses.put(ModelResponseMessage(id=req.id, return_value=None))

    head = await agen.__anext__()
    assert head == b"head"

    await agen.aclose()
    assert req.id not in dispatcher._async_responses._streams


def test_input_channel_ref_is_picklable():
    ref = InputChannelRef(id=generate_uuid(), channel_id="in-0")
    # Ensure no unpicklable payload leaks across process boundary
    data = pickle.dumps(ref)
    ref2 = pickle.loads(data)
    assert ref2.id == ref.id and ref2.channel_id == ref.channel_id


class _RaisingFirstItem:
    def __aiter__(self): return self
    async def __anext__(self): raise RuntimeError("boom")
    async def aclose(self): self.closed = True

@pytest.mark.asyncio
async def test_unary_fallback_first_item_raises(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher
    a_it = _RaisingFirstItem()
    a_it.closed = False
    b_it = _RaisingFirstItem()
    b_it.closed = False
    with pytest.raises(InferenceError, match="boom"):
        agen = pm._send_stream("add", a_it, b=b_it)  # type: ignore[attr-defined]
        async for _ in agen:
            pass
    assert a_it.closed and b_it.closed

@pytest.mark.asyncio
async def test_custom_handler_precedence(pm_and_dispatcher, mocker):
    pm, dispatcher, model = pm_and_dispatcher

    async def custom_predict(req: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(model_name=pm.settings.name, outputs=[])

    mocker.patch("woprserver.parallel.model.get_custom_handlers",
                 return_value=[("predict", custom_predict)])
    mocker.patch("woprserver.parallel.model.register_custom_handler",
                 side_effect=lambda name, fn: setattr(pm, name, fn))
    pm._register_custom_handlers()

    async def _fake(req_msg):
        assert req_msg.method_name == "custom_predict"  # custom path chosen
        return SimpleNamespace(return_value=InferenceResponse(model_name=pm.settings.name, outputs=[]))

    mocker.patch.object(dispatcher, "dispatch_request", side_effect=_fake)
    out = await pm.predict(InferenceRequest(inputs=[]))
    assert isinstance(out, InferenceResponse)
