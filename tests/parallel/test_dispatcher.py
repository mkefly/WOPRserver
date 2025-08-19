import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from mlserver.utils import generate_uuid

from woprserver.parallel.dispatcher import Dispatcher
from woprserver.parallel.messages import (
    ModelRequestMessage,
    ModelResponseMessage,
    ModelStreamChunkMessage,
    ModelStreamEndMessage,
    ModelStreamInputChunk,
    ModelStreamInputEnd,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_request(
    method_name: str = "predict", *, model: str = "m", version: str = "1"
) -> ModelRequestMessage:
    return ModelRequestMessage(
        id=generate_uuid(),
        model_name=model,
        model_version=version,
        method_name=method_name,
        method_args=[],
        method_kwargs={},
    )


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dispatch_streaming_happy_path(dispatcher: Dispatcher):
    stream_req = make_request(method_name="stream_tokens")

    async def feeder():
        await asyncio.sleep(0)
        dispatcher._responses.put(ModelStreamChunkMessage(id=stream_req.id, chunk=b"a"))
        dispatcher._responses.put(ModelStreamChunkMessage(id=stream_req.id, chunk=b"b"))
        dispatcher._responses.put(ModelStreamEndMessage(id=stream_req.id))
        dispatcher._responses.put(ModelResponseMessage(id=stream_req.id, return_value=None))

    feeder_task = asyncio.create_task(feeder())

    chunks: list[bytes] = []
    agen = await dispatcher.dispatch_request_stream(stream_req)
    async for ch in agen:
        chunks.append(ch)

    # Ensure background producer finished (and satisfy RUF006)
    await feeder_task

    assert chunks == [b"a", b"b"]


@pytest.mark.asyncio
async def test_dispatch_streaming_error_propagates(dispatcher: Dispatcher):
    stream_req = make_request(method_name="stream_tokens_err")

    async def feeder():
        await asyncio.sleep(0)
        dispatcher._responses.put(ModelStreamChunkMessage(id=stream_req.id, chunk=b"x"))
        dispatcher._responses.put(
            ModelStreamEndMessage(id=stream_req.id, exception=RuntimeError("fail mid-stream"))
        )
        dispatcher._responses.put(ModelResponseMessage(id=stream_req.id, return_value=None))

    feeder_task = asyncio.create_task(feeder())

    agen = await dispatcher.dispatch_request_stream(stream_req)
    first = await agen.__anext__()
    assert first == b"x"

    with pytest.raises(RuntimeError) as err:
        await agen.__anext__()

    await feeder_task

    assert "fail mid-stream" in str(err.value)


@pytest.mark.asyncio
async def test_stream_ignores_wrong_id_and_malformed_objects(dispatcher: Dispatcher):
    stream_req = make_request(method_name="stream_tokens")

    class BadThing:
        pass

    async def feeder():
        await asyncio.sleep(0)
        dispatcher._responses.put(ModelStreamChunkMessage(id="WRONG-ID", chunk=b"?"))
        dispatcher._responses.put(BadThing())
        dispatcher._responses.put(ModelStreamChunkMessage(id=stream_req.id, chunk=b"ok"))
        dispatcher._responses.put(ModelStreamEndMessage(id=stream_req.id))
        dispatcher._responses.put(ModelResponseMessage(id=stream_req.id, return_value=None))

    feeder_task = asyncio.create_task(feeder())

    agen = await dispatcher.dispatch_request_stream(stream_req)
    out = [chunk async for chunk in agen]

    await feeder_task

    assert out == [b"ok"]


@pytest.mark.asyncio
async def test_stream_queue_is_cleaned_up_on_end(dispatcher: Dispatcher):
    stream_req = make_request(method_name="stream_tokens")

    async def feeder():
        await asyncio.sleep(0)
        dispatcher._responses.put(ModelStreamEndMessage(id=stream_req.id))
        dispatcher._responses.put(ModelResponseMessage(id=stream_req.id, return_value=None))

    feeder_task = asyncio.create_task(feeder())

    agen = await dispatcher.dispatch_request_stream(stream_req)
    res = [c async for c in agen]

    await feeder_task

    assert res == []
    assert dispatcher._async_responses._streams.get(stream_req.id) is None


@pytest.mark.asyncio
async def test_bridge_async_iterable_arg_sends_chunks_and_end(dispatcher: Dispatcher):
    """
    Ensure bridge sends chunk -> chunk -> end to the worker's request queue.
    """
    worker = next(iter(dispatcher._workers.values()))
    req_id = generate_uuid()
    channel = "in-0"

    async def payloads() -> AsyncIterator[bytes]:
        yield b"one"
        yield b"two"

    await dispatcher._bridge_async_iterable_arg(worker, req_id, payloads(), channel_id=channel)

    got: list[Any] = []
    deadline = time.monotonic() + 5.0
    while len(got) < 3 and time.monotonic() < deadline:
        try:
            msg = worker._requests.get(timeout=0.2)  # type: ignore[attr-defined]
        except Exception:
            continue
        if (
            isinstance(msg, (ModelStreamInputChunk, ModelStreamInputEnd))
            and msg.id == req_id
            and getattr(msg, "channel_id", "") == channel
        ):
            got.append(msg)

    assert got and isinstance(got[0], ModelStreamInputChunk)
    assert got[0].item == b"one"
    assert isinstance(got[1], ModelStreamInputChunk) and got[1].item == b"two"
    assert isinstance(got[2], ModelStreamInputEnd)


# ---------------------------------------------------------------------------
# Additional coverage
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_unary_dispatch_round_trip(dispatcher: Dispatcher):
    req = make_request(method_name="predict")

    async def feeder():
        # Simulate a worker putting a unary response on the responses queue
        await asyncio.sleep(0)
        dispatcher._responses.put(ModelResponseMessage(id=req.id, return_value={"ok": True}))

    feeder_task = asyncio.create_task(feeder())

    # Call the public API; background response loop should resolve the future
    resp = await dispatcher.dispatch_request(req)

    await feeder_task

    assert isinstance(resp, ModelResponseMessage)
    assert resp.id == req.id
    assert resp.return_value == {"ok": True}


@pytest.mark.asyncio
async def test_start_stop_idempotent(dispatcher: Dispatcher):
    # Start/stop multiple times should not raise and should shut down cleanly
    dispatcher.start()
    dispatcher.start()
    await dispatcher.stop()
    await dispatcher.stop()


@pytest.mark.asyncio
async def test_no_workers_raises(dispatcher: Dispatcher):
    # Preserve current workers and restore after to avoid leaking state to other tests
    original_workers = dict(dispatcher._workers)
    try:
        dispatcher._workers.clear()
        dispatcher._reset_round_robin()
        with pytest.raises(RuntimeError):
            await dispatcher.dispatch_request(make_request())
    finally:
        dispatcher._workers = original_workers
        dispatcher._reset_round_robin()


@pytest.mark.asyncio
async def test_stream_consumer_cancellation_cleanup(dispatcher: Dispatcher):
    stream_req = make_request(method_name="stream_tokens")

    async def feeder():
        await asyncio.sleep(0)
        dispatcher._responses.put(ModelStreamChunkMessage(id=stream_req.id, chunk=b"first"))
        await asyncio.sleep(0)
        dispatcher._responses.put(ModelStreamEndMessage(id=stream_req.id))
        dispatcher._responses.put(ModelResponseMessage(id=stream_req.id, return_value=None))

    feeder_task = asyncio.create_task(feeder())

    agen = await dispatcher.dispatch_request_stream(stream_req)

    # Consume one chunk, then explicitly close the generator
    first = await agen.__anext__()
    assert first == b"first"
    await agen.aclose()  # <-- important: runs agen()'s finally (cancels ack_task)

    # Also bound the wait for safety
    await asyncio.wait_for(feeder_task, timeout=1.0)

    # Stream mapping should already be gone (also true even without aclose(), thanks to auto-cleanup on END)
    assert dispatcher._async_responses._streams.get(stream_req.id) is None

