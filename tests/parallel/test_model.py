import asyncio
import inspect
from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from mlserver.codecs import StringCodec
from mlserver.errors import InferenceError
from mlserver.types import InferenceRequest, InferenceResponse, MetadataModelResponse
from mlserver.utils import generate_uuid

from woprserver.parallel.messages import InputChannelRef
from woprserver.parallel.model import ModelMethods

from .fixtures import (
    ClosableInput,
    TextModel,
    TextStreamModel,
    _EmptyAsyncIter,
    _ServerAsyncGen,
)


@pytest.mark.asyncio
async def test_predict_returns_inference_response(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher
    expected = InferenceResponse(model_name=pm.settings.name, outputs=[])

    async def _fake(req_msg):
        assert req_msg.method_name == ModelMethods.Predict.value
        assert len(req_msg.method_args) == 1 and isinstance(req_msg.method_args[0], InferenceRequest)
        return SimpleNamespace(return_value=expected)

    mocker.patch.object(dispatcher, "dispatch_request", side_effect=_fake)

    out = await pm.predict(InferenceRequest(inputs=[]))
    assert out is expected


@pytest.mark.asyncio
async def test_predict_raises_on_wrong_return_type(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    async def _fake(_req_msg):
        return SimpleNamespace(return_value="not-a-response")

    mocker.patch.object(dispatcher, "dispatch_request", side_effect=_fake)

    with pytest.raises(InferenceError):
        await pm.predict(InferenceRequest(inputs=[]))


@pytest.mark.asyncio
async def test_metadata_caches_and_returns_single_dispatch(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher
    expected = MetadataModelResponse(name=pm.settings.name, platform="x")

    async def _fake(_req_msg):
        return SimpleNamespace(return_value=expected)

    spy = mocker.patch.object(dispatcher, "dispatch_request", side_effect=_fake)

    m1 = await pm.metadata()
    m2 = await pm.metadata()
    m3 = await pm.metadata()
    assert m1 is expected and m2 is expected and m3 is expected
    spy.assert_called_once()


@pytest.mark.asyncio
async def test_metadata_cache_is_concurrency_safe(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher
    expected = MetadataModelResponse(name=pm.settings.name, platform="x")

    async def _fake(_req_msg):
        await asyncio.sleep(0.05)
        return SimpleNamespace(return_value=expected)

    spy = mocker.patch.object(dispatcher, "dispatch_request", side_effect=_fake)

    out = await asyncio.gather(*[pm.metadata() for _ in range(10)])
    assert all(o is expected for o in out)
    spy.assert_called_once()


@pytest.mark.asyncio
async def test_metadata_raises_on_wrong_type(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    async def _fake(_req_msg):
        return SimpleNamespace(return_value=None)

    mocker.patch.object(dispatcher, "dispatch_request", side_effect=_fake)

    with pytest.raises(InferenceError):
        await pm.metadata()


@pytest.mark.asyncio
async def test_predict_stream_replaces_input_with_channel_ref(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher
    r1 = InferenceResponse(model_name=pm.settings.name, id=generate_uuid(), outputs=[])
    r2 = InferenceResponse(model_name=pm.settings.name, id=generate_uuid(), outputs=[])

    async def _fake_stream(req_msg) -> AsyncIterator[InferenceResponse]:
        assert req_msg.method_name == ModelMethods.PredictStream.value
        assert len(req_msg.method_args) == 1
        ref = req_msg.method_args[0]
        assert isinstance(ref, InputChannelRef)
        assert ref.id == req_msg.id
        async def _gen():
            yield r1
            yield r2
        return _gen()

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    async def _reqs() -> AsyncIterator[InferenceRequest]:
        yield InferenceRequest(inputs=[])
        yield InferenceRequest(inputs=[])

    seen: list[InferenceResponse] = []
    async for ch in pm.predict_stream(_reqs()):
        seen.append(ch)

    assert seen == [r1, r2]


@pytest.mark.asyncio
async def test_unary_fallback_consumes_first_item_and_closes_inputs(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    a_it = ClosableInput([10, 999])
    b_it = ClosableInput([20, 999])

    got_args: list | None = None
    got_kwargs: dict | None = None

    async def _fake_stream(req_msg):
        nonlocal got_args, got_kwargs
        assert req_msg.method_name == "add"
        got_args = req_msg.method_args
        got_kwargs = req_msg.method_kwargs
        async def _gen():
            yield InferenceResponse(model_name=pm.settings.name, outputs=[])
        return _gen()

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    agen = pm._send_stream("add", a_it, b=b_it)  # type: ignore[attr-defined]
    async for _ in agen:
        pass

    assert got_args == [10]
    assert got_kwargs == {"b": 20}
    assert a_it.closed is True and b_it.closed is True


@pytest.mark.asyncio
async def test_multiple_async_inputs_create_distinct_channels_and_bridge(pm_and_dispatcher, mocker):
    pm, dispatcher, model = pm_and_dispatcher

    mocker.patch("woprserver.parallel.model.get_custom_handlers", return_value=[("multi", model.multi)])
    mocker.patch("woprserver.parallel.model.register_custom_handler", side_effect=lambda name, fn: setattr(pm, name, fn))
    pm._register_custom_handlers()

    calls = []

    async def _bridge(worker, req_id, stream, channel_id: str):
        calls.append((req_id, channel_id, id(stream)))

    dispatcher._bridge_async_iterable_arg = _bridge

    async def _fake_stream(req_msg):
        assert req_msg.method_name == "multi"
        assert isinstance(req_msg.method_args[0], InputChannelRef)
        assert isinstance(req_msg.method_kwargs["ys"], InputChannelRef)
        assert req_msg.method_args[0].channel_id != req_msg.method_kwargs["ys"].channel_id
        async def _gen():
            yield "x"
            yield "y"
        return _gen()

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    async def xs():
        yield 1
        yield 2

    async def ys():
        yield "a"

    out = []
    async for item in pm.multi(xs(), ys=ys()):  # type: ignore[attr-defined]
        out.append(item)

    await asyncio.sleep(0)
    assert out == ["x", "y"]
    assert len(calls) == 2
    ch_ids = {c[1] for c in calls}
    assert len(ch_ids) == 2


@pytest.mark.asyncio
async def test_custom_handler_unary_path(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    async def my_payload(lst: list[int]) -> int:  # pragma: no cover
        return sum(lst)

    mocker.patch("woprserver.parallel.model.get_custom_handlers", return_value=[("my_payload", my_payload)])
    mocker.patch("woprserver.parallel.model.register_custom_handler", side_effect=lambda name, fn: setattr(pm, name, fn))
    pm._register_custom_handlers()

    async def _fake(req_msg):
        assert req_msg.method_name == "my_payload"
        assert req_msg.method_args == [[1, 2, 3]]
        return SimpleNamespace(return_value=6)

    mocker.patch.object(dispatcher, "dispatch_request", side_effect=_fake)

    got = await pm.my_payload([1, 2, 3])  # type: ignore[attr-defined]
    assert got == 6


@pytest.mark.asyncio
async def test_custom_handler_streaming_path(pm_and_dispatcher, mocker):
    pm, dispatcher, model = pm_and_dispatcher

    mocker.patch("woprserver.parallel.model.get_custom_handlers", return_value=[("tokens", model.tokens)])
    mocker.patch("woprserver.parallel.model.register_custom_handler", side_effect=lambda name, fn: setattr(pm, name, fn))
    pm._register_custom_handlers()

    async def _fake_stream(req_msg):
        assert req_msg.method_name == "tokens"
        async def _gen():
            for i in (1, 2, 3):
                yield i
        return _gen()

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    out = []
    async for x in pm.tokens():  # type: ignore[attr-defined]
        out.append(x)
    assert out == [1, 2, 3]


@pytest.mark.asyncio
async def test_predict_stream_propagates_dispatcher_chunks(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    r1 = InferenceResponse(model_name=pm.settings.name, id=generate_uuid(), outputs=[])
    r2 = InferenceResponse(model_name=pm.settings.name, id=generate_uuid(), outputs=[])

    async def _fake_stream(_req_msg) -> AsyncIterator[InferenceResponse]:
        async def _gen():
            yield r1
            yield r2
        return _gen()

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    async def _reqs():
        yield InferenceRequest(inputs=[])

    seen = []
    async for ch in pm.predict_stream(_reqs()):
        seen.append(ch)
    assert seen == [r1, r2]


@pytest.mark.asyncio
async def test_metadata_and_predict_use_model_settings(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher
    expected_md = MetadataModelResponse(name=pm.settings.name, platform="z")

    async def _meta(req_msg):
        assert req_msg.method_name == ModelMethods.Metadata.value
        assert req_msg.model_name == pm.settings.name
        return SimpleNamespace(return_value=expected_md)

    async def _pred(req_msg):
        assert req_msg.method_name == ModelMethods.Predict.value
        assert req_msg.model_name == pm.settings.name
        return SimpleNamespace(return_value=InferenceResponse(model_name=pm.settings.name, outputs=[]))

    mocker.patch.object(
        dispatcher,
        "dispatch_request",
        side_effect=lambda rm: _meta(rm) if rm.method_name == "metadata" else _pred(rm),
    )

    md = await pm.metadata()
    pr = await pm.predict(InferenceRequest(inputs=[]))
    assert md is expected_md
    assert isinstance(pr, InferenceResponse)
    assert pr.model_name == pm.settings.name

async def stream_generator(generate_request):
    yield generate_request


async def test_predict_stream_fallback(
    text_model: TextModel,
    generate_request: InferenceRequest,
):
    generator = text_model.predict_stream(stream_generator(generate_request))
    assert inspect.isasyncgen(generator)

    responses = []
    async for response in generator:
        responses.append(response)

    assert len(responses) == 1
    assert len(responses[0].outputs) > 0


async def test_predict_stream(
    text_stream_model: TextStreamModel,
    generate_request: InferenceRequest,
):
    generator = text_stream_model.predict_stream(stream_generator(generate_request))
    assert inspect.isasyncgen(generator)

    responses = []
    async for response in generator:
        responses.append(response)

    ref_text = ["What", " is", " the", " capital", " of", " France?"]
    assert len(responses) == len(ref_text)

    for idx in range(len(ref_text)):
        assert ref_text[idx] == StringCodec.decode_output(responses[idx].outputs[0])[0]

# --------------------------------------------------------------------
# 1) Unary fallback: empty input streams should raise & close inputs
# --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_unary_fallback_raises_on_empty_stream(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    # Ensure dispatcher.dispatch_request_stream is NOT called (error before that)
    spy = mocker.patch.object(dispatcher, "dispatch_request_stream")

    a_it = _EmptyAsyncIter()
    b_it = _EmptyAsyncIter()

    # Call a unary method ("add") with async-iterable args -> unary fallback path
    with pytest.raises(InferenceError):
        agen = pm._send_stream("add", a_it, b=b_it)  # type: ignore[attr-defined]
        async for _ in agen:
            pass

    spy.assert_not_called()
    assert a_it.closed is True and b_it.closed is True


# --------------------------------------------------------------------
# 2) Expect async input, but no worker available -> skip bridging
# --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_no_worker_skips_bridging(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    # Make get_worker_for return None so the proxy does not start bridge tasks
    mocker.patch.object(dispatcher, "get_worker_for", return_value=None)
    bridge_spy = mocker.spy(dispatcher, "_bridge_async_iterable_arg")

    # Return a simple server stream with two chunks
    r1 = InferenceResponse(model_name=pm.settings.name, outputs=[])
    r2 = InferenceResponse(model_name=pm.settings.name, outputs=[])

    async def _fake_stream(req_msg):
        assert req_msg.method_name == ModelMethods.PredictStream.value
        assert isinstance(req_msg.method_args[0], InputChannelRef)  # input replaced with channel ref
        return _ServerAsyncGen([r1, r2])

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    async def _payloads():
        yield InferenceRequest(inputs=[])
        yield InferenceRequest(inputs=[])

    out = []
    async for ch in pm.predict_stream(_payloads()):
        out.append(ch)

    assert out == [r1, r2]
    bridge_spy.assert_not_called()


# --------------------------------------------------------------------
# 3) Bridge tasks are awaited on completion (no dangling tasks)
# --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_bridge_tasks_are_awaited(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    done = asyncio.Event()

    async def _bridge_async_iterable_arg(_worker, _rid, stream, channel_id: str):
        # Consume the client stream fully, then signal completion
        async for _ in stream:
            pass
        done.set()

    mocker.patch.object(dispatcher, "_bridge_async_iterable_arg", side_effect=_bridge_async_iterable_arg)

    # Server yields one chunk then finishes
    r = InferenceResponse(model_name=pm.settings.name, outputs=[])

    async def _fake_stream(_req_msg):
        return _ServerAsyncGen([r])

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    async def _payloads():
        yield InferenceRequest(inputs=[])
        yield InferenceRequest(inputs=[])  # ensure bridge consumes multiple client items

    # Drain client stream completely
    seen = []
    async for ch in pm.predict_stream(_payloads()):
        seen.append(ch)

    assert seen == [r]
    # If the proxy awaited bridge tasks in finally, this should be set already
    assert done.is_set()


# --------------------------------------------------------------------
# 4) Server stream is aclosed on normal completion
# --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_server_stream_aclose_on_completion(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    r1 = InferenceResponse(model_name=pm.settings.name, outputs=[])
    r2 = InferenceResponse(model_name=pm.settings.name, outputs=[])

    server_stream = _ServerAsyncGen([r1, r2])

    async def _fake_stream(_req_msg):
        return server_stream

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    async def _payloads():
        yield InferenceRequest(inputs=[])

    # Drain the stream
    out = []
    async for ch in pm.predict_stream(_payloads()):
        out.append(ch)

    assert out == [r1, r2]
    assert server_stream.closed is True

# --------------------------------------------------------------------
# 5) Server error mid-stream: error propagates & server stream closed,
#    and bridge tasks are awaited.
# --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_server_stream_error_propagates_and_cleanup(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    bridge_done = asyncio.Event()

    async def _bridge_async_iterable_arg(_worker, _rid, stream, channel_id: str):
        # Consume 1 item then finish.
        async for _ in stream:
            break
        bridge_done.set()

    mocker.patch.object(dispatcher, "_bridge_async_iterable_arg", side_effect=_bridge_async_iterable_arg)

    r1 = InferenceResponse(model_name=pm.settings.name, outputs=[])
    server_stream = _ServerAsyncGen([r1, r1, r1], raise_after=1)  # raise during iteration

    async def _fake_stream(_req_msg):
        return server_stream

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    async def _payloads():
        yield InferenceRequest(inputs=[])
        # keep the client side open long enough for bridge to see one item
        await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="server stream error"):
        async for _ in pm.predict_stream(_payloads()):
            pass

    # Even on error, the proxy should aclose the server stream
    assert server_stream.closed is True
    # And ensure bridge task was awaited (finished)
    assert bridge_done.is_set()

# --------------------------------------------------------------------
# 6) Multiple async inputs: stable order & channel ids are assigned
# --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_multiple_streams_channel_id_order(pm_and_dispatcher, mocker):
    pm, dispatcher, model = pm_and_dispatcher

    # Register a custom streaming handler that expects two async inputs
    mocker.patch("woprserver.parallel.model.get_custom_handlers", return_value=[("multi", model.multi)])
    mocker.patch(
        "woprserver.parallel.model.register_custom_handler",
        side_effect=lambda name, fn: setattr(pm, name, fn),
    )
    pm._register_custom_handlers()

    captured = {}

    async def _fake_stream(req_msg):
        # Expect two distinct channel refs, in discovery order (positional first, then kw)
        ref_pos = req_msg.method_args[0]
        ref_kw = req_msg.method_kwargs["ys"]
        assert isinstance(ref_pos, InputChannelRef)
        assert isinstance(ref_kw, InputChannelRef)
        captured["pos_cid"] = ref_pos.channel_id
        captured["kw_cid"] = ref_kw.channel_id
        # yield some chunks
        async def _gen():
            yield "x"
            yield "y"
        return _gen()

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    async def xs():
        yield 1

    async def ys():
        yield "a"

    out = []
    async for item in pm.multi(xs(), ys=ys()):  # type: ignore[attr-defined]
        out.append(item)

    assert out == ["x", "y"]
    # Order must match discovery order (positional first, then kw)
    assert captured["pos_cid"].startswith("ch-")
    assert captured["kw_cid"].startswith("ch-")
    assert captured["pos_cid"] != captured["kw_cid"]


# --------------------------------------------------------------------
# 7) predict_stream: still works when no client items (empty async input)
#    (channel plan path but client produces no payloads)
# --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_predict_stream_empty_client_input(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    r = InferenceResponse(model_name=pm.settings.name, outputs=[])

    async def _fake_stream(_req_msg):
        return _ServerAsyncGen([r])

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    async def _payloads():
        if False:
            yield  # pragma: no cover
        return

    out = []
    async for ch in pm.predict_stream(_payloads()):
        out.append(ch)

    assert out == [r]

# --------------------------------------------------------------------
# 8) Custom unary handler: kwargs-only, ensures _send packs kwargs correctly
# --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_custom_unary_kwargs_only(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    async def my_payload(a: int, b: int) -> int:  # pragma: no cover
        return a + b

    mocker.patch("woprserver.parallel.model.get_custom_handlers", return_value=[("my_payload", my_payload)])
    mocker.patch("woprserver.parallel.model.register_custom_handler", side_effect=lambda name, fn: setattr(pm, name, fn))
    pm._register_custom_handlers()

    async def _fake(req_msg):
        # Ensure kwargs preserved
        assert req_msg.method_name == "my_payload"
        assert req_msg.method_args == []
        assert req_msg.method_kwargs == {"a": 10, "b": 32}
        return SimpleNamespace(return_value=42)

    mocker.patch.object(dispatcher, "dispatch_request", side_effect=_fake)

    got = await pm.my_payload(a=10, b=32)  # type: ignore[attr-defined]
    assert got == 42


# --------------------------------------------------------------------
# 9) predict_stream: generator type is asyncgen (smoke test on wrapper)
# --------------------------------------------------------------------
@pytest.mark.asyncio
async def test_predict_stream_returns_asyncgen(pm_and_dispatcher, mocker):
    pm, dispatcher, _ = pm_and_dispatcher

    r = InferenceResponse(model_name=pm.settings.name, outputs=[])

    async def _fake_stream(_req_msg):
        return _ServerAsyncGen([r])

    mocker.patch.object(dispatcher, "dispatch_request_stream", side_effect=_fake_stream)

    async def _payloads():
        yield InferenceRequest(inputs=[StringCodec.encode_input("x", payload=b"", use_bytes=True)])

    gen = pm.predict_stream(_payloads())
    assert inspect.isasyncgen(gen)

    out = []
    async for ch in gen:
        out.append(ch)

    assert out == [r]
