# tests/parallel/fixtures.py
from __future__ import annotations

import asyncio
import os
import random
import string
from collections.abc import AsyncIterator
from typing import Annotated, Optional

import numpy as np
from fastapi import Body

from mlserver.codecs import NumpyCodec, StringCodec
from mlserver.errors import MLServerError
from mlserver.handlers.custom import custom_handler
from mlserver.model import MLModel
from mlserver.repository import DEFAULT_MODEL_SETTINGS_FILENAME
from mlserver.settings import ModelSettings
from mlserver.types import (
    InferenceRequest,
    InferenceResponse,
    MetadataModelResponse,
    Parameters,
    ResponseOutput,
)
from mlserver.utils import generate_uuid


class SumModel(MLModel):
    @custom_handler(rest_path="/my-custom-endpoint")
    async def my_payload(self, payload: Annotated[list, Body(...)]) -> int:
        return sum(payload)

    @custom_handler(rest_path="/custom-endpoint-with-long-response")
    async def long_response_endpoint(self, length: int = Body(...)) -> dict[str, str]:
        alphabet = string.ascii_lowercase
        response = "".join(random.choice(alphabet) for _ in range(length))
        return {"foo": response}

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        decoded = self.decode(payload.inputs[0])
        total = decoded.sum(axis=1, keepdims=True)

        output = NumpyCodec.encode_output(name="total", payload=total)
        response = InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[output],
        )

        # Echo incoming headers which already start with `x-`
        if payload.parameters and payload.parameters.headers:
            hdrs_in = payload.parameters.headers
            response.parameters = Parameters(headers={k: v for k, v in hdrs_in.items() if k.startswith("x-")})

        return response


class TextModel(MLModel):
    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        text = StringCodec.decode_input(payload.inputs[0])[0]
        return InferenceResponse(
            model_name=self._settings.name,
            outputs=[StringCodec.encode_output(name="output", payload=[text], use_bytes=True)],
        )


class TextStreamModel(MLModel):
    async def predict_stream(self, payloads: AsyncIterator[InferenceRequest]) -> AsyncIterator[InferenceResponse]:
        # Pull first payload (or exit if none)
        try:
            payload = await anext(payloads)  # py3.11+
        except NameError:  # pragma: no cover - py<3.11
            async def _anext(it):
                return await it.__anext__()
            try:
                payload = await _anext(payloads)
            except StopAsyncIteration:
                return
        except StopAsyncIteration:
            return

        text = StringCodec.decode_input(payload.inputs[0])[0]
        words = text.split(" ")
        chunks = [words[0], *(" " + w for w in words[1:])]

        for word in chunks:
            await asyncio.sleep(0.5)
            yield InferenceResponse(
                model_name=self._settings.name,
                outputs=[StringCodec.encode_output(name="output", payload=[word], use_bytes=True)],
            )


class ErrorModel(MLModel):
    error_message = "something really bad happened"

    async def load(self) -> bool:
        if self._settings.parameters and getattr(self._settings.parameters, "load_error", False):
            raise MLServerError(self.error_message)
        return True

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:  # pragma: no cover - raised
        raise MLServerError(self.error_message)


class SimpleModel(MLModel):
    async def predict(self, foo: np.ndarray, bar: list[str]) -> InferenceResponse:  # noqa: ARG002
        
        output = NumpyCodec.encode_output(name="total", payload=foo.sum(axis=1, keepdims=True))
        response = InferenceResponse(
            model_name=self.name,
            model_version=self.version,
            outputs=[output],
        )
        return response


class SlowModel(MLModel):
    async def load(self) -> bool:
        await asyncio.sleep(10)
        return True

    async def infer(self, payload: InferenceRequest) -> InferenceResponse:
        await asyncio.sleep(10)
        return InferenceResponse(id=payload.id, model_name=self.name, outputs=[])


class EnvModel(MLModel):
    async def load(self):
        # Optional dependency available in custom env
        import sklearn
        self._sklearn_version = sklearn.__version__ if sklearn else "missing"
        return True

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            model_name=self.name,
            outputs=[StringCodec.encode_output("sklearn_version", [self._sklearn_version])],
        )


class EchoModel(MLModel):
    async def load(self) -> bool:
        print("Echo Model Initialized")
        return await super().load()

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                ResponseOutput(
                    name=inp.name,
                    shape=inp.shape,
                    datatype=inp.datatype,
                    data=inp.data,
                    parameters=inp.parameters,
                )
                for inp in payload.inputs
            ],
        )


class PidUnaryModel(MLModel):
    async def load(self) -> bool:
        return True

    async def metadata(self) -> MetadataModelResponse:
        return MetadataModelResponse(name=self.settings.name, platform="test")

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:  # noqa: ARG002
        return InferenceResponse(
            model_name=self.settings.name,
            outputs=[StringCodec.encode_output(name="output", payload=[str(os.getpid())], use_bytes=False)],
        )


class PidStreamModel(MLModel):
    async def load(self) -> bool:
        return True

    async def predict_stream(self, payloads: Optional[AsyncIterator[InferenceRequest]] = None) -> AsyncIterator[InferenceResponse]:  # noqa: ARG002
        pid = str(os.getpid())
        await asyncio.sleep(0)  # give the loop a tick
        yield InferenceResponse(model_name=self.settings.name, outputs=[StringCodec.encode_output("output", [pid], use_bytes=True)])


class FakeModel(MLModel):
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        settings_path = os.path.join(base_dir, "testdata", DEFAULT_MODEL_SETTINGS_FILENAME)
        super().__init__(ModelSettings.parse_file(settings_path))

    async def metadata(self) -> MetadataModelResponse:  # pragma: no cover
        return MetadataModelResponse(name=self.settings.name, platform="mlserver")

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:  # pragma: no cover
        return InferenceResponse(model_name=self.settings.name, id=payload.id or generate_uuid(), outputs=[])

    async def predict_stream(self, payloads: AsyncIterator[InferenceRequest]) -> AsyncIterator[InferenceResponse]:  # pragma: no cover
        async for _ in payloads:
            yield InferenceResponse(model_name=self.settings.name, outputs=[])

    async def tokens(self) -> AsyncIterator[int]:  # pragma: no cover
        for i in (1, 2, 3):
            yield i

    async def multi(self, xs: AsyncIterator[int], ys: AsyncIterator[str]) -> AsyncIterator[str]:  # pragma: no cover
        async for _ in xs:
            yield "x"
        async for _ in ys:
            yield "y"

    async def add(self, a: int, b: int) -> int:  # pragma: no cover
        return a + b


class ClosableInput:
    """Async generator wrapper that supports aclose()."""

    def __init__(self, items):
        self._it = iter(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            return StopAsyncIteration

    async def aclose(self):
        self.closed = True


class _EmptyAsyncIter:
    """Async-iterable that yields nothing but supports aclose()."""

    def __init__(self):
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self):
        self.closed = True


class _ServerAsyncGen:
    """
    Async iterator returned by dispatcher.dispatch_request_stream().
    Records whether aclose() was called; can raise mid-stream.
    """

    def __init__(self, chunks, raise_after=None):
        self._chunks = list(chunks)
        self._i = 0
        self.closed = False
        self.raise_after = raise_after  # int or None

    def __aiter__(self):
        return self

    async def __anext__(self):
        await asyncio.sleep(0)  # let loop tick
        if self.raise_after is not None and self._i == self.raise_after:
            raise RuntimeError("server stream error")
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        ch = self._chunks[self._i]
        self._i += 1
        return ch

    async def aclose(self):
        self.closed = True
