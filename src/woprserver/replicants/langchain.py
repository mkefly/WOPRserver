
from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import mlflow
from mlflow.langchain.api_request_parallel_processor import (
    process_api_requests,
    process_stream_request,
)
from mlflow.models.utils import _convert_llm_input_data
from mlflow.version import VERSION
from mlserver import types
from mlserver.errors import InferenceError
from mlserver.handlers import custom_handler
from mlserver.model import MLModel
from mlserver.settings import ModelParameters
from mlserver.types import InferenceRequest, InferenceResponse
from mlserver.utils import get_model_uri

# MLflow codec
from mlserver_mlflow.codecs import TensorDictCodec
from mlserver_mlflow.metadata import (
    DefaultInputPrefix,
    DefaultOutputPrefix,
    to_metadata_tensors,
    to_model_content_type,
)

from ..logging import get_logger

# Shared IO helpers
from .shared.io import (
    STREAM_DONE,
    extract_params,
    extract_text,
    fallback_inputs_to_native,
    unwrap_first_request,
)

# ------------------------------ helpers ------------------------------
logger = get_logger()

def _resolve_response_id(req: InferenceRequest) -> str:
    rid = getattr(req, "id", None)
    if isinstance(rid, str) and rid.strip():
        return rid
    try:
        params = getattr(req, "parameters", None)
        headers = getattr(params, "headers", None) or {}
        norm = {str(k).lower(): str(v) for k, v in headers.items()}
        for k in ("ce-requestid", "ce-id", "x-request-id"):
            if k in norm and norm[k].strip():
                return norm[k].strip()
    except Exception:
        pass
    return f"gen-{uuid.uuid4()}"


def _response_from_text(model_name: str, response_id: str, text: str) -> InferenceResponse:
    return types.InferenceResponse(
        id=response_id,
        model_name=model_name,
        outputs=[
            types.ResponseOutput(
                name="text",
                shape=[1],
                datatype="BYTES",
                data=[text.encode("utf-8", errors="replace")],
            )
        ],
    )


def _response_from_error(model_name: str, response_id: str, exc: Exception) -> InferenceResponse:
    return _response_from_text(model_name, response_id, f"[error] {exc.__class__.__name__}: {exc}")


# ------------------------------ runtime ------------------------------

class LangChainRuntime(MLModel):
    """
    MLServer runtime for MLflow's `langchain` flavor with:
      - /ping, /health, /version helpers
      - V2 predict (batch) returning a single "text" tensor
      - V2 predict_stream that always yields >=1 chunk and a final "[DONE]"
    """

    # ---- simple health/version ----
    @custom_handler(rest_path="/ping", rest_method="GET")
    async def ping(self) -> str:
        return "\\n"

    @custom_handler(rest_path="/health", rest_method="GET")
    async def health(self) -> str:
        return "\\n"

    @custom_handler(rest_path="/version", rest_method="GET")
    async def mlflow_version(self) -> str:
        return VERSION

    # ---- lifecycle ----
    async def load(self) -> bool:
        model_uri = await get_model_uri(self._settings)
        self._model = mlflow.langchain.load_model(model_uri)

        self._signature = getattr(self._model, "metadata", None)
        try:
            self._input_schema = self._signature.get_input_schema() if self._signature and hasattr(self._model, "metadata") else None
        except Exception:
            self._input_schema = None

        self._sync_metadata()
        return True

    # ---- V2 predict (non-stream) ----
    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        response_id = _resolve_response_id(payload)

        try:
            try:
                decoded = self.decode_request(payload)  # type: ignore
                logger.debug(f"[predict] decoded request: {decoded!r}")
            except Exception as e:
                logger.warning(f"[predict] decode_request failed: {e!r}; using fallback")
                decoded = fallback_inputs_to_native(payload)

            params = extract_params(payload) or None

            texts = await self._predict_batch(decoded, params)
            model_output = {"text": texts if len(texts) > 1 else texts[0]}

            logger.info(f"[predict] id={response_id} success")
            return self.encode_response(model_output, default_codec=TensorDictCodec)

        except Exception as e:
            logger.error(f"[predict] id={response_id} error={e!r}", exc_info=True)
            return _response_from_error(self.name, response_id, e)

    # ---- V2 predict_stream ----
    async def predict_stream(
        self, payloads: AsyncIterator[InferenceRequest] | InferenceRequest
    ) -> AsyncIterator[InferenceResponse]:
        req = await unwrap_first_request(payloads)

        if req is None:
            rid = f"gen-{uuid.uuid4()}"
            yield _response_from_text(self.name, rid, STREAM_DONE)
            return

        response_id = _resolve_response_id(req)

        try:
            native = fallback_inputs_to_native(req)
            single = _convert_llm_input_data(native)
            if isinstance(single, list):
                if len(single) != 1:
                    raise InferenceError(
                        f"predict_stream requires a single input, but got {len(single)} items."
                    )
                single = single[0]

            params = extract_params(req)
            callback_handlers = self._build_tracing_callbacks()
            emitted_any = False

            for chunk in process_stream_request(
                lc_model=self._model,
                request_json=single,
                params=params or {},
                callback_handlers=callback_handlers,
            ):
                text = extract_text(chunk)
                if text is None:
                    continue
                emitted_any = True
                yield _response_from_text(self.name, response_id, text)

            if not emitted_any:
                yield _response_from_text(self.name, response_id, "")

            yield _response_from_text(self.name, response_id, STREAM_DONE)

        except Exception as e:
            yield _response_from_error(self.name, response_id, e)
            yield _response_from_text(self.name, response_id, STREAM_DONE)

    # ---- internals ----
    async def _predict_batch(self, data: Any, params: dict[str, Any] | None) -> list[str]:
        requests = _convert_llm_input_data(data)
        single = not isinstance(requests, list)
        if single:
            requests = [requests]  # type: ignore[list-item]

        results = process_api_requests(
            lc_model=self._model,
            requests=requests,
            callback_handlers=self._build_tracing_callbacks(),
            convert_chat_responses=False,
            params=params or {},
            context=None,
        )
        texts = [(extract_text(r) or "") for r in results]
        return texts if not single else [texts[0]]

    def _build_tracing_callbacks(self):
        enable = os.getenv("ENABLE_MLFLOW_TRACING", "false").lower() == "true"
        enable_legacy = os.getenv("MLFLOW_ENABLE_TRACE_IN_SERVING", "false").lower() == "true"
        if not (enable or enable_legacy):
            return None
        try:
            from mlflow.langchain.langchain_tracer import MlflowLangchainTracer
            return [MlflowLangchainTracer()]
        except Exception as e:
            logger.warning(f"Tracing requested but unavailable: {e}")
            return None

    def _sync_metadata(self) -> None:
        if not self._settings.parameters:
            self._settings.parameters = ModelParameters()
        if not getattr(self._settings.parameters, "content_type", None):
            self._settings.parameters.content_type = "application/json"

        sig = getattr(self, "_signature", None)
        inputs = getattr(sig, "inputs", None)
        if not inputs:
            return

        if self.inputs:
            logger.warning("Overwriting existing inputs metadata with model signature")
        self.inputs = to_metadata_tensors(schema=inputs, prefix=DefaultInputPrefix)

        if self.outputs:
            logger.warning("Overwriting existing outputs metadata with model signature")
        self.outputs = to_metadata_tensors(schema=sig.outputs, prefix=DefaultOutputPrefix)

        self._settings.parameters.content_type = to_model_content_type(schema=inputs)
