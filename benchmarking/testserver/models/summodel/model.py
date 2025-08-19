from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Iterable
from typing import Any

from mlserver import MLModel, types


class SumModel(MLModel):
    """
    Simple numeric model that sums all input arrays.

    Streaming behavior:
    - Streams a sequence of partial totals from 0 up to the final sum (inclusive).
    - Each chunk emits a single FP32 value under output name "partial_total".

    Robustness:
    - Accepts a single `InferenceRequest`, a list/tuple, a sync/async iterator of
      `InferenceRequest`, OR raw numeric inputs (int/float) / containers of numerics.
      If a numeric (or container of numerics) is received instead of a request,
      it's treated as an already-summed total.
    """

    # ---------- helpers ----------

    async def _first_request_or_scalar(self, payload: Any) -> tuple[types.InferenceRequest | None, float | None]:
        """
        Try to extract the first InferenceRequest from various payload shapes.
        If the payload is numeric or a container of numerics, return (None, total).
        """
        # Fast-path numeric scalar
        if isinstance(payload, (int, float)):
            return None, float(payload)

        # List / tuple: could be requests OR numerics
        if isinstance(payload, (list, tuple)):
            if not payload:
                return None, 0.0
            first = payload[0]
            # numeric batch
            if isinstance(first, (int, float)):
                # if it's a flat list/tuple of numerics, sum all
                try:
                    total = sum(float(x) for x in payload)  # type: ignore[arg-type]
                    return None, float(total)
                except Exception:
                    pass
            # else, fall-through to request handling
            if isinstance(first, types.InferenceRequest):
                return first, None

        # Single request
        if isinstance(payload, types.InferenceRequest):
            return payload, None

        # Sync iterator: try first item
        try:
            iterator = iter(payload)  # type: ignore[arg-type]
        except TypeError:
            iterator = None
        if iterator is not None:
            try:
                first = next(iterator)
            except StopIteration:
                return None, 0.0
            if isinstance(first, (int, float)):
                return None, float(first)
            if isinstance(first, types.InferenceRequest):
                return first, None

        # Async iterator
        try:
            aiter = payload.__aiter__()  # type: ignore[attr-defined]
        except AttributeError:
            aiter = None
        if aiter is not None:
            try:
                first = await aiter.__anext__()  # type: ignore[attr-defined]
            except StopAsyncIteration:
                return None, 0.0
            if isinstance(first, (int, float)):
                return None, float(first)
            if isinstance(first, types.InferenceRequest):
                return first, None

        # Unknown shape: treat as zero total
        return None, 0.0

    @staticmethod
    def _flatten_numeric(data: Any) -> Iterable[float]:
        if data is None:
            return []
        if isinstance(data, (int, float)):
            return [float(data)]
        if isinstance(data, (list, tuple)):
            out: list[float] = []
            for v in data:
                try:
                    if isinstance(v, (list, tuple)):
                        out.extend(float(x) for x in v)  # type: ignore[arg-type]
                    else:
                        out.append(float(v))  # type: ignore[arg-type]
                except Exception:
                    continue
            return out
        try:
            return [float(data)]  # type: ignore[arg-type]
        except Exception:
            return []

    def _sum_inputs(self, req: types.InferenceRequest | None) -> float:
        """Sum all numeric elements across all inputs in an InferenceRequest."""
        if not req or not getattr(req, "inputs", None):
            return 0.0
        total = 0.0
        for inp in req.inputs:
            try:
                total += sum(self._flatten_numeric(getattr(inp, "data", None)))
            except Exception:
                continue
        return total

    # ---------- unary API ----------

    async def predict(self, payload: Any) -> types.InferenceResponse:
        req, scalar_total = await self._first_request_or_scalar(payload)
        total = scalar_total if scalar_total is not None else self._sum_inputs(req)
        pid = str(os.getpid())
        rid = (req.id if req and getattr(req, "id", None) else None) or "1"

        output = types.ResponseOutput(
            name="total",
            shape=[1, 1],
            datatype="FP32",
            data=[total],
            parameters=types.Parameters(headers={"X-Worker-PID": pid}),
        )
        return types.InferenceResponse(
            model_name=self.name,
            id=rid,
            outputs=[output],
        )

    # ---------- streaming API ----------

    async def _predict_stream_single(
        self, req: types.InferenceRequest | None, preset_total: float | None
    ) -> AsyncIterator[types.InferenceResponse]:
        """
        Emit a ramp [0, total] in up to 100 steps for a single request or a preset numeric total.
        """
        total = preset_total if preset_total is not None else self._sum_inputs(req)

        steps = max(1, min(100, int(abs(total)) if abs(total) >= 1 else 10))
        increment = total / steps if steps else 0.0

        current = 0.0
        rid = (req.id if req and getattr(req, "id", None) else None)
        pid = str(os.getpid())

        for i in range(steps + 1):
            out = types.ResponseOutput(
                name="partial_total",
                shape=[1, 1],
                datatype="FP32",
                data=[current],
                parameters=types.Parameters(
                    headers={
                        "X-Worker-PID": pid,
                        "X-Stream-Index": str(i),
                        "X-Stream-Steps": str(steps),
                    }
                ),
            )
            yield types.InferenceResponse(
                id=rid,
                model_name=self.name,
                outputs=[out],
            )
            await asyncio.sleep(0.01)
            current += increment

    async def predict_stream(self, payload: Any):
        """
        Server/bidi streaming endpoint (REST /infer_stream, gRPC ModelStreamInfer).

        Accepts a single request, batch/iterator, or raw numeric input and uses the first
        element to drive the server-streamed sequence. Emits FP32 chunks under output name
        "partial_total".
        """
        req, scalar_total = await self._first_request_or_scalar(payload)
        async for resp in self._predict_stream_single(req, scalar_total):
            yield resp
