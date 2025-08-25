"""
Worker/Dispatcher RPC Message Schema
====================================

This module defines the **wire protocol** (Pydantic models) used between the
dispatcher (parent process) and worker processes. Messages fall into three
families:

1. **Request / Response**
   Invoke a model method inside a worker and return either a value or an error.

2. **Control Plane (Model Updates)**
   Load or unload models within a worker process.

3. **Streaming**
   Bidirectional streaming of data and control signals:
   - *worker → client*: response chunks and a terminal end/exception marker.
   - *client → worker*: input chunks for streaming parameters, scoped by
     request ``id`` and per-parameter ``channel_id``.

Design Principles
-----------------
* **Explicit schemas** – unexpected fields are rejected (``extra='forbid'``).
* **Safe defaults** – lists/dicts use ``default_factory`` to avoid shared
  mutable state.
* **Lazy ModelSettings parsing** – settings objects are serialized to JSON
  across process boundaries and deserialized only when accessed, avoiding
  cross-version Pydantic friction.
* **Exception-friendly responses** – responses can embed arbitrary exceptions
  while keeping the rest of the schema strict.

Compatibility
-------------
``ModelSettings`` (de)serialization supports **Pydantic v1** and **Pydantic v2**:

1) Try v2 API: ``model_validate_json``  
2) Fallback to v1 API: ``parse_raw``  
3) As a last resort, ``model_validate`` from a decoded dict
"""

import json
from asyncio import CancelledError
from enum import IntEnum
from typing import Any

from mlserver.settings import ModelSettings
from mlserver.utils import generate_uuid
from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---- Enums -----------------------------------------------------------------


class ModelUpdateType(IntEnum):
    """Control-plane updates that can be applied inside a worker."""
    Load = 1
    Unload = 2


# ---- Base message ----------------------------------------------------------


class Message(BaseModel):
    """
    Base type for all messages.

    Each message carries a unique ``id`` used to correlate requests with
    responses. The ``id`` is generated via ``mlserver.utils.generate_uuid``.
    """
    model_config = ConfigDict(
        protected_namespaces=(),   # allow attributes like "model_config" on subclasses
        extra="forbid",            # reject unexpected fields for safety
    )

    id: str = Field(default_factory=generate_uuid)


# ---- Request / Response ----------------------------------------------------


class ModelRequestMessage(Message):
    """
    Request to call a model method inside a worker.

    Fields
    ------
    model_name / model_version : str
        Identify the target model instance (version may be ``None``).
    method_name : str
        The method to invoke (e.g. ``predict``).
    method_args / method_kwargs : list / dict
        Positional and keyword arguments. May include ``InputChannelRef``
        placeholders for streaming parameters.
    """
    model_name: str
    model_version: str | None = None
    method_name: str
    method_args: list[Any] = Field(default_factory=list)            # avoid shared mutable defaults
    method_kwargs: dict[str, Any] = Field(default_factory=dict)


class ModelResponseMessage(Message):
    """
    Response to a :class:`ModelRequestMessage`.

    Exactly one of ``return_value`` or ``exception`` should be present.
    ``arbitrary_types_allowed`` is enabled so real exception objects can be
    transported without coercion.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    return_value: Any | None = None
    exception: Exception | CancelledError | None = None

    def ok(self) -> bool:
        """True if the request completed successfully (no exception)."""
        return self.exception is None

    def raise_for_exception(self) -> None:
        """Helper: raise the contained exception if present."""
        if self.exception is not None:
            raise self.exception


# ---- Model updates ---------------------------------------------------------


class ModelUpdateMessage(Message):
    """
    Control-plane message to (un)load a model in a worker.

    Accepted construction forms
    ---------------------------
    * ``serialised_model_settings`` — JSON string representation of a
      ``ModelSettings`` instance (preferred across process boundaries).
    * ``model_settings`` — concrete ``ModelSettings`` object which will be
      serialized during validation.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    update_type: ModelUpdateType
    serialised_model_settings: str

    # Accept `model_settings` as input and turn it into JSON before validation completes.
    @model_validator(mode="before")
    @classmethod
    def _serialize_model_settings(cls, data: Any):
        """
        Normalize inputs so ``serialised_model_settings`` always contains JSON.

        If the caller provides a ``model_settings`` object (or dict-like), we
        serialize it to a compact JSON string and stash it under
        ``serialised_model_settings``.
        """

        if not isinstance(data, dict):
            return data

        if "model_settings" in data and "serialised_model_settings" not in data:
            ms = data.pop("model_settings")
            if isinstance(ms, ModelSettings):
                as_dict = ms.model_dump(by_alias=True)
                # Preserve private _source if present (useful for provenance).
                if getattr(ms, "_source", None):
                    as_dict["_source"] = ms._source
                data["serialised_model_settings"] = json.dumps(
                    as_dict, ensure_ascii=False, separators=(",", ":")
                )
            else:
                # If it's already a dict-like structure, serialize that too.
                data["serialised_model_settings"] = json.dumps(
                    ms, ensure_ascii=False, separators=(",", ":")
                )
        return data

    @property
    def model_settings(self):
        """
        Lazily deserialize the JSON payload into a ``ModelSettings`` instance.

        Tries Pydantic v2 first, then v1, then a generic validate-from-dict.
        """
        from mlserver.settings import ModelSettings  # lazy import

        parse_v2 = getattr(ModelSettings, "model_validate_json", None)
        if callable(parse_v2):
            return parse_v2(self.serialised_model_settings)  # type: ignore[arg-type]

        parse_v1 = getattr(ModelSettings, "parse_raw", None)
        if callable(parse_v1):
            return parse_v1(self.serialised_model_settings)  # type: ignore[arg-type]

        return ModelSettings.model_validate(json.loads(self.serialised_model_settings))  # type: ignore[arg-type]

    @classmethod
    def from_model_settings(cls, update_type: ModelUpdateType, model_settings: ModelSettings) -> "ModelUpdateMessage":
        """Convenience constructor for callers holding a concrete settings object."""
        return cls(update_type=update_type, model_settings=model_settings)


# ---- Streaming (outbound: worker -> client) --------------------------------


class ModelStreamChunkMessage(Message):
    """
    One chunk of a streaming response for the request with the same ``id``.

    ``chunk`` is intentionally typed as ``Any`` to accommodate text tokens,
    JSON fragments, binary blobs, etc., depending on the model/method.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    chunk: Any


class ModelStreamEndMessage(Message):
    """
    Terminal signal for a streaming response for the request with the same ``id``.

    If ``exception`` is set, the stream ended due to an error; clients should
    raise or handle it appropriately.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    exception: Exception | CancelledError | None = None


# ---- Streaming (inbound: client -> worker) ---------------------------------


class InputChannelRef(BaseModel):
    """
    Placeholder used in ``method_args`` / ``method_kwargs`` for inbound
    client→worker streaming parameters.

    * ``id`` — the request id (matches the surrounding request).
    * ``channel_id`` — unique per *parameter* within a request. An empty string
      denotes legacy single-channel mode.
    """
    model_config = ConfigDict(extra="forbid")

    id: str
    channel_id: str = ""  # empty = legacy single-channel


class ModelStreamInputChunk(Message):
    """
    One inbound input item for the request with the same ``id``.

    The worker routes chunks to the correct method parameter using
    ``channel_id``.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    item: Any
    channel_id: str = ""  # empty = legacy


class ModelStreamInputEnd(Message):
    """
    Marks the end of an inbound input stream for a given channel.

    The absence of further :class:`ModelStreamInputChunk` messages for the same
    (``id``, ``channel_id``) pair implies completion on that channel only; other
    channels for the same request may continue independently.
    """
    model_config = ConfigDict(extra="forbid")

    channel_id: str = ""  # empty = legacy
