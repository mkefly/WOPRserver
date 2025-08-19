import json

import pytest
from mlserver.settings import ModelSettings

from woprserver.parallel.messages import (
    ModelResponseMessage,
    ModelUpdateMessage,
    ModelUpdateType,
)

from .fixtures import SumModel

_DUMMY_FULL = ModelSettings(name="foo", implementation=SumModel).model_dump(by_alias=True)


def _json_eq(a: str, b: str) -> bool:
    return json.loads(a) == json.loads(b)


@pytest.mark.parametrize(
    "kwargs, expected_serialised_json, preserves_exact_string",
    [
        (
            {
                "id": "foo",
                "update_type": ModelUpdateType.Load,
                "model_settings": ModelSettings(name="foo", implementation=SumModel),
            },
            json.dumps({**_DUMMY_FULL, "name": "foo", "implementation": "parallel.fixtures.SumModel"}),
            False,
        ),
        (
            {
                "id": "foo",
                "update_type": ModelUpdateType.Load,
                "serialised_model_settings": '{"name":"foo","implementation":"parallel.fixtures.SumModel"}',
            },
            '{"name":"foo","implementation":"parallel.fixtures.SumModel"}',
            True,
        ),
    ],
)
def test_model_update_message(kwargs: dict, expected_serialised_json: str, preserves_exact_string: bool):
    msg = ModelUpdateMessage(**kwargs)
    assert msg.id == kwargs.get("id", msg.id)
    assert msg.update_type == kwargs["update_type"]
    if preserves_exact_string:
        assert msg.serialised_model_settings == expected_serialised_json
    else:
        assert _json_eq(msg.serialised_model_settings, expected_serialised_json)


@pytest.mark.parametrize(
    "model_update_message, expected",
    [
        (
            ModelUpdateMessage(update_type=ModelUpdateType.Load, model_settings=ModelSettings(name="foo", implementation=SumModel)),
            ModelSettings(name="foo", implementation=SumModel),
        ),
        (
            ModelUpdateMessage(update_type=ModelUpdateType.Load, serialised_model_settings='{"name":"foo","implementation":"parallel.fixtures.SumModel"}'),
            ModelSettings(name="foo", implementation=SumModel),
        ),
    ],
)
def test_model_settings_roundtrip(model_update_message: ModelUpdateMessage, expected: ModelSettings):
    ms = model_update_message.model_settings
    assert ms.implementation_ == expected.implementation_
    assert ms.name == expected.name


def test_model_response_helpers_raise_and_ok():
    # ok()
    ok = ModelResponseMessage(return_value=123)
    assert ok.ok()
    # raise_for_exception()
    err = ModelResponseMessage(exception=RuntimeError("boom"))
    assert not err.ok()
    with pytest.raises(RuntimeError):
        err.raise_for_exception()
