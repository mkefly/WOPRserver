from logging import INFO

import pytest
from mlserver import ModelSettings
from mlserver.context import model_context
from mlserver.logging import (
    ModelLoggerFormatter,
    logger,
)
from mlserver.settings import ModelParameters, Settings

from .fixtures import SumModel


@pytest.mark.parametrize(
    "name, version, expected_model_fmt, fmt_present_in_all",
    [
        (
            "foo",
            "v1.0",
            "[foo:v1.0]",
            False,
        ),
        (
            "foo",
            "",
            "[foo]",
            False,
        ),
        (
            "",
            "v1.0",
            "",
            True,
        ),
        (
            "",
            "",
            "",
            True,
        ),
    ],
)
def test_model_logging_formatter_unstructured(
    name: str,
    version: str,
    expected_model_fmt: str,
    fmt_present_in_all: bool,
    settings: Settings,
    caplog,
):
    settings.use_structured_logging = False
    caplog.handler.setFormatter(ModelLoggerFormatter(settings))
    caplog.set_level(INFO)

    model_settings = ModelSettings(
        name=name, implementation=SumModel, parameters=ModelParameters(version=version)
    )

    logger.info("Before model context")
    with model_context(model_settings):
        logger.info("Inside model context")
    logger.info("After model context")

    log_records = caplog.get_records("call")
    assert len(log_records) == 3

    assert all(hasattr(lr, "model") for lr in log_records)

    if fmt_present_in_all:
        assert all(lr.model == expected_model_fmt for lr in log_records)
    else:
        assert expected_model_fmt != log_records[0].model
        assert expected_model_fmt == log_records[1].model
        assert expected_model_fmt != log_records[2].model
