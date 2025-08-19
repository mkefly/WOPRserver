
from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

from .utils import map_ns

# Request/model context via contextvars (clean with context manager)
model_name_var: ContextVar[str] = ContextVar("model_name", default="")
model_version_var: ContextVar[str] = ContextVar("model_version", default="")
environment_var: ContextVar[str] = ContextVar("environment", default="")
request_id_var: ContextVar[str] = ContextVar("request_id", default="")
http_method_var: ContextVar[str] = ContextVar("http_method", default="")
component_var: ContextVar[str] = ContextVar("component", default="")

@contextmanager
def context(**kwargs) -> Iterator[None]:
    tokens = {}
    try:
        if "model_name" in kwargs:
            tokens["model_name"] = model_name_var.set(kwargs["model_name"])
        if "model_version" in kwargs:
            tokens["model_version"] = model_version_var.set(kwargs["model_version"])
        if "environment" in kwargs:
            tokens["environment"] = environment_var.set(kwargs["environment"])
        if "request_id" in kwargs:
            tokens["request_id"] = request_id_var.set(kwargs["request_id"])
        if "http_method" in kwargs:
            tokens["http_method"] = http_method_var.set(kwargs["http_method"])
        if "component" in kwargs:
            tokens["component"] = component_var.set(kwargs["component"])
        yield
    finally:
        for k, tok in tokens.items():
            try:
                if tok is not None:
                    getattr(globals()[f"{k}_var"], "reset")(tok)  # type: ignore
            except Exception:
                pass

def set_context(**kwargs) -> None:
    # non-context-manager setter (sticky in current task)
    if "model_name" in kwargs:
        model_name_var.set(kwargs["model_name"])
    if "model_version" in kwargs:
        model_version_var.set(kwargs["model_version"])
    if "environment" in kwargs:
        environment_var.set(kwargs["environment"])
    if "request_id" in kwargs:
        request_id_var.set(kwargs["request_id"])
    if "http_method" in kwargs:
        http_method_var.set(kwargs["http_method"])
    if "component" in kwargs:
        component_var.set(kwargs["component"])

# LogRecord factory installer
def install_logrecord_factory() -> None:
    if getattr(install_logrecord_factory, "_installed", False):
        return
    install_logrecord_factory._installed = True

    orig_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = orig_factory(*args, **kwargs)
        name = record.name or ""

        # namespace mapping
        if name.startswith("woprserver."):
            ns = "woprserver"; scope = name.split(".", 1)[1]
        elif name == "woprserver":
            ns = "woprserver"; scope = ""
        elif name.startswith("mlserver.") or name.startswith("mlflow."):
            ns = "woprserver"; scope = name.split(".", 1)[1]
        elif name in ("mlserver", "mlflow"):
            ns = "woprserver"; scope = ""
        else:
            ns = name.split(".", 1)[0] if name else "root"
            scope = name.split(".", 1)[1] if "." in name else ""

        record.wopr_ns = ns
        record.wopr_scope = scope
        if not hasattr(record, "system_name"):
            record.system_name = ns

        # inject contextvars into record
        record.model_name = getattr(record, "model_name", "") or model_name_var.get("")
        record.model_version = getattr(record, "model_version", "") or model_version_var.get("")
        record.environment = getattr(record, "environment", "") or environment_var.get("")
        record.request_id = getattr(record, "request_id", "") or request_id_var.get("")
        record.http_method = getattr(record, "http_method", "") or http_method_var.get("")
        record.component = getattr(record, "component", "") or component_var.get("")

        # ensure mapped name for display
        try:
            record.name = map_ns(getattr(record, "name", ""))
        except Exception:
            pass

        return record

    logging.setLogRecordFactory(factory)
