
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any


class ModelAdapter(logging.LoggerAdapter):
    """Injects model_name/model_version/environment into all log calls."""
    def process(self, msg: str, kwargs: Mapping[str, Any]):
        extra = dict(kwargs.get("extra", {}))
        for k in ("model_name", "model_version", "environment"):
            if k not in extra and k in self.extra:
                extra[k] = self.extra[k]
        kw = dict(kwargs)
        kw["extra"] = extra
        return msg, kw

class RequestAdapter(logging.LoggerAdapter):
    """Injects request-scoped info such as request_id and http_method."""
    def process(self, msg: str, kwargs: Mapping[str, Any]):
        extra = dict(kwargs.get("extra", {}))
        for k in ("request_id", "http_method", "component", "environment"):
            if k not in extra and k in self.extra:
                extra[k] = self.extra[k]
        kw = dict(kwargs)
        kw["extra"] = extra
        return msg, kw
