
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, Union

def supports_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

def supports_unicode() -> bool:
    try:
        enc = (sys.stdout.encoding or "").upper()
        return "UTF" in enc
    except Exception:
        return False

def map_ns(name: str) -> str:
    if not name:
        return name
    for bad in ("mlserver", "mlflow"):
        if name == bad or name.startswith(bad + "."):
            return name.replace(bad, "woprserver", 1)
    return name

class NameRewriteFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            record.name = map_ns(getattr(record, "name", ""))
        except Exception:
            pass
        return True

class DeDupeFilter(logging.Filter):
    def __init__(self) -> None:
        super().__init__()
        self._last = None
    def filter(self, record: logging.LogRecord) -> bool:
        key = (record.name, record.levelno, record.getMessage(), record.process, record.thread)
        if key == self._last:
            return False
        self._last = key
        return True
