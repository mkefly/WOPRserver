
from __future__ import annotations
import logging

from .utils import map_ns

def install_hard_router() -> None:
    if getattr(install_hard_router, "_installed", False):
        return
    install_hard_router._installed = True

    _orig_callHandlers = logging.Logger.callHandlers

    def callHandlers(self: logging.Logger, record: logging.LogRecord):  # type: ignore[override]
        try:
            record.name = map_ns(getattr(record, "name", ""))
        except Exception:
            pass

        root = logging.getLogger()
        if self is root:
            return _orig_callHandlers(self, record)

        if getattr(record, "_wopr_forced", False):
            return _orig_callHandlers(self, record)

        try:
            setattr(record, "_wopr_forced", True)
            root.handle(record)
        finally:
            try:
                delattr(record, "_wopr_forced")
            except Exception:
                pass
        return

    logging.Logger.callHandlers = callHandlers  # type: ignore[assignment]

def install_global_handler_router(our_handler: logging.Handler) -> None:
    if getattr(install_global_handler_router, "_installed", False):
        return
    install_global_handler_router._installed = True

    orig_handle = logging.Handler.handle

    def handle(self: logging.Handler, record: logging.LogRecord):  # type: ignore[override]
        if getattr(self, "_wopr_handler", False):
            return orig_handle(self, record)
        if getattr(record, "_wopr_routed2", False):
            return orig_handle(self, record)

        try:
            record._wopr_routed2 = True  # type: ignore[attr-defined]
            try:
                record.name = map_ns(getattr(record, "name", ""))
            except Exception:
                pass
            logging.getLogger().handle(record)
        finally:
            try:
                delattr(record, "_wopr_routed2")  # type: ignore[attr-defined]
            except Exception:
                pass
        return

    logging.Handler.handle = handle  # type: ignore[assignment]

def lock_root(root: logging.Logger, our_handler: logging.Handler) -> None:
    if getattr(root, "_wopr_locked", False):
        return
    root._wopr_locked = True  # type: ignore[attr-defined]
    _orig_add = root.addHandler
    def addHandler(h):  # type: ignore[override]
        if h is our_handler:
            return _orig_add(h)
        if isinstance(h, logging.StreamHandler):
            return
        return _orig_add(h)
    root.addHandler = addHandler  # type: ignore[assignment]
