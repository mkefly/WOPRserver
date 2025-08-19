
from __future__ import annotations

import json
import logging
import logging.config
from pathlib import Path
from typing import Optional, Dict, Union

from .config import LoggingConfig, default_config
from .record import install_logrecord_factory
from .routing import install_hard_router, install_global_handler_router, lock_root
from .handlers import create_root_handler
from .neutralizers import neutralize_mlserver, neutralize_mlflow, disarm_uvicorn

# --- Package Facade / Orchestrator -----------------------------------------

def _install_logger_creation_hooks() -> None:
    if getattr(_install_logger_creation_hooks, "_installed", False):
        return
    _install_logger_creation_hooks._installed = True

    _orig_getLogger = logging.getLogger
    from .utils import map_ns
    def getLogger(name: str | None = None):  # type: ignore[override]
        return _orig_getLogger(map_ns(name) if name else name)
    logging.getLogger = getLogger  # type: ignore[assignment]

    _orig_mgr_getLogger = logging.Manager.getLogger
    def mgr_getLogger(self: logging.Manager, name: str):  # type: ignore[override]
        return _orig_mgr_getLogger(self, map_ns(name))
    logging.Manager.getLogger = mgr_getLogger  # type: ignore[assignment]

def _strip_all_handlers_and_enable_propagation() -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        try: root.removeHandler(h)
        except Exception: pass
    for entry in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(entry, logging.Logger):
            entry.propagate = True
            for h in list(entry.handlers):
                try: entry.removeHandler(h)
                except Exception: pass

def install_root_logging(cfg: LoggingConfig) -> None:
    if getattr(install_root_logging, "_installed", False):
        return
    install_root_logging._installed = True

    if cfg.capture_warnings:
        logging.captureWarnings(True)

    _install_logger_creation_hooks()
    install_hard_router()

    # third-party neutralization
    neutralize_mlserver()
    disarm_uvicorn()
    neutralize_mlflow()

    _strip_all_handlers_and_enable_propagation()
    install_logrecord_factory()

    root = logging.getLogger()
    level = getattr(logging, str(cfg.level).upper(), logging.INFO)
    root.setLevel(level if isinstance(level, int) else logging.INFO)

    handler = create_root_handler(cfg)
    root.addHandler(handler)
    install_global_handler_router(handler)
    lock_root(root, handler)

    # Route warnings logger
    pyw = logging.getLogger("py.warnings")
    pyw.propagate = True
    for h in list(pyw.handlers):
        try: pyw.removeHandler(h)
        except Exception: pass

def setup_once() -> None:
    if getattr(setup_once, "_did", False):
        return
    setup_once._did = True
    install_root_logging(default_config())

# Public API

logger = logging.getLogger("woprserver")

def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "woprserver")

def configure_logger(settings: Optional[object] = None) -> logging.Logger:
    setup_once()
    debug = bool(getattr(settings, "debug", False)) if settings is not None else False
    root = logging.getLogger()
    if debug:
        root.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    else:
        if root.level == logging.NOTSET:
            root.setLevel(logging.INFO)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)

    try:
        _maybe_apply_logging_file(getattr(settings, "logging_settings", None) if settings else None)
    except Exception:
        pass
    return logger

def _maybe_apply_logging_file(logging_settings: Optional[Union[str, Dict]]) -> None:
    if not logging_settings:
        return
    if isinstance(logging_settings, str) and Path(logging_settings).is_file():
        path = Path(logging_settings)
        if path.suffix.lower() == ".json":
            with open(path) as f:
                cfg = json.load(f)
            logging.config.dictConfig(cfg)
        else:
            logging.config.fileConfig(fname=str(path), disable_existing_loggers=False)
    elif isinstance(logging_settings, dict):
        logging.config.dictConfig(logging_settings)
