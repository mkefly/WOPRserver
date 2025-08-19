
from __future__ import annotations
import logging
import os

def neutralize_mlserver() -> None:
    try:
        import mlserver.logging as mlog  # type: ignore
    except Exception:
        return
    def _noop(*a, **k): return None
    for n in ("configure_logging","setup_logging","init_logging","configure"):
        if hasattr(mlog, n):
            try:
                setattr(mlog, n, _noop)
            except Exception:
                pass

def disarm_uvicorn() -> None:
    try:
        import uvicorn.config as ucfg  # type: ignore
        ucfg.LOGGING_CONFIG = None
    except Exception:
        pass

def neutralize_mlflow() -> None:
    os.environ.setdefault("MLFLOW_CONFIGURE_LOGGING","false")
    try:
        import mlflow.utils.logging_utils as mlog_utils  # type: ignore
        try:
            mlog_utils.configure_logging = lambda *a, **k: None  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        return
    for n in ("mlflow","mlflow.models","mlflow.store","mlflow.tracking","mlflow.utils"):
        lg = logging.getLogger(n)
        lg.propagate = True
        for h in list(lg.handlers):
            try: lg.removeHandler(h)
            except Exception: pass
