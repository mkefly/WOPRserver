
from __future__ import annotations
import logging
from .formatting import WoprFormatter
from .utils import NameRewriteFilter, DeDupeFilter, supports_color
from .config import LoggingConfig

def create_root_handler(cfg: LoggingConfig) -> logging.Handler:
    use_color = supports_color() if cfg.use_color is None else bool(cfg.use_color)
    handler = logging.StreamHandler()
    handler._wopr_handler = True  # type: ignore[attr-defined]
    handler.addFilter(NameRewriteFilter())
    handler.addFilter(DeDupeFilter())
    handler.setFormatter(WoprFormatter(use_color=use_color, banner_char=cfg.banner_char))
    return handler
