
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Union

@dataclass
class LoggingConfig:
    level: Union[int, str] = "INFO"
    use_color: Optional[bool] = None
    capture_warnings: bool = True
    banner_char: str = "="

def default_config() -> LoggingConfig:
    return LoggingConfig()
