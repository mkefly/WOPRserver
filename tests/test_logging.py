# tests/test_logging_formatter.py
from __future__ import annotations

import logging
import sys
import traceback
from types import SimpleNamespace

import woprserver.logging as wopr_logging

# ---------- tiny helpers ----------

def _mk_record(
    *,
    name: str = "woprserver.testlogger",
    level: int = logging.INFO,
    msg: str = "hello",
    pathname: str = __file__,
    lineno: int = 1,
    exc_info=None,
    stack_info: str | None = None,
    extra: dict | None = None,
):
    """
    Create a LogRecord similar to what logging would emit.
    """
    rec = logging.LogRecord(
        name=name,
        level=level,
        pathname=pathname,
        lineno=lineno,
        msg=msg,
        args=(),
        exc_info=exc_info,
        func=None,
        sinfo=stack_info,
    )
    if extra:
        for k, v in extra.items():
            setattr(rec, k, v)
    return rec


# --------------------------------- _supports_color ---------------------------------

def test_supports_color_respects_NO_COLOR_env(monkeypatch):
    # Simulate TTY true
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True, encoding="UTF-8"))
    monkeypatch.setenv("NO_COLOR", "1")

    assert wopr_logging._supports_color() is False

    # Without NO_COLOR
    monkeypatch.delenv("NO_COLOR", raising=False)
    assert wopr_logging._supports_color() is True


# --------------------------------- header formatting ---------------------------------

def test_header_formats_without_color(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    fmt = wopr_logging.WoprFormatter(use_color=False)

    rec = _mk_record(level=logging.INFO, msg="hello")
    out = fmt.format(rec).strip()

    # Expect timestamp, [logger], level, hyphen, message (without ANSI codes)
    assert " [woprserver.testlogger] INFO - hello" in out


def test_header_uses_system_name_when_present(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    fmt = wopr_logging.WoprFormatter(use_color=False)

    rec = _mk_record(level=logging.DEBUG, msg="boot", extra={"system_name": "WOPR"})
    out = fmt.format(rec).strip()
    assert " [WOPR] DEBUG - boot" in out


# --------------------------------- warnings path ---------------------------------

def test_pywarnings_emit_banner_block_no_color(monkeypatch):
    # No color -> still bannered
    monkeypatch.setenv("NO_COLOR", "1")
    fmt = wopr_logging.WoprFormatter(use_color=False)

    # The formatter looks only at name == "py.warnings" and level >= WARNING.
    payload = "file.py:12: UserWarning: Danger!\n  x = 1"
    rec = _mk_record(name="py.warnings", level=logging.WARNING, msg=payload)
    out = fmt.format(rec)

    # Has the blank-line+banner structure
    assert out.startswith("\n")
    assert "WARNING START TRACE" in out
    assert "WARNING END TRACE" in out
    assert "UserWarning" in out
    assert "Danger!" in out
    # payload is indented in the block
    assert "\n    file.py:12: UserWarning: Danger!" in out


# --------------------------------- error / critical ---------------------------------

def test_error_includes_bannered_traceback_block(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    fmt = wopr_logging.WoprFormatter(use_color=False)

    try:
        raise ValueError("oops")
    except Exception:
        exc = sys.exc_info()

    rec = _mk_record(level=logging.ERROR, msg="failed", exc_info=exc)
    out = fmt.format(rec)

    assert "[woprserver.testlogger] ERROR - failed" in out
    assert "ERROR START TRACE" in out
    assert "ERROR END TRACE" in out
    assert "ValueError" in out
    assert "oops" in out
    assert 'File "' in out  # traceback header


def test_warning_with_stack_has_banner_block(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    fmt = wopr_logging.WoprFormatter(use_color=False)

    # Build a fake stack text (logging uses traceback.format_stack() normally)
    stack = "".join(traceback.format_stack(limit=2))
    rec = _mk_record(level=logging.WARNING, msg="careful", stack_info=stack)
    out = fmt.format(rec)

    assert "[woprserver.testlogger] WARNING - careful" in out
    assert "WARNING START TRACE" in out
    assert "WARNING END TRACE" in out
    # Stack lines should appear indented
    assert "\n    " in out


# --------------------------------- info/debug single line ---------------------------------

def test_info_and_debug_are_single_line(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    fmt = wopr_logging.WoprFormatter(use_color=False)

    out_info = fmt.format(_mk_record(level=logging.INFO, msg="one"))
    out_debug = fmt.format(_mk_record(level=logging.DEBUG, msg="two"))

    # No START/END banners expected
    assert "START TRACE" not in out_info
    assert "END TRACE" not in out_info
    assert "START TRACE" not in out_debug
    assert "END TRACE" not in out_debug

    assert " INFO - one" in out_info
    assert " DEBUG - two" in out_debug

# --------------------------------- colorized path ---------------------------------

def test_colorized_output_injects_ansi(monkeypatch):
    # Force terminal support and allow colors
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys, "stdout", SimpleNamespace(isatty=lambda: True, encoding="UTF-8"))

    fmt = wopr_logging.WoprFormatter(use_color=True)
    out = fmt.format(_mk_record(level=logging.INFO, msg="colored"))

    # Look for ANSI CSI sequence \x1b[
    assert "\x1b[" in out
