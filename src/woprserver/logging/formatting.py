from __future__ import annotations

import logging
import time
import re
from dataclasses import dataclass
from typing import Protocol, Dict, Optional, Callable, List, Tuple

from .utils import supports_color, supports_unicode


# ─────────────────────────────────────────────────────────────────────────────
# Color Strategy
# ─────────────────────────────────────────────────────────────────────────────

class ColorStrategy(Protocol):
    def color(self, text: str, role: str) -> str: ...


class NoColorStrategy:
    def color(self, text: str, role: str) -> str:
        return text


class AnsiColorStrategy:
    COLORS: Dict[str, str] = {
        "reset": "\x1b[0m",
        "dim": "\x1b[38;5;244m",
        "name": "\x1b[38;5;177m",
        "info": "\x1b[38;5;45m",
        "warn": "\x1b[38;5;214m",
        "err":  "\x1b[38;5;196m",
        "file": "\x1b[38;5;81m",
        "line": "\x1b[38;5;207m",
        "func": "\x1b[38;5;149m",
        "exc":  "\x1b[38;5;199m",
        "code": "\x1b[38;5;252m",
        "env":  "\x1b[38;5;112m",
        "pid":  "\x1b[38;5;39m",
        "meth": "\x1b[38;5;178m",
        "model":"\x1b[38;5;141m",
        "ts":   "\x1b[38;5;244m",

        # banner cosmetics
        "banner_line_warn": "\x1b[38;5;214m",
        "banner_line_err":  "\x1b[38;5;196m",
        "banner_label_warn":"\x1b[38;5;214m",
        "banner_label_err": "\x1b[38;5;196m",
    }

    def color(self, text: str, role: str) -> str:
        c = self.COLORS.get(role, "")
        r = self.COLORS["reset"]
        return f"{c}{text}{r}" if c else text


# ─────────────────────────────────────────────────────────────────────────────
# Rendering Context (shared services)
# ─────────────────────────────────────────────────────────────────────────────

class RenderingContext:
    def __init__(self, use_color: bool, banner_char: str = "=") -> None:
        self.use_color = bool(use_color) and supports_color()
        self.use_unicode = self.use_color and supports_unicode()
        self.colors: ColorStrategy = AnsiColorStrategy() if self.use_color else NoColorStrategy()
        self.banner_char = banner_char

    # time / level helpers
    @staticmethod
    def ts(record: logging.LogRecord) -> str:
        t = time.localtime(record.created)
        base = time.strftime("%Y-%m-%d %H:%M:%S", t)
        ms = int(record.msecs)
        return f"{base},{ms:03d}"

    @staticmethod
    def level_role(levelno: int) -> str:
        return {
            logging.DEBUG: "dim",
            logging.INFO: "info",
            logging.WARNING: "warn",
            logging.ERROR: "err",
            logging.CRITICAL: "err",
        }.get(levelno, "dim")

    # indentation
    @staticmethod
    def indent_block(text: str) -> str:
        return "\n".join(("    " + ln) if ln else "" for ln in text.splitlines())

    # banners
    def banner(self, title: str, start: bool) -> str:
        # title: "ERROR" | "WARNING"
        line_role = f"banner_line_{'err' if title == 'ERROR' else 'warn'}"
        label_role = f"banner_label_{'err' if title == 'ERROR' else 'warn'}"
        border = self.colors.color(self.banner_char * 23, line_role)
        label = self.colors.color(f"{title} {'START' if start else 'END'} TRACE", label_role)
        return f"{border}  {label}  {border}"

    def banner_block(self, title: str, body: str) -> str:
        return f"\n{self.banner(title, True)}\n{self.indent_block(body)}\n{self.banner(title, False)}\n"


# ─────────────────────────────────────────────────────────────────────────────
# Header Segment Factory (eliminates if-chains)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentSpec:
    """
    value_fn: returns a value (or tuple) to render, or None to skip
    render_fn: builds the segment string from that value
    role: color role for the rendered segment (coarse). Sub-part coloring can
          still be done inside render_fn.
    """
    name: str
    value_fn: Callable[[logging.LogRecord], Optional[object]]
    render_fn: Callable[[object, logging.LogRecord, RenderingContext], str]
    role: str = "name"


class SegmentFactory:
    def __init__(self, specs: List[SegmentSpec]):
        self.specs = specs

    def build(self, record: logging.LogRecord, ctx: RenderingContext) -> List[Tuple[str, str]]:
        out: List[Tuple[str, str]] = []
        for spec in self.specs:
            val = spec.value_fn(record)
            if val is None or val == "" or val == [] or val == ():
                continue
            out.append((spec.render_fn(val, record, ctx), spec.role))
        return out


# helpers
def _get(name: str, record: logging.LogRecord, default=None):
    v = getattr(record, name, default)
    return v if v not in ("", None) else None


def make_default_segment_factory() -> SegmentFactory:
    def render_file(v, r, ctx: RenderingContext) -> str:
        filename, lineno = v
        return "[" + \
            f"{ctx.colors.color('file', 'dim')}:" + \
            f"{ctx.colors.color(filename, 'file')}:" + \
            f"{ctx.colors.color(str(lineno or ''), 'line')}" + \
            "]"

    def render_model(v, r, ctx: RenderingContext) -> str:
        name, ver = v
        return f"[{name}:{ver}]" if ver else f"[{name}]"

    specs = [
        SegmentSpec(
            "pid",
            lambda r: _get("process", r),
            lambda v, r, ctx: f"[pid:{v}]",
            role="pid",
        ),
        SegmentSpec(
            "system",
            lambda r: (_get("system_name", r) or _get("name", r) or "root"),
            lambda v, r, ctx: f"[{v}]",
            role="name",
        ),
        SegmentSpec(
            "model",
            lambda r: (_get("model_name", r), _get("model_version", r)) if _get("model_name", r) else None,
            render_model,
            role="name",
        ),
        SegmentSpec(
            "env",
            lambda r: _get("environment", r),
            lambda v, r, ctx: f"[env:{v}]",
            role="env",
        ),
        SegmentSpec(
            "component",
            lambda r: _get("component", r),
            lambda v, r, ctx: f"[component:{v}]",
            role="dim",
        ),
        SegmentSpec(
            "file",
            lambda r: (_get("filename", r), _get("lineno", r)) if _get("filename", r) else None,
            render_file,
            role="name",  # coarse role; sub-parts colored in render_file
        ),
        SegmentSpec(
            "func",
            lambda r: _get("funcName", r),
            lambda v, r, ctx: f"[func:{v}]",
            role="func",
        ),
        SegmentSpec(
            "method",
            lambda r: _get("http_method", r),
            lambda v, r, ctx: f"[method:{v}]",
            role="meth",
        ),
        SegmentSpec(
            "req",
            lambda r: _get("request_id", r),
            lambda v, r, ctx: f"[req:{v}]",
            role="dim",
        ),
    ]
    return SegmentFactory(specs)


# ─────────────────────────────────────────────────────────────────────────────
# Highlighters
# ─────────────────────────────────────────────────────────────────────────────

class TracebackHighlighter:
    file_re = re.compile(r'^(  File )(".*?")(, line )(\d+)(, in )([^\n]+)$')
    exc_re  = re.compile(r"^([A-Za-z_][A-Za-z0-9_\.]*Error|Exception)(: .*)$")
    code_re = re.compile(r"^(\s{2,})(?!File )(?!\^)(.*)$")

    @staticmethod
    def highlight(text: str, ctx: RenderingContext) -> str:
        if not ctx.use_color:
            return text

        def _file_sub(m: "re.Match[str]") -> str:
            return ''.join([
                m.group(1),
                ctx.colors.color(m.group(2), "file"),
                m.group(3),
                ctx.colors.color(m.group(4), "line"),
                m.group(5),
                ctx.colors.color(m.group(6), "func"),
            ])

        out = []
        for line in text.splitlines():
            if TracebackHighlighter.file_re.match(line):
                line = TracebackHighlighter.file_re.sub(_file_sub, line)
            elif TracebackHighlighter.exc_re.match(line):
                g = TracebackHighlighter.exc_re.match(line); assert g is not None
                line = ctx.colors.color(g.group(1), "exc") + (g.group(2) or "")
            elif TracebackHighlighter.code_re.match(line):
                g = TracebackHighlighter.code_re.match(line); assert g is not None
                line = g.group(1) + ctx.colors.color(g.group(2), "code")
            out.append(line)
        return "\n".join(out)


class PyWarningHighlighter:
    head = re.compile(r"^(.+?):(\d+):\s*([A-Za-z_][\w\.]*Warning):(.*)$")
    code_re = re.compile(r"^(\s{2,})(.*)$")

    @staticmethod
    def highlight(text: str, ctx: RenderingContext) -> str:
        if not ctx.use_color:
            return text
        out = []
        for line in text.splitlines():
            m = PyWarningHighlighter.head.match(line)
            if m:
                path, lineno, cat, rest = m.groups()
                line = f"{ctx.colors.color(path, 'file')}:{ctx.colors.color(lineno, 'line')}: {ctx.colors.color(cat, 'exc')}:{rest}"
            else:
                mc = PyWarningHighlighter.code_re.match(line)
                if mc:
                    line = mc.group(1) + ctx.colors.color(mc.group(2), 'code')
            out.append(line)
        return "\n".join(out)


# ─────────────────────────────────────────────────────────────────────────────
# Chain of Responsibility for record rendering
# ─────────────────────────────────────────────────────────────────────────────

class RecordHandler(Protocol):
    def can_handle(self, record: logging.LogRecord) -> bool: ...
    def render(self, record: logging.LogRecord, ctx: RenderingContext, header: str) -> str: ...


class PyWarningsHandler:
    def can_handle(self, record: logging.LogRecord) -> bool:
        return record.name == "py.warnings" and record.levelno >= logging.WARNING

    def render(self, record: logging.LogRecord, ctx: RenderingContext, header: str) -> str:
        payload = record.getMessage().rstrip("\n")
        payload = PyWarningHighlighter.highlight(payload, ctx)
        return ctx.banner_block("WARNING", payload)


class ErrorHandler:
    def can_handle(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR

    def render(self, record: logging.LogRecord, ctx: RenderingContext, header: str) -> str:
        msg = record.getMessage()
        out = [f"{header}{msg}"]
        tb_text: Optional[str] = None
        if record.exc_info:
            import traceback
            tb_text = "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
        elif record.stack_info:
            tb_text = record.stack_info.rstrip("\n")
        if tb_text:
            tb_text = TracebackHighlighter.highlight(tb_text, ctx)
            out.append(ctx.banner_block("ERROR", tb_text))
        return "\n".join(out)


class WarningHandler:
    def can_handle(self, record: logging.LogRecord) -> bool:
        return record.levelno == logging.WARNING

    def render(self, record: logging.LogRecord, ctx: RenderingContext, header: str) -> str:
        msg = record.getMessage()
        tb_text: Optional[str] = None
        if record.exc_info:
            import traceback
            tb_text = "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
        elif record.stack_info:
            tb_text = record.stack_info.rstrip("\n")
        if tb_text:
            tb_text = TracebackHighlighter.highlight(tb_text, ctx)
            return f"{header}{msg}\n{ctx.banner_block('WARNING', tb_text)}"
        return f"{header}{msg}"


class DefaultHandler:
    def can_handle(self, record: logging.LogRecord) -> bool:
        return True

    def render(self, record: logging.LogRecord, ctx: RenderingContext, header: str) -> str:
        return f"{header}{record.getMessage()}"


# ─────────────────────────────────────────────────────────────────────────────
# Formatter
# ─────────────────────────────────────────────────────────────────────────────

class WoprFormatter(logging.Formatter):
    """
    Header fields (when available):
    [pid:<id>] [<system>] [<model>:<version>] [env:<environment>] [file:<filename>:<lineno>]
    [func:<funcName>] [method:<http_method>] [req:<request_id>] [component:<component>]
    """

    def __init__(self, *, use_color: bool, banner_char: str = "=") -> None:
        super().__init__(fmt="%(message)s")
        self._ctx = RenderingContext(use_color=use_color, banner_char=banner_char)
        self._seg_factory = make_default_segment_factory()
        self._handlers: List[RecordHandler] = [
            PyWarningsHandler(),
            ErrorHandler(),
            WarningHandler(),
            DefaultHandler(),
        ]

    # header building (no if-chains)
    def _header(self, record: logging.LogRecord) -> str:
        ts = self._ctx.ts(record)
        lvl_role = self._ctx.level_role(record.levelno)

        segments = self._seg_factory.build(record, self._ctx)

        if self._ctx.use_color:
            ts_col = self._ctx.colors.color(ts, "ts")
            segs_col = "".join(self._ctx.colors.color(seg, role) for seg, role in segments)
            lvl = self._ctx.colors.color(record.levelname, lvl_role)
            return f"{ts_col} {segs_col} {lvl} - "
        else:
            segs_plain = "".join(seg for seg, _ in segments)
            return f"{ts} {segs_plain} {record.levelname} - "

    # Python's logging.Formatter API
    def format(self, record: logging.LogRecord) -> str:
        header = self._header(record)
        for handler in self._handlers:
            if handler.can_handle(record):
                return handler.render(record, self._ctx, header)
        # Should not reach here because DefaultHandler handles all
        return f"{header}{record.getMessage()}"
