
from __future__ import annotations

import logging
import time
import re
from typing import Protocol, Dict

from .utils import supports_color, supports_unicode

# ---- Strategy Pattern for Colorization ------------------------------------

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
    }
    def color(self, text: str, role: str) -> str:
        c = self.COLORS.get(role, "")
        r = self.COLORS["reset"]
        return f"{c}{text}{r}" if c else text

# ---- Banner Builder --------------------------------------------------------

class BannerBuilder:
    def __init__(self, char: str = "=") -> None:
        self.char = char
    def build(self, title: str, start: bool) -> str:
        label = f"{title} {'START' if start else 'END'} TRACE"
        ch = self.char
        return f"{ch*24}  {label}  {ch*24}"

# ---- The Formatter ---------------------------------------------------------

class WoprFormatter(logging.Formatter):
    """
    Header fields (when available):
    [pid:<id>] [<system>] [<model>:<version>] [env:<environment>] [file:<filename>:<lineno>]
    [func:<funcName>] [method:<http_method>] [req:<request_id>] [component:<component>]
    """

    def __init__(self, *, use_color: bool, banner_char: str = "=") -> None:
        super().__init__(fmt="%(message)s")
        self._use_color = bool(use_color) and supports_color()
        self._use_unicode = self._use_color and supports_unicode()
        self._colors = AnsiColorStrategy() if self._use_color else NoColorStrategy()
        self._banner = BannerBuilder(char=banner_char)

    # --- small helpers
    def _ts(self, record: logging.LogRecord) -> str:
        t = time.localtime(record.created)
        base = time.strftime("%Y-%m-%d %H:%M:%S", t)
        ms = int(record.msecs)
        return f"{base},{ms:03d}"

    def _lvl_color_role(self, levelno: int) -> str:
        return {logging.DEBUG: "dim", logging.INFO: "info",
                logging.WARNING: "warn", logging.ERROR: "err",
                logging.CRITICAL: "err"}.get(levelno, "dim")

    # --- header construction
    def _header(self, record: logging.LogRecord) -> str:
        ts = self._ts(record)
        lvl_role = self._lvl_color_role(record.levelno)
        system = getattr(record, "system_name", None) or getattr(record, "name", "root")
        pid = getattr(record, "process", None)
        func = getattr(record, "funcName", "")
        lineno = getattr(record, "lineno", None)
        filename = getattr(record, "filename", "")
        model = getattr(record, "model_name", "") or ""
        mver = getattr(record, "model_version", "") or ""
        env = getattr(record, "environment", "") or ""
        http_method = getattr(record, "http_method", "") or ""
        req_id = getattr(record, "request_id", "") or ""
        component = getattr(record, "component", "") or ""

        segs = []
        if pid:
            segs.append(f"[pid:{pid}]")
        segs.append(f"[{system}]")
        if model:
            segs.append(f"[{model}:{mver}]" if mver else f"[{model}]")
        if env:
            segs.append(f"[env:{env}]")
        if component:
            segs.append(f"[component:{component}]")
        if filename:
            segs.append(f"[file:{filename}:{lineno if lineno is not None else ''}]")
        if func:
            segs.append(f"[func:{func}]")
        if http_method:
            segs.append(f"[method:{http_method}]")
        if req_id:
            segs.append(f"[req:{req_id}]")

        # apply colors per role
        if self._use_color:
            colored = []
            colored.append(self._colors.color(ts, "ts"))
            # system name highlighted
            tmp = []
            for s in segs:
                if s.startswith("[pid:"):
                    tmp.append(self._colors.color(s, "pid"))
                elif s.startswith("[env:"):
                    tmp.append(self._colors.color(s, "env"))
                elif s.startswith("[file:"):
                    # color "file", filename, and line separately for readability
                    m = re.match(r"\[file:(.*?):(.*?)\]", s)
                    if m:
                        fn, ln = m.groups()
                        tiny = "[" + self._colors.color("file", "dim") + ":" + self._colors.color(fn, "file") + ":" + self._colors.color(ln, "line") + "]"
                        tmp.append(tiny)
                    else:
                        tmp.append(s)
                elif s.startswith("[func:"):
                    tmp.append(self._colors.color(s, "func"))
                elif s.startswith("[method:"):
                    tmp.append(self._colors.color(s, "meth"))
                elif s.startswith("[req:"):
                    tmp.append(self._colors.color(s, "dim"))
                elif s.startswith("[component:"):
                    tmp.append(self._colors.color(s, "dim"))
                elif s.startswith("[woprserver]") or s.startswith("[root]") or s.startswith("["):
                    tmp.append(self._colors.color(s, "name"))
                else:
                    tmp.append(s)
            colored.append(" ".join(tmp))
            lvl = self._colors.color(record.levelname, lvl_role)
            return f"{colored[0]} {colored[1]} {lvl} - "
        else:
            return f"{ts} {' '.join(segs)} {record.levelname} - "

    # --- Highlight helpers
    def _hl_traceback(self, text: str) -> str:
        if not self._use_color:
            return text
        file_re = re.compile(r'^(  File )(".*?")(, line )(\d+)(, in )([^\n]+)$')
        exc_re  = re.compile(r"^([A-Za-z_][A-Za-z0-9_\.]*Error|Exception)(: .*)$")
        code_re = re.compile(r"^(\s{2,})(?!File )(?!\^)(.*)$")

        def _file_sub(m: "re.Match[str]") -> str:
            return ''.join([
                m.group(1),
                self._colors.color(m.group(2), "file"),
                m.group(3),
                self._colors.color(m.group(4), "line"),
                m.group(5),
                self._colors.color(m.group(6), "func"),
            ])

        out = []
        for line in text.splitlines():
            if file_re.match(line):
                line = file_re.sub(_file_sub, line)
            elif exc_re.match(line):
                g = exc_re.match(line); assert g is not None
                line = self._colors.color(g.group(1), "exc") + (g.group(2) or "")
            elif code_re.match(line):
                g = code_re.match(line); assert g is not None
                line = g.group(1) + self._colors.color(g.group(2), "code")
            out.append(line)
        return "\n".join(out)

    def _hl_pywarning(self, text: str) -> str:
        if not self._use_color:
            return text
        head = re.compile(r"^(.+?):(\d+):\s*([A-Za-z_][\w\.]*Warning):(.*)$")
        code_re = re.compile(r"^(\s{2,})(.*)$")

        out = []
        for line in text.splitlines():
            m = head.match(line)
            if m:
                path, lineno, cat, rest = m.groups()
                line = f"{self._colors.color(path, 'file')}:{self._colors.color(lineno, 'line')}: {self._colors.color(cat, 'exc')}:{rest}"
            else:
                mc = code_re.match(line)
                if mc:
                    line = mc.group(1) + self._colors.color(mc.group(2), 'code')
            out.append(line)
        return "\n".join(out)

    # --- Main format
    def format(self, record: logging.LogRecord) -> str:
        # Python warnings
        if record.name == "py.warnings" and record.levelno >= logging.WARNING:
            payload = record.getMessage().rstrip("\n")
            payload = self._hl_pywarning(payload)
            start_banner = self._banner.build("WARNING", True)
            end_banner   = self._banner.build("WARNING", False)
            indented = "\n".join(("    " + ln) if ln else "" for ln in payload.splitlines())
            return f"\n{start_banner}\n{indented}\n{end_banner}\n"

        head = self._header(record)
        msg = record.getMessage()

        # Errors: attach traceback banner if present
        if record.levelno >= logging.ERROR:
            out = [f"{head}{msg}"]
            tb_text: str | None = None
            if record.exc_info:
                import traceback
                tb_text = "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
            elif record.stack_info:
                tb_text = record.stack_info.rstrip("\n")
            if tb_text:
                tb_text = self._hl_traceback(tb_text)
                indented = "\n".join(("    " + ln) if ln else "" for ln in tb_text.splitlines())
                start_banner = self._banner.build("ERROR", True)
                end_banner   = self._banner.build("ERROR", False)
                out += [f"\n{start_banner}", indented, f"{end_banner}\n"]
            return "\n".join(out)

        # Warnings with traceback
        if record.levelno == logging.WARNING:
            if record.exc_info or record.stack_info:
                out = [f"{head}{msg}"]
                tb_text: str | None = None
                if record.exc_info:
                    import traceback
                    tb_text = "".join(traceback.format_exception(*record.exc_info)).rstrip("\n")
                elif record.stack_info:
                    tb_text = record.stack_info.rstrip("\n")
                if tb_text:
                    tb_text = self._hl_traceback(tb_text)
                    indented = "\n".join(("    " + ln) if ln else "" for ln in tb_text.splitlines())
                    start_banner = self._banner.build("WARNING", True)
                    end_banner   = self._banner.build("WARNING", False)
                    out += [f"\n{start_banner}", indented, f"{end_banner}\n"]
                return "\n".join(out)
            return f"{head}{msg}"

        # Info/Debug
        return f"{head}{msg}"
