from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

from .utils import map_ns

# ─────────────────────────────────────────────────────────────────────────────
# Context registry (single source of truth for contextvars)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CtxKey:
    # attribute name on LogRecord and in kwargs
    attr: str
    # default value for the ContextVar
    default: str = ""


class ContextRegistry:
    """
    Manages a set of ContextVars. Provides uniform set/reset and bulk utilities.
    """
    def __init__(self, keys: Iterable[CtxKey]) -> None:
        self._keys: Tuple[CtxKey, ...] = tuple(keys)
        self._vars: Dict[str, ContextVar[str]] = {
            k.attr: ContextVar(k.attr, default=k.default) for k in self._keys
        }

    def var(self, attr: str) -> ContextVar[str]:
        return self._vars[attr]

    def get(self, attr: str) -> str:
        return self._vars[attr].get()

    def set_many(self, values: Mapping[str, str]) -> Dict[str, Token]:
        tokens: Dict[str, Token] = {}
        for k, v in values.items():
            if k in self._vars:
                tokens[k] = self._vars[k].set(v)
        return tokens

    def reset_many(self, tokens: Mapping[str, Token]) -> None:
        # Reset only what we set; ignore missing to be robust
        for k, tok in tokens.items():
            try:
                self._vars[k].reset(tok)
            except Exception:
                pass

    def inject_into_record(self, record: logging.LogRecord) -> None:
        """
        For each registered key, ensure the LogRecord has an attribute:
        explicit attribute on record wins; otherwise fill from ContextVar.
        """
        for k in self._keys:
            current = getattr(record, k.attr, "")
            if not current:
                setattr(record, k.attr, self._vars[k.attr].get(""))

    @contextmanager
    def context(self, **kwargs: str) -> Iterator[None]:
        tokens = self.set_many(kwargs)
        try:
            yield
        finally:
            self.reset_many(tokens)

    # Convenience non-CM setter (sticky)
    def set_context(self, **kwargs: str) -> None:
        self.set_many(kwargs)


# Declare the context keys (add here to extend)
CTX_KEYS = [
    CtxKey("model_name"),
    CtxKey("model_version"),
    CtxKey("environment"),
    CtxKey("request_id"),
    CtxKey("http_method"),
    CtxKey("component"),
]
CTX = ContextRegistry(CTX_KEYS)

# Public API compatible with your previous names
context = CTX.context
set_context = CTX.set_context


# ─────────────────────────────────────────────────────────────────────────────
# Namespace resolution (extensible rules instead of if/elif chains)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NamespaceResult:
    namespace: str
    scope: str


class NamespaceResolver:
    """
    Resolves (namespace, scope) from a logger name using ordered rules.
    """
    def __init__(self) -> None:
        self._rules: List[Callable[[str], Optional[NamespaceResult]]] = []
        self._install_default_rules()

    def _install_default_rules(self) -> None:
        def startswith_rule(prefixes: Tuple[str, ...], ns: str):
            def _rule(name: str) -> Optional[NamespaceResult]:
                for p in prefixes:
                    if name.startswith(p + "."):
                        return NamespaceResult(ns, name.split(".", 1)[1])
                    if name == p:
                        return NamespaceResult(ns, "")
                return None
            return _rule

        # Map mlserver/mlflow into "woprserver"
        self._rules.append(startswith_rule(("woprserver",), "woprserver"))
        self._rules.append(startswith_rule(("mlserver", "mlflow"), "woprserver"))

        # Fallback: first segment or "root"
        def fallback(name: str) -> Optional[NamespaceResult]:
            if not name:
                return NamespaceResult("root", "")
            parts = name.split(".", 1)
            ns = parts[0]
            scope = parts[1] if len(parts) > 1 else ""
            return NamespaceResult(ns, scope)

        self._rules.append(fallback)

    def resolve(self, logger_name: str) -> NamespaceResult:
        for rule in self._rules:
            out = rule(logger_name)
            if out is not None:
                return out
        # Should never hit fallback here, but keep safe:
        return NamespaceResult("root", "")


NAMESPACE_RESOLVER = NamespaceResolver()


# ─────────────────────────────────────────────────────────────────────────────
# LogRecord enrichment pipeline (small pluggable steps)
# ─────────────────────────────────────────────────────────────────────────────

class LogRecordEnricher:
    """
    Applies an ordered set of enrichment steps to a LogRecord.
    Steps mutate the record in-place and return None.
    """

    def __init__(self) -> None:
        self._steps: List[Callable[[logging.LogRecord], None]] = []
        self._install_default_steps()

    def add_step(self, step: Callable[[logging.LogRecord], None]) -> None:
        self._steps.append(step)

    def enrich(self, record: logging.LogRecord) -> None:
        for step in self._steps:
            step(record)

    # Default steps
    def _install_default_steps(self) -> None:
        # 1) Resolve namespace/scope and system_name
        def step_namespace(record: logging.LogRecord) -> None:
            name = getattr(record, "name", "") or ""
            res = NAMESPACE_RESOLVER.resolve(name)
            record.wopr_ns = res.namespace
            record.wopr_scope = res.scope
            if not hasattr(record, "system_name"):
                record.system_name = res.namespace

        # 2) Inject contextvars (only if missing on the record)
        def step_context(record: logging.LogRecord) -> None:
            CTX.inject_into_record(record)

        # 3) Map display name via map_ns (safe)
        def step_display_name(record: logging.LogRecord) -> None:
            try:
                record.name = map_ns(getattr(record, "name", ""))
            except Exception:
                # Don't let mapping failures break logging
                pass

        self.add_step(step_namespace)
        self.add_step(step_context)
        self.add_step(step_display_name)


ENRICHER = LogRecordEnricher()


# ─────────────────────────────────────────────────────────────────────────────
# LogRecord factory installer
# ─────────────────────────────────────────────────────────────────────────────

def install_logrecord_factory() -> None:
    """
    Installs a log record factory exactly once, enriching each record via
    the pluggable `ENRICHER` pipeline.
    """
    if getattr(install_logrecord_factory, "_installed", False):
        return
    install_logrecord_factory._installed = True

    orig_factory = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = orig_factory(*args, **kwargs)
        ENRICHER.enrich(record)
        return record

    logging.setLogRecordFactory(factory)
