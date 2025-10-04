"""Tracing and diagnostic helpers for the service layer.

The tracing module exposes a module-level `LOGGER` whose verbosity can be
controlled through the `APP_TRACE` environment variable. Setting the
variable to `"on"` promotes the logger to DEBUG while keeping INFO as the
default level otherwise. The logger intentionally relies on propagation so
that Uvicorn's logging configuration remains in control of handler setup.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping

__all__ = ["LOGGER", "TraceContext", "is_trace_enabled", "trace", "_TRACE_ENABLED"]

_TRACE_ENABLED = os.getenv("APP_TRACE", "").lower() == "on"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG if _TRACE_ENABLED else logging.INFO)
# `LOGGER.propagate` remains untouched (True by default) to allow upstream
# handlers configured by Uvicorn to process emitted records.


def is_trace_enabled() -> bool:
    """Return whether detailed trace logging is enabled."""

    return _TRACE_ENABLED


def _iso_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _make_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(2)}"


class TraceContext:
    """Structured logging helper that emits JSON trace events."""

    __slots__ = ("cid", "rid", "_static", "_enabled")

    def __init__(
        self,
        *,
        cid: str | None = None,
        rid: str | None = None,
        enabled: bool | None = None,
        **static: Any,
    ) -> None:
        self.cid = cid or _make_id("c")
        self.rid = rid or _make_id("r")
        self._static: MutableMapping[str, Any] = {}
        for key, value in static.items():
            if value is not None:
                self._static[key] = value
        if "cid" not in self._static:
            self._static["cid"] = self.cid
        if "rid" not in self._static:
            self._static["rid"] = self.rid
        self._enabled = is_trace_enabled() if enabled is None else bool(enabled)

    # ------------------------------------------------------------------
    # Context helpers
    # ------------------------------------------------------------------
    def child(self, **extra: Any) -> "TraceContext":
        payload = dict(self._static)
        payload.update(extra)
        return TraceContext(cid=self.cid, **payload)

    def bind(self, **extra: Any) -> "TraceContext":
        return self.child(**extra)

    # ------------------------------------------------------------------
    # Logging primitives
    # ------------------------------------------------------------------
    def _emit(self, level: str, event: str, *, override_rid: str | None = None, **fields: Any) -> None:
        level_name = level.upper()
        level_no = _LEVEL_MAP.get(level_name, logging.INFO)
        if not LOGGER.isEnabledFor(level_no):
            return
        if not self._enabled and level_no < logging.INFO:
            return
        payload: dict[str, Any] = {"ts": _iso_now(), "level": level_name, "event": event}
        payload.update(self._static)
        if override_rid:
            payload["rid"] = override_rid
        for key, value in fields.items():
            if value is None:
                continue
            payload[key] = value
        serialized = json.dumps(payload, separators=(",", ":"), ensure_ascii=True, default=str)
        LOGGER.log(level_no, serialized)

    def debug(self, event: str, **fields: Any) -> None:
        self._emit("DEBUG", event, **fields)

    def info(self, event: str, **fields: Any) -> None:
        self._emit("INFO", event, **fields)

    def warn(self, event: str, **fields: Any) -> None:
        self._emit("WARN", event, **fields)

    def error(self, event: str, **fields: Any) -> None:
        self._emit("ERROR", event, **fields)

    def span(self, event: str, **fields: Any) -> "TraceContext":
        """Return a child context with a fresh request id and emit a start event."""

        child_rid = _make_id("r")
        payload = dict(self._static)
        payload["rid"] = child_rid
        span_ctx = TraceContext(cid=self.cid, rid=child_rid, **payload)
        span_ctx.info(event, **fields)
        return span_ctx


def trace(event: str, extra: Mapping[str, Any] | None = None) -> None:
    """Emit a structured INFO log entry for the provided event."""

    payload = dict(extra or {})
    TraceContext(**payload).info(event)
