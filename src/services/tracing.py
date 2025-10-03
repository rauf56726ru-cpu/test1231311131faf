"""Utilities for structured tracing logs with correlation identifiers."""
from __future__ import annotations

import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, MutableMapping

LOGGER = logging.getLogger("app.trace")

_TRACE_ENABLED = os.getenv("APP_TRACE", "off").lower() == "on"

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}


def _isoformat(value: Any) -> str | None:
    """Convert milliseconds or ISO strings into canonical UTC ISO strings."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            timestamp = max(0, int(value)) / 1000.0
        except (TypeError, ValueError):  # pragma: no cover - defensive guard
            return None
        return (
            datetime.fromtimestamp(timestamp, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
    if isinstance(value, str) and value:
        if value.endswith("Z"):
            return value
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:  # pragma: no cover - defensive guard
            return value
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        else:
            parsed = parsed.astimezone(timezone.utc)
        return parsed.isoformat().replace("+00:00", "Z")
    return None


def _normalise_window(window: Any) -> Mapping[str, Any] | None:
    """Normalise window payloads into a from/to ISO mapping."""

    if window is None:
        return None
    if isinstance(window, Mapping):
        from_value = (
            window.get("from_utc")
            or window.get("from")
            or window.get("start")
            or window.get("start_ms")
            or window.get("from_ms")
        )
        to_value = (
            window.get("to_utc")
            or window.get("to")
            or window.get("end")
            or window.get("end_ms")
            or window.get("to_ms")
        )
        payload: dict[str, Any] = {}
        start_iso = _isoformat(from_value)
        if start_iso:
            payload["from_utc"] = start_iso
        end_iso = _isoformat(to_value)
        if end_iso:
            payload["to_utc"] = end_iso
        if payload:
            return payload
        return None
    if isinstance(window, (tuple, list)) and len(window) >= 2:
        start_iso = _isoformat(window[0])
        end_iso = _isoformat(window[1])
        payload: dict[str, Any] = {}
        if start_iso:
            payload["from_utc"] = start_iso
        if end_iso:
            payload["to_utc"] = end_iso
        if payload:
            return payload
    return None


def _serialise_metrics(metrics: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not metrics:
        return None
    filtered: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            filtered[key] = value
    return filtered or None


def _should_emit(level: str) -> bool:
    normalised = level.upper()
    if normalised == "DEBUG" and not _TRACE_ENABLED:
        return False
    return normalised in _LEVELS


def new_cid(prefix: str = "c") -> str:
    """Generate a correlation identifier."""

    return f"{prefix}_{secrets.token_hex(4)}"


def new_rid(prefix: str = "r") -> str:
    """Generate a span/request identifier."""

    return f"{prefix}_{secrets.token_hex(4)}"


@dataclass(slots=True)
class TraceContext:
    """Context for correlating structured trace events."""

    cid: str
    user_action: str | None = None

    def new_rid(self) -> str:
        return new_rid()


def log_event(
    *,
    level: str,
    event: str,
    cid: str | None = None,
    rid: str | None = None,
    user_action: str | None = None,
    symbol: str | None = None,
    tf: str | None = None,
    window: Any = None,
    details: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Emit a structured JSON log event respecting the trace configuration."""

    level_name = level.upper()
    if not _should_emit(level_name):
        return

    payload: MutableMapping[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "level": level_name,
        "event": event,
    }
    if cid:
        payload["cid"] = cid
    if rid:
        payload["rid"] = rid
    if user_action:
        payload["user_action"] = user_action
    if symbol:
        payload["symbol"] = symbol
    if tf:
        payload["tf"] = tf
    window_payload = _normalise_window(window)
    if window_payload:
        payload["window"] = window_payload
    if details:
        payload["details"] = details
    metrics_payload = _serialise_metrics(metrics)
    if metrics_payload:
        payload["metrics"] = metrics_payload
    if extra:
        for key, value in extra.items():
            if value is None:
                continue
            if key in payload:
                continue
            payload[key] = value

    target_logger = logger or LOGGER
    target_logger.log(_LEVELS[level_name], json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


__all__ = ["TraceContext", "log_event", "new_cid", "new_rid"]
