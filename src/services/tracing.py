"""Tracing and diagnostic helpers for the service layer.

The tracing module exposes a module-level ``LOGGER`` whose verbosity can be
controlled through the ``APP_TRACE`` environment variable. Setting the
variable to ``"on"`` promotes the logger to DEBUG while keeping INFO as the
default level otherwise. The logger intentionally relies on propagation so
that Uvicorn's logging configuration remains in control of handler setup.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Mapping

__all__ = ["LOGGER", "is_trace_enabled", "trace", "_TRACE_ENABLED"]

_TRACE_ENABLED = os.getenv("APP_TRACE", "").lower() == "on"

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG if _TRACE_ENABLED else logging.INFO)
# ``LOGGER.propagate`` remains untouched (True by default) to allow upstream
# handlers configured by Uvicorn to process emitted records.


def is_trace_enabled() -> bool:
    """Return whether detailed trace logging is enabled."""

    return _TRACE_ENABLED


def trace(event: str, extra: Mapping[str, Any] | None = None) -> None:
    """Emit a structured INFO log entry for the provided event.

    Parameters
    ----------
    event:
        The event name or message to log.
    extra:
        Optional dictionary merged into the log record via the ``extra``
        parameter. The mapping defaults to an empty dictionary when omitted to
        avoid mutating logging internals with ``None`` values.
    """

    if extra is None:
        LOGGER.info(event)
    else:
        LOGGER.info(event, extra=dict(extra))
