"""Utilities for working with market-data timestamps."""
from __future__ import annotations

from datetime import date, datetime, timezone
import inspect
import logging
import math
from typing import Any

from src.common.ts import ensure_epoch_ms


LOGGER = logging.getLogger(__name__)

_UTC = timezone.utc
_INVALID_TOTAL = 0
_INVALID_LOGGED = 0
_INVALID_THRESHOLD = 20


def _record_invalid(value: Any) -> None:
    global _INVALID_TOTAL, _INVALID_LOGGED
    _INVALID_TOTAL += 1
    if _INVALID_LOGGED < _INVALID_THRESHOLD:
        caller = "<unknown>"
        try:
            frame = inspect.stack()[2]
            caller = f"{frame.filename}:{frame.lineno}"
        except Exception:  # pragma: no cover - defensive
            pass
        LOGGER.warning(
            "timestamp_invalid %s type=%s value=%r",
            caller,
            type(value).__name__,
            value,
        )
        _INVALID_LOGGED += 1
    elif _INVALID_LOGGED == _INVALID_THRESHOLD:
        LOGGER.warning(
            "timestamp_invalid.suppressed",
            extra={"suppressed": _INVALID_TOTAL - _INVALID_THRESHOLD, "last_value": value},
        )
        _INVALID_LOGGED += 1
    else:
        # Suppress further logs to avoid flooding output.
        _INVALID_LOGGED += 1


def _coerce_int(value: Any) -> int | None:
    """Best-effort conversion of ``value`` to an integer."""

    if value is None:
        return None

    if isinstance(value, bool):  # bool is an ``int`` subclass, skip explicitly
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return int(value)

    try:
        text = str(value).strip()
    except Exception:  # pragma: no cover - defensive guard
        return None

    if not text:
        return None

    try:
        return int(text)
    except ValueError:
        try:
            numeric = float(text)
        except ValueError:
            return None
        if not math.isfinite(numeric):
            return None
        return int(numeric)


def ensure_ms_epoch(value: Any) -> int | None:
    """Normalise mixed timestamp units to milliseconds since Unix epoch."""

    numeric: int | None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        numeric = int(dt.timestamp() * 1000)
    elif isinstance(value, date):
        dt = datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
        numeric = int(dt.timestamp() * 1000)
    else:
        numeric = _coerce_int(value)
        if numeric is None:
            return None

    try:
        return ensure_epoch_ms(numeric)
    except ValueError:
        if numeric >= 1_000_000_000:
            _record_invalid(value)
        return None


def safe_datetime_from_ms(ms: int, tz: timezone) -> datetime | None:
    """Convert milliseconds to ``datetime`` guarding against runtime errors."""

    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=tz)
    except (OverflowError, OSError, ValueError):
        LOGGER.warning("Failed to convert timestamp to datetime", extra={"ms": ms})
        return None
