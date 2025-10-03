"""Utilities for working with market-data timestamps."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
import math
from typing import Any


LOGGER = logging.getLogger(__name__)

_FUTURE_DRIFT = timedelta(days=1)
_UTC = timezone.utc


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
    """Normalise timestamps to milliseconds since the Unix epoch."""

    numeric = _coerce_int(value)
    if numeric is None:
        return None

    abs_value = abs(numeric)
    if abs_value >= 10**14:
        numeric //= 1000
    elif abs_value < 10**10:
        numeric *= 1000

    if numeric <= 0:
        return None

    now = datetime.now(_UTC)
    max_allowed = int((now + _FUTURE_DRIFT).timestamp() * 1000)
    if numeric > max_allowed:
        return None

    return numeric


def safe_datetime_from_ms(ms: int, tz: timezone) -> datetime | None:
    """Convert milliseconds to ``datetime`` guarding against runtime errors."""

    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=tz)
    except (OverflowError, OSError, ValueError):
        LOGGER.warning("Failed to convert timestamp to datetime", extra={"ms": ms})
        return None

