"""Shared helpers for timestamp math and numeric coercion in pipeline code."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Iterable, List

from .timeutils import safe_datetime_from_ms

UTC = timezone.utc

__all__ = [
    "align_to_interval",
    "build_expected_times",
    "coerce_float",
    "coerce_iso_timestamp",
    "isoformat_utc",
    "safe_float",
    "safe_int",
]


def safe_int(value: Any) -> int | None:
    """Best-effort conversion of ``value`` to an integer."""

    if value is None or isinstance(value, bool):
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


def safe_float(value: Any) -> float | None:
    """Attempt to convert ``value`` to a finite ``float``."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def coerce_float(value: Any) -> float:
    """Convert to float, returning 0.0 when conversion fails."""

    result = safe_float(value)
    return result if result is not None else 0.0


def align_to_interval(value: int, interval_ms: int) -> int:
    """Snap ``value`` down to the nearest multiple of ``interval_ms``."""

    if interval_ms <= 0:
        return value
    return (int(value) // int(interval_ms)) * int(interval_ms)


def build_expected_times(start_ms: int, end_ms: int, interval_ms: int) -> List[int]:
    """Return a list of aligned timestamps between ``start_ms`` and ``end_ms``."""

    if end_ms < start_ms or interval_ms <= 0:
        return []
    steps = ((end_ms - start_ms) // interval_ms) + 1
    return [start_ms + index * interval_ms for index in range(steps)]


def isoformat_utc(timestamp_ms: int) -> str:
    """Return a Z-suffixed ISO string for a millisecond timestamp."""

    clamped_ms = max(0, int(timestamp_ms))
    dt = safe_datetime_from_ms(clamped_ms, UTC)
    if dt is None:
        return datetime.fromtimestamp(0, tz=UTC).isoformat().replace("+00:00", "Z")
    dt = dt.replace(microsecond=0)
    return dt.isoformat().replace("+00:00", "Z")


def coerce_iso_timestamp(value: Any) -> str | None:
    """Return a best-effort ISO-8601 string for mixed timestamp inputs."""

    if isinstance(value, str) and value:
        return value

    ts_ms = safe_int(value)
    if ts_ms is None:
        return None

    if ts_ms < 10_000_000_000:  # treat as seconds
        ts_ms *= 1000

    return isoformat_utc(ts_ms)
