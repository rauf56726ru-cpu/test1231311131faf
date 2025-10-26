"""Timestamp utilities shared across the application."""

from __future__ import annotations

from datetime import datetime, timezone

_MIN_EPOCH_MS = int(datetime(2015, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
_MAX_EPOCH_MS = int(datetime(2035, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def ensure_epoch_ms(ts: int | float) -> int:
    """Normalise ``ts`` to milliseconds since epoch within supported bounds."""

    try:
        value = int(ts)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("invalid ts") from exc

    if value <= 0:
        raise ValueError("invalid ts<=0")

    if value < 1_000_000_000_000:
        value *= 1000
    elif value > 10_000_000_000_000:
        # reduce microsecond or nanosecond inputs down to milliseconds
        while value > 10_000_000_000_000 and value % 1000 == 0:
            value //= 1000
        if value > 10_000_000_000_000:
            value //= 1000

    if not (_MIN_EPOCH_MS <= value <= _MAX_EPOCH_MS):
        raise ValueError(f"epoch-ms out of bounds: {value}")

    return value


__all__ = ["ensure_epoch_ms"]
