"""Utilities for working with exchange timeframes."""
from __future__ import annotations

from typing import Dict

INTERVAL_SECONDS: Dict[str, float] = {
    "1s": 1.0,
    "1m": 60.0,
    "3m": 180.0,
    "5m": 300.0,
    "10m": 600.0,
    "15m": 900.0,
    "1h": 3_600.0,
    "4h": 14_400.0,
    "1d": 86_400.0,
    "1w": 604_800.0,
}


def interval_to_seconds(interval: str) -> float:
    """Convert a Binance/CCXT style timeframe to seconds."""
    interval = (interval or "").strip()
    if not interval:
        return 60.0
    if interval in INTERVAL_SECONDS:
        return INTERVAL_SECONDS[interval]
    try:
        value = float(interval[:-1])
        unit = interval[-1]
    except (ValueError, IndexError):
        return 60.0
    if unit.lower() == "s":
        return max(1.0, value)
    if unit.lower() == "m":
        return max(1.0, value * 60.0)
    if unit.lower() == "h":
        return max(1.0, value * 3_600.0)
    if unit.lower() == "d":
        return max(60.0, value * 86_400.0)
    return 60.0


def compute_refresh_seconds(
    interval: str,
    *,
    fraction: float,
    min_seconds: float = 1.0,
    max_seconds: float = 15.0,
) -> float:
    """Return a polling cadence based on the interval with clamps.

    Args:
        interval: Exchange interval string (e.g. ``"1m"`` or ``"1d"``).
        fraction: Multiplier applied to the base interval length. For example,
            a fraction of ``0.25`` polls four times per interval while ``1/12``
            mirrors the legacy ``interval / 12`` behaviour used by the
            websocket tick loop.
        min_seconds: Lower bound for the cadence to avoid tight loops.
        max_seconds: Upper bound to ensure large intervals refresh frequently.

    Returns:
        A number of seconds satisfying the requested clamps. ``max_seconds``
        may be set to ``<= 0`` to disable the upper bound.
    """

    base_seconds = max(1.0, interval_to_seconds(interval))
    clamped_fraction = max(0.0, float(fraction))
    candidate = base_seconds * clamped_fraction
    if max_seconds > 0:
        candidate = min(candidate, max_seconds)
    return max(min_seconds, candidate)
