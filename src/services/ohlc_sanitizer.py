"""Helpers for sanitising OHLCV candle payloads."""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any, Iterable, Mapping, MutableMapping

from .timeutils import ensure_ms_epoch


LOGGER = logging.getLogger(__name__)


_TS_KEYS = ("t", "time", "openTime")
_OPEN_KEYS = ("o", "open")
_HIGH_KEYS = ("h", "high")
_LOW_KEYS = ("l", "low")
_CLOSE_KEYS = ("c", "close")


@dataclass(slots=True)
class SanitizedCandles:
    """Result of sanitising a collection of OHLC candles."""

    candles: list[dict[str, Any]]
    invalid_ts: int
    invalid_ohlc: int
    earliest_ms: int | None
    latest_ms: int | None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return float(numeric)


def _pick_first(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def sanitize_candles(
    candles: Iterable[Mapping[str, Any]] | Iterable[MutableMapping[str, Any]] | None,
    *,
    stage: str,
) -> SanitizedCandles:
    """Normalise timestamps and filter invalid OHLC records."""

    sanitized: list[dict[str, Any]] = []
    invalid_ts = 0
    invalid_ohlc = 0
    input_count = 0
    earliest_ms: int | None = None
    latest_ms: int | None = None

    if candles is None:
        return SanitizedCandles([], 0, 0, None, None)

    for raw in candles:
        input_count += 1
        if not isinstance(raw, Mapping):
            continue

        ts_value = _pick_first(raw, _TS_KEYS)
        timestamp_ms = ensure_ms_epoch(ts_value)
        if timestamp_ms is None:
            invalid_ts += 1
            LOGGER.warning(
                "Invalid candle timestamp discarded", extra={"stage": stage, "ts": ts_value}
            )
            continue

        open_value = _coerce_float(_pick_first(raw, _OPEN_KEYS))
        high_value = _coerce_float(_pick_first(raw, _HIGH_KEYS))
        low_value = _coerce_float(_pick_first(raw, _LOW_KEYS))
        close_value = _coerce_float(_pick_first(raw, _CLOSE_KEYS))

        if (
            open_value is None
            or high_value is None
            or low_value is None
            or close_value is None
            or low_value > high_value
            or low_value > open_value
            or low_value > close_value
            or high_value < open_value
            or high_value < close_value
        ):
            invalid_ohlc += 1
            LOGGER.warning(
                "Invalid candle range discarded",
                extra={
                    "stage": stage,
                    "ts": timestamp_ms,
                    "o": open_value,
                    "h": high_value,
                    "l": low_value,
                    "c": close_value,
                },
            )
            continue

        sanitized_candle = dict(raw)
        sanitized_candle["t"] = timestamp_ms
        for key in _TS_KEYS:
            if key in sanitized_candle:
                sanitized_candle[key] = timestamp_ms

        if any(key in sanitized_candle for key in _OPEN_KEYS):
            for key in _OPEN_KEYS:
                if key in sanitized_candle:
                    sanitized_candle[key] = open_value
        else:
            sanitized_candle["o"] = open_value

        if any(key in sanitized_candle for key in _HIGH_KEYS):
            for key in _HIGH_KEYS:
                if key in sanitized_candle:
                    sanitized_candle[key] = high_value
        else:
            sanitized_candle["h"] = high_value

        if any(key in sanitized_candle for key in _LOW_KEYS):
            for key in _LOW_KEYS:
                if key in sanitized_candle:
                    sanitized_candle[key] = low_value
        else:
            sanitized_candle["l"] = low_value

        if any(key in sanitized_candle for key in _CLOSE_KEYS):
            for key in _CLOSE_KEYS:
                if key in sanitized_candle:
                    sanitized_candle[key] = close_value
        else:
            sanitized_candle["c"] = close_value

        sanitized.append(sanitized_candle)

        earliest_ms = timestamp_ms if earliest_ms is None else min(earliest_ms, timestamp_ms)
        latest_ms = timestamp_ms if latest_ms is None else max(latest_ms, timestamp_ms)

    if sanitized:
        sanitized.sort(key=lambda candle: candle.get("t", 0))

    LOGGER.info(
        "Sanitised candles",
        extra={
            "stage": stage,
            "input_count": input_count,
            "output_count": len(sanitized),
            "invalid_ts": invalid_ts,
            "invalid_ohlc": invalid_ohlc,
            "earliest_ms": earliest_ms,
            "latest_ms": latest_ms,
        },
    )

    return SanitizedCandles(sanitized, invalid_ts, invalid_ohlc, earliest_ms, latest_ms)

