"""Utilities for normalising OHLCV data collected by the chart frontend."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence

# Mapping of supported timeframes to their window sizes.
TIMEFRAME_WINDOWS: Dict[str, timedelta] = {
    "1m": timedelta(hours=8),
    "3m": timedelta(hours=24),
    "5m": timedelta(hours=48),
    "15m": timedelta(hours=72),
    "1h": timedelta(days=7),
    "4h": timedelta(days=30),
    "1d": timedelta(days=90),
}

# Explicit duration of a single candle in milliseconds.
TIMEFRAME_TO_MS: Dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


@dataclass(slots=True)
class Candle:
    """Normalized representation of a single OHLCV bar."""

    t: int
    o: float
    h: float
    l: float
    c: float
    v: float
    missing: bool = False

    def as_dict(self, *, include_missing: bool = False) -> Dict[str, float]:
        data: Dict[str, float | int | bool] = {
            "t": self.t,
            "o": self.o,
            "h": self.h,
            "l": self.l,
            "c": self.c,
            "v": self.v,
        }
        if include_missing:
            data["missing"] = self.missing
        return data  # type: ignore[return-value]


def _align_to_interval(timestamp_ms: int, interval_ms: int) -> int:
    """Floor the timestamp to the closest interval boundary."""

    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return (timestamp_ms // interval_ms) * interval_ms


def _to_candle_mapping(row: Mapping[str, object] | Sequence[object]) -> Candle | None:
    """Convert raw data from the frontend into an internal candle structure."""

    open_time: int | None = None
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    volume: float | None = None

    if isinstance(row, Mapping):
        # Accepted keys: time/ts/ts_ms_utc or t; open/o; high/h; low/l; close/c; volume/v.
        time_value = None
        for key in ("ts_ms_utc", "t", "time", "openTime", "open_time"):
            candidate = row.get(key)
            if candidate is not None:
                time_value = candidate
                break
        if isinstance(time_value, (int, float)):
            open_time = int(time_value)

        def _num(key: str, fallback: str | None = None) -> float | None:
            value = row.get(key)
            if value is None and fallback is not None:
                value = row.get(fallback)
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        open_price = _num("o", "open")
        high_price = _num("h", "high")
        low_price = _num("l", "low")
        close_price = _num("c", "close")
        volume = _num("v", "volume")
    else:
        try:
            open_time = int(row[0])
            open_price = float(row[1])
            high_price = float(row[2])
            low_price = float(row[3])
            close_price = float(row[4])
            volume = float(row[5]) if len(row) > 5 else 0.0
        except (IndexError, TypeError, ValueError):
            return None

    if not all(
        value is not None and math.isfinite(value)
        for value in (open_time, open_price, high_price, low_price, close_price)
    ):
        return None

    if volume is None or not math.isfinite(volume):
        volume = 0.0

    return Candle(
        t=int(open_time),
        o=float(open_price),
        h=float(high_price),
        l=float(low_price),
        c=float(close_price),
        v=float(volume),
    )


def _limit_window(interval_ms: int, window: timedelta) -> int:
    candles_required = max(1, math.ceil(window.total_seconds() * 1000 / interval_ms))
    return min(1_000, candles_required)


def _prepare_index(
    raw_rows: Iterable[Mapping[str, object] | Sequence[object]]
) -> tuple[MutableMapping[int, Candle], List[int]]:
    candles_by_time: MutableMapping[int, Candle] = {}
    duplicates: List[int] = []
    for row in raw_rows:
        candle = _to_candle_mapping(row)
        if candle is None:
            continue
        if candle.t in candles_by_time:
            duplicates.append(candle.t)
        candles_by_time[candle.t] = candle
    return candles_by_time, duplicates


def normalise_ohlcv(
    symbol: str,
    timeframe: str,
    raw_rows: Sequence[Mapping[str, object] | Sequence[object]],
    *,
    include_diagnostics: bool = False,
    window_limit: int | None = None,
) -> Dict[str, object]:
    """Normalise raw OHLC candles gathered by the frontend into aligned bars."""

    timeframe = timeframe.lower()
    if timeframe not in TIMEFRAME_WINDOWS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    interval_ms = TIMEFRAME_TO_MS[timeframe]
    window = TIMEFRAME_WINDOWS[timeframe]
    candles_by_time, duplicate_times = _prepare_index(raw_rows)
    limit_default = _limit_window(interval_ms, window)
    if window_limit is not None:
        try:
            window_limit = int(window_limit)
        except (TypeError, ValueError):
            window_limit = None
        if window_limit is not None and window_limit <= 0:
            window_limit = None
    if not candles_by_time:
        payload: Dict[str, object] = {
            "symbol": symbol.upper(),
            "tf": timeframe,
            "candles": [],
        }
        if include_diagnostics:
            payload["diagnostics"] = {
                "interval_ms": interval_ms,
                "expected_candles": window_limit or limit_default,
                "unique_candles": 0,
                "duplicates": duplicate_times,
                "missing_bars": [],
                "series": [],
            }
        return payload

    ordered_times = sorted(candles_by_time)
    last_open_ms = _align_to_interval(ordered_times[-1], interval_ms)
    first_open_ms = _align_to_interval(ordered_times[0], interval_ms)
    span_candles = max(1, (last_open_ms - first_open_ms) // interval_ms + 1)
    limit_cap = window_limit if window_limit is not None else limit_default
    limit = min(limit_cap, span_candles)
    start_open_ms = last_open_ms - (limit - 1) * interval_ms

    normalized: List[Candle] = []
    missing_records: List[Dict[str, float | int]] = []
    previous_close = None
    fallback_close = candles_by_time[ordered_times[0]].c

    for offset in range(limit):
        open_time = start_open_ms + offset * interval_ms
        candle = candles_by_time.get(open_time)
        if candle is None:
            fill_close = previous_close if previous_close is not None else fallback_close
            candle = Candle(
                t=open_time,
                o=fill_close,
                h=fill_close,
                l=fill_close,
                c=fill_close,
                v=0.0,
                missing=True,
            )
            missing_records.append({"t": open_time, "filled_with": fill_close})
        else:
            previous_close = candle.c
        normalized.append(candle)

    export_candles = [candle.as_dict() for candle in normalized]
    last_candle = normalized[-1]
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    next_close_ms = last_candle.t + interval_ms

    payload: Dict[str, object] = {
        "symbol": symbol.upper(),
        "tf": timeframe,
        "candles": export_candles,
        "last_price": last_candle.c,
        "last_ts": last_candle.t,
        "next_close_ts": next_close_ms,
        "time_to_close_ms": max(0, next_close_ms - now_ms),
    }

    if include_diagnostics:
        diagnostics = {
            "interval_ms": interval_ms,
            "expected_candles": limit,
            "unique_candles": len(candles_by_time),
            "duplicates": sorted(set(duplicate_times)),
            "missing_bars": missing_records,
            "series": [candle.as_dict(include_missing=True) for candle in normalized],
        }
        payload["diagnostics"] = diagnostics

    return payload


def normalise_ohlcv_sync(
    symbol: str, timeframe: str, raw_rows: Sequence[Mapping[str, object] | Sequence[object]]
) -> Dict[str, object]:
    """Synchronous helper for normalising OHLCV snapshots."""

    return normalise_ohlcv(symbol, timeframe, raw_rows)
