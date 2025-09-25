"""Utilities for fetching Binance OHLCV data for the chart backend."""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import httpx

BINANCE_FAPI_REST = "https://fapi.binance.com/fapi/v1/klines"

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

    def as_dict(self) -> Dict[str, float]:
        return {
            "t": self.t,
            "o": self.o,
            "h": self.h,
            "l": self.l,
            "c": self.c,
            "v": self.v,
        }


def _align_to_interval(timestamp_ms: int, interval_ms: int) -> int:
    """Floor the timestamp to the closest interval boundary."""

    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return (timestamp_ms // interval_ms) * interval_ms


def _compute_limits(timeframe: str) -> tuple[int, int, int]:
    """Return (interval_ms, limit, start_open_ms) for the timeframe."""

    interval_ms = TIMEFRAME_TO_MS[timeframe]
    window = TIMEFRAME_WINDOWS[timeframe]
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    # Use the last fully closed candle to avoid partially formed bars.
    last_open_ms = _align_to_interval(now_ms - interval_ms, interval_ms)
    candles_required = max(1, math.ceil(window.total_seconds() * 1000 / interval_ms))
    limit = min(1000, candles_required)
    start_open_ms = last_open_ms - (limit - 1) * interval_ms
    return interval_ms, limit, start_open_ms


async def _fetch_klines(
    symbol: str,
    interval: str,
    start_open_ms: int,
    limit: int,
) -> List[List[float]]:
    """Fetch klines for the requested parameters."""

    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": str(limit),
        "startTime": str(start_open_ms),
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(BINANCE_FAPI_REST, params=params)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, list):  # pragma: no cover - defensive
        raise ValueError("Unexpected Binance response")
    return data


def _normalise_row(row: List[float]) -> Candle | None:
    try:
        open_time = int(row[0])
        open_price = float(row[1])
        high_price = float(row[2])
        low_price = float(row[3])
        close_price = float(row[4])
        volume = float(row[5])
    except (IndexError, TypeError, ValueError):
        return None
    return Candle(
        t=open_time,
        o=open_price,
        h=high_price,
        l=low_price,
        c=close_price,
        v=volume,
    )


async def fetch_ohlcv(symbol: str, timeframe: str) -> Dict[str, object]:
    """Fetch OHLCV data and return the normalized JSON structure."""

    timeframe = timeframe.lower()
    if timeframe not in TIMEFRAME_WINDOWS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    interval_ms, limit, start_open_ms = _compute_limits(timeframe)
    rows = await _fetch_klines(symbol, timeframe, start_open_ms, limit)

    # Deduplicate and sort the candles by their open time.
    candles_by_time: Dict[int, Candle] = {}
    for row in rows:
        candle = _normalise_row(row)
        if candle is None:
            continue
        # Keep only candles aligned with the timeframe and >= start.
        if candle.t < start_open_ms:
            continue
        if candle.t % interval_ms != 0:
            continue
        candles_by_time[candle.t] = candle

    ordered_times = sorted(candles_by_time)
    candles = [candles_by_time[ts].as_dict() for ts in ordered_times]

    last_candle = candles_by_time[ordered_times[-1]] if ordered_times else None
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    next_close_ms = (
        last_candle.t + interval_ms if last_candle else now_ms
    )

    payload: Dict[str, object] = {
        "symbol": symbol.upper(),
        "tf": timeframe,
        "candles": candles,
    }
    if last_candle:
        payload["last_price"] = last_candle.c
        payload["last_ts"] = last_candle.t
        payload["next_close_ts"] = next_close_ms
        payload["time_to_close_ms"] = max(0, next_close_ms - now_ms)
    return payload


def fetch_ohlcv_sync(symbol: str, timeframe: str) -> Dict[str, object]:
    """Synchronous helper mostly for tests or scripts."""

    return asyncio.run(fetch_ohlcv(symbol, timeframe))
