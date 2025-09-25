"""VWAP utilities for daily and session calculations."""
from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import httpx

from ..meta import Meta
from .ohlc import BINANCE_FAPI_REST, TIMEFRAME_TO_MS

VWAP_INTERVAL = "1m"
INTERVAL_MS = TIMEFRAME_TO_MS[VWAP_INTERVAL]


@dataclass(slots=True)
class MinuteBar:
    """Simplified kline representation for VWAP computation."""

    open_ms: int
    high: float
    low: float
    close: float
    volume: float


def _align_to_interval(timestamp_ms: int, interval_ms: int = INTERVAL_MS) -> int:
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return (timestamp_ms // interval_ms) * interval_ms


def _normalise_row(row: Sequence[object]) -> MinuteBar | None:
    try:
        open_ms = int(row[0])
        high = float(row[2])
        low = float(row[3])
        close = float(row[4])
        volume = float(row[5])
    except (IndexError, TypeError, ValueError):
        return None
    return MinuteBar(open_ms=open_ms, high=high, low=low, close=close, volume=volume)


async def _fetch_minute_bars(symbol: str, start_ms: int, end_ms: int) -> List[MinuteBar]:
    """Download minute bars between start and end timestamps."""

    if start_ms >= end_ms:
        return []

    bars: Dict[int, MinuteBar] = {}
    cursor = start_ms
    limit = 1000

    async with httpx.AsyncClient(timeout=15.0) as client:
        while cursor < end_ms:
            params = {
                "symbol": symbol.upper(),
                "interval": VWAP_INTERVAL,
                "startTime": str(cursor),
                "endTime": str(end_ms),
                "limit": str(limit),
            }
            response = await client.get(BINANCE_FAPI_REST, params=params)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list):  # pragma: no cover
                break
            if not data:
                break

            last_open = None
            for row in data:
                bar = _normalise_row(row)
                if bar is None:
                    continue
                if bar.open_ms < start_ms or bar.open_ms >= end_ms:
                    continue
                bars[bar.open_ms] = bar
                last_open = bar.open_ms

            if last_open is None:
                break
            cursor = max(last_open + INTERVAL_MS, cursor + INTERVAL_MS)
            if len(data) < limit:
                break

    ordered_times = sorted(bars)
    return [bars[ts] for ts in ordered_times]


def _in_session(moment: dtime, start: dtime, end: dtime) -> bool:
    if start <= end:
        return start <= moment < end
    return moment >= start or moment < end


def _compute_vwap(bars: Iterable[MinuteBar]) -> float:
    total_pv = 0.0
    total_volume = 0.0
    for bar in bars:
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        total_pv += typical_price * bar.volume
        total_volume += bar.volume
    if total_volume <= 0:
        return 0.0
    return total_pv / total_volume


async def fetch_session_vwap(symbol: str) -> Dict[str, object]:
    """Return VWAP metrics for the last Meta-configured number of days."""

    lookback_days = Meta.VWAP_LOOKBACK_DAYS
    now = datetime.now(timezone.utc)
    last_closed_open_ms = _align_to_interval(int(now.timestamp() * 1000))
    end_ms = last_closed_open_ms + INTERVAL_MS
    start_date = (now - timedelta(days=lookback_days - 1)).date()
    start_dt = datetime.combine(start_date, dtime.min, tzinfo=timezone.utc)
    start_ms = _align_to_interval(int(start_dt.timestamp() * 1000))

    bars = await _fetch_minute_bars(symbol, start_ms, end_ms)
    if not bars:
        return {"symbol": symbol.upper(), "vwap": []}

    sessions = list(Meta.iter_vwap_sessions())
    daily_buckets: DefaultDict[str, List[MinuteBar]] = defaultdict(list)
    session_buckets: DefaultDict[Tuple[str, str], List[MinuteBar]] = defaultdict(list)

    for bar in bars:
        dt = datetime.fromtimestamp(bar.open_ms / 1000.0, tz=timezone.utc)
        date_key = dt.date().isoformat()
        if dt.date() < start_date:
            continue
        daily_buckets[date_key].append(bar)
        moment = dt.time()
        for session_name, start_time, end_time in sessions:
            if _in_session(moment, start_time, end_time):
                session_buckets[(date_key, session_name)].append(bar)

    ordered_dates = sorted(daily_buckets.keys())[-lookback_days:]
    results: List[Dict[str, object]] = []

    for date_key in ordered_dates:
        daily_value = _compute_vwap(daily_buckets[date_key])
        results.append({"date": date_key, "session": "daily", "value": daily_value})
        for session_name, _, _ in sessions:
            bars_in_session = session_buckets.get((date_key, session_name), [])
            value = _compute_vwap(bars_in_session) if bars_in_session else 0.0
            results.append({"date": date_key, "session": session_name, "value": value})

    return {"symbol": symbol.upper(), "vwap": results}


def fetch_session_vwap_sync(symbol: str) -> Dict[str, object]:
    """Synchronous helper for VWAP calculations."""

    return asyncio.run(fetch_session_vwap(symbol))
