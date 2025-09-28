"""VWAP utilities for daily and session calculations."""
from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, date, time as dtime, timedelta, timezone
from typing import (
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
)

import httpx

from ..meta import Meta
from .ohlc import TIMEFRAME_TO_MS
BINANCE_SPOT_REST = "https://api.binance.com/api/v3/klines"
BINANCE_FAPI_REST = "https://fapi.binance.com/fapi/v1/klines"

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


@dataclass(slots=True)
class VWAPStats:
    """Holds VWAP aggregates along with sigma channels."""

    value: float
    sigma: float

    def as_sigma_payload(self, *, basis: str) -> Dict[str, object]:
        levels = _build_sigma_levels(self.value, self.sigma)
        return {"basis": basis, "sigma": levels}


def _build_sigma_levels(center: float, sigma: float) -> List[Dict[str, float]]:
    return [
        {"k": k, "price_minus": center - sigma * k, "price_plus": center + sigma * k}
        for k in (1, 2)
    ]


def _compute_vwap_stats(bars: Iterable[MinuteBar]) -> Optional[VWAPStats]:
    total_pv = 0.0
    total_p2v = 0.0
    total_volume = 0.0
    valid = 0
    for bar in bars:
        volume = float(bar.volume)
        if volume <= 0.0:
            continue
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        if not math.isfinite(typical_price):
            continue
        total_pv += typical_price * volume
        total_p2v += typical_price * typical_price * volume
        total_volume += volume
        valid += 1
    if total_volume <= 0.0:
        return None
    value = total_pv / total_volume
    if valid < 2:
        sigma = 0.0
    else:
        variance = max(total_p2v / total_volume - value * value, 0.0)
        sigma = math.sqrt(variance)
    return VWAPStats(value=value, sigma=sigma)


def _compute_vwap(bars: Iterable[MinuteBar]) -> float:
    stats = _compute_vwap_stats(bars)
    return stats.value if stats is not None else 0.0



async def fetch_daily_vwap(
    symbol: str,
    target_date: Optional[date] = None,
    tz_offset_minutes: int = 0,
    session: str = "UTC",
    *,
    now_ms: Optional[int] = None,
    client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
) -> Dict[str, object]:
    """Fetch deterministic daily VWAP for the latest closed minute candle."""

    now_utc_ms = int(time.time() * 1000) if now_ms is None else int(now_ms)
    now_dt = datetime.fromtimestamp(now_utc_ms / 1000.0, tz=timezone.utc)
    tz_delta = timedelta(minutes=tz_offset_minutes)

    if target_date is None:
        local_dt = now_dt + tz_delta
        target_date = local_dt.date()

    base_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    if session.upper() == "UTC":
        day_start_ms = int(base_start.timestamp() * 1000)
    else:
        day_start_ms = int((base_start - tz_delta).timestamp() * 1000)
    day_end_ms = day_start_ms + 24 * 60 * 60 * 1000
    end_ms = min(day_end_ms, now_utc_ms)

    params = {
        "symbol": symbol.upper(),
        "interval": "1m",
        "startTime": day_start_ms,
        "endTime": end_ms,
        "limit": 1000,
    }

    batches: List[List[object]] = []
    factory = client_factory or (lambda: httpx.AsyncClient(timeout=10.0))
    async with factory() as client:
        while True:
            response = await client.get(BINANCE_SPOT_REST, params=params)
            response.raise_for_status()
            batch = response.json()
            if not isinstance(batch, list) or not batch:
                break
            batches.extend(batch)
            last_close = int(batch[-1][6])
            if last_close >= end_ms or len(batch) < params["limit"]:
                break
            params["startTime"] = last_close + 1

    closed: List[List[object]] = []
    for row in batches:
        try:
            close_time = int(row[6])
        except (IndexError, TypeError, ValueError):
            continue
        if close_time > now_utc_ms:
            continue
        closed.append(row)
    closed.sort(key=lambda item: int(item[0]))

    if not closed:
        raise ValueError("no data")

    last_closed = closed[-1]
    cum_tpv = 0.0
    cum_tp2v = 0.0
    cum_volume = 0.0
    candles_used = 0
    eps = 1e-12

    for row in closed:
        try:
            high = float(row[2])
            low = float(row[3])
            close_price = float(row[4])
            volume = float(row[5])
        except (IndexError, TypeError, ValueError):
            continue
        if volume <= 0.0:
            continue
        tp = (high + low + close_price) / 3.0
        if not math.isfinite(tp):
            continue
        cum_tpv += tp * volume
        cum_tp2v += tp * tp * volume
        cum_volume += volume
        candles_used += 1

    if cum_volume <= eps:
        raise ValueError("no data")

    vwap_value = cum_tpv / max(cum_volume, eps)
    if candles_used < 2:
        sigma_value = 0.0
    else:
        variance = max(cum_tp2v / cum_volume - vwap_value * vwap_value, 0.0)
        sigma_value = math.sqrt(variance)
    last_close_iso = datetime.fromtimestamp(int(last_closed[6]) / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    return {
        "symbol": symbol.upper(),
        "vwap_at_last_closed": vwap_value,
        "last_closed_candle_time": last_close_iso,
        "cum_volume": cum_volume,
        "candles_used": candles_used,
        "vwap_sigma": VWAPStats(value=vwap_value, sigma=sigma_value).as_sigma_payload(basis="daily"),
    }


def fetch_daily_vwap_sync(**kwargs: object) -> Dict[str, object]:
    """Synchronous helper around :func:`fetch_daily_vwap`."""

    return asyncio.run(fetch_daily_vwap(**kwargs))

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
        return {"symbol": symbol.upper(), "vwap": [], "vwap_sigma": []}

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
    sigma_results: List[Dict[str, object]] = []

    for date_key in ordered_dates:
        daily_stats = _compute_vwap_stats(daily_buckets[date_key])
        if daily_stats is None:
            daily_value = 0.0
            daily_sigma = 0.0
        else:
            daily_value = daily_stats.value
            daily_sigma = daily_stats.sigma
        results.append({"date": date_key, "session": "daily", "value": daily_value})
        if daily_stats is None:
            sigma_payload = {
                "basis": "daily",
                "sigma": _build_sigma_levels(daily_value, daily_sigma),
            }
        else:
            sigma_payload = daily_stats.as_sigma_payload(basis="daily")
        sigma_payload.update({"date": date_key, "session": "daily"})
        sigma_results.append(sigma_payload)
        for session_name, _, _ in sessions:
            bars_in_session = session_buckets.get((date_key, session_name), [])
            if bars_in_session:
                stats = _compute_vwap_stats(bars_in_session)
                if stats is None:
                    value = 0.0
                    sigma_value = 0.0
                else:
                    value = stats.value
                    sigma_value = stats.sigma
            else:
                value = 0.0
                sigma_value = 0.0
            results.append({"date": date_key, "session": session_name, "value": value})
            sigma_results.append(
                {
                    "date": date_key,
                    "session": session_name,
                    "basis": "session",
                    "sigma": _build_sigma_levels(value, sigma_value),
                }
            )

    return {"symbol": symbol.upper(), "vwap": results, "vwap_sigma": sigma_results}


def fetch_session_vwap_sync(symbol: str) -> Dict[str, object]:
    """Synchronous helper for VWAP calculations."""

    return asyncio.run(fetch_session_vwap(symbol))
