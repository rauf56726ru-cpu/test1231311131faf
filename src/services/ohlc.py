"""Utilities for normalising and fetching OHLCV data."""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
)

import httpx

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

MINUTE_INTERVAL_MS = TIMEFRAME_TO_MS["1m"]
MS_IN_DAY = 86_400_000

TARGET_TIMEFRAMES: Tuple[str, ...] = ("1m", "3m", "5m", "15m", "1h", "4h", "1d")

DEFAULT_WINDOW_MS: Dict[str, int] = {
    "1m": 4 * 3_600_000,
    "3m": 4 * 3_600_000,
    "5m": 4 * 3_600_000,
    "15m": 3 * MS_IN_DAY,
    "1h": 3 * MS_IN_DAY,
    "4h": 3 * MS_IN_DAY,
    "1d": 3 * MS_IN_DAY,
}

MINIMUM_BARS: Dict[str, int] = {
    "1m": 4 * 60,  # 4 hours of minute candles
    "3m": 4 * 60 // 3,
    "5m": 4 * 60 // 5,
    "15m": (3 * 24 * 60) // 15,
    "1h": (3 * 24 * 60) // 60,
    "4h": (3 * 24 * 60) // 240,
    "1d": 3,
}


class OhlcvValidationError(RuntimeError):
    """Raised when required OHLCV timeframes cannot be produced."""

    def __init__(self, message: str, *, detail: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.detail = dict(detail or {})


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


def resample_ohlcv(
    candles: Sequence[Mapping[str, Any]],
    interval_ms: int,
) -> List[Dict[str, float | int]]:
    """Aggregate minute candles into a higher timeframe bucket."""

    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")

    buckets: Dict[int, Dict[str, float]] = {}
    for candle in candles:
        if not isinstance(candle, Mapping):
            continue
        timestamp = candle.get("t")
        open_price = candle.get("o")
        high_price = candle.get("h")
        low_price = candle.get("l")
        close_price = candle.get("c")
        volume_value = candle.get("v", 0.0)

        if not isinstance(timestamp, (int, float)):
            continue

        try:
            o_value = float(open_price)
            h_value = float(high_price)
            l_value = float(low_price)
            c_value = float(close_price)
            v_value = float(volume_value) if volume_value is not None else 0.0
        except (TypeError, ValueError):
            continue

        if not math.isfinite(o_value) or not math.isfinite(h_value) or not math.isfinite(l_value) or not math.isfinite(c_value):
            continue

        bucket_start = _align_to_interval(int(timestamp), interval_ms)
        bucket = buckets.get(bucket_start)
        if bucket is None:
            bucket = {
                "t": int(bucket_start),
                "o": o_value,
                "h": h_value,
                "l": l_value,
                "c": c_value,
                "v": v_value,
            }
            buckets[bucket_start] = bucket
        else:
            bucket["h"] = max(bucket["h"], h_value)
            bucket["l"] = min(bucket["l"], l_value)
            bucket["c"] = c_value
            bucket["v"] += v_value

    return [buckets[key] for key in sorted(buckets)]


def _safe_int(value: object | None) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object | None) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalise_series(
    candles: Sequence[Mapping[str, Any]] | None,
) -> List[Dict[str, float]]:
    if not candles:
        return []
    normalised: Dict[int, Dict[str, float]] = {}
    for candle in candles:
        if not isinstance(candle, Mapping):
            continue
        ts = _safe_int(candle.get("t"))
        if ts is None:
            continue
        open_price = _safe_float(candle.get("o"))
        high_price = _safe_float(candle.get("h"))
        low_price = _safe_float(candle.get("l"))
        close_price = _safe_float(candle.get("c"))
        volume = _safe_float(candle.get("v"))
        if None in (open_price, high_price, low_price, close_price):
            continue
        normalised[ts] = {
            "t": ts,
            "o": float(open_price),
            "h": float(high_price),
            "l": float(low_price),
            "c": float(close_price),
            "v": float(volume or 0.0),
        }
    return [normalised[key] for key in sorted(normalised)]


def _validate_series(
    series: Sequence[Mapping[str, Any]],
    interval_ms: int,
) -> Tuple[bool, Dict[str, Any]]:
    """Validate that a series is aligned and monotonic for the given interval."""

    diagnostics: Dict[str, Any] = {
        "interval_ms": interval_ms,
        "count": 0,
        "first_ts": None,
        "last_ts": None,
        "issues": [],
    }

    if not series:
        diagnostics["issues"].append("empty")
        return False, diagnostics

    last_ts: int | None = None
    seen: set[int] = set()
    for candle in series:
        ts = _safe_int(candle.get("t"))
        o_val = _safe_float(candle.get("o"))
        h_val = _safe_float(candle.get("h"))
        l_val = _safe_float(candle.get("l"))
        c_val = _safe_float(candle.get("c"))
        v_val = _safe_float(candle.get("v"))
        if None in (ts, o_val, h_val, l_val, c_val):
            diagnostics["issues"].append({"ts": ts, "reason": "non_numeric"})
            return False, diagnostics
        if ts in seen:
            diagnostics["issues"].append({"ts": ts, "reason": "duplicate"})
            return False, diagnostics
        seen.add(ts)
        if ts % interval_ms != 0:
            diagnostics["issues"].append({"ts": ts, "reason": "misaligned"})
            return False, diagnostics
        if last_ts is not None and ts <= last_ts:
            diagnostics["issues"].append({"ts": ts, "reason": "non_monotonic"})
            return False, diagnostics
        if last_ts is not None and ts - last_ts != interval_ms:
            diagnostics["issues"].append({"ts": ts, "reason": "gap", "delta": ts - last_ts})
            return False, diagnostics
        last_ts = ts
    diagnostics["count"] = len(series)
    diagnostics["first_ts"] = series[0]["t"]
    diagnostics["last_ts"] = series[-1]["t"]
    return True, diagnostics


def _aggregate_bucket(
    minute_index: Mapping[int, Mapping[str, Any]],
    start_ms: int,
    interval_ms: int,
) -> Dict[str, float]:
    end_ms = start_ms + interval_ms - MINUTE_INTERVAL_MS
    cursor = start_ms
    high = float("-inf")
    low = float("inf")
    volume_sum = 0.0
    open_price: float | None = None
    close_price: float | None = None

    while cursor <= end_ms:
        candle = minute_index.get(cursor)
        if candle is None:
            raise OhlcvValidationError(
                "Missing 1m candle required for aggregation",
                detail={"missing_ts": cursor, "start_ms": start_ms, "interval_ms": interval_ms},
            )
        o_val = _safe_float(candle.get("o"))
        h_val = _safe_float(candle.get("h"))
        l_val = _safe_float(candle.get("l"))
        c_val = _safe_float(candle.get("c"))
        v_val = _safe_float(candle.get("v")) or 0.0
        if None in (o_val, h_val, l_val, c_val):
            raise OhlcvValidationError(
                "Non numeric 1m candle encountered during aggregation",
                detail={"ts": cursor},
            )
        if open_price is None:
            open_price = float(o_val)
        high = max(high, float(h_val))
        low = min(low, float(l_val))
        close_price = float(c_val)
        volume_sum += float(v_val)
        cursor += MINUTE_INTERVAL_MS

    if open_price is None or close_price is None:
        raise OhlcvValidationError(
            "Aggregation failed due to incomplete bucket",
            detail={"start_ms": start_ms, "interval_ms": interval_ms},
        )

    return {
        "t": start_ms,
        "o": open_price,
        "h": high,
        "l": low,
        "c": close_price,
        "v": volume_sum,
    }


def ensure_complete_ohlcv(
    minute_candles: Sequence[Mapping[str, Any]],
    *,
    selection_start: int | None = None,
    selection_end: int | None = None,
    existing_frames: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    logger: logging.Logger | None = None,
) -> Tuple[Dict[str, List[Dict[str, float]]], Dict[str, Any]]:
    """Build a complete OHLCV matrix for all target timeframes."""

    if logger is None:
        logger = logging.getLogger(__name__)

    minute_series = _normalise_series(minute_candles)
    if not minute_series:
        raise OhlcvValidationError("Minute candles are required to build OHLCV")

    valid, minute_diag = _validate_series(minute_series, MINUTE_INTERVAL_MS)
    if not valid:
        raise OhlcvValidationError(
            "Minute series failed validation",
            detail={"series": "1m", "diagnostics": minute_diag},
        )

    minute_index = {candle["t"]: candle for candle in minute_series}
    first_minute = minute_series[0]["t"]
    last_minute = minute_series[-1]["t"]

    if selection_start is not None and selection_end is not None and selection_start > selection_end:
        selection_start, selection_end = selection_end, selection_start

    effective_end = last_minute
    if selection_end is not None:
        selection_end_aligned = _align_to_interval(selection_end, MINUTE_INTERVAL_MS)
        effective_end = min(effective_end, selection_end_aligned)
        if effective_end < first_minute:
            raise OhlcvValidationError(
                "Selection end precedes available minute data",
                detail={"selection_end": selection_end, "first_minute": first_minute},
            )

    diagnostics: Dict[str, Any] = {}
    ohlcv_bundle: Dict[str, List[Dict[str, float]]] = {}
    existing_map = existing_frames or {}
    selection_start_aligned = (
        _align_to_interval(selection_start, MINUTE_INTERVAL_MS)
        if selection_start is not None
        else None
    )

    for tf in TARGET_TIMEFRAMES:
        interval_ms = TIMEFRAME_TO_MS[tf]
        default_window = DEFAULT_WINDOW_MS[tf]
        min_bars = max(1, MINIMUM_BARS.get(tf, 1))

        last_start = _align_to_interval(effective_end, interval_ms)
        while last_start + interval_ms - MINUTE_INTERVAL_MS > effective_end:
            last_start -= interval_ms
        if last_start < first_minute:
            raise OhlcvValidationError(
                "Insufficient minute data to build timeframe",
                detail={"timeframe": tf, "required_start": last_start, "available_start": first_minute},
            )

        selection_bars = 0
        if selection_start_aligned is not None:
            if selection_start_aligned > last_start:
                selection_bars = 1
            else:
                selection_bars = ((last_start - selection_start_aligned) // interval_ms) + 1

        default_bars = max(min_bars, max(1, default_window // interval_ms))
        required_bars = max(default_bars, selection_bars, min_bars)
        start_candidate = last_start - (required_bars - 1) * interval_ms
        start_aligned = _align_to_interval(start_candidate, interval_ms)

        first_boundary = _align_to_interval(first_minute, interval_ms)
        while first_boundary < first_minute:
            first_boundary += interval_ms

        if start_aligned < first_boundary:
            detail = {
                "timeframe": tf,
                "required_start": start_aligned,
                "first_available": first_boundary,
                "required_bars": required_bars,
            }
            raise OhlcvValidationError("Not enough 1m data for timeframe window", detail=detail)

        bars: List[Dict[str, float]] = []
        cursor = start_aligned
        while cursor <= last_start:
            bucket = _aggregate_bucket(minute_index, cursor, interval_ms)
            bars.append(bucket)
            cursor += interval_ms

        valid_tf, tf_diag = _validate_series(bars, interval_ms)
        if not valid_tf:
            raise OhlcvValidationError(
                "Generated timeframe failed validation",
                detail={"timeframe": tf, "diagnostics": tf_diag},
            )

        existing_series = _normalise_series(existing_map.get(tf)) if existing_map else []
        status = "rebuilt"
        if existing_series and len(existing_series) == len(bars):
            mismatch = False
            for lhs, rhs in zip(existing_series, bars):
                if lhs["t"] != rhs["t"]:
                    mismatch = True
                    break
                if not math.isclose(lhs["o"], rhs["o"], rel_tol=1e-6, abs_tol=1e-9):
                    mismatch = True
                    break
                if not math.isclose(lhs["h"], rhs["h"], rel_tol=1e-6, abs_tol=1e-9):
                    mismatch = True
                    break
                if not math.isclose(lhs["l"], rhs["l"], rel_tol=1e-6, abs_tol=1e-9):
                    mismatch = True
                    break
                if not math.isclose(lhs["c"], rhs["c"], rel_tol=1e-6, abs_tol=1e-9):
                    mismatch = True
                    break
                if not math.isclose(lhs["v"], rhs["v"], rel_tol=1e-6, abs_tol=1e-9):
                    mismatch = True
                    break
            if not mismatch:
                status = "existing"

        diagnostics[tf] = {
            "status": status,
            "bars": len(bars),
            "expected_bars": required_bars,
            "start_ms": bars[0]["t"] if bars else None,
            "end_ms": bars[-1]["t"] if bars else None,
            "interval_ms": interval_ms,
        }

        ohlcv_bundle[tf] = bars

    missing = [tf for tf, item in diagnostics.items() if item.get("status") != "existing"]
    if missing:
        logger.debug(
            "OHLCV frames rebuilt from 1m",
            extra={"timeframes": missing, "diagnostics": diagnostics},
        )
    else:
        logger.debug("OHLCV frames validated without rebuild", extra={"diagnostics": diagnostics})

    return ohlcv_bundle, diagnostics


def aggregate_1m_to_1h(
    candles: Sequence[Mapping[str, Any] | Sequence[object] | Candle],
) -> List[Dict[str, float | int]]:
    """Aggregate one-minute OHLCV candles into one-hour buckets.

    The function aligns candle timestamps to UTC hour boundaries and produces
    standard OHLCV fields for each one-hour bar. Invalid rows are ignored.
    """

    hour_ms = TIMEFRAME_TO_MS.get("1h")
    if not hour_ms:
        raise ValueError("1h timeframe is not defined in TIMEFRAME_TO_MS")

    parsed: List[Candle] = []
    for row in candles:
        candle: Candle | None
        if isinstance(row, Candle):
            candle = row
        elif isinstance(row, Mapping) or isinstance(row, Sequence):
            candle = _to_candle_mapping(row)  # type: ignore[arg-type]
        else:
            candle = None
        if candle is None:
            continue
        parsed.append(candle)

    if not parsed:
        return []

    parsed.sort(key=lambda item: item.t)

    buckets: Dict[int, Dict[str, float]] = {}
    for candle in parsed:
        bucket_start = _align_to_interval(int(candle.t), hour_ms)
        bucket = buckets.get(bucket_start)
        if bucket is None:
            buckets[bucket_start] = {
                "t": float(bucket_start),
                "o": float(candle.o),
                "h": float(candle.h),
                "l": float(candle.l),
                "c": float(candle.c),
                "v": float(candle.v),
            }
            continue

        bucket["h"] = max(bucket["h"], float(candle.h))
        bucket["l"] = min(bucket["l"], float(candle.l))
        bucket["c"] = float(candle.c)
        bucket["v"] += float(candle.v)

    ordered: List[Dict[str, float | int]] = []
    for ts in sorted(buckets):
        bucket = buckets[ts]
        ordered.append(
            {
                "t": int(ts),
                "o": float(bucket["o"]),
                "h": float(bucket["h"]),
                "l": float(bucket["l"]),
                "c": float(bucket["c"]),
                "v": float(bucket["v"]),
            }
        )

    return ordered


def _ensure_cache_entry(symbol: str, timeframe: str) -> CandleCache:
    key = (symbol.upper(), timeframe)
    entry = _CANDLE_CACHE.get(key)
    interval_ms = TIMEFRAME_TO_MS[timeframe]
    if entry is None or entry.interval_ms != interval_ms:
        entry = CandleCache(interval_ms=interval_ms)
        _CANDLE_CACHE[key] = entry
    return entry


def _merge_candles(entry: CandleCache, new_candles: Sequence[Candle]) -> None:
    if not new_candles:
        return
    mapping: Dict[int, Candle] = {candle.t: candle for candle in entry.candles}
    for candle in new_candles:
        mapping[candle.t] = candle
    entry.candles = [mapping[key] for key in sorted(mapping)]


def compute_window_end_from_store_or_exchange(
    entry: CandleCache | None,
    *,
    interval_ms: int,
    exchange_latest_open: int | None = None,
) -> int:
    """Compute the exclusive end timestamp for the rolling window."""

    latest_open = exchange_latest_open
    if entry and entry.candles:
        latest_open = entry.candles[-1].t if latest_open is None else max(latest_open, entry.candles[-1].t)
    if latest_open is None:
        raise ValueError("No candles available to anchor the window")
    return latest_open + interval_ms


async def slice_or_fetch_missing(
    symbol: str,
    timeframe: str,
    *,
    start_ms: int,
    end_ms: int,
    fetcher: Callable[[str, str, int | None, int | None, int | None], Awaitable[Sequence[Mapping[str, object] | Sequence[object]]]],
    limit: int,
) -> List[Candle]:
    """Return candles covering [start_ms, end_ms) fetching only missing tails."""

    interval_ms = TIMEFRAME_TO_MS[timeframe]
    entry = _ensure_cache_entry(symbol, timeframe)
    existing: Dict[int, Candle] = {
        candle.t: candle
        for candle in entry.candles
        if start_ms <= candle.t < end_ms
    }

    cursor = start_ms
    required_times: List[int] = []
    while cursor < end_ms:
        if cursor not in existing:
            required_times.append(cursor)
            break
        cursor += interval_ms

    if required_times:
        fetch_start = required_times[0]
        raw_rows = await fetcher(symbol, timeframe, fetch_start, end_ms, limit)
        fetched = []
        for row in raw_rows:
            candle = _to_candle_mapping(row)
            if candle is None:
                continue
            if candle.t < fetch_start or candle.t >= end_ms:
                continue
            fetched.append(candle)
        _merge_candles(entry, fetched)
        existing = {
            candle.t: candle
            for candle in entry.candles
            if start_ms <= candle.t < end_ms
        }

    ordered: List[Candle] = []
    cursor = start_ms
    while cursor < end_ms:
        candle = existing.get(cursor)
        if candle is not None:
            ordered.append(candle)
        cursor += interval_ms

    return ordered


async def _fetch_binance_klines(
    symbol: str,
    timeframe: str,
    start_ms: int | None,
    end_ms: int | None,
    limit: int | None,
) -> Sequence[Sequence[object]]:
    params = {
        "symbol": symbol.upper(),
        "interval": timeframe,
    }
    if start_ms is not None:
        params["startTime"] = str(start_ms)
    if end_ms is not None:
        params["endTime"] = str(end_ms)
    if limit is not None:
        params["limit"] = str(limit)
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(BINANCE_FAPI_REST, params=params)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, Sequence):
        return []
    return data  # type: ignore[return-value]


async def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    *,
    hours: int | None = None,
    fetcher: Callable[[str, str, int | None, int | None, int | None], Awaitable[Sequence[Mapping[str, object] | Sequence[object]]]] | None = None,
) -> Dict[str, object]:
    """Fetch OHLC candles anchored to the latest available Binance bar."""

    timeframe = timeframe.lower()
    if timeframe not in TIMEFRAME_TO_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    interval_ms = TIMEFRAME_TO_MS[timeframe]
    if hours is not None and hours <= 0:
        raise ValueError("hours must be positive")

    window_ms = (
        hours * 3_600_000
        if hours is not None
        else int(TIMEFRAME_WINDOWS[timeframe].total_seconds() * 1000)
    )
    candles_required = max(1, math.ceil(window_ms / interval_ms))

    entry = _ensure_cache_entry(symbol, timeframe)
    fetcher_fn = fetcher or _fetch_binance_klines

    exchange_latest_open: int | None = None
    if not entry.candles:
        raw_rows = await fetcher_fn(symbol, timeframe, None, None, candles_required)
        fetched = []
        for row in raw_rows:
            candle = _to_candle_mapping(row)
            if candle is None:
                continue
            fetched.append(candle)
        if fetched:
            exchange_latest_open = fetched[-1].t
        _merge_candles(entry, fetched)
    else:
        tail_start = entry.candles[-1].t + interval_ms
        raw_rows = await fetcher_fn(symbol, timeframe, tail_start, None, candles_required)
        fetched = []
        for row in raw_rows:
            candle = _to_candle_mapping(row)
            if candle is None:
                continue
            if candle.t <= entry.candles[-1].t:
                continue
            fetched.append(candle)
        if fetched:
            _merge_candles(entry, fetched)
            exchange_latest_open = fetched[-1].t
        else:
            exchange_latest_open = entry.candles[-1].t

    if exchange_latest_open is None and entry.candles:
        exchange_latest_open = entry.candles[-1].t

    if exchange_latest_open is None:
        return {"symbol": symbol.upper(), "tf": timeframe, "candles": [], "last_ts": None}

    end_ms = compute_window_end_from_store_or_exchange(
        entry,
        interval_ms=interval_ms,
        exchange_latest_open=exchange_latest_open,
    )
    start_ms = end_ms - candles_required * interval_ms

    candles = await slice_or_fetch_missing(
        symbol,
        timeframe,
        start_ms=start_ms,
        end_ms=end_ms,
        fetcher=fetcher_fn,
        limit=candles_required,
    )

    payload = {
        "symbol": symbol.upper(),
        "tf": timeframe,
        "candles": [candle.as_dict() for candle in candles],
        "last_ts": candles[-1].t if candles else None,
    }
    return payload


def fetch_ohlcv_sync(
    symbol: str,
    timeframe: str,
    *,
    hours: int | None = None,
) -> Dict[str, object]:
    """Synchronous helper for fetching OHLC candles."""

    return asyncio.run(fetch_ohlcv(symbol, timeframe, hours=hours))


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
    use_full_span: bool = False,
) -> Dict[str, object]:
    """Normalise raw OHLC candles gathered by the frontend into aligned bars."""

    timeframe = timeframe.lower()
    if timeframe not in TIMEFRAME_WINDOWS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    interval_ms = TIMEFRAME_TO_MS[timeframe]
    window = TIMEFRAME_WINDOWS[timeframe]
    candles_by_time, duplicate_times = _prepare_index(raw_rows)
    limit_default = _limit_window(interval_ms, window)
    if not candles_by_time:
        payload: Dict[str, object] = {
            "symbol": symbol.upper(),
            "tf": timeframe,
            "candles": [],
        }
        if include_diagnostics:
            payload["diagnostics"] = {
                "interval_ms": interval_ms,
                "expected_candles": limit_default,
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
    if use_full_span:
        limit = span_candles
    else:
        limit = min(limit_default, span_candles)
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
    symbol: str,
    timeframe: str,
    raw_rows: Sequence[Mapping[str, object] | Sequence[object]],
    *,
    include_diagnostics: bool = False,
    use_full_span: bool = False,
) -> Dict[str, object]:
    """Synchronous helper for normalising OHLCV snapshots."""

    return normalise_ohlcv(
        symbol,
        timeframe,
        raw_rows,
        include_diagnostics=include_diagnostics,
        use_full_span=use_full_span,
    )
BINANCE_FAPI_REST = "https://fapi.binance.com/fapi/v1/klines"


@dataclass(slots=True)
class CandleCache:
    """In-memory cache of Binance OHLC candles."""

    interval_ms: int
    candles: List["Candle"] = field(default_factory=list)


_CANDLE_CACHE: Dict[Tuple[str, str], CandleCache] = {}

