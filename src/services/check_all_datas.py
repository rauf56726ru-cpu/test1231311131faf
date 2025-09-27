"""Snapshot diagnostics builder for the inspection check-all endpoint."""
from __future__ import annotations

import logging
import time
import math
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

import httpx

from .inspection import build_htf_section
from .presets import resolve_profile_config
from .profile import build_profile_package
from .zones import Config as ZonesConfig, detect_zones
from ..meta import Meta


UTC = timezone.utc
MS_IN_HOUR = 3_600_000
MS_IN_DAY = 86_400_000
VALID_HOUR_WINDOWS = {1, 2, 3, 4}
VALUE_AREA_PCT = 0.70

try:
    from .ohlc import TIMEFRAME_TO_MS
except ImportError:  # pragma: no cover - circular import guard
    TIMEFRAME_TO_MS = {"1m": MS_IN_HOUR // 60}

MINUTE_INTERVAL_MS = TIMEFRAME_TO_MS.get("1m", MS_IN_HOUR // 60)

BINANCE_FAPI_REST = "https://fapi.binance.com/fapi/v1/klines"
_RETRYABLE_STATUS = {418, 429, 500, 502, 503, 504}
_MAX_RETRIES = 5


class DataQualityError(RuntimeError):
    """Raised when the inspected snapshot fails deterministic data checks."""

    def __init__(self, detail: Mapping[str, Any]):
        super().__init__("Market data continuity validation failed")
        self.detail = dict(detail)


class BinanceDownloadError(RuntimeError):
    """Raised when Binance minute candles could not be fetched fully."""

    def __init__(self, downloaded: int, message: str):
        super().__init__(message)
        self.downloaded = int(downloaded)


def _align_to_interval(value: int, interval_ms: int) -> int:
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return (value // interval_ms) * interval_ms


def _deduplicate_sorted(
    candles: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return candles sorted by timestamp with the last occurrence kept."""

    seen: Dict[int, Dict[str, Any]] = {}
    for candle in candles:
        ts = _safe_int(candle.get("t"))
        if ts is None:
            continue
        seen[ts] = {
            "t": ts,
            "o": _coerce_float(candle.get("o")),
            "h": _coerce_float(candle.get("h")),
            "l": _coerce_float(candle.get("l")),
            "c": _coerce_float(candle.get("c")),
            "v": _coerce_float(candle.get("v")),
        }

    ordered_times = sorted(seen)
    return [seen[ts] for ts in ordered_times]


def _build_expected_times(start_ms: int, end_ms: int, interval_ms: int) -> List[int]:
    if end_ms < start_ms:
        return []
    steps = ((end_ms - start_ms) // interval_ms) + 1
    return [start_ms + index * interval_ms for index in range(steps)]


def _summarise_missing_times(
    expected: Sequence[int],
    available: Mapping[int, Mapping[str, Any]],
) -> List[Dict[str, int]]:
    gaps: List[Dict[str, int]] = []
    current_start: int | None = None
    current_count = 0

    for ts in expected:
        if ts not in available:
            if current_start is None:
                current_start = ts
                current_count = 1
            else:
                current_count += 1
        elif current_start is not None:
            gaps.append({"from": current_start, "to": ts - MINUTE_INTERVAL_MS, "count": current_count})
            current_start = None
            current_count = 0

    if current_start is not None:
        last_missing_ts = expected[-1]
        gaps.append({"from": current_start, "to": last_missing_ts, "count": current_count})

    return gaps


def _normalise_binance_row(row: Sequence[object]) -> Dict[str, Any] | None:
    try:
        open_time = int(row[0])
        open_price = float(row[1])
        high_price = float(row[2])
        low_price = float(row[3])
        close_price = float(row[4])
        volume = float(row[5])
    except (IndexError, TypeError, ValueError):
        return None
    return {
        "t": open_time,
        "o": open_price,
        "h": high_price,
        "l": low_price,
        "c": close_price,
        "v": volume,
    }


def _request_binance_minutes(
    client: httpx.Client,
    symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int,
) -> List[Sequence[object]]:
    params = {
        "symbol": symbol.upper(),
        "interval": "1m",
        "startTime": str(start_ms),
        "endTime": str(end_ms),
        "limit": str(limit),
    }

    delay = 0.5
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.get(BINANCE_FAPI_REST, params=params)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                return data  # type: ignore[return-value]
            return []
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
        except httpx.RequestError:
            if attempt < _MAX_RETRIES - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    return []


def _download_missing_minutes(
    symbol: str,
    start_ms: int,
    end_ms: int,
    gaps: Sequence[Mapping[str, int]],
) -> List[Dict[str, Any]]:
    if not gaps:
        return []

    fetched: List[Dict[str, Any]] = []
    downloaded = 0

    try:
        with httpx.Client(timeout=15.0) as client:
            for gap in gaps:
                gap_start = int(gap["from"])
                gap_end = int(gap["to"])
                cursor = gap_start
                while cursor <= gap_end:
                    chunk_end = min(
                        gap_end,
                        cursor + (1000 - 1) * MINUTE_INTERVAL_MS,
                    )
                    request_end = chunk_end + MINUTE_INTERVAL_MS
                    raw_rows = _request_binance_minutes(
                        client,
                        symbol,
                        cursor,
                        request_end,
                        limit=1000,
                    )
                    if not raw_rows:
                        break

                    last_open = None
                    for row in raw_rows:
                        candle = _normalise_binance_row(row)
                        if candle is None:
                            continue
                        ts = candle["t"]
                        if ts < start_ms or ts > end_ms:
                            continue
                        fetched.append(candle)
                        downloaded += 1
                        last_open = ts

                    if last_open is None:
                        break
                    cursor = last_open + MINUTE_INTERVAL_MS
                    if cursor > gap_end:
                        break
    except (httpx.HTTPError, httpx.TransportError) as exc:  # pragma: no cover - defensive
        raise BinanceDownloadError(downloaded, str(exc)) from exc

    return fetched


def _aggregate_from_minutes(
    minute_index: Mapping[int, Mapping[str, Any]],
    open_time: int,
    interval_ms: int,
) -> Dict[str, Any] | None:
    end_exclusive = open_time + interval_ms
    cursor = open_time
    bucket: List[Mapping[str, Any]] = []

    while cursor < end_exclusive:
        candle = minute_index.get(cursor)
        if candle is None:
            return None
        bucket.append(candle)
        cursor += MINUTE_INTERVAL_MS

    if not bucket:
        return None

    high = max(item["h"] for item in bucket)
    low = min(item["l"] for item in bucket)
    return {
        "t": open_time,
        "o": bucket[0]["o"],
        "h": high,
        "l": low,
        "c": bucket[-1]["c"],
        "v": sum(item["v"] for item in bucket),
    }


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_candle(entry: Mapping[str, Any]) -> MutableMapping[str, Any] | None:
    """Normalise a raw candle mapping into numeric OHLCV fields."""

    raw_ts = (
        entry.get("t")
        or entry.get("time")
        or entry.get("openTime")
        or entry.get("open_time")
    )
    timestamp_ms = _safe_int(raw_ts)
    if timestamp_ms is None:
        return None

    candle: MutableMapping[str, Any] = {
        "t": timestamp_ms,
        "o": _coerce_float(entry.get("o", entry.get("open"))),
        "h": _coerce_float(entry.get("h", entry.get("high"))),
        "l": _coerce_float(entry.get("l", entry.get("low"))),
        "c": _coerce_float(entry.get("c", entry.get("close"))),
        "v": _coerce_float(entry.get("v", entry.get("volume"))),
    }

    return candle


def _extract_raw_candles(snapshot: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    frames = snapshot.get("frames")
    primary_tf = str(snapshot.get("tf") or snapshot.get("timeframe") or "1m").lower()

    if isinstance(frames, Mapping):
        target = frames.get(primary_tf)
        if target is None and frames:
            target = next(iter(frames.values()))
        if isinstance(target, Mapping):
            candles = target.get("candles", [])
        else:
            candles = target
    else:
        candles = snapshot.get("candles", [])

    if candles is None:
        return []

    try:
        return list(candles)  # type: ignore[arg-type]
    except TypeError:
        return []


def _normalise_candles(snapshot: Mapping[str, Any]) -> List[MutableMapping[str, Any]]:
    candles: List[MutableMapping[str, Any]] = []
    for entry in _extract_raw_candles(snapshot):
        if not isinstance(entry, Mapping):
            continue
        candle = _coerce_candle(entry)
        if candle is None:
            continue
        candles.append(candle)

    candles.sort(key=lambda item: item["t"])
    return candles


def _normalise_frames(snapshot: Mapping[str, Any]) -> Dict[str, List[MutableMapping[str, Any]]]:
    frames: Dict[str, List[MutableMapping[str, Any]]] = {}
    raw_frames = snapshot.get("frames")

    if isinstance(raw_frames, Mapping):
        for key, frame in raw_frames.items():
            candles: List[MutableMapping[str, Any]] = []
            raw_candles = []
            if isinstance(frame, Mapping):
                raw_candles = frame.get("candles", [])
            else:
                raw_candles = frame
            if raw_candles is None:
                raw_candles = []
            try:
                iterator = list(raw_candles)  # type: ignore[arg-type]
            except TypeError:
                iterator = []
            for entry in iterator:
                if not isinstance(entry, Mapping):
                    continue
                candle = _coerce_candle(entry)
                if candle is None:
                    continue
                candles.append(candle)
            candles.sort(key=lambda item: item["t"])
            frames[str(key).lower()] = candles

    if not frames:
        default_tf = str(snapshot.get("tf") or snapshot.get("timeframe") or "1m").lower()
        frames[default_tf] = _normalise_candles(snapshot)

    return frames


def _timeframe_interval_ms(tf_key: str) -> int | None:
    return TIMEFRAME_TO_MS.get(tf_key)


def _ensure_minute_frame(
    frames: MutableMapping[str, List[MutableMapping[str, Any]]],
    *,
    primary_key: str,
    primary_candles: Sequence[Mapping[str, Any]],
) -> None:
    existing = frames.get("1m")
    if isinstance(existing, list) and existing:
        return

    if primary_key == "1m":
        frames["1m"] = [dict(candle) for candle in primary_candles]
        return

    ordered_frames = sorted(
        frames.items(),
        key=lambda item: _timeframe_interval_ms(item[0]) or float("inf"),
    )

    for tf_key, candles in ordered_frames:
        interval_ms = _timeframe_interval_ms(tf_key)
        if interval_ms is None or interval_ms < MINUTE_INTERVAL_MS:
            continue
        ratio = interval_ms // MINUTE_INTERVAL_MS
        if ratio <= 0:
            continue
        expanded: List[MutableMapping[str, Any]] = []
        for candle in candles:
            ts = _safe_int(candle.get("t"))
            if ts is None:
                continue
            open_price = _coerce_float(candle.get("o"))
            high_price = _coerce_float(candle.get("h"))
            low_price = _coerce_float(candle.get("l"))
            close_price = _coerce_float(candle.get("c"))
            volume = _coerce_float(candle.get("v"))
            portion = volume / ratio if ratio else volume
            for idx in range(ratio):
                expanded.append(
                    {
                        "t": ts + idx * MINUTE_INTERVAL_MS,
                        "o": open_price,
                        "h": high_price,
                        "l": low_price,
                        "c": close_price,
                        "v": portion,
                    }
                )
        if expanded:
            frames["1m"] = expanded
            return

    frames["1m"] = [
        {
            "t": _safe_int(candle.get("t")) or 0,
            "o": _coerce_float(candle.get("o")),
            "h": _coerce_float(candle.get("h")),
            "l": _coerce_float(candle.get("l")),
            "c": _coerce_float(candle.get("c")),
            "v": _coerce_float(candle.get("v")),
        }
        for candle in primary_candles
    ]


def _latest_candle_before(
    candles: Sequence[Mapping[str, Any]],
    *,
    end_ms: int | None,
) -> Mapping[str, Any] | None:
    if not candles:
        return None
    if end_ms is None:
        return candles[-1]
    for entry in reversed(candles):
        if not isinstance(entry, Mapping):
            continue
        ts = _safe_int(entry.get("t"))
        if ts is None:
            continue
        if ts <= end_ms:
            return entry
    return candles[-1]


def _primary_frame_key(snapshot: Mapping[str, Any], frames: Mapping[str, Sequence[Mapping[str, Any]]]) -> str | None:
    preferred = str(snapshot.get("tf") or snapshot.get("timeframe") or "").lower()
    if preferred and preferred in frames:
        return preferred
    if frames:
        return next(iter(frames))
    return None


def _filter_candles(
    candles: Sequence[Mapping[str, Any]],
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> List[Dict[str, Any]]:
    if start_ms is None and end_ms is None:
        return [dict(candle) for candle in candles]

    result: List[Dict[str, Any]] = []
    for candle in candles:
        ts = _safe_int(candle.get("t"))
        if ts is None:
            continue
        if start_ms is not None and ts < start_ms:
            continue
        if end_ms is not None and ts > end_ms:
            continue
        result.append({
            "t": ts,
            "o": _coerce_float(candle.get("o")),
            "h": _coerce_float(candle.get("h")),
            "l": _coerce_float(candle.get("l")),
            "c": _coerce_float(candle.get("c")),
            "v": _coerce_float(candle.get("v")),
        })
    return result


def _summarise(candles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not candles:
        return {"count": 0, "open": None, "close": None, "high": None, "low": None, "volume": 0.0}

    highs = [float(item.get("h", 0.0)) for item in candles]
    lows = [float(item.get("l", 0.0)) for item in candles]
    volumes = [float(item.get("v", 0.0)) for item in candles]

    return {
        "count": len(candles),
        "open": float(candles[0].get("o", 0.0)),
        "close": float(candles[-1].get("c", 0.0)),
        "high": max(highs) if highs else None,
        "low": min(lows) if lows else None,
        "volume": float(sum(volumes)),
    }


def _build_delta_series(candles: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    series: List[Dict[str, Any]] = []
    cumulative = 0.0
    for candle in candles:
        open_price = float(candle.get("o", 0.0))
        close_price = float(candle.get("c", 0.0))
        volume = float(candle.get("v", 0.0))
        net = (close_price - open_price) * volume
        cumulative += net
        delta_pct = ((close_price - open_price) / open_price * 100.0) if open_price else 0.0
        series.append(
            {
                "t": candle.get("t"),
                "delta": net,
                "deltaPct": delta_pct,
                "cvd": cumulative,
            }
        )
    return series


def _summarise_delta_series(series: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not series:
        return {"count": 0, "net_delta": 0.0, "cvd_change": 0.0, "delta_pct_total": 0.0}
    net = sum(float(item.get("delta", 0.0)) for item in series)
    cvd_change = float(series[-1].get("cvd", 0.0)) - float(series[0].get("cvd", 0.0))
    delta_pct_total = sum(float(item.get("deltaPct", 0.0)) for item in series)
    return {
        "count": len(series),
        "net_delta": net,
        "cvd_change": cvd_change,
        "delta_pct_total": delta_pct_total,
    }


def _compute_vwap(candles: Sequence[Mapping[str, Any]]) -> float:
    total_pv = 0.0
    total_volume = 0.0
    for candle in candles:
        high = float(candle.get("h", 0.0))
        low = float(candle.get("l", 0.0))
        close = float(candle.get("c", 0.0))
        volume = float(candle.get("v", 0.0))
        typical_price = (high + low + close) / 3.0
        total_pv += typical_price * volume
        total_volume += volume
    if total_volume <= 0:
        return 0.0
    return total_pv / total_volume


def _typical_price(candle: Mapping[str, Any]) -> float:
    high = float(candle.get("h", 0.0))
    low = float(candle.get("l", 0.0))
    close = float(candle.get("c", 0.0))
    return (high + low + close) / 3.0


def _determine_bin_size(prices: Sequence[float], tick_size: float | None) -> float | None:
    finite_prices = [price for price in prices if math.isfinite(price)]
    if not finite_prices:
        return float(tick_size) if tick_size and tick_size > 0 else None

    average_price = sum(finite_prices) / len(finite_prices)
    adaptive_step = abs(average_price) * 1e-4
    if adaptive_step <= 0:
        adaptive_step = max(abs(finite_prices[0]) * 1e-4, 1e-6)

    tick = float(tick_size) if tick_size and tick_size > 0 else None
    step = adaptive_step if adaptive_step > 0 else None
    if tick is not None:
        if step is None:
            return tick
        return max(tick, step)
    return step


def _build_volume_profile_stats(
    candles: Sequence[Mapping[str, Any]],
    *,
    start_ms: int,
    end_ms: int,
    tick_size: float | None,
    value_area_pct: float = VALUE_AREA_PCT,
) -> Dict[str, Any]:
    window_start_iso = datetime.fromtimestamp(start_ms / 1000.0, tz=UTC).isoformat()
    window_end_iso = datetime.fromtimestamp(end_ms / 1000.0, tz=UTC).isoformat()

    if end_ms < start_ms:
        return {
            "vwap": 0.0,
            "poc": None,
            "vah": None,
            "val": None,
            "window": {"start": window_start_iso, "end": window_end_iso},
        }

    scoped = [
        candle
        for candle in candles
        if isinstance(candle, Mapping)
        and (ts := _safe_int(candle.get("t"))) is not None
        and start_ms <= ts <= end_ms
    ]

    if not scoped:
        return {
            "vwap": 0.0,
            "poc": None,
            "vah": None,
            "val": None,
            "window": {"start": window_start_iso, "end": window_end_iso},
        }

    vwap_value = _compute_vwap(scoped)
    prices: List[float] = []
    volumes: List[float] = []
    for candle in scoped:
        volume = float(candle.get("v", 0.0))
        if volume <= 0:
            continue
        price = _typical_price(candle)
        if not math.isfinite(price):
            continue
        prices.append(price)
        volumes.append(volume)

    if not prices or not volumes:
        return {
            "vwap": vwap_value,
            "poc": None,
            "vah": None,
            "val": None,
            "window": {"start": window_start_iso, "end": window_end_iso},
        }

    bin_size = _determine_bin_size(prices, tick_size)
    if not bin_size or bin_size <= 0:
        return {
            "vwap": vwap_value,
            "poc": None,
            "vah": None,
            "val": None,
            "window": {"start": window_start_iso, "end": window_end_iso},
        }

    min_price = min(prices)
    max_price = max(prices)
    start_bin = math.floor(min_price / bin_size) * bin_size
    bins_count = max(1, int(math.floor((max_price - start_bin) / bin_size)) + 1)

    histogram = [0.0 for _ in range(bins_count)]
    for price, volume in zip(prices, volumes):
        index = int(math.floor((price - start_bin) / bin_size + 1e-9))
        if index < 0:
            index = 0
        elif index >= bins_count:
            index = bins_count - 1
        histogram[index] += volume

    total_volume = sum(histogram)
    if total_volume <= 0:
        return {
            "vwap": vwap_value,
            "poc": None,
            "vah": None,
            "val": None,
            "window": {"start": window_start_iso, "end": window_end_iso},
        }

    poc_index = max(range(len(histogram)), key=lambda idx: histogram[idx])
    poc_price = start_bin + poc_index * bin_size

    threshold = total_volume * max(0.0, min(1.0, value_area_pct))
    coverage = histogram[poc_index]
    left = right = poc_index

    while coverage < threshold and (left > 0 or right < len(histogram) - 1):
        next_left = histogram[left - 1] if left > 0 else -1.0
        next_right = histogram[right + 1] if right < len(histogram) - 1 else -1.0

        if next_left < 0 and next_right < 0:
            break

        if next_right > next_left:
            right += 1
            coverage += max(0.0, next_right)
        elif next_left > next_right:
            left -= 1
            coverage += max(0.0, next_left)
        else:
            if next_left >= 0 and left > 0:
                left -= 1
                coverage += max(0.0, next_left)
            if coverage < threshold and next_right >= 0 and right < len(histogram) - 1:
                right += 1
                coverage += max(0.0, next_right)

    val_price = start_bin + left * bin_size
    vah_price = start_bin + right * bin_size

    return {
        "vwap": vwap_value,
        "poc": round(poc_price, 12),
        "vah": round(vah_price, 12),
        "val": round(val_price, 12),
        "window": {"start": window_start_iso, "end": window_end_iso},
    }


def _start_of_day_ms(timestamp_ms: int) -> int:
    dt = datetime.fromtimestamp(timestamp_ms / 1000.0, tz=UTC)
    start_dt = datetime(dt.year, dt.month, dt.day, tzinfo=UTC)
    return int(start_dt.timestamp() * 1000)


def _session_window(
    anchor_ms: int,
    start_time: dtime,
    end_time: dtime,
) -> tuple[int, int]:
    anchor_aligned = _align_to_interval(anchor_ms, MINUTE_INTERVAL_MS)
    anchor_dt = datetime.fromtimestamp(anchor_aligned / 1000.0, tz=UTC)
    day_start = datetime(anchor_dt.year, anchor_dt.month, anchor_dt.day, tzinfo=UTC)
    session_start_dt = datetime.combine(day_start.date(), start_time, tzinfo=UTC)
    session_end_dt = datetime.combine(day_start.date(), end_time, tzinfo=UTC)

    if end_time <= start_time:
        session_end_dt += timedelta(days=1)

    start_ms = int(session_start_dt.timestamp() * 1000)
    raw_end_ms = int(session_end_dt.timestamp() * 1000) - MINUTE_INTERVAL_MS
    if raw_end_ms < start_ms:
        raw_end_ms = start_ms

    end_ms = min(raw_end_ms, anchor_aligned)
    return start_ms, end_ms


def _extract_range(candidate: Mapping[str, Any]) -> tuple[int, int] | None:
    for key in ("t", "time", "timestamp", "ts"):
        ts = _safe_int(candidate.get(key))
        if ts is not None:
            return ts, ts
    start = candidate.get("start") or candidate.get("from") or candidate.get("begin")
    end = candidate.get("end") or candidate.get("to") or candidate.get("finish")
    start_ts = _safe_int(start)
    end_ts = _safe_int(end)
    if start_ts is None and end_ts is None:
        return None
    if start_ts is None:
        start_ts = end_ts or 0
    if end_ts is None:
        end_ts = start_ts
    return start_ts, end_ts


def _range_intersects(range_tuple: tuple[int, int], start_ms: int | None, end_ms: int | None) -> bool:
    if start_ms is None and end_ms is None:
        return True
    start, end = range_tuple
    if end_ms is not None and start > end_ms:
        return False
    if start_ms is not None and end < start_ms:
        return False
    return True


def _filter_indicator_block(value: Any, start_ms: int | None, end_ms: int | None) -> Any:
    if isinstance(value, Mapping):
        range_tuple = _extract_range(value)
        if range_tuple and not _range_intersects(range_tuple, start_ms, end_ms):
            return None
        result: Dict[str, Any] = {}
        for key, inner in value.items():
            filtered = _filter_indicator_block(inner, start_ms, end_ms)
            if filtered is None and isinstance(inner, (Mapping, list, tuple, set)):
                continue
            result[key] = filtered if filtered is not None else inner
        return result

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        filtered_items: List[Any] = []
        has_range = False
        for item in value:
            if isinstance(item, Mapping):
                item_range = _extract_range(item)
                if item_range:
                    has_range = True
                    if not _range_intersects(item_range, start_ms, end_ms):
                        continue
            filtered = _filter_indicator_block(item, start_ms, end_ms)
            if filtered is None and isinstance(item, (Mapping, list, tuple, set)):
                continue
            filtered_items.append(filtered if filtered is not None else item)
        return filtered_items if has_range else filtered_items

    return value


def _select_indicator_timeframes(data: Any, targets: Sequence[str]) -> Any:
    if not isinstance(data, Mapping):
        return data
    lowered = {str(key).lower(): key for key in data.keys()}
    result: Dict[str, Any] = {}
    for target in targets:
        key = lowered.get(target)
        if key is None:
            continue
        result[key] = data[key]
    return result


def _filter_agg_trades(
    payload: Any,
    *,
    start_ms: int | None,
    end_ms: int | None,
    include_trades: bool,
) -> Dict[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None

    trades = payload.get("agg")
    filtered_trades: List[Dict[str, Any]] = []
    if isinstance(trades, Sequence):
        for entry in trades:
            if not isinstance(entry, Mapping):
                continue
            ts = _safe_int(entry.get("t"))
            if ts is None:
                continue
            if start_ms is not None and ts < start_ms:
                continue
            if end_ms is not None and ts > end_ms:
                continue
            filtered_trades.append({
                "t": ts,
                "p": _coerce_float(entry.get("p")),
                "q": _coerce_float(entry.get("q")),
                "side": entry.get("side"),
            })

    summary = {
        "count": len(filtered_trades),
        "buy": sum(1 for trade in filtered_trades if str(trade.get("side")).lower() == "buy"),
        "sell": sum(1 for trade in filtered_trades if str(trade.get("side")).lower() == "sell"),
        "volume": sum(float(trade.get("q", 0.0)) for trade in filtered_trades),
    }

    result: Dict[str, Any] = {
        "symbol": payload.get("symbol"),
        "summary": summary,
    }
    if include_trades:
        result["trades"] = filtered_trades
    return result


def _build_daily_vwap(
    frames: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    start_ms: int | None,
    end_ms: int | None,
) -> Dict[str, Any] | None:
    daily_keys = [key for key in ("1d", "1h", "4h") if key in frames]
    if not daily_keys:
        return None

    source_key = daily_keys[0]
    filtered = _filter_candles(frames[source_key], start_ms=start_ms, end_ms=end_ms)
    if not filtered:
        return None
    return {
        "timeframe": source_key,
        "value": _compute_vwap(filtered),
        "summary": _summarise(filtered),
    }


def build_check_all_datas(
    snapshot: Mapping[str, Any],
    *,
    now_utc: datetime | None = None,
    selection_start_ms: int | None = None,
    selection_end_ms: int | None = None,
    hours: int | None = None,
) -> Dict[str, Any] | None:
    """Create an enriched diagnostics payload for the snapshot health endpoint."""

    frames = _normalise_frames(snapshot)
    if not frames:
        return None

    primary_key = _primary_frame_key(snapshot, frames)
    if not primary_key:
        return None

    primary_candles = frames.get(primary_key, [])
    if not primary_candles:
        return None

    # Drop unused granularities to keep the payload focused on the requested set.
    frames.pop("3m", None)
    frames.pop("5m", None)

    minute_candles = _deduplicate_sorted(frames.get("1m", []))
    frames["1m"] = minute_candles

    symbol = str(snapshot.get("symbol") or snapshot.get("pair") or "UNKNOWN").upper()
    profile_config = resolve_profile_config(symbol, snapshot.get("meta") if isinstance(snapshot.get("meta"), Mapping) else None)
    sessions = list(Meta.iter_vwap_sessions())
    profile_tpo: List[Dict[str, Any]] = []
    profile_flat: List[Dict[str, float]] = []
    profile_zones: List[Dict[str, Any]] = []
    detected_zones: Dict[str, Any] = {
        "symbol": symbol,
        "zones": {"fvg": [], "ob": [], "inducement": [], "cisd": []},
    }

    target_tf_key = profile_config.get("target_tf_key", "1m")
    base_candidates = frames.get(target_tf_key, [])
    if not base_candidates:
        base_candidates = minute_candles
    if not base_candidates and frames:
        base_candidates = next(iter(frames.values()))
    base_candles = _deduplicate_sorted(base_candidates)
    frames[target_tf_key] = base_candles

    if primary_key == "1m":
        primary_candles = minute_candles
    elif primary_key == target_tf_key:
        primary_candles = base_candles
    else:
        primary_candles = _deduplicate_sorted(frames.get(primary_key, []))
        frames[primary_key] = primary_candles
        if not primary_candles:
            primary_candles = base_candles

    if profile_config.get("preset") and base_candles and sessions:
        cache_token = (
            "check_all",
            snapshot.get("id"),
            symbol,
            target_tf_key,
        )
        (profile_tpo, profile_flat, profile_zones) = build_profile_package(
            base_candles,
            sessions=sessions,
            last_n=int(profile_config.get("last_n", 3)),
            tick_size=profile_config.get("tick_size"),
            adaptive_bins=bool(profile_config.get("adaptive_bins", True)),
            value_area_pct=float(profile_config.get("value_area_pct", 0.7)),
            atr_multiplier=float(profile_config.get("atr_multiplier", 0.5)),
            target_bins=int(profile_config.get("target_bins", 80)),
            clip_threshold=float(profile_config.get("clip_threshold", 0.0)),
            smooth_window=int(profile_config.get("smooth_window", 1)),
            cache_token=cache_token,
            tf_key=target_tf_key,
        )

    snapshot_selection = snapshot.get("selection") if isinstance(snapshot.get("selection"), Mapping) else None
    selection_start = selection_start_ms or _safe_int(snapshot_selection.get("start")) if snapshot_selection else None
    selection_end = selection_end_ms or _safe_int(snapshot_selection.get("end")) if snapshot_selection else None

    if selection_start is None:
        selection_start = primary_candles[0]["t"]
    if selection_end is None:
        selection_end = primary_candles[-1]["t"]

    if selection_start > selection_end:
        selection_start, selection_end = selection_end, selection_start

    hours_window = hours if hours in VALID_HOUR_WINDOWS else min(VALID_HOUR_WINDOWS)

    if now_utc is not None:
        if now_utc.tzinfo is None:
            now_dt = now_utc.replace(tzinfo=UTC)
        else:
            now_dt = now_utc.astimezone(UTC)
        now_ms = int(now_dt.timestamp() * 1000)
        window_end_ms = _align_to_interval(now_ms, MINUTE_INTERVAL_MS) - MINUTE_INTERVAL_MS
    else:
        window_end_ms = minute_candles[-1]["t"] if minute_candles else None

    target_interval_ms = _timeframe_interval_ms(target_tf_key) or MINUTE_INTERVAL_MS

    if window_end_ms is None and base_candles:
        window_end_ms = base_candles[-1]["t"] + max(target_interval_ms - MINUTE_INTERVAL_MS, 0)

    if window_end_ms is None and primary_candles:
        primary_interval = _timeframe_interval_ms(primary_key) or MINUTE_INTERVAL_MS
        window_end_ms = primary_candles[-1]["t"] + max(primary_interval - MINUTE_INTERVAL_MS, 0)

    if window_end_ms is None:
        return None

    window_end_ms = max(0, _align_to_interval(window_end_ms, MINUTE_INTERVAL_MS))

    raw_window_start = window_end_ms - hours_window * MS_IN_HOUR
    window_start_ms = max(0, _align_to_interval(raw_window_start, MINUTE_INTERVAL_MS))
    if target_interval_ms > MINUTE_INTERVAL_MS:
        window_start_ms = max(0, _align_to_interval(window_start_ms, target_interval_ms))

    minute_index_all = {candle["t"]: candle for candle in minute_candles}
    minute_window_index = {
        ts: candle
        for ts, candle in minute_index_all.items()
        if window_start_ms <= ts <= window_end_ms
    }

    expected_minutes = _build_expected_times(window_start_ms, window_end_ms, MINUTE_INTERVAL_MS)
    time_gaps = _summarise_missing_times(expected_minutes, minute_window_index)
    minute_missing_before = sum(gap["count"] for gap in time_gaps)

    fetched_unique = 0
    if time_gaps:
        try:
            downloaded_minutes = _download_missing_minutes(
                symbol,
                window_start_ms,
                window_end_ms,
                time_gaps,
            )
        except BinanceDownloadError as exc:
            detail = {
                "tf": target_tf_key,
                "window": {"start_ms": window_start_ms, "end_ms": window_end_ms},
                "minute_missing_before": minute_missing_before,
                "minute_missing_after": minute_missing_before,
                "fetched_1m_count": exc.downloaded,
                "tf_missing_before": 0,
                "tf_missing_after": 0,
                "time_gaps": time_gaps,
                "downloaded": exc.downloaded,
            }
            raise DataQualityError(detail) from exc
        for candle in downloaded_minutes:
            ts = candle["t"]
            if ts < window_start_ms or ts > window_end_ms:
                continue
            if ts not in minute_window_index:
                fetched_unique += 1
            minute_window_index[ts] = candle
            minute_index_all[ts] = candle

    minute_missing_after = sum(1 for ts in expected_minutes if ts not in minute_window_index)
    data_quality = {
        "tf": target_tf_key,
        "window": {"start_ms": window_start_ms, "end_ms": window_end_ms},
        "minute_missing_before": minute_missing_before,
        "minute_missing_after": minute_missing_after,
        "fetched_1m_count": fetched_unique,
        "tf_missing_before": 0,
        "tf_missing_after": 0,
        "time_gaps": time_gaps,
    }

    if minute_missing_after > 0:
        data_quality["downloaded"] = fetched_unique
        raise DataQualityError(data_quality)

    frames["1m"] = [minute_index_all[ts] for ts in sorted(minute_index_all)]
    minute_candles = frames["1m"]

    base_index_all = {candle["t"]: candle for candle in base_candles}

    if target_interval_ms <= MINUTE_INTERVAL_MS:
        expected_tf_times = expected_minutes
    else:
        expected_tf_times: List[int] = []
        cursor = window_start_ms
        while True:
            last_minute = cursor + target_interval_ms - MINUTE_INTERVAL_MS
            if last_minute > window_end_ms:
                break
            expected_tf_times.append(cursor)
            cursor += target_interval_ms

    tf_missing_before = sum(1 for ts in expected_tf_times if ts not in base_index_all)
    aggregated_added = 0
    if tf_missing_before:
        for open_ts in expected_tf_times:
            if open_ts in base_index_all:
                continue
            aggregated = _aggregate_from_minutes(minute_window_index, open_ts, target_interval_ms)
            if aggregated is None:
                continue
            base_index_all[open_ts] = aggregated
            aggregated_added += 1

    tf_missing_after = sum(1 for ts in expected_tf_times if ts not in base_index_all)
    data_quality["tf_missing_before"] = tf_missing_before
    data_quality["tf_missing_after"] = tf_missing_after

    if tf_missing_after > 0:
        data_quality["downloaded"] = fetched_unique
        raise DataQualityError(data_quality)

    frames[target_tf_key] = [base_index_all[ts] for ts in sorted(base_index_all)]
    base_candles = frames[target_tf_key]
    if primary_key == target_tf_key:
        primary_candles = base_candles

    selection_payload: Dict[str, Any] = {
        "start": selection_start,
        "end": selection_end,
    }
    htf_section, htf_quality = build_htf_section(symbol, frames, selection_payload)

    reference_ts = window_end_ms + MINUTE_INTERVAL_MS
    reference_dt = datetime.fromtimestamp(reference_ts / 1000.0, tz=UTC)
    detailed_start_ts = window_start_ms

    detection_candles: List[Dict[str, Any]] = []
    if base_candles:
        detection_candles = _filter_candles(
            base_candles,
            start_ms=detailed_start_ts,
            end_ms=window_end_ms,
        )

    if detection_candles:
        try:
            zone_cfg = ZonesConfig(tick_size=profile_config.get("tick_size"))
            detected_zones = detect_zones(
                detection_candles,
                target_tf_key,
                symbol,
                zone_cfg,
            )
        except Exception:  # pragma: no cover - defensive logging guard
            logging.getLogger(__name__).exception(
                "Failed to detect zones for check-all payload",
                extra={
                    "snapshot_id": snapshot.get("id"),
                    "symbol": symbol,
                    "timeframe": target_tf_key,
                },
            )
            detected_zones = {
                "symbol": symbol,
                "zones": {"fvg": [], "ob": [], "inducement": [], "cisd": []},
            }

    movement_anchor_ts = detailed_start_ts
    movement_start_ts = min(selection_start, movement_anchor_ts)
    movement_end_ts = max(selection_start, movement_anchor_ts)
    if movement_end_ts > window_end_ms:
        movement_end_ts = window_end_ms

    latest_minute_candle = minute_window_index.get(window_end_ms)
    latest_primary_candle = None
    if expected_tf_times:
        latest_primary_candle = base_index_all.get(expected_tf_times[-1])
    elif base_candles:
        latest_primary_candle = base_candles[-1]

    latest_candle_source = (
        latest_minute_candle
        or latest_primary_candle
        or (base_candles[-1] if base_candles else None)
    )
    latest_candle_ts = _safe_int(latest_candle_source.get("t")) if latest_candle_source else None
    if latest_candle_ts is None and base_candles:
        latest_candle_ts = base_candles[-1]["t"]
    if latest_candle_ts is None:
        latest_candle_ts = window_end_ms
    latest_candle_dt = datetime.fromtimestamp(latest_candle_ts / 1000.0, tz=UTC)

    detailed_start_dt = datetime.fromtimestamp(detailed_start_ts / 1000.0, tz=UTC)
    movement_start_dt = datetime.fromtimestamp(movement_start_ts / 1000.0, tz=UTC)
    movement_end_dt = datetime.fromtimestamp(movement_end_ts / 1000.0, tz=UTC)

    detailed_frames: Dict[str, Dict[str, Any]] = {}
    for tf_key, candles in frames.items():
        filtered = _filter_candles(candles, start_ms=detailed_start_ts, end_ms=reference_ts)
        delta_series = _build_delta_series(filtered)
        detailed_frames[tf_key] = {
            "summary": _summarise(filtered),
            "candles": filtered,
            "delta_cvd": delta_series,
            "vwap": _compute_vwap(filtered),
        }

    zones_detailed = _filter_indicator_block(snapshot.get("zones"), detailed_start_ts, reference_ts)
    smt_detailed = _filter_indicator_block(snapshot.get("smt"), detailed_start_ts, reference_ts)
    agg_trades_detailed = _filter_agg_trades(
        snapshot.get("agg_trades"),
        start_ms=detailed_start_ts,
        end_ms=reference_ts,
        include_trades=True,
    )
    daily_vwap_detailed = _build_daily_vwap(frames, start_ms=detailed_start_ts, end_ms=reference_ts)

    detailed_section = {
        "hours": hours_window,
        "range": {
            "start_utc": detailed_start_dt.isoformat(),
            "end_utc": reference_dt.isoformat(),
        },
        "frames": detailed_frames,
        "indicators": {
            "zones": zones_detailed,
            "smt": smt_detailed,
            "delta_cvd": {tf: details["delta_cvd"] for tf, details in detailed_frames.items()},
            "vwap_daily": daily_vwap_detailed,
            "agg_trades": agg_trades_detailed,
        },
    }

    movement_frames: Dict[str, Dict[str, Any]] = {}
    delta_summaries: Dict[str, Dict[str, Any]] = {}
    vwap_summaries: Dict[str, Dict[str, Any]] = {}
    for tf_key in ("4h", "1d"):
        candles = frames.get(tf_key)
        if not candles:
            continue
        filtered = _filter_candles(candles, start_ms=movement_start_ts, end_ms=movement_end_ts)
        if not filtered:
            continue
        delta_series = _build_delta_series(filtered)
        movement_frames[tf_key] = {
            "summary": _summarise(filtered),
            "first_candle_utc": datetime.fromtimestamp(filtered[0]["t"] / 1000.0, tz=UTC).isoformat(),
            "last_candle_utc": datetime.fromtimestamp(filtered[-1]["t"] / 1000.0, tz=UTC).isoformat(),
        }
        delta_summaries[tf_key] = _summarise_delta_series(delta_series)
        vwap_summaries[tf_key] = {
            "value": _compute_vwap(filtered),
            "summary": _summarise(filtered),
        }

    zones_movement = _select_indicator_timeframes(
        _filter_indicator_block(snapshot.get("zones"), movement_start_ts, movement_end_ts),
        ("4h", "1d"),
    )
    smt_movement = _select_indicator_timeframes(
        _filter_indicator_block(snapshot.get("smt"), movement_start_ts, movement_end_ts),
        ("4h", "1d"),
    )
    agg_trades_movement = _filter_agg_trades(
        snapshot.get("agg_trades"),
        start_ms=movement_start_ts,
        end_ms=movement_end_ts,
        include_trades=False,
    )

    movement_days = 0
    if selection_start is not None and selection_end is not None:
        movement_days = max(0, int((selection_end - selection_start) // MS_IN_DAY))

    movement_section = {
        "days": movement_days,
        "range": {
            "start_utc": movement_start_dt.isoformat(),
            "end_utc": movement_end_dt.isoformat(),
        },
        "frames": movement_frames,
        "indicators": {
            "zones": zones_movement,
            "smt": smt_movement,
            "delta_cvd": delta_summaries,
            "vwap": vwap_summaries,
            "agg_trades": agg_trades_movement,
        },
    }

    movement_key = f"movement_datas_for_{movement_days}_days"

    tick_size_value = profile_config.get("tick_size") if isinstance(profile_config, Mapping) else None
    tick_size_numeric: float | None = None
    if isinstance(tick_size_value, (int, float)):
        tick_size_numeric = float(tick_size_value)

    minute_series = frames.get("1m", [])
    daily_start_ms = _start_of_day_ms(window_end_ms)
    daily_vwap_profile = _build_volume_profile_stats(
        minute_series,
        start_ms=daily_start_ms,
        end_ms=window_end_ms,
        tick_size=tick_size_numeric,
        value_area_pct=VALUE_AREA_PCT,
    )

    session_profiles: Dict[str, Dict[str, Any]] = {}
    for session_name, session_start, session_end in sessions:
        session_start_ms, session_end_ms = _session_window(window_end_ms, session_start, session_end)
        session_profiles[session_name] = _build_volume_profile_stats(
            minute_series,
            start_ms=session_start_ms,
            end_ms=session_end_ms,
            tick_size=tick_size_numeric,
            value_area_pct=VALUE_AREA_PCT,
        )

    vwap_payload = {
        "daily": daily_vwap_profile,
        "sessions": session_profiles,
    }

    latest_candle_payload_source = (
        latest_candle_source
        or (primary_candles[-1] if primary_candles else None)
    )
    latest_candle_payload = (
        dict(latest_candle_payload_source)
        if isinstance(latest_candle_payload_source, Mapping)
        else {}
    )

    return {
        "snapshot_id": snapshot.get("id"),
        "symbol": snapshot.get("symbol"),
        "timeframe": snapshot.get("tf"),
        "selection": {"start": selection_start, "end": selection_end},
        "asof_utc": reference_dt.isoformat(),
        "latest_candle_utc": latest_candle_dt.isoformat(),
        "latest_candle": dict(latest_candle_payload),
        "datas_for_last_N_hours": detailed_section,
        movement_key: movement_section,
        "tpo": {"sessions": profile_tpo, "zones": profile_zones},
        "profile": profile_flat,
        "zones": detected_zones,
        "data_quality": data_quality,
        "htf": htf_section,
        "data_quality_htf": htf_quality,
        "profile_preset": profile_config.get("preset_payload"),
        "profile_preset_required": bool(profile_config.get("preset_required", False)),
        "vwap": vwap_payload,
    }
