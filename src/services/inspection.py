"""Inspection payload assembly and UI rendering."""
from __future__ import annotations

import html as html_utils
import json
import logging
import math
import os
import re
from collections import OrderedDict, defaultdict
from datetime import datetime, timedelta, timezone, time as dtime
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    Tuple,
)

import httpx

from .binance import BINANCE_FAPI_REST
from .liquidity import (
    build_liquidity_snapshot,
    normalise_symbol_for_tick,
    resolve_liquidity_tick_size,
)
from .ohlc import (
    TIMEFRAME_WINDOWS,
    TIMEFRAME_TO_MS,
    aggregate_1m_to_1h,
    normalise_ohlcv,
    resample_ohlcv,
)
from .profile import build_profile_package
from .presets import resolve_profile_config
from .zones import Config as ZonesConfig, detect_zones
from ..meta import Meta

Snapshot = Dict[str, Any]

DEFAULT_SYMBOL = "BTCUSDT"

_MAX_STORED_SNAPSHOTS = 16
_SNAPSHOT_STORE: "OrderedDict[str, Snapshot]" = OrderedDict()

_DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parents[2] / "var" / "snapshots"
SNAPSHOT_STORAGE_DIR = Path(
    os.environ.get("INSPECTION_SNAPSHOT_DIR", str(_DEFAULT_STORAGE_ROOT))
).expanduser()
_SNAPSHOT_ID_SANITISER = re.compile(r"[^A-Za-z0-9._-]")

MS_IN_DAY = 86_400_000
HTF_TIMEFRAMES: Tuple[str, ...] = ("15m", "1h", "4h", "1d")
MINUTE_INTERVAL_MS = TIMEFRAME_TO_MS.get("1m", 60_000)



def _safe_int(value: object | None) -> int | None:
    try:
        if isinstance(value, bool):  # Guard against bools masquerading as ints
            return int(value) if value else 0
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object | None) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _align_to_interval(timestamp_ms: int, interval_ms: int) -> int:
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return (timestamp_ms // interval_ms) * interval_ms


def _extract_frame_candles(
    frames: Mapping[str, Any], timeframe: str
) -> Sequence[Mapping[str, object] | Sequence[object]]:
    frame = frames.get(timeframe)
    if isinstance(frame, Mapping):
        candles = frame.get("candles")
        if isinstance(candles, Sequence):
            return candles  # type: ignore[return-value]
        return []
    if isinstance(frame, Sequence):
        return frame  # type: ignore[return-value]
    return []


def _parse_minute_row(
    row: Mapping[str, object] | Sequence[object]
) -> Dict[str, float] | None:
    open_time: int | None = None
    open_price: float | None = None
    high_price: float | None = None
    low_price: float | None = None
    close_price: float | None = None
    volume: float | None = None

    if isinstance(row, Mapping):
        open_time = _safe_int(
            row.get("t")
            or row.get("time")
            or row.get("openTime")
            or row.get("open_time")
        )
        open_price = _coerce_float(row.get("o") or row.get("open"))
        high_price = _coerce_float(row.get("h") or row.get("high"))
        low_price = _coerce_float(row.get("l") or row.get("low"))
        close_price = _coerce_float(row.get("c") or row.get("close"))
        volume = _coerce_float(row.get("v") or row.get("volume"))
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

    if (
        open_time is None
        or open_price is None
        or high_price is None
        or low_price is None
        or close_price is None
    ):
        return None

    return {
        "t": int(open_time),
        "o": float(open_price),
        "h": float(high_price),
        "l": float(low_price),
        "c": float(close_price),
        "v": float(volume or 0.0),
    }


def _normalise_minute_rows(
    rows: Sequence[Mapping[str, object] | Sequence[object]]
) -> Dict[int, Dict[str, float]]:
    minutes: Dict[int, Dict[str, float]] = {}
    for row in rows:
        candle = _parse_minute_row(row)
        if candle is None:
            continue
        minutes[candle["t"]] = candle
    return minutes


def ensure_higher_timeframes(
    candles_by_tf: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Guarantee 15m and 1h frames by resampling minute candles when missing."""

    logger = logging.getLogger(__name__)
    minute_seed = candles_by_tf.get("1m")
    minute_candles: List[Mapping[str, Any]] = (
        list(minute_seed) if isinstance(minute_seed, Sequence) else []
    )
    logger.debug(
        "Ensuring higher timeframes for liquidity",
        extra={"seed_tf": "1m", "seed_candles": len(minute_candles)},
    )

    generated: Dict[str, List[Dict[str, Any]]] = {}
    if not minute_candles:
        return generated

    for target_tf in ("15m", "1h"):
        existing = candles_by_tf.get(target_tf)
        existing_count = len(existing) if isinstance(existing, Sequence) else 0
        if existing_count:
            logger.debug(
                "Skipping resample for timeframe with existing data",
                extra={"tf": target_tf, "candles": existing_count},
            )
            continue
        interval_ms = TIMEFRAME_TO_MS.get(target_tf)
        if not interval_ms:
            continue
        aggregated = resample_ohlcv(minute_candles, interval_ms)
        generated[target_tf] = aggregated
        logger.debug(
            "Generated higher timeframe from minute seed",
            extra={
                "tf": target_tf,
                "candles": len(aggregated),
                "interval_ms": interval_ms,
            },
        )
    return generated


def _expected_minute_sequence(start_ms: int, end_ms: int) -> List[int]:
    if end_ms < start_ms:
        return []
    steps = ((end_ms - start_ms) // MINUTE_INTERVAL_MS) + 1
    return [start_ms + index * MINUTE_INTERVAL_MS for index in range(steps)]


def _summarise_missing_minutes(
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
            gaps.append(
                {"from": current_start, "to": ts - MINUTE_INTERVAL_MS, "count": current_count}
            )
            current_start = None
            current_count = 0

    if current_start is not None and expected:
        gaps.append({"from": current_start, "to": expected[-1], "count": current_count})

    return gaps


def _fetch_binance_minutes(
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> Sequence[Sequence[object]]:
    params = {
        "symbol": symbol.upper(),
        "interval": "1m",
        "startTime": str(start_ms),
        "endTime": str(end_ms),
        "limit": str(limit),
    }
    with httpx.Client(timeout=15.0) as client:
        response = client.get(BINANCE_FAPI_REST, params=params)
        response.raise_for_status()
        data = response.json()
    if isinstance(data, Sequence):
        return data  # type: ignore[return-value]
    return []


MinuteFetcher = Callable[[str, int, int, int], Sequence[Mapping[str, object] | Sequence[object]]]
_DEFAULT_MINUTE_FETCHER: MinuteFetcher = _fetch_binance_minutes


def _download_missing_minutes(
    symbol: str,
    gaps: Sequence[Mapping[str, int]],
    *,
    fetcher: MinuteFetcher,
    target: MutableMapping[int, Dict[str, float]],
) -> int:
    downloaded_unique = 0

    for gap in gaps:
        start = _safe_int(gap.get("from"))
        end = _safe_int(gap.get("to"))
        if start is None or end is None or end < start:
            continue
        if end < 0:
            continue
        cursor = max(start, 0)
        attempts = 0
        while cursor <= end and attempts < 2048:
            request_end = min(end, cursor + MINUTE_INTERVAL_MS * 999)
            try:
                raw_rows = fetcher(symbol, cursor, request_end, 1_000)
            except Exception as exc:  # pragma: no cover - network guard
                logging.getLogger(__name__).warning(
                    "Failed to download 1m candles for HTF aggregation",
                    exc_info=exc,
                    extra={
                        "symbol": symbol,
                        "start_ms": cursor,
                        "end_ms": request_end,
                    },
                )
                break
            if not raw_rows:
                break
            last_open = None
            for row in raw_rows:
                candle = _parse_minute_row(row)
                if candle is None:
                    continue
                ts = candle["t"]
                if ts < cursor or ts > end:
                    continue
                if ts not in target:
                    downloaded_unique += 1
                target[ts] = candle
                if last_open is None or ts > last_open:
                    last_open = ts
            if last_open is None:
                break
            cursor = last_open + MINUTE_INTERVAL_MS
            attempts += 1

    return downloaded_unique


def _aggregate_from_minutes(
    minute_index: Mapping[int, Mapping[str, float]],
    open_time: int,
    interval_ms: int,
) -> Dict[str, float] | None:
    end_exclusive = open_time + interval_ms
    cursor = open_time
    bucket: List[Mapping[str, float]] = []

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


def build_htf_section(
    symbol: str,
    frames: Mapping[str, Any],
    selection: Mapping[str, Any] | None,
    *,
    fetcher: MinuteFetcher | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Collect high timeframe candles aggregated from 1m data."""

    minute_rows = _normalise_minute_rows(_extract_frame_candles(frames, "1m"))

    if not minute_rows:
        htf_payload = {
            "symbol": symbol,
            "candles": {tf: [] for tf in HTF_TIMEFRAMES},
            "window": None,
            "days": 0,
        }
        dq = {
            "window": None,
            "days": 0,
            "minute_missing_before": 0,
            "minute_missing_after": 0,
            "downloaded_1m": 0,
            "timeframes": {tf: {"missing_before": 0, "missing_after": 0, "expected": 0} for tf in HTF_TIMEFRAMES},
        }
        return htf_payload, dq

    ordered_minutes = sorted(minute_rows)
    minute_start = ordered_minutes[0]
    minute_end = ordered_minutes[-1]

    selection_start = _safe_int(selection.get("start")) if selection else None
    selection_end = _safe_int(selection.get("end")) if selection else None
    if selection_start is None:
        selection_start = minute_start
    if selection_end is None:
        selection_end = minute_end
    if selection_start > selection_end:
        selection_start, selection_end = selection_end, selection_start

    span_ms = max(0, selection_end - selection_start)
    days = max(1, math.ceil(span_ms / MS_IN_DAY)) if span_ms else 1
    total_span_ms = max(days * MS_IN_DAY, MINUTE_INTERVAL_MS)

    fetcher_fn = fetcher or _DEFAULT_MINUTE_FETCHER

    timeframe_ranges: Dict[str, Dict[str, int]] = {}
    minute_window_start = minute_end
    minute_window_end = minute_start

    last_close_candidate = minute_end + MINUTE_INTERVAL_MS

    for tf in HTF_TIMEFRAMES:
        interval_ms = TIMEFRAME_TO_MS.get(tf)
        if interval_ms is None or interval_ms <= 0:
            continue

        last_open = _align_to_interval(last_close_candidate - interval_ms, interval_ms)
        if last_open < 0:
            last_open = 0
        bars_required = max(1, math.ceil(total_span_ms / interval_ms))
        start_open = last_open - (bars_required - 1) * interval_ms
        if start_open < 0:
            start_open = 0
        if last_open < start_open:
            last_open = start_open
        effective_bars = ((last_open - start_open) // interval_ms) + 1 if last_open >= start_open else 0
        if effective_bars <= 0:
            effective_bars = 0

        timeframe_ranges[tf] = {
            "start": start_open,
            "end": last_open,
            "interval": interval_ms,
            "bars": effective_bars,
        }

        window_start_candidate = start_open
        window_end_candidate = last_open + interval_ms - MINUTE_INTERVAL_MS
        minute_window_start = min(minute_window_start, window_start_candidate)
        minute_window_end = max(minute_window_end, window_end_candidate)

    if not timeframe_ranges:
        empty_htf = {tf: [] for tf in HTF_TIMEFRAMES}
        empty_quality = {
            "window": None,
            "days": days,
            "minute_missing_before": 0,
            "minute_missing_after": 0,
            "downloaded_1m": 0,
            "timeframes": {tf: {"start_ms": None, "end_ms": None, "expected": 0, "missing_before": 0, "missing_after": 0} for tf in HTF_TIMEFRAMES},
        }
        return {
            "symbol": symbol,
            "window": None,
            "days": days,
            "candles": empty_htf,
        }, empty_quality

    minute_window_start = max(0, _align_to_interval(minute_window_start, MINUTE_INTERVAL_MS))
    minute_window_end = max(minute_window_start, _align_to_interval(minute_window_end, MINUTE_INTERVAL_MS))
    if minute_window_end < minute_window_start:
        minute_window_end = minute_window_start

    expected_minutes = _expected_minute_sequence(minute_window_start, minute_window_end)
    gaps_before = _summarise_missing_minutes(expected_minutes, minute_rows)
    minute_missing_before = sum(gap["count"] for gap in gaps_before)

    minute_rows_initial = dict(minute_rows)

    downloaded_unique = 0
    if gaps_before:
        downloaded_unique += _download_missing_minutes(
            symbol,
            gaps_before,
            fetcher=fetcher_fn,
            target=minute_rows,
        )

    gaps_after = _summarise_missing_minutes(expected_minutes, minute_rows)
    minute_missing_after = sum(gap["count"] for gap in gaps_after)

    candles_by_tf: Dict[str, List[Dict[str, float]]] = {}
    dq_timeframes: Dict[str, Dict[str, Any]] = {}

    for tf, spec in timeframe_ranges.items():
        start_open = spec["start"]
        last_open = spec["end"]
        interval_ms = spec["interval"]
        expected = spec["bars"]

        candles: List[Dict[str, float]] = []
        missing_before = 0
        missing_after = 0

        cursor = start_open
        while cursor <= last_open:
            if _aggregate_from_minutes(minute_rows_initial, cursor, interval_ms) is None:
                missing_before += 1
            cursor += interval_ms

        cursor = start_open
        while cursor <= last_open:
            candle = _aggregate_from_minutes(minute_rows, cursor, interval_ms)
            if candle is None:
                missing_after += 1
            else:
                candles.append(candle)
            cursor += interval_ms

        candles.sort(key=lambda item: item["t"])
        candles_by_tf[tf] = candles
        dq_timeframes[tf] = {
            "start_ms": start_open,
            "end_ms": last_open,
            "expected": expected,
            "missing_before": missing_before,
            "missing_after": missing_after,
        }

    for tf in HTF_TIMEFRAMES:
        if tf not in candles_by_tf:
            candles_by_tf[tf] = []
        if tf not in dq_timeframes:
            dq_timeframes[tf] = {
                "start_ms": None,
                "end_ms": None,
                "expected": 0,
                "missing_before": 0,
                "missing_after": 0,
            }

    htf_payload = {
        "symbol": symbol,
        "window": {"start_ms": minute_window_start, "end_ms": minute_window_end},
        "days": days,
        "candles": candles_by_tf,
    }

    dq = {
        "window": {"start_ms": minute_window_start, "end_ms": minute_window_end},
        "days": days,
        "minute_missing_before": minute_missing_before,
        "minute_missing_after": minute_missing_after,
        "downloaded_1m": downloaded_unique,
        "timeframes": dq_timeframes,
    }

    return htf_payload, dq


def _ensure_storage_dir() -> None:
    """Backward compatible no-op for legacy callers.

    The new runtime avoids persisting snapshots to disk entirely, therefore this
    helper simply exists to keep the old call-sites intact without creating any
    directories on the filesystem.
    """
    return None


def _snapshot_path(snapshot_id: str) -> Path:
    safe_id = _SNAPSHOT_ID_SANITISER.sub("_", snapshot_id or "snapshot")
    return SNAPSHOT_STORAGE_DIR / f"{safe_id}.json"


def _persist_snapshot(snapshot: Snapshot) -> None:
    """Retained for compatibility but intentionally does nothing.

    Historically snapshots were serialised to the local filesystem. The new
    workflow performs all analytics immediately and avoids long-lived storage,
    so the persistence hook became redundant. The helper now simply keeps the
    call-sites valid while guaranteeing that no files are ever written.
    """
    return None


def _remove_snapshot_file(snapshot_id: str) -> None:
    """Filesystem clean-up hook preserved as a harmless no-op."""
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_snapshot_limit() -> None:
    while len(_SNAPSHOT_STORE) > _MAX_STORED_SNAPSHOTS:
        _SNAPSHOT_STORE.popitem(last=False)


def _load_existing_snapshots() -> None:
    """Legacy shim kept for import side-effects.

    Snapshots are no longer loaded from disk which keeps the start-up path fast
    and avoids resurrecting stale data. The ordered dictionary remains empty
    until new runtime payloads are supplied explicitly by the caller.
    """
    _SNAPSHOT_STORE.clear()


_load_existing_snapshots()


def build_placeholder_snapshot(*, symbol: str = DEFAULT_SYMBOL, timeframe: str = "1m") -> Snapshot:
    """Create a synthetic snapshot used to populate the inspection UI by default."""

    timeframe_key = timeframe.lower()
    if timeframe_key not in TIMEFRAME_TO_MS:
        timeframe_key = "1m"
    interval_ms = TIMEFRAME_TO_MS[timeframe_key]

    total_candles = 120
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - (total_candles - 1) * interval_ms

    base_price = 100_000.0
    candles = []
    rolling_price = base_price
    for idx in range(total_candles):
        ts = start_ms + idx * interval_ms
        wave = math.sin(idx / 6.0) * 140.0
        drift = idx * 6.5
        open_price = rolling_price + wave + drift
        close_variation = math.sin(idx / 3.5) * 60.0 + math.cos(idx / 5.0) * 35.0
        close_price = max(1.0, open_price + close_variation)
        high_price = max(open_price, close_price) + abs(math.sin(idx / 4.5)) * 55.0
        low_price = min(open_price, close_price) - abs(math.cos(idx / 3.8)) * 55.0
        volume = 180.0 + abs(math.sin(idx / 4.2)) * 90.0

        candles.append(
            {
                "t": ts,
                "o": round(open_price, 2),
                "h": round(high_price, 2),
                "l": round(low_price, 2),
                "c": round(close_price, 2),
                "v": round(volume, 2),
            }
        )
        rolling_price = close_price

    selection = None
    if candles:
        window = min(40, len(candles))
        selection = {"start": candles[-window]["t"], "end": candles[-1]["t"]}

    agg_trades: List[Dict[str, Any]] = []
    sample = candles[-80:] if candles else []
    for candle in sample:
        ts = int(candle["t"])
        close = float(candle.get("c", candle.get("o", 0.0)))
        open_ = float(candle.get("o", close))
        qty = max(0.01, abs(close - open_) / max(1.0, interval_ms / 60_000))
        trade_time = ts + interval_ms // 2
        agg_trades.append({
            "t": trade_time,
            "p": round(close, 2),
            "q": round(qty, 4),
            "side": "buy" if close >= open_ else "sell",
        })

    agg_payload = {
        "symbol": symbol.upper(),
        "agg": agg_trades,
    }

    return {
        "id": "placeholder",
        "symbol": symbol.upper(),
        "tf": timeframe_key,
        "frames": {timeframe_key: {"tf": timeframe_key, "candles": candles}},
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "selection": selection,
        "agg_trades": agg_payload,
        "meta": {"source": {"kind": "placeholder", "generated": True}},
    }



def _in_session(moment: dtime, start: dtime, end: dtime) -> bool:
    if start <= end:
        return start <= moment < end
    return moment >= start or moment < end


def _build_sigma_levels(center: float, sigma: float) -> List[Dict[str, float]]:
    return [
        {"k": k, "price_minus": center - sigma * k, "price_plus": center + sigma * k}
        for k in (1, 2)
    ]


def _compute_vwap_stats(entries: Iterable[Mapping[str, Any]]) -> Tuple[float, float] | None:
    total_pv = 0.0
    total_p2v = 0.0
    total_volume = 0.0
    valid = 0
    for entry in entries:
        high = float(entry.get("h", entry.get("high", 0.0)))
        low = float(entry.get("l", entry.get("low", 0.0)))
        close = float(entry.get("c", entry.get("close", 0.0)))
        volume = float(entry.get("v", entry.get("volume", 0.0)))
        if volume <= 0.0:
            continue
        typical_price = (high + low + close) / 3.0
        if not math.isfinite(typical_price):
            continue
        total_pv += typical_price * volume
        total_p2v += typical_price * typical_price * volume
        total_volume += volume
        valid += 1
    if total_volume <= 0.0:
        return None
    vwap_value = total_pv / total_volume
    if valid < 2:
        sigma_value = 0.0
    else:
        variance = max(total_p2v / total_volume - vwap_value * vwap_value, 0.0)
        sigma_value = math.sqrt(variance)
    return vwap_value, sigma_value


def compute_session_vwaps(symbol: str, candles: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute VWAP for daily and configured sessions across recent days."""

    if not candles:
        return {"symbol": symbol.upper(), "vwap": []}

    bars: List[Tuple[int, float, float, float, float]] = []
    for candle in candles:
        if not isinstance(candle, Mapping):
            continue
        raw_ts = (
            candle.get("t")
            or candle.get("time")
            or candle.get("openTime")
        )
        if raw_ts is None:
            continue
        try:
            open_ms = int(raw_ts)
            high = float(candle.get("h", candle.get("high", 0.0)))
            low = float(candle.get("l", candle.get("low", 0.0)))
            close = float(candle.get("c", candle.get("close", 0.0)))
            volume = float(candle.get("v", candle.get("volume", 0.0)))
        except (TypeError, ValueError):
            continue
        bars.append((open_ms, high, low, close, volume))

    if not bars:
        return {"symbol": symbol.upper(), "vwap": []}

    bars.sort(key=lambda item: item[0])
    last_date = datetime.fromtimestamp(bars[-1][0] / 1000.0, tz=timezone.utc).date()
    lookback = max(1, int(Meta.VWAP_LOOKBACK_DAYS))
    start_date = last_date - timedelta(days=lookback - 1)
    sessions = list(Meta.iter_vwap_sessions())

    daily_buckets: defaultdict = defaultdict(list)  # type: ignore[var-annotated]
    session_buckets: DefaultDict[
        Tuple[datetime.date, str], List[Mapping[str, Any]]
    ] = defaultdict(list)
    session_extrema: Dict[Tuple[datetime.date, str], Tuple[float, float]] = {}

    for open_ms, high, low, close, volume in bars:
        dt = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc)
        if dt.date() < start_date:
            continue
        entry = {"h": high, "l": low, "c": close, "v": volume}
        daily_buckets[dt.date()].append(entry)
        moment = dt.time()
        for session_name, start_time, end_time in sessions:
            if not _in_session(moment, start_time, end_time):
                continue
            bucket_key = (dt.date(), session_name)
            session_buckets[bucket_key].append(entry)
            high_value = float(entry["h"])
            low_value = float(entry["l"])
            if bucket_key in session_extrema:
                prev_high, prev_low = session_extrema[bucket_key]
                high_value = max(prev_high, high_value)
                low_value = min(prev_low, low_value)
            session_extrema[bucket_key] = (high_value, low_value)

    ordered_dates = sorted(daily_buckets.keys())
    if len(ordered_dates) > lookback:
        ordered_dates = ordered_dates[-lookback:]

    results: List[Dict[str, object]] = []
    sigma_results: List[Dict[str, object]] = []
    for date_key in ordered_dates:
        daily_stats = _compute_vwap_stats(daily_buckets[date_key])
        if daily_stats is None:
            continue
        daily_value, daily_sigma = daily_stats
        date_str = date_key.isoformat()
        results.append({"date": date_str, "session": "daily", "value": daily_value})
        sigma_results.append(
            {
                "date": date_str,
                "session": "daily",
                "basis": "daily",
                "sigma": _build_sigma_levels(daily_value, daily_sigma),
            }
        )
        for session_name, _, _ in sessions:
            entries = session_buckets.get((date_key, session_name))
            if not entries:
                continue
            session_stats = _compute_vwap_stats(entries)
            if session_stats is None:
                continue
            session_value, session_sigma = session_stats
            result_entry = {
                "date": date_str,
                "session": session_name,
                "value": session_value,
            }
            extrema = session_extrema.get((date_key, session_name))
            if extrema is not None:
                session_high, session_low = extrema
                result_entry["session_high"] = session_high
                result_entry["session_low"] = session_low
            results.append(result_entry)
            sigma_results.append(
                {
                    "date": date_str,
                    "session": session_name,
                    "basis": "session",
                    "sigma": _build_sigma_levels(session_value, session_sigma),
                }
            )

    return {"symbol": symbol.upper(), "vwap": results, "vwap_sigma": sigma_results}
def _coerce_frame(tf_key: str, frame: Mapping[str, Any] | Sequence[Any]) -> Dict[str, Any]:
    if tf_key not in TIMEFRAME_WINDOWS:
        raise ValueError(f"Unsupported timeframe: {tf_key}")

    if isinstance(frame, Mapping):
        raw_candles = frame.get("candles", [])
    else:
        raw_candles = frame

    try:
        candles = list(raw_candles)  # type: ignore[arg-type]
    except TypeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Frame candles must be iterable") from exc

    return {"tf": tf_key, "candles": candles}


def _extract_frames(snapshot: Mapping[str, Any], primary_tf: str) -> Dict[str, Dict[str, Any]]:
    frames: Dict[str, Dict[str, Any]] = {}
    raw_frames = snapshot.get("frames")
    if isinstance(raw_frames, Mapping):
        for key, frame in raw_frames.items():
            tf_value = None
            if isinstance(frame, Mapping):
                tf_value = frame.get("tf")
            tf_key = str(tf_value or key or primary_tf).lower()
            frames[tf_key] = _coerce_frame(tf_key, frame)  # type: ignore[arg-type]
    elif "candles" in snapshot:
        try:
            candles = list(snapshot["candles"])  # type: ignore[index]
        except TypeError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Snapshot candles must be iterable") from exc
        frames[primary_tf] = {"tf": primary_tf, "candles": candles}
    else:
        raise ValueError("Snapshot must include candles or frames")

    if not frames:
        raise ValueError("Snapshot did not include any frames")

    return frames


def register_snapshot(snapshot: Mapping[str, Any]) -> str:
    """Store a snapshot captured by the chart frontend or inspection UI."""

    symbol = str(snapshot.get("symbol") or snapshot.get("ticker") or "UNKNOWN").upper()
    primary_tf = str(snapshot.get("tf") or snapshot.get("timeframe") or "1m").lower()
    snapshot_id = str(
        snapshot.get("id")
        or snapshot.get("snapshot_id")
        or snapshot.get("snapshot")
        or f"snap-{int(datetime.now(timezone.utc).timestamp()*1000)}"
    )

    frames = _extract_frames(snapshot, primary_tf)
    if primary_tf not in frames:
        primary_tf = next(iter(frames))

    meta: MutableMapping[str, Any] = {}
    for key in ("meta", "diagnostics", "source"):
        value = snapshot.get(key)
        if isinstance(value, Mapping):
            meta[key] = dict(value)

    primary_frame = frames.get(primary_tf, {})
    t_values: List[int] = []
    if isinstance(primary_frame, Mapping):
        raw_candles = primary_frame.get("candles")
        if isinstance(raw_candles, Sequence):
            for candle in raw_candles:
                ts: int | None = None
                if isinstance(candle, Mapping):
                    ts = _safe_int(
                        candle.get("t")
                        or candle.get("time")
                        or candle.get("openTime")
                        or candle.get("open_time")
                    )
                elif isinstance(candle, Sequence) and candle:
                    ts = _safe_int(candle[0])
                if ts is not None:
                    t_values.append(ts)
    if t_values:
        t_values.sort()
        meta["t_first"] = t_values[0]
        meta["t_last"] = t_values[-1]

    existing_source = meta.get("source")
    if isinstance(existing_source, Mapping):
        meta["source_details"] = dict(existing_source)
    meta["source"] = "futures"
    meta["market"] = "USDT-M Futures"

    selection = snapshot.get("selection") if isinstance(snapshot.get("selection"), Mapping) else None
    selection_data = None
    if selection is not None:
        try:
            selection_data = {
                "start": int(selection.get("start", 0)),
                "end": int(selection.get("end", 0)),
            }
        except (TypeError, ValueError):
            selection_data = None

    stored: Snapshot = {
        "id": snapshot_id,
        "symbol": symbol,
        "tf": primary_tf,
        "frames": frames,
        "captured_at": snapshot.get("captured_at") or _now_iso(),
        "meta": meta,
    }

    if selection_data:
        stored["selection"] = selection_data

    for key in ("delta", "vwap", "zones", "smt", "agg_trades"):
        if key in snapshot:
            stored[key] = snapshot[key]

    _SNAPSHOT_STORE[snapshot_id] = stored
    _SNAPSHOT_STORE.move_to_end(snapshot_id)
    _persist_snapshot(stored)
    _ensure_snapshot_limit()
    return snapshot_id


def get_snapshot(snapshot_id: str) -> Snapshot | None:
    """Return a stored snapshot if present."""

    snapshot = _SNAPSHOT_STORE.get(snapshot_id)
    if snapshot is not None:
        _SNAPSHOT_STORE.move_to_end(snapshot_id)
    return snapshot


def list_snapshots() -> List[Dict[str, Any]]:
    """Return metadata about stored snapshots ordered from newest to oldest."""

    entries: List[Dict[str, Any]] = []
    for snapshot in reversed(_SNAPSHOT_STORE.values()):
        entries.append(
            {
                "id": snapshot.get("id"),
                "symbol": snapshot.get("symbol"),
                "tf": snapshot.get("tf"),
                "captured_at": snapshot.get("captured_at"),
                "selection": snapshot.get("selection"),
            }
        )
    return entries


def _filter_by_selection(
    candles: Sequence[Mapping[str, Any]],
    *,
    start: int | None,
    end: int | None,
) -> List[Dict[str, Any]]:
    if start is None and end is None:
        return [dict(candle) for candle in candles]
    filtered: List[Dict[str, Any]] = []
    for candle in candles:
        ts = int(candle.get("t", 0))
        if start is not None and ts < start:
            continue
        if end is not None and ts > end:
            continue
        filtered.append(dict(candle))
    return filtered


def _compute_delta_series(candles: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
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


def build_inspection_payload(snapshot: Snapshot) -> Dict[str, Any]:
    """Build a combined inspection payload from a stored snapshot."""

    symbol = snapshot.get("symbol", "UNKNOWN")
    frames: Mapping[str, Mapping[str, Any]] = snapshot.get("frames", {})  # type: ignore[assignment]
    selection = snapshot.get("selection") if isinstance(snapshot.get("selection"), Mapping) else None
    start = int(selection.get("start")) if selection and selection.get("start") else None
    end = int(selection.get("end")) if selection and selection.get("end") else None

    htf_section, htf_quality = build_htf_section(symbol, frames, selection)

    normalised_frames: Dict[str, Dict[str, Any]] = {}
    diagnostics_frames: Dict[str, Any] = {}
    delta_frames: Dict[str, List[Dict[str, Any]]] = {}
    vwap_frames: Dict[str, Dict[str, Any]] = {}
    full_candles_by_tf: Dict[str, List[Mapping[str, Any]]] = {}
    liquidity_sources: Dict[str, str] = {}

    for tf_key, frame in frames.items():
        candles = frame.get("candles", []) if isinstance(frame, Mapping) else []
        if not isinstance(candles, Sequence):
            try:
                candles = list(candles)  # type: ignore[arg-type]
            except TypeError:
                candles = []

        result = normalise_ohlcv(
            symbol,
            tf_key,
            candles,
            include_diagnostics=True,
            use_full_span=(tf_key == "1m"),
        )
        diagnostics = result.pop("diagnostics", {})

        raw_candles_seq = result.get("candles", [])
        raw_candles = [
            candle
            for candle in raw_candles_seq
            if isinstance(candle, Mapping)
        ]
        full_candles_by_tf[tf_key] = raw_candles
        if tf_key == "1m":
            liquidity_sources[tf_key] = "minute"
        elif tf_key == "1d":
            liquidity_sources.setdefault(tf_key, "frame")

        filtered_candles = _filter_by_selection(raw_candles, start=start, end=end)
        result["candles"] = filtered_candles

        if isinstance(diagnostics, Mapping):
            diagnostics_series = diagnostics.get("series") if isinstance(diagnostics.get("series"), Sequence) else []
            diagnostics_missing = diagnostics.get("missing_bars") if isinstance(diagnostics.get("missing_bars"), Sequence) else []
            diagnostics = dict(diagnostics)
            diagnostics["series"] = _filter_by_selection(diagnostics_series, start=start, end=end)
            diagnostics["missing_bars"] = _filter_by_selection(diagnostics_missing, start=start, end=end)
        else:
            diagnostics = {}

        normalised_frames[tf_key] = result
        diagnostics_frames[tf_key] = diagnostics
        delta_frames[tf_key] = _compute_delta_series(filtered_candles)
        vwap_frames[tf_key] = {
            "selection": {"start": start, "end": end},
            "value": _compute_vwap(filtered_candles),
        }

    htf_candles_map = (
        htf_section.get("candles")
        if isinstance(htf_section, Mapping) and isinstance(htf_section.get("candles"), Mapping)
        else {}
    )
    if isinstance(htf_candles_map, Mapping):
        for tf_key in ("15m", "1h", "1d"):
            series = htf_candles_map.get(tf_key)
            if not isinstance(series, Sequence):
                continue
            cleaned = [c for c in series if isinstance(c, Mapping)]
            if not cleaned:
                continue
            full_candles_by_tf[tf_key] = cleaned
            liquidity_sources[tf_key] = "htf"

    generated_frames = ensure_higher_timeframes(full_candles_by_tf)
    for tf_key, candles in generated_frames.items():
        filtered_candles = _filter_by_selection(candles, start=start, end=end)
        frame_payload: Dict[str, Any] = {
            "symbol": symbol,
            "tf": tf_key,
            "candles": filtered_candles,
        }
        if candles:
            last_candle = candles[-1]
            frame_payload["last_price"] = _coerce_float(last_candle.get("c"))
            frame_payload["last_ts"] = last_candle.get("t")
        normalised_frames[tf_key] = frame_payload
        diagnostics_frames.setdefault(tf_key, {})
        delta_frames[tf_key] = _compute_delta_series(filtered_candles)
        vwap_frames[tf_key] = {
            "selection": {"start": start, "end": end},
            "value": _compute_vwap(filtered_candles),
        }
        full_candles_by_tf[tf_key] = candles
        if liquidity_sources.get(tf_key) != "htf":
            liquidity_sources[tf_key] = "aggregated"

    for tf_key in ("15m", "1h"):
        candles = full_candles_by_tf.get(tf_key)
        if not candles:
            continue
        filtered_candles = _filter_by_selection(candles, start=start, end=end)
        frame_payload: Dict[str, Any] = {
            "symbol": symbol,
            "tf": tf_key,
            "candles": filtered_candles,
        }
        last_candle = candles[-1]
        frame_payload["last_price"] = _coerce_float(last_candle.get("c")) if isinstance(last_candle, Mapping) else None
        frame_payload["last_ts"] = last_candle.get("t") if isinstance(last_candle, Mapping) else None
        normalised_frames[tf_key] = frame_payload
        diagnostics_frames.setdefault(tf_key, {})
        delta_frames[tf_key] = _compute_delta_series(filtered_candles)
        vwap_frames[tf_key] = {
            "selection": {"start": start, "end": end},
            "value": _compute_vwap(filtered_candles),
        }

    profile_config = resolve_profile_config(symbol, snapshot.get("meta"))
    preset = profile_config["preset"]
    raw_profile_defaults = profile_config.get("raw_defaults")
    preset_payload = profile_config.get("preset_payload")
    preset_required = profile_config.get("preset_required", False)
    target_tf_key = profile_config.get("target_tf_key", "1m")

    base_candles = normalised_frames.get(target_tf_key, {}).get("candles", [])
    if not base_candles:
        base_candles = normalised_frames.get("1m", {}).get("candles", [])
    if not base_candles and normalised_frames:
        first_key = next(iter(normalised_frames))
        base_candles = normalised_frames[first_key].get("candles", [])

    session_vwap = compute_session_vwaps(symbol, base_candles)

    sessions = list(Meta.iter_vwap_sessions())
    tpo_entries: List[Dict[str, object]] = []
    tpo_zone_items: List[Dict[str, Any]] = []
    flattened_profile: List[Dict[str, float]] = []
    detected_zones: Dict[str, Any] = {
        "zones": {
            "fvg": [],
            "ob": [],
            "mb": [],
            "bb": [],
            "rb": [],
            "pb": [],
            "sr": [],
            "profile_levels": [],
        },
        "meta": {},
    }
    profile_candles: List[Dict[str, Any]] = []

    tick_size_value = profile_config.get("tick_size")
    adaptive_bins_flag = bool(profile_config.get("adaptive_bins", True))
    value_area_pct = float(profile_config.get("value_area_pct", 0.7))
    profile_last_n = int(profile_config.get("last_n", 3))
    atr_multiplier = float(profile_config.get("atr_multiplier", 0.5))
    target_bins = int(profile_config.get("target_bins", 80))
    clip_threshold = float(profile_config.get("clip_threshold", 0.0))
    smooth_window = int(profile_config.get("smooth_window", 1))

    if base_candles:
        closed_candles = [
            candle
            for candle in base_candles
            if bool(candle.get("closed", True))
        ]
        profile_candles = closed_candles or list(base_candles)

    profile_ready = bool(profile_candles)

    if preset and profile_candles and sessions:
        cache_token = (
            "inspection",
            snapshot.get("id"),
            symbol,
            target_tf_key,
        )
        try:
            (tpo_entries, flattened_profile, tpo_zone_items) = build_profile_package(
                profile_candles,
                sessions=sessions,
                last_n=profile_last_n,
                tick_size=tick_size_value,
                adaptive_bins=adaptive_bins_flag,
                value_area_pct=value_area_pct,
                atr_multiplier=atr_multiplier,
                target_bins=target_bins,
                clip_threshold=clip_threshold,
                smooth_window=smooth_window,
                cache_token=cache_token,
                tf_key=target_tf_key,
            )
        except Exception:  # pragma: no cover - defensive guard against upstream errors
            logging.getLogger(__name__).exception(
                "Failed to build profile package for inspection payload",
                extra={
                    "snapshot_id": snapshot.get("id"),
                    "symbol": symbol,
                    "timeframe": target_tf_key,
                },
            )
            tpo_entries = []
            flattened_profile = []
            tpo_zone_items = []
            detected_zones = {
                "zones": {
                    "fvg": [],
                    "ob": [],
                    "mb": [],
                    "bb": [],
                    "rb": [],
                    "pb": [],
                    "sr": [],
                    "profile_levels": [],
                },
                "meta": {},
            }
            profile_ready = False

    if profile_ready and profile_candles:
        try:
            zone_cfg = ZonesConfig(tick_size=tick_size_value)
            zone_frames: Dict[str, Sequence[Mapping[str, Any]]] = {
                target_tf_key: profile_candles
            }
            detected_zones = detect_zones(
                frames=zone_frames,
                config=zone_cfg,
            )
        except Exception:  # pragma: no cover - defensive guard
            logging.getLogger(__name__).exception(
                "Failed to detect zones for inspection payload",
                extra={
                    "snapshot_id": snapshot.get("id"),
                    "symbol": symbol,
                    "timeframe": target_tf_key,
                },
            )
            detected_zones = {
                "zones": {
                    "fvg": [],
                    "ob": [],
                    "mb": [],
                    "bb": [],
                    "rb": [],
                    "pb": [],
                    "sr": [],
                    "profile_levels": [],
                },
                "meta": {},
            }

    raw_meta = snapshot.get("meta") if isinstance(snapshot.get("meta"), Mapping) else {}
    liquidity_config = raw_meta.get("liquidity") if isinstance(raw_meta, Mapping) else None
    tick_size, tick_size_source = resolve_liquidity_tick_size(
        symbol,
        tick_size_value,
        full_candles_by_tf,
        meta=raw_meta,
        logger=logging.getLogger(__name__),
    )
    liquidity_frames: Dict[str, Dict[str, Any]] = {}
    for tf, candles in full_candles_by_tf.items():
        payload: Dict[str, Any] = {"candles": list(candles)}
        source_label = liquidity_sources.get(tf)
        if source_label:
            payload["source"] = source_label
        liquidity_frames[tf] = payload
    liquidity_payload = build_liquidity_snapshot(
        liquidity_frames,
        symbol=symbol,
        tick_size=tick_size,
        meta=raw_meta,
        selection=selection,
        config=liquidity_config if isinstance(liquidity_config, Mapping) else None,
    )

    liquidity_diagnostics: Dict[str, Any] = {}
    if isinstance(liquidity_payload, Mapping):
        diagnostics_payload = liquidity_payload.get("diagnostics")
        if isinstance(diagnostics_payload, Mapping):
            liquidity_diagnostics = diagnostics_payload
        liquidity_payload = dict(liquidity_payload)
        liquidity_payload.pop("diagnostics", None)
    else:
        liquidity_payload = {
            "eqh": [],
            "eql": [],
            "pdh": None,
            "pdl": None,
            "sweeps": [],
        }

    logging.getLogger(__name__).debug(
        "Liquidity tick size applied",
        extra={
            "symbol": symbol,
            "normalized_symbol": normalise_symbol_for_tick(symbol) or "UNKNOWN",
            "tick_size": tick_size,
            "tick_size_source": tick_size_source,
        },
    )

    minute_htf_source: Sequence[Mapping[str, Any]] = []
    minute_frame_payload = normalised_frames.get("1m")
    has_minute_frame = isinstance(minute_frame_payload, Mapping)
    if has_minute_frame:
        minute_series = minute_frame_payload.get("candles")
        if isinstance(minute_series, Sequence):
            minute_htf_source = [
                candle
                for candle in minute_series
                if isinstance(candle, Mapping)
            ]  # type: ignore[list-item]

    hourly_htf = aggregate_1m_to_1h(minute_htf_source) if has_minute_frame else []
    htf_blocks = []
    if has_minute_frame:
        htf_blocks.append({"tf": "1h", "candles": hourly_htf})

    data_section = {
        "symbol": symbol,
        "frames": normalised_frames,
        "selection": selection,
        "htf": htf_blocks,
        "htf_details": htf_section,
        "session_vwap": session_vwap,
        "tpo": {"sessions": tpo_entries, "zones": tpo_zone_items},
        "profile": flattened_profile,
        "zones": detected_zones,
        "zones_raw": snapshot.get("zones"),
        "agg_trades": snapshot.get("agg_trades")
        or {
            "status": "unavailable",
            "detail": "Agg trade data is not present in the snapshot.",
        },
        "delta_cvd": delta_frames,
        "vwap_tpo": vwap_frames,
        "smt": snapshot.get("smt")
        or {
            "status": "unavailable",
            "detail": "SMT provider is not configured in the snapshot.",
        },
        "liquidity": liquidity_payload,
        "profile_preset": preset_payload,
        "profile_preset_required": bool(preset_required),
        "profile_defaults": raw_profile_defaults,
        "meta": {
            "requested": {
                "symbol": symbol,
                "frames": sorted(normalised_frames),
            },
            "source": snapshot.get("meta", {}),
            "data_quality_htf": htf_quality,
        },
    }

    diagnostics_section = {
        "generated_at": _now_iso(),
        "snapshot_id": snapshot.get("id"),
        "captured_at": snapshot.get("captured_at"),
        "frames": diagnostics_frames,
        "liquidity": liquidity_diagnostics,
    }

    return {"DATA": data_section, "DIAGNOSTICS": diagnostics_section}


def render_inspection_page(
    payload: Dict[str, Any],
    *,
    snapshot_id: str | None,
    symbol: str,
    timeframe: str,
    snapshots: List[Dict[str, Any]],
) -> str:
    """Render the simplified diagnostics dashboard without snapshot storage."""

    def _format_json_block(value: Any) -> str:
        try:
            formatted = json.dumps(
                value if value is not None else None,
                ensure_ascii=False,
                indent=2,
            )
        except (TypeError, ValueError):
            formatted = json.dumps(None, ensure_ascii=False, indent=2)
        return html_utils.escape(formatted)

    data_section = payload.get("DATA") if isinstance(payload, Mapping) else None
    diagnostics_section = payload.get("DIAGNOSTICS") if isinstance(payload, Mapping) else None
    check_section = None
    if isinstance(payload, Mapping):
        if isinstance(payload.get("CHECK_ALL_DATAS"), Mapping):
            check_section = payload.get("CHECK_ALL_DATAS")
        elif isinstance(data_section, Mapping):
            check_section = data_section

    interval_info: Dict[str, Any] = {}
    interval_raw = payload.get("interval") if isinstance(payload, Mapping) else None
    if isinstance(interval_raw, Mapping):
        interval_info = dict(interval_raw)

    def _format_interval_value(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not value.is_integer():
                return html_utils.escape(f"{value:.3f}")
            return html_utils.escape(str(int(value)))
        return html_utils.escape(str(value))

    interval_from = _format_interval_value(interval_info.get("from_utc"))
    interval_to = _format_interval_value(interval_info.get("to_utc"))
    interval_completed = _format_interval_value(interval_info.get("last_completed_ts"))
    interval_run = _format_interval_value(interval_info.get("last_run_ts"))

    check_json_initial = _format_json_block(check_section)
    diagnostics_json_initial = _format_json_block(diagnostics_section)

    initial_state_payload = {
        "check_all": check_section,
        "diagnostics": diagnostics_section,
        "interval": interval_info,
        "snapshot": snapshot_id,
        "symbol": symbol,
        "timeframe": timeframe,
    }
    initial_state_json = html_utils.escape(
        json.dumps(initial_state_payload, ensure_ascii=False)
    )

    style_block = """
    :root {{
      color-scheme: dark;
      --bg: #0f172a;
      --fg: #e2e8f0;
      --muted: #94a3b8;
      --border: rgba(148, 163, 184, 0.28);
      --accent: #38bdf8;
      --accent-strong: #0ea5e9;
      --panel: rgba(15, 23, 42, 0.88);
      --error: #f87171;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Inter", "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(circle at 15% -10%, #1e293b 0%, #0f172a 45%, #020617 100%);
      color: var(--fg);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }}
    header {{
      padding: 2.5rem 1.5rem 1.5rem;
      max-width: 1100px;
      width: 100%;
      margin: 0 auto;
    }}
    header h1 {{
      margin: 0 0 0.5rem;
      font-size: clamp(1.9rem, 2.8vw, 2.6rem);
    }}
    header p {{
      margin: 0;
      color: var(--muted);
      max-width: 720px;
    }}
    main {{
      width: min(1100px, 95vw);
      margin: 0 auto 3rem;
      flex: 1;
      display: grid;
      gap: 1.5rem;
      grid-template-columns: minmax(280px, 320px) 1fr;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.4rem;
      display: flex;
      flex-direction: column;
      gap: 1rem;
      box-shadow: 0 24px 48px rgba(8, 47, 73, 0.35);
    }}
    .panel h2 {{
      margin: 0;
      font-size: 0.82rem;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: rgba(148, 163, 184, 0.9);
    }}
    .actions {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 0.6rem;
    }}
    button {{
      cursor: pointer;
      border: none;
      border-radius: 999px;
      padding: 0.65rem 1rem;
      font-weight: 600;
      font-size: 0.95rem;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 0.4rem;
      transition: transform 0.18s ease, box-shadow 0.18s ease, opacity 0.2s ease;
    }}
    button.primary {{
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
      color: #0f172a;
      box-shadow: 0 14px 36px rgba(56, 189, 248, 0.32);
    }}
    button.secondary {{
      background: rgba(148, 163, 184, 0.18);
      color: var(--fg);
    }}
    button:hover:not(:disabled) {{
      transform: translateY(-1px);
      box-shadow: 0 12px 28px rgba(14, 165, 233, 0.28);
    }}
    button:disabled {{
      opacity: 0.6;
      cursor: not-allowed;
      box-shadow: none;
    }}
    .loader {{
      font-size: 0.85rem;
      color: var(--muted);
    }}
    .error-banner {{
      border-radius: 12px;
      padding: 0.6rem 0.75rem;
      background: rgba(248, 113, 113, 0.12);
      color: var(--error);
      font-size: 0.85rem;
    }}
    dl.interval {{
      margin: 0;
      display: grid;
      gap: 0.75rem;
    }}
    dl.interval div {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 0.45rem 0.9rem;
      align-items: baseline;
    }}
    dl.interval dt {{
      font-size: 0.72rem;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: rgba(148, 163, 184, 0.72);
    }}
    dl.interval dd {{
      margin: 0;
      font-family: "JetBrains Mono", "SFMono-Regular", ui-monospace, monospace;
      font-size: 0.9rem;
      color: var(--fg);
    }}
    .view-switch {{
      display: inline-flex;
      gap: 0.5rem;
      background: rgba(15, 23, 42, 0.7);
      padding: 0.3rem;
      border-radius: 999px;
      border: 1px solid rgba(148, 163, 184, 0.28);
    }}
    .view-tab {{
      padding: 0.45rem 1.05rem;
      border-radius: 999px;
      background: transparent;
      color: var(--muted);
    }}
    .view-tab.active {{
      background: rgba(148, 163, 184, 0.16);
      color: var(--fg);
      box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.35);
    }}
    section.content {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 1.4rem 1.6rem;
      display: flex;
      flex-direction: column;
      gap: 1.2rem;
      box-shadow: 0 26px 50px rgba(8, 47, 73, 0.4);
    }}
    article.view {{
      display: none;
      flex-direction: column;
      gap: 0.9rem;
    }}
    article.view.active {{
      display: flex;
    }}
    article.view header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 0.8rem;
    }}
    article.view h3 {{
      margin: 0;
      font-size: 1rem;
    }}
    pre {{
      margin: 0;
      background: rgba(15, 23, 42, 0.78);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 14px;
      padding: 1rem;
      font-size: 0.85rem;
      overflow-x: auto;
      color: var(--fg);
      font-family: "JetBrains Mono", "SFMono-Regular", ui-monospace, monospace;
      max-height: 520px;
    }}
    .helper-text {{
      font-size: 0.78rem;
      color: var(--muted);
    }}
    @media (max-width: 960px) {{
      main {{
        grid-template-columns: 1fr;
      }}
    }}
    """

    html = f"""<!DOCTYPE html>
<html lang="ru">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Diagnostics &amp; Check All Datas</title>
    <style>{style_block}</style>
  </head>
  <body>
    <header>
      <h1>Diagnostics &amp; Check All Datas</h1>
      <p>
        Интерфейс для мгновенного расчёта отчётов без сохранения снапшотов. Используйте кнопки ниже,
        чтобы пересчитать данные за последние дни или дозагрузить только новый хвост истории.
      </p>
    </header>
    <main>
      <aside class="panel">
        <h2>Сбор данных</h2>
        <div class="actions">
          <button id="action-previous" class="primary" type="button">Check previous 3 days</button>
          <button id="action-resume" class="secondary" type="button">Дособрать данные</button>
        </div>
        <div id="runtime-loader" class="loader" hidden>Сбор данных…</div>
        <div id="runtime-error" class="error-banner" hidden></div>
        <dl class="interval" id="interval-stats">
          <div><dt>from_utc</dt><dd id="interval-from">{interval_from}</dd></div>
          <div><dt>to_utc</dt><dd id="interval-to">{interval_to}</dd></div>
          <div><dt>last_completed_ts</dt><dd id="interval-completed">{interval_completed}</dd></div>
          <div><dt>last_run_ts</dt><dd id="interval-run">{interval_run}</dd></div>
        </dl>
        <section>
          <h2>Просмотр</h2>
          <div class="view-switch">
            <button class="view-tab active" type="button" data-switch-view="check">Check All Datas</button>
            <button class="view-tab" type="button" data-switch-view="diagnostics">Diagnostics</button>
          </div>
        </section>
      </aside>
      <section class="content">
        <article class="view active" data-view="check">
          <header>
            <h3>Результат</h3>
            <span class="helper-text">JSON ответа check_all_datas</span>
          </header>
          <pre id="check-json">{check_json_initial}</pre>
        </article>
        <article class="view" data-view="diagnostics">
          <header>
            <h3>Diagnostics</h3>
            <span class="helper-text">Сводка счётчиков и reject-причин</span>
          </header>
          <pre id="diagnostics-json">{diagnostics_json_initial}</pre>
        </article>
      </section>
    </main>
    <script id="initial-state" type="application/json">{initial_state_json}</script>
    <script>
      (() => {{
        const stateNode = document.getElementById("initial-state");
        let runtimeState = Object.create(null);
        try {{
          runtimeState = JSON.parse(stateNode.textContent || "{{}}") || Object.create(null);
        }} catch (error) {{
          runtimeState = Object.create(null);
        }}

        const viewTabs = Array.from(document.querySelectorAll("[data-switch-view]"));
        const viewMap = new Map(
          Array.from(document.querySelectorAll("[data-view]")).map((node) => [node.dataset.view, node])
        );

        function setActiveView(view) {{
          viewTabs.forEach((btn) => {{
            btn.classList.toggle("active", btn.dataset.switchView === view);
          }});
          viewMap.forEach((node, key) => {{
            node.classList.toggle("active", key === view);
          }});
        }}

        viewTabs.forEach((btn) => {{
          btn.addEventListener("click", () => setActiveView(btn.dataset.switchView || "check"));
        }});

        const loader = document.getElementById("runtime-loader");
        const errorBanner = document.getElementById("runtime-error");
        const intervalFrom = document.getElementById("interval-from");
        const intervalTo = document.getElementById("interval-to");
        const intervalCompleted = document.getElementById("interval-completed");
        const intervalRun = document.getElementById("interval-run");
        const checkNode = document.getElementById("check-json");
        const diagnosticsNode = document.getElementById("diagnostics-json");

        function formatValue(value) {{
          if (value === null || value === undefined || value === "") {{
            return "—";
          }}
          if (typeof value === "number") {{
            if (Number.isFinite(value) && !Number.isInteger(value)) {{
              return value.toFixed(3);
            }}
            return String(Math.trunc(value));
          }}
          return String(value);
        }}

        function updateInterval(info) {{
          const payload = info && typeof info === "object" ? info : Object.create(null);
          if (intervalFrom) intervalFrom.textContent = formatValue(payload.from_utc);
          if (intervalTo) intervalTo.textContent = formatValue(payload.to_utc);
          if (intervalCompleted) intervalCompleted.textContent = formatValue(payload.last_completed_ts);
          if (intervalRun) intervalRun.textContent = formatValue(payload.last_run_ts);
        }}

        function updateOutputs(payload) {{
          if (!payload || typeof payload !== "object") {{
            return;
          }}
          runtimeState = Object.assign(Object.create(null), runtimeState, payload);
          if (checkNode) {{
            try {{
              checkNode.textContent = JSON.stringify(payload.check_all ?? null, null, 2);
            }} catch (error) {{
              checkNode.textContent = "null";
            }}
          }}
          if (diagnosticsNode) {{
            try {{
              diagnosticsNode.textContent = JSON.stringify(payload.diagnostics ?? null, null, 2);
            }} catch (error) {{
              diagnosticsNode.textContent = "null";
            }}
          }}
          updateInterval(payload.interval);
        }}

        function setLoading(active) {{
          if (!loader) return;
          loader.hidden = !active;
        }}

        function showError(message) {{
          if (!errorBanner) return;
          if (!message) {{
            errorBanner.hidden = true;
            errorBanner.textContent = "";
            return;
          }}
          errorBanner.hidden = false;
          errorBanner.textContent = message;
        }}

        async function trigger(action) {{
          setLoading(true);
          showError("");
          try {{
            const response = await fetch("/inspection/runtime/" + action, {{
              method: "POST",
              headers: {{"Content-Type": "application/json"}},
            }});
            if (!response.ok) {{
              throw new Error("Запрос завершился с ошибкой " + response.status);
            }}
            const data = await response.json();
            updateOutputs(data);
          }} catch (error) {{
            showError(error && error.message ? error.message : "Не удалось обновить данные");
          }} finally {{
            setLoading(false);
          }}
        }}

        const prevBtn = document.getElementById("action-previous");
        const resumeBtn = document.getElementById("action-resume");
        if (prevBtn) {{
          prevBtn.addEventListener("click", () => trigger("previous-3d"));
        }}
        if (resumeBtn) {{
          resumeBtn.addEventListener("click", () => trigger("resume"));
        }}

        updateOutputs(runtimeState);
        setActiveView("check");
      }})();
    </script>
  </body>
</html>
"""

    return html
