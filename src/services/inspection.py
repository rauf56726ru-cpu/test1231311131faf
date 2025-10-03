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

LOGGER = logging.getLogger(__name__)

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
from .ohlc_sanitizer import sanitize_candles
from .timeutils import ensure_ms_epoch, safe_datetime_from_ms
from .presets import resolve_profile_config
from .zones import Config as ZonesConfig, detect_zones
from ..meta import Meta
from ..static_version import STATIC_VERSION

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
    try:
        SNAPSHOT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Failing to create the directory should not break the request flow; in-memory
        # storage will continue to function even without persistence on disk.
        pass


def _snapshot_path(snapshot_id: str) -> Path:
    safe_id = _SNAPSHOT_ID_SANITISER.sub("_", snapshot_id or "snapshot")
    return SNAPSHOT_STORAGE_DIR / f"{safe_id}.json"


def _persist_snapshot(snapshot: Snapshot) -> None:
    if not snapshot:
        return

    _ensure_storage_dir()

    snapshot_id = str(snapshot.get("id") or "snapshot")
    path = _snapshot_path(snapshot_id)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot, handle, ensure_ascii=False, indent=2)
        tmp_path.replace(path)
    except OSError:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _remove_snapshot_file(snapshot_id: str) -> None:
    path = _snapshot_path(snapshot_id)
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_snapshot_limit() -> None:
    while len(_SNAPSHOT_STORE) > _MAX_STORED_SNAPSHOTS:
        removed_id, _ = _SNAPSHOT_STORE.popitem(last=False)
        _remove_snapshot_file(removed_id)


def _load_existing_snapshots() -> None:
    if not SNAPSHOT_STORAGE_DIR.exists():
        return

    try:
        files = sorted(
            SNAPSHOT_STORAGE_DIR.glob("*.json"),
            key=lambda item: item.stat().st_mtime,
        )
    except OSError:
        return

    for path in files:
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, Mapping):
            continue
        snapshot_id = str(data.get("id") or path.stem)
        snapshot = dict(data)
        snapshot["id"] = snapshot_id
        _SNAPSHOT_STORE[snapshot_id] = snapshot
        _SNAPSHOT_STORE.move_to_end(snapshot_id)

    _ensure_snapshot_limit()


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

    input_count = len(candles) if isinstance(candles, Sequence) else 0
    stage = "session_vwaps"

    sanitized = sanitize_candles(candles, stage=stage)
    filtered_candles = sanitized.candles

    meta: Dict[str, Any] = {
        "sanitized": True,
        "invalid_ts_count": sanitized.invalid_ts,
        "invalid_ohlc_count": sanitized.invalid_ohlc,
        "sessions_empty": False,
        "input_count": input_count,
        "output_count": len(filtered_candles),
    }
    if sanitized.earliest_ms is not None:
        meta["earliest_ms"] = sanitized.earliest_ms
    if sanitized.latest_ms is not None:
        meta["latest_ms"] = sanitized.latest_ms

    if not filtered_candles:
        meta["sessions_empty"] = bool(input_count)
        LOGGER.info(
            "Session VWAP sanitisation produced no candles",
            extra={
                "stage": stage,
                "symbol": symbol,
                "input_count": input_count,
                "invalid_ts": sanitized.invalid_ts,
                "invalid_ohlc": sanitized.invalid_ohlc,
            },
        )
        return {"symbol": symbol.upper(), "vwap": [], "vwap_sigma": [], "meta": meta}

    bars: List[Tuple[int, datetime, float, float, float, float]] = []
    for candle in filtered_candles:
        if not isinstance(candle, Mapping):
            continue
        timestamp_ms = ensure_ms_epoch(candle.get("t"))
        if timestamp_ms is None:
            meta["invalid_ts_count"] += 1
            LOGGER.warning(
                "Invalid candle timestamp encountered after sanitisation",
                extra={"stage": stage, "symbol": symbol, "ts": candle.get("t")},
            )
            continue
        dt = safe_datetime_from_ms(timestamp_ms, timezone.utc)
        if dt is None:
            meta["invalid_ts_count"] += 1
            LOGGER.warning(
                "Timestamp conversion failed for session VWAP",
                extra={"stage": stage, "symbol": symbol, "ts": timestamp_ms},
            )
            continue
        try:
            high = float(candle.get("h", candle.get("high", 0.0)))
            low = float(candle.get("l", candle.get("low", 0.0)))
            close = float(candle.get("c", candle.get("close", 0.0)))
        except (TypeError, ValueError):
            meta["invalid_ohlc_count"] += 1
            LOGGER.warning(
                "Invalid OHLC values encountered after sanitisation",
                extra={"stage": stage, "symbol": symbol, "ts": timestamp_ms},
            )
            continue
        try:
            volume = float(candle.get("v", candle.get("volume", 0.0)))
        except (TypeError, ValueError):
            volume = 0.0
        if not math.isfinite(volume) or volume < 0.0:
            volume = 0.0
        bars.append((timestamp_ms, dt, high, low, close, volume))

    if not bars:
        meta["sessions_empty"] = True
        LOGGER.info(
            "No usable candles for session VWAP after validation",
            extra={"stage": stage, "symbol": symbol},
        )
        return {"symbol": symbol.upper(), "vwap": [], "vwap_sigma": [], "meta": meta}

    bars.sort(key=lambda item: item[0])
    last_dt = bars[-1][1]
    last_date = last_dt.date()
    lookback = max(1, int(Meta.VWAP_LOOKBACK_DAYS))
    start_date = last_date - timedelta(days=lookback - 1)
    sessions = list(Meta.iter_vwap_sessions())

    daily_buckets: defaultdict = defaultdict(list)  # type: ignore[var-annotated]
    session_buckets: DefaultDict[
        Tuple[datetime.date, str], List[Mapping[str, Any]]
    ] = defaultdict(list)
    session_extrema: Dict[Tuple[datetime.date, str], Tuple[float, float]] = {}

    for open_ms, dt, high, low, close, volume in bars:
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

    LOGGER.info(
        "Computed session VWAP payload",
        extra={
            "stage": stage,
            "symbol": symbol,
            "bars": len(bars),
            "start_ms": bars[0][0],
            "end_ms": bars[-1][0],
        },
    )

    return {
        "symbol": symbol.upper(),
        "vwap": results,
        "vwap_sigma": sigma_results,
        "meta": meta,
    }
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
    """Render the inspection dashboard HTML."""

    static_version = STATIC_VERSION
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    snapshots_json = json.dumps(snapshots, ensure_ascii=False).replace("</", "<\\/")

    symbol_clean = (symbol or "").strip().upper()
    if not symbol_clean or symbol_clean in {"—", "-", "UNKNOWN"}:
        symbol_clean = DEFAULT_SYMBOL
    symbol_value = html_utils.escape(symbol_clean)
    timeframe_value = html_utils.escape(timeframe)
    snapshot_value = html_utils.escape(snapshot_id or "")

    data_section = payload.get("DATA") if isinstance(payload, Mapping) else None
    diagnostics_section = payload.get("DIAGNOSTICS") if isinstance(payload, Mapping) else None

    def _format_json_block(value: Any) -> str:
        try:
            formatted = json.dumps(value if value is not None else None, ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            formatted = json.dumps(None, ensure_ascii=False, indent=2)
        return html_utils.escape(formatted)

    diagnostics_json_initial = _format_json_block(diagnostics_section)
    check_all_json_initial = _format_json_block(None)

    style_block = """
    :root {
      color-scheme: dark;
      --bg: #020617;
      --fg: #e2e8f0;
      --muted: #94a3b8;
      --border: rgba(148, 163, 184, 0.24);
      --accent: #38bdf8;
      --accent-strong: #0ea5e9;
      --panel: rgba(15, 23, 42, 0.72);
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: "Inter", "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(circle at 20% -10%, #1e293b 0%, #0f172a 40%, #020617 100%);
      color: var(--fg);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 2rem 1.5rem 1rem;
      max-width: 1200px;
      margin: 0 auto;
      width: 100%;
    }
    header h1 {
      margin: 0 0 0.35rem;
      font-size: clamp(1.8rem, 2.8vw, 2.6rem);
    }
    header p {
      margin: 0;
      color: var(--muted);
    }
    main {
      width: min(1200px, 96vw);
      margin: 0 auto 3rem;
      flex: 1;
      display: grid;
      grid-template-columns: minmax(320px, 360px) minmax(580px, 1fr);
      gap: 1.5rem;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 1.5rem;
      box-shadow: 0 28px 70px rgba(2, 6, 23, 0.45);
      display: flex;
      flex-direction: column;
      gap: 1.2rem;
    }
    .panel--collection {
      grid-column: 1;
    }
    .panel--view {
      grid-column: 2;
      justify-self: center;
      width: min(100%, 920px);
    }
    @media (max-width: 960px) {
      main {
        grid-template-columns: 1fr;
      }
      .panel--collection,
      .panel--view {
        grid-column: 1;
      }
      .panel--view {
        justify-self: stretch;
      }
    }
    h2 {
      margin: 0;
      font-size: 0.95rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(148, 163, 184, 0.9);
    }
    label span {
      display: block;
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 0.3rem;
      color: rgba(148, 163, 184, 0.75);
    }
    input, select, button {
      font: inherit;
    }
    select, input[type="text"], input[type="number"] {
      width: 100%;
      background: rgba(15, 23, 42, 0.75);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 0.55rem 0.75rem;
      color: var(--fg);
    }
    button {
      cursor: pointer;
      border-radius: 999px;
      border: none;
      padding: 0.55rem 1.1rem;
      font-weight: 600;
      transition: transform 0.18s ease, box-shadow 0.18s ease;
      display: inline-flex;
      align-items: center;
      gap: 0.45rem;
    }
    button.primary {
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%);
      color: #0f172a;
      box-shadow: 0 16px 40px rgba(14, 165, 233, 0.35);
    }
    button.secondary {
      background: rgba(148, 163, 184, 0.15);
      color: var(--fg);
    }
    button:disabled {
      opacity: 0.55;
      cursor: not-allowed;
      box-shadow: none;
    }
    button:hover:not(:disabled) {
      transform: translateY(-1px);
      box-shadow: 0 14px 30px rgba(8, 47, 73, 0.4);
    }
    .panel-lead {
      margin: 0 0 1rem;
      color: rgba(148, 163, 184, 0.8);
      font-size: 0.95rem;
    }
    .collection-actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      margin-bottom: 1rem;
      justify-content: flex-start;
    }
    .collection-actions .primary {
      min-width: 260px;
    }
    .collection-actions .secondary {
      min-width: 200px;
    }
    .preset-chip-bar {
      min-height: 1.6rem;
      display: flex;
      align-items: center;
      margin-bottom: 0.5rem;
    }
    .controls-grid {
      display: grid;
      gap: 1rem;
    }
    .selection-bar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      flex-wrap: wrap;
      margin-top: 1rem;
    }
    .selection-bar > div {
      display: inline-flex;
      gap: 0.6rem;
      align-items: center;
      flex-wrap: wrap;
    }
    .live-meta {
      margin-top: 1rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 0.75rem;
      padding: 0.85rem 1rem;
      border-radius: 14px;
      border: 1px solid rgba(148, 163, 184, 0.22);
      background: rgba(15, 23, 42, 0.68);
      box-shadow: inset 0 1px 0 rgba(148, 163, 184, 0.08);
    }
    .live-meta__item span {
      display: block;
      font-size: 0.75rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .live-meta__value {
      font-size: 1.05rem;
      font-weight: 600;
      color: #f8fafc;
    }
    .live-meta__value[data-state="stale"] {
      color: #f97316;
    }
    .live-meta__value[data-state="fresh"] {
      color: #22c55e;
    }
    .chart-toolbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 1rem;
      flex-wrap: wrap;
    }
    .chart-toolbar__symbol {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.25rem 0.6rem;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid rgba(148, 163, 184, 0.25);
    }
    .chart-toolbar__symbol strong {
      font-size: 1.05rem;
      letter-spacing: 0.08em;
      color: #f8fafc;
    }
    .chart-toolbar__frames {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
      align-items: flex-start;
    }
    .tf-toggle {
      display: inline-flex;
      gap: 0.35rem;
      padding: 0.2rem;
      border-radius: 999px;
      background: rgba(30, 41, 59, 0.6);
      border: 1px solid rgba(148, 163, 184, 0.24);
      flex-wrap: wrap;
    }
    .tf-toggle button {
      border-radius: 999px;
      background: transparent;
      padding: 0.45rem 0.9rem;
      color: var(--fg);
      border: none;
      font-weight: 600;
    }
    .tf-toggle button.active {
      background: var(--accent);
      color: #0f172a;
      box-shadow: 0 12px 26px rgba(14, 165, 233, 0.25);
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.2rem 0.6rem;
      border-radius: 999px;
      font-size: 0.75rem;
      background: rgba(148, 163, 184, 0.18);
      color: var(--muted);
    }
    .chart-shell {
      height: 420px;
      min-height: 360px;
      width: 100%;
      border-radius: 14px;
      overflow: hidden;
      border: 1px solid rgba(148, 163, 184, 0.25);
      background: rgba(15, 23, 42, 0.9);
      position: relative;
    }
    .chart-shell::after {
      content: attr(data-selection-label);
      position: absolute;
      inset: auto 1rem 1rem auto;
      background: rgba(8, 47, 73, 0.85);
      border-radius: 999px;
      padding: 0.35rem 0.8rem;
      font-size: 0.75rem;
      color: rgba(248, 250, 252, 0.88);
      pointer-events: none;
    }
    .json-panels {
      display: grid;
      gap: 1rem;
    }
    .collapse {
      border-radius: 14px;
      border: 1px solid rgba(148, 163, 184, 0.2);
      overflow: hidden;
      background: rgba(15, 23, 42, 0.9);
    }
    .collapse header {
      margin: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0.85rem 1rem;
      background: rgba(14, 165, 233, 0.18);
      cursor: pointer;
      gap: 1rem;
    }
    .collapse header .actions {
      display: inline-flex;
      gap: 0.5rem;
      align-items: center;
    }
    .collapse header h3 {
      margin: 0;
      font-size: 0.9rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    .checkall-control {
      display: flex;
      flex-direction: column;
      align-items: stretch;
      gap: 0.75rem;
      padding: 0.75rem 1rem 0.5rem;
      background: rgba(15, 23, 42, 0.85);
      border-top: 1px solid rgba(148, 163, 184, 0.16);
    }
    .checkall-control__row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 0.6rem;
    }
    .checkall-control span,
    .checkall-control__row span {
      font-size: 0.75rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: rgba(148, 163, 184, 0.78);
    }
    .checkall-control select {
      background: rgba(15, 23, 42, 0.65);
      border: 1px solid rgba(148, 163, 184, 0.3);
      border-radius: 12px;
      padding: 0.4rem 0.7rem;
      color: var(--fg);
      min-width: 110px;
    }
    .collapse pre {
      margin: 0;
      padding: 1rem;
      max-height: 260px;
      overflow: auto;
      font-size: 0.85rem;
      background: rgba(15, 23, 42, 0.78);
      border-top: 1px solid rgba(148, 163, 184, 0.18);
    }
    .collapse.collapsed pre {
      display: none;
    }
    .meta-grid {
      display: grid;
      gap: 0.8rem;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
    .meta-tile {
      padding: 0.75rem 1rem;
      border-radius: 12px;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
    }
    .meta-tile span {
      display: block;
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: rgba(148, 163, 184, 0.7);
      margin-bottom: 0.4rem;
    }
    .status-banner {
      padding: 0.5rem 0.75rem;
      border-radius: 10px;
      border: 1px solid rgba(148, 163, 184, 0.35);
      background: rgba(15, 23, 42, 0.6);
      font-size: 0.85rem;
    }
    .status-banner[data-tone="success"] {
      border-color: rgba(34, 197, 94, 0.6);
      background: rgba(22, 101, 52, 0.3);
    }
    .status-banner[data-tone="error"] {
      border-color: rgba(248, 113, 113, 0.6);
      background: rgba(127, 29, 29, 0.35);
    }
    .status-banner[data-tone="warning"] {
      border-color: rgba(250, 204, 21, 0.6);
      background: rgba(113, 63, 18, 0.35);
    }
    .preset-controls {
      margin-top: 0.8rem;
      margin-bottom: 0.8rem;
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      align-items: center;
    }
    .preset-chip {
      display: inline-flex;
      align-items: center;
      padding: 0.35rem 0.75rem;
      border-radius: 999px;
      font-size: 0.82rem;
      font-weight: 600;
      background: rgba(14, 165, 233, 0.25);
      color: #f8fafc;
    }
    .preset-chip[data-variant="warning"] {
      background: rgba(249, 115, 22, 0.5);
      color: #0f172a;
    }
    .preset-chip[data-variant="success"] {
      background: rgba(34, 197, 94, 0.55);
      color: #0f172a;
    }
    .preset-chip[data-variant="info"] {
      background: rgba(14, 165, 233, 0.55);
      color: #0f172a;
    }
    .modal {
      position: fixed;
      inset: 0;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      z-index: 40;
    }
    .modal[hidden] {
      display: none;
    }
    .modal__backdrop {
      position: absolute;
      inset: 0;
      background: rgba(15, 23, 42, 0.7);
    }
    .modal__dialog {
      position: relative;
      margin: 6vh auto;
      max-width: 640px;
      width: min(92vw, 640px);
      background: rgba(15, 23, 42, 0.96);
      border-radius: 16px;
      padding: 1.5rem;
      border: 1px solid rgba(148, 163, 184, 0.35);
      color: #e2e8f0;
      box-shadow: 0 32px 64px rgba(15, 23, 42, 0.45);
    }
    .modal__header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1rem;
    }
    .modal__header h3 {
      margin: 0;
      font-size: 1.2rem;
    }
    .modal__close {
      background: none;
      border: none;
      color: #94a3b8;
      font-size: 1.5rem;
      cursor: pointer;
      padding: 0;
      line-height: 1;
    }
    .modal__body {
      display: flex;
      flex-direction: column;
      gap: 1rem;
      max-height: 70vh;
      overflow-y: auto;
    }
    .preset-form-grid {
      display: grid;
      gap: 0.8rem;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    }
    .preset-form-grid label {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      font-size: 0.85rem;
    }
    .preset-form-grid input,
    .preset-form-grid select {
      padding: 0.45rem 0.6rem;
      border-radius: 8px;
      border: 1px solid rgba(148, 163, 184, 0.28);
      background: rgba(15, 23, 42, 0.8);
      color: #e2e8f0;
    }
    .preset-form-actions {
      display: flex;
      justify-content: flex-end;
      gap: 0.75rem;
      margin-top: 0.5rem;
    }
    .preset-list {
      display: flex;
      flex-direction: column;
      gap: 0.75rem;
    }
    .preset-list__item {
      padding: 0.75rem 0.9rem;
      border-radius: 10px;
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(148, 163, 184, 0.25);
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.5rem;
    }
    .preset-list__title {
      font-weight: 600;
      font-size: 1rem;
    }
    .preset-list__meta {
      font-size: 0.82rem;
      color: #94a3b8;
    }
    .preset-list__actions {
      display: flex;
      gap: 0.5rem;
    }
    .preset-list__empty {
      margin: 0;
      color: #94a3b8;
    }
    .btn-danger {
      background: rgba(239, 68, 68, 0.85);
      color: #0f172a;
      border: none;
      border-radius: 8px;
      padding: 0.45rem 0.75rem;
      cursor: pointer;
      font-weight: 600;
    }
    .btn-danger:disabled {
      opacity: 0.4;
      cursor: not-allowed;
    }

    @media (max-width: 960px) {
      main {
        grid-template-columns: 1fr;
      }
      .snapshot-select {
        flex-direction: column;
        align-items: stretch;
      }
    }
    """

    script_block = (
        "window.__INSPECTION_INITIAL__ = {\n"
        f"  payload: {payload_json},\n"
        f"  snapshotId: {json.dumps(snapshot_id or '')},\n"
        f"  symbol: {json.dumps(symbol_value)},\n"
        f"  timeframe: {json.dumps(timeframe_value)},\n"
        f"  snapshots: {snapshots_json},\n"
        f"  defaultSymbol: {json.dumps(DEFAULT_SYMBOL)}\n"
        "};\n"
    )

    ui_script = """
(function () {
  const TIMEFRAME_TO_MS = {
    "1s": 1000,
    "3s": 3000,
    "5s": 5000,
    "15s": 15000,
    "30s": 30000,
    "1m": 60000,
    "3m": 180000,
    "5m": 300000,
    "15m": 900000,
    "30m": 1800000,
    "1h": 3600000,
    "2h": 7200000,
    "4h": 14400000,
    "6h": 21600000,
    "8h": 28800000,
    "12h": 43200000,
    "1d": 86400000,
  };
  const DEFAULT_TEST_TIMEFRAMES = ["1m", "3m", "5m", "15m", "1h", "4h", "1d"];
  const SHARED_MAX_BARS = 5000;
  let LightweightCharts = window.LightweightCharts || null;
  const BinanceCandles = window.BinanceCandles || null;
  const ChartGapWatcher = window.ChartGapWatcher || null;
  const SharedCandles = window.SharedCandles || null;


  function toChartBars(candles) {
    return (candles || [])
      .map((candle) => {
        const rawTs = Number(candle?.ts_ms_utc ?? candle?.t ?? candle?.time ?? 0);
        const open = Number(candle?.o ?? candle?.open ?? 0);
        const high = Number(candle?.h ?? candle?.high ?? open);
        const low = Number(candle?.l ?? candle?.low ?? open);
        const close = Number(candle?.c ?? candle?.close ?? open);
        if (
          !Number.isFinite(rawTs) ||
          !Number.isFinite(open) ||
          !Number.isFinite(high) ||
          !Number.isFinite(low) ||
          !Number.isFinite(close)
        ) {
          return null;
        }
        const time = Math.floor(rawTs / 1000);
        return {
          time,
          open,
          high,
          low,
          close,
          ts_ms_utc: Math.floor(rawTs),
        };
      })
      .filter((bar) => bar !== null);
  }

  async function fetchSnapshots() {
    const response = await fetch("/inspection/snapshots", { cache: "no-store" });
    if (!response.ok) throw new Error("Failed to fetch snapshots");
    return response.json();
  }

  function formatTs(ts) {
    if (!Number.isFinite(ts)) return "—";
    const date = new Date(ts);
    if (Number.isNaN(date.getTime())) return "—";
    return date.toISOString().replace("T", " ").replace(".000Z", "Z");
  }

  function computeCandleDisplayTime(candle, intervalMs, lastUpdateMs) {
    if (!candle) return Number.NaN;
    const openMs = Number(
      candle.ts_ms_utc ?? candle.t ?? candle.time ?? (candle.openTime ?? candle.open_time ?? 0),
    );
    if (!Number.isFinite(openMs)) return Number.NaN;
    const safeInterval = Math.max(1, Number(intervalMs) || 0);
    const closingMs = openMs + safeInterval;
    const specificUpdate = Number(
      candle.last_update_ms ?? candle.lastUpdateMs ?? candle.last_update ?? candle.updated_at ?? 0,
    );
    const referenceSource = Number.isFinite(specificUpdate) ? specificUpdate : lastUpdateMs;
    const reference = Number.isFinite(referenceSource) ? Number(referenceSource) : Date.now();
    const alignedReference = Math.max(openMs, reference);
    return Math.min(closingMs, alignedReference);
  }

  function intervalToMs(value) {
    const numeric = Number(value);
    if (Number.isFinite(numeric)) {
      return Math.max(1, numeric) * 60000;
    }
    return TIMEFRAME_TO_MS[value] || 60000;
  }

  function normaliseBar(bar) {
    if (!bar) return null;
    const open = Number(bar.open ?? bar.o ?? 0);
    const high = Number(bar.high ?? bar.h ?? open);
    const low = Number(bar.low ?? bar.l ?? open);
    const close = Number(bar.close ?? bar.c ?? open);
    const time = Number(bar.time ?? bar.t ?? 0);
    if (
      !Number.isFinite(time) ||
      !Number.isFinite(open) ||
      !Number.isFinite(high) ||
      !Number.isFinite(low) ||
      !Number.isFinite(close)
    ) {
      return null;
    }
    const ts = Number(bar.ts_ms_utc ?? time * 1000);
    return {
      time: Math.floor(time),
      open,
      high,
      low,
      close,
      ts_ms_utc: Number.isFinite(ts) ? Math.floor(ts) : Math.floor(time * 1000),
    };
  }

  async function fetchHistory(symbol, interval, limit = 1000) {
    if (!BinanceCandles || typeof BinanceCandles.fetchHistory !== "function") {
      throw new Error("Binance helper is unavailable");
    }
    const rows = await BinanceCandles.fetchHistory(symbol, interval, limit);
    return rows.map((bar) => normaliseBar(bar)).filter((bar) => bar !== null);
  }

  async function fetchRange(symbol, interval, startMs, endMs, limit = 1000) {
    const url = new URL("https://fapi.binance.com/fapi/v1/klines");
    url.searchParams.set("symbol", symbol);
    url.searchParams.set("interval", interval);
    if (Number.isFinite(startMs)) {
      url.searchParams.set("startTime", Math.floor(startMs));
    }
    if (Number.isFinite(endMs)) {
      url.searchParams.set("endTime", Math.floor(endMs));
    }
    url.searchParams.set("limit", String(Math.max(1, Math.min(limit, 1500))));
    const response = await fetch(url.toString(), { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to fetch range: ${response.status}`);
    }
    const data = await response.json();
    if (!BinanceCandles || typeof BinanceCandles.transformKlines !== "function") {
      throw new Error("Binance helper is unavailable");
    }
    return BinanceCandles.transformKlines(data)
      .map((bar) => normaliseBar(bar))
      .filter((bar) => bar !== null);
  }

  function ensurePreviewBar(input) {
    if (!input) return null;
    const candidateTime = Number(input.time);
    const candidateOpen = Number(input.open);
    const candidateHigh = Number(input.high ?? candidateOpen);
    const candidateLow = Number(input.low ?? candidateOpen);
    const candidateClose = Number(input.close ?? candidateOpen);
    let candidateTs = Number(input.ts_ms_utc ?? input.t ?? 0);
    if (
      Number.isFinite(candidateTime) &&
      Number.isFinite(candidateOpen) &&
      Number.isFinite(candidateHigh) &&
      Number.isFinite(candidateLow) &&
      Number.isFinite(candidateClose)
    ) {
      if (!Number.isFinite(candidateTs)) {
        candidateTs = candidateTime * 1000;
      }
      return {
        time: Math.floor(candidateTime),
        open: candidateOpen,
        high: candidateHigh,
        low: candidateLow,
        close: candidateClose,
        ts_ms_utc: Math.floor(candidateTs),
      };
    }
    const normalized = normaliseBar(input);
    if (!normalized) return null;
    return {
      time: Math.floor(normalized.time),
      open: normalized.open,
      high: normalized.high,
      low: normalized.low,
      close: normalized.close,
      ts_ms_utc: normalized.ts_ms_utc,
      ...(Number.isFinite(normalized.last_update_ms)
        ? { last_update_ms: Number(normalized.last_update_ms) }
        : {}),
    };
  }


  function setJson(pre, data) {
    if (!pre) return;
    pre.textContent = JSON.stringify(data ?? null, null, 2);
  }

  function selectionLabel(start, end) {
    if (!start || !end) return "Выделите диапазон";
    const from = formatTs(start);
    const to = formatTs(end);
    return `${from} → ${to}`;
  }

  function normaliseSymbol(input) {
    const trimmed = (input || "").trim().toUpperCase();
    const cleaned = trimmed.replace(/[^A-Z0-9]/g, "");
    return cleaned;
  }

  async function fetchCandles(symbol, interval, startMs, endMs, options = {}) {
    const collected = [];
    const seen = new Set();
    const intervalMs = TIMEFRAME_TO_MS[interval] || 60000;
    const hintedLimit = Number.isFinite(options.limit) ? Math.floor(options.limit) : null;
    const batchLimit = Math.max(1, Math.min(1500, hintedLimit || 1000));
    const hasStart = Number.isFinite(startMs);
    const hasEnd = Number.isFinite(endMs);
    let startBound = null;
    if (hasStart && hasEnd) {
      startBound = Math.floor(Math.min(startMs, endMs));
    } else if (hasStart) {
      startBound = Math.floor(startMs);
    }
    let endBound = null;
    if (hasStart && hasEnd) {
      endBound = Math.floor(Math.max(startMs, endMs));
    } else if (hasEnd) {
      endBound = Math.floor(endMs);
    }
    let cursor = startBound;
    let guard = 0;
    const guardLimit = 4096;

    while (true) {
      const url = new URL("https://fapi.binance.com/fapi/v1/klines");
      url.searchParams.set("symbol", symbol.toUpperCase());
      url.searchParams.set("interval", interval);
      if (Number.isFinite(cursor)) {
        url.searchParams.set("startTime", String(cursor));
      } else if (Number.isFinite(startBound)) {
        url.searchParams.set("startTime", String(startBound));
      }
      if (Number.isFinite(endBound)) {
        url.searchParams.set("endTime", String(endBound));
      }
      url.searchParams.set("limit", String(batchLimit));

      const resp = await fetch(url.toString(), { cache: "no-store" });
      if (!resp.ok) {
        throw new Error(`klines ${resp.status}`);
      }
      const rows = await resp.json();
      if (!Array.isArray(rows) || !rows.length) {
        break;
      }

      let lastOpen = null;
      for (const row of rows) {
        if (!row) continue;
        const openMs = Number(row[0]);
        const open = Number(row[1]);
        const high = Number(row[2]);
        const low = Number(row[3]);
        const close = Number(row[4]);
        const volume = Number(row[5]);
        if (!Number.isFinite(openMs)) continue;
        lastOpen = openMs;
        if (Number.isFinite(endBound) && openMs > endBound) {
          continue;
        }
        if (seen.has(openMs)) continue;
        seen.add(openMs);
        collected.push({ t: openMs, o: open, h: high, l: low, c: close, v: volume });
      }

      if (!hasStart || !Number.isFinite(cursor)) {
        break;
      }
      if (!Number.isFinite(lastOpen)) {
        break;
      }
      const nextCursor = lastOpen + intervalMs;
      if (!Number.isFinite(nextCursor)) {
        break;
      }
      if (Number.isFinite(endBound) && nextCursor > endBound) {
        break;
      }
      if (nextCursor <= cursor) {
        break;
      }
      cursor = nextCursor;
      guard += 1;
      if (guard >= guardLimit) {
        break;
      }
      if (rows.length < batchLimit) {
        break;
      }
    }

    collected.sort((a, b) => a.t - b.t);
    if (Number.isFinite(startBound) || Number.isFinite(endBound)) {
      return collected.filter((candle) => {
        const ts = Number(candle?.t ?? 0);
        if (!Number.isFinite(ts)) return false;
        if (Number.isFinite(startBound) && ts < startBound) return false;
        if (Number.isFinite(endBound) && ts > endBound) return false;
        return true;
      });
    }
    return collected;
  }

  async function postSnapshot(payload) {
    const response = await fetch("/inspection/snapshot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
      body: JSON.stringify(payload),
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`Snapshot failed: ${response.status} ${text}`);
    }
    return response.json();
  }

  async function fetchPayload(id) {
    const response = await fetch(`/inspection?snapshot=${encodeURIComponent(id)}`, {
      headers: { Accept: "application/json" },
      cache: "no-store",
    });
    if (!response.ok) {
      throw new Error("Failed to fetch payload");
    }
    return response.json();
  }

  function initCollapsibles() {
    document.querySelectorAll("[data-collapse-toggle]").forEach((toggle) => {
      toggle.addEventListener("click", () => {
        const target = toggle.closest(".collapse");
        if (!target) return;
        target.classList.toggle("collapsed");
      });
    });
    document.querySelectorAll("[data-copy-target]").forEach((button) => {
      button.addEventListener("click", async () => {
        const id = button.getAttribute("data-copy-target");
        if (!id) return;
        const el = document.getElementById(id);
        if (!el) return;
        try {
          await navigator.clipboard.writeText(el.textContent || "");
          const original = button.textContent;
          button.textContent = "Скопировано";
          setTimeout(() => (button.textContent = original), 1200);
        } catch (error) {
          console.warn("copy failed", error);
        }
      });
    });
  }

  document.addEventListener("DOMContentLoaded", async () => {
    const initial = window.__INSPECTION_INITIAL__ || {};
    const defaultSymbol = normaliseSymbol(initial.defaultSymbol) || "BTCUSDT";
    const snapshotSelect = document.getElementById("snapshot-select");
    const refreshButton = document.getElementById("refresh-snapshot");
    const diagnosticsPre = document.getElementById("diagnostics-json");
    const checkAllPre = document.getElementById("checkall-json");
    const summaryButton = document.getElementById("collect-summary");
    const sessionDetailedButton = document.getElementById("btn_collect_last_session_detailed");
    const topupButton = document.getElementById("collect-topup");
    const collectSelectionButton = document.getElementById("collect-selection");
    const checkAllHours = document.getElementById("checkall-hours");
    const snapshotMeta = document.getElementById("snapshot-meta");
    const chartContainer = document.getElementById("inspection-chart");
    const selectionInfo = document.getElementById("selection-info");
    const chartSymbolLabel = document.getElementById("chart-symbol");
    const buildButton = document.getElementById("build-session");
    const clearSelection = document.getElementById("clear-selection");
    const timeframeCheckboxes = Array.from(document.querySelectorAll("[data-tf-checkbox]"));
    const timeframeToggle = document.getElementById("chart-tf-toggle");
    const statusEl = document.getElementById("inspection-status");
    const symbolInput = document.getElementById("symbol-input");
    const presetChip = document.getElementById("preset-chip");
    const managePresetsButton = document.getElementById("manage-presets");
    const presetModal = document.getElementById("preset-modal");
    const presetBackdrop = document.getElementById("preset-modal-backdrop");
    const presetForm = document.getElementById("preset-form");
    const presetListView = document.getElementById("preset-list-view");
    const presetListContainer = document.getElementById("preset-list");
    const presetFormSection = document.getElementById("preset-form-section");
    const presetModalTitle = document.getElementById("preset-modal-title");
    const presetModeField = document.getElementById("preset-mode-input");
    const presetSymbolField = document.getElementById("preset-symbol");
    const presetTfField = document.getElementById("preset-tf");
    const presetLastNField = document.getElementById("preset-last-n");
    const presetValueAreaField = document.getElementById("preset-value-area");
    const presetBinningModeField = document.getElementById("preset-binning-mode");
    const presetTickSizeField = document.getElementById("preset-tick-size");
    const presetAtrField = document.getElementById("preset-atr-multiplier");
    const presetBinsField = document.getElementById("preset-target-bins");
    const presetClipField = document.getElementById("preset-clip-tail");
    const presetSmoothField = document.getElementById("preset-smooth-window");
    const presetTickRow = document.querySelector("[data-preset-tick]");
    const presetAdaptiveRow = document.querySelector("[data-preset-adaptive]");
    const presetSubmitButton = document.getElementById("preset-submit");
    const presetCancelButtons = Array.from(document.querySelectorAll("[data-close-preset]"));
    const presetCreateButton = document.getElementById("preset-create-button");
    const livePriceEl = document.getElementById("live-last-price");
    const liveTsEl = document.getElementById("live-last-ts");
    const liveTfEl = document.getElementById("live-last-tf");
    const liveAgeEl = document.getElementById("live-age-sec");
    const liveStateEl = document.getElementById("live-stale-flag");
    const mismatchBanner = document.getElementById("stream-mismatch");

    initCollapsibles();

    const resolveHours = (value) => {
      const parsed = Number(value);
      if (!Number.isFinite(parsed)) return 1;
      return Math.min(4, Math.max(1, Math.floor(parsed)));
    };

    const PREFERRED_CHART_FRAMES = ["1m", "3m", "15m", "30m", "1h", "4h", "1d", "1w"];

    let state = null;

    function frameHasCandles(frames, tf) {
      if (!tf) return false;
      const liveEntry = state?.liveFrames?.[tf];
      if (liveEntry && Array.isArray(liveEntry.candles) && liveEntry.candles.length) {
        return true;
      }
      if (!frames) return false;
      const entry = frames[tf];
      if (!entry || typeof entry !== "object") return false;
      const candles = Array.isArray(entry.candles) ? entry.candles : [];
      return candles.length > 0;
    }

    function selectPreferredFrame(frames, desired) {
      const frameMap = frames || {};
      if (desired && frameHasCandles(frameMap, desired)) {
        return desired;
      }
      for (const tf of PREFERRED_CHART_FRAMES) {
        if (frameHasCandles(frameMap, tf)) {
          return tf;
        }
      }
      const keys = Object.keys(frameMap);
      if (keys.length) {
        return keys.sort()[0];
      }
      return desired || PREFERRED_CHART_FRAMES[0];
    }

    const initialFrameMap = initial.payload?.DATA?.frames || {};
    const defaultFrame = selectPreferredFrame(initialFrameMap, initial.timeframe);

    state = {
      payload: initial.payload || null,
      snapshotId: initial.snapshotId || null,
      selection: initial.payload?.DATA?.selection || null,
      frame: defaultFrame,
      chart: null,
      series: null,
      candles: [],
      gapWatcher: null,
      gapSymbol: null,
      gapInterval: null,
      intervalMs: intervalToMs(defaultFrame),
      lastUpdateMs: null,
      availableFrames: initialFrameMap,
      checkAll: null,
      hours: checkAllHours ? resolveHours(checkAllHours.value) : 1,
      profilePreset: initial.payload?.DATA?.profile_preset || null,
      presetRequired: Boolean(initial.payload?.DATA?.profile_preset_required),
      presetDefaults: initial.payload?.DATA?.profile_defaults || null,
      presetList: [],
      presetModalMode: null,
      managingSymbol: null,
      presetModalOpen: false,
      liveFrames: {},
      liveMeta: null,
      liveMismatch: false,
    };

    const AUTO_PRESET_SYMBOLS = new Set(["BTCUSDT", "ETHUSDT", "SOLUSDT"]);

    function autoPresetSymbol(symbol) {
      const normalised = normaliseSymbol(symbol);
      return normalised ? AUTO_PRESET_SYMBOLS.has(normalised) : false;
    }

    function activeSymbol() {
      return (
        (symbolInput && normaliseSymbol(symbolInput.value)) ||
        normaliseSymbol(state.payload?.DATA?.symbol) ||
        normaliseSymbol(initial.payload?.DATA?.symbol) ||
        normaliseSymbol(initial.symbol) ||
        defaultSymbol
      );
    }

    const MarketDataStore = window.MarketDataStore || null;
    const chartStore =
      MarketDataStore &&
      new MarketDataStore({
        symbol: activeSymbol(),
        interval: state.frame || "1m",
        pollIntervalMs: 1500,
        historyLimit: 1500,
      });
    const priceStore =
      MarketDataStore &&
      new MarketDataStore({
        symbol: activeSymbol(),
        interval: "1m",
        pollIntervalMs: 1500,
        historyLimit: 1500,
      });

    function formatLivePrice(value) {
      if (!Number.isFinite(value)) return "—";
      const abs = Math.abs(value);
      if (abs >= 100_000) return value.toFixed(1);
      if (abs >= 10_000) return value.toFixed(2);
      if (abs >= 1_000) return value.toFixed(2);
      if (abs >= 100) return value.toFixed(2);
      if (abs >= 10) return value.toFixed(3);
      if (abs >= 1) return value.toFixed(4);
      if (abs >= 0.1) return value.toFixed(5);
      return value.toFixed(6);
    }

    function renderLiveMeta(meta) {
      state.liveMeta = meta || null;
      if (!meta) {
        if (livePriceEl) livePriceEl.textContent = "—";
        if (liveTsEl) liveTsEl.textContent = "—";
        if (liveTfEl) liveTfEl.textContent = "—";
        if (liveAgeEl) liveAgeEl.textContent = "—";
        if (liveStateEl) {
          liveStateEl.textContent = "—";
          liveStateEl.dataset.state = "unknown";
        }
        if (mismatchBanner) mismatchBanner.hidden = true;
        return;
      }
      if (livePriceEl) livePriceEl.textContent = formatLivePrice(meta.last_price);
      if (liveTsEl) liveTsEl.textContent = meta.last_ts_ms ? formatTs(meta.last_ts_ms) : "—";
      if (liveTfEl) liveTfEl.textContent = meta.last_tf || "—";
      if (liveAgeEl) liveAgeEl.textContent = Number.isFinite(meta.age_sec) ? `${meta.age_sec}s` : "—";
      if (liveStateEl) {
        liveStateEl.textContent = meta.stale ? "Stale" : "Live";
        liveStateEl.dataset.state = meta.stale ? "stale" : "fresh";
      }
      state.liveMismatch = Boolean(meta.mismatch);
      if (mismatchBanner) {
        mismatchBanner.hidden = !state.liveMismatch;
        if (state.liveMismatch) {
          mismatchBanner.dataset.tone = "warning";
          mismatchBanner.textContent = "Stream vs OHLCV mismatch > 1 tick";
        }
      }
    }

    function syncLiveStores({ force = false } = {}) {
      const symbol = activeSymbol();
      const chartInterval = state.frame || "1m";
      if (chartStore) {
        chartStore.setSymbol(symbol, chartInterval);
        if (force) chartStore.restart();
      }
      if (priceStore) {
        priceStore.setSymbol(symbol, "1m");
        if (force) priceStore.restart();
      }
    }

    function handleChartStoreEvent(event) {
      if (!event || !chartStore) return;
      const eventSymbol = normaliseSymbol(event.symbol);
      const currentSymbol = normaliseSymbol(activeSymbol());
      if (eventSymbol && currentSymbol && eventSymbol !== currentSymbol) return;
      const tf = event.interval || (state.frame || "1m");
      const latest = chartStore.candles ? chartStore.candles.slice() : [];
      if (!state.liveFrames) state.liveFrames = {};
      if (latest.length) {
        state.liveFrames[tf] = { tf, candles: latest };
        state.availableFrames = { ...(state.availableFrames || {}), [tf]: { tf, candles: latest } };
      }

      if (event.type === "status") {
        if (event.status === "connected") {
          updateStatus("Лайв-данные подключены", "info");
        } else if (event.status === "closed") {
          updateStatus("Лайв-канал закрыт, используем пуллинг", "warning");
        } else if (event.status === "error") {
          updateStatus("Ошибка потока, переподключение...", "warning");
        }
        return;
      }
      if (event.type === "error") {
        updateStatus("Ошибка получения лайв-данных", "error");
        return;
      }

      if (tf === (state.frame || "1m")) {
        if (event.type === "snapshot") {
          mergeChartBars(latest, { reset: true, persist: true });
          renderChart({ resetRequestedKeys: true, fitContent: true });
        } else if (event.type === "update" && event.candle) {
          mergeChartBars([event.candle], { reset: false, persist: true });
        } else if (event.type === "poll") {
          const pollBars = Array.isArray(event.candles) ? event.candles : [];
          if (pollBars.length) {
            mergeChartBars(pollBars, { reset: false, persist: false });
          }
        }
      }

      updateTimeframeToggle();
    }

    function handlePriceStoreEvent(event) {
      if (!event || !priceStore) return;
      const eventSymbol = normaliseSymbol(event.symbol);
      const currentSymbol = normaliseSymbol(activeSymbol());
      if (eventSymbol && currentSymbol && eventSymbol !== currentSymbol) return;
      if (event.meta) {
        renderLiveMeta(event.meta);
      }
    }

    if (chartStore) {
      chartStore.subscribe(handleChartStoreEvent);
      chartStore.start();
    }

    if (priceStore) {
      priceStore.subscribe(handlePriceStoreEvent);
      priceStore.start();
    }

    if (chartStore || priceStore) {
      syncLiveStores({ force: true });
    }

    function renderChartSymbol() {
      if (!chartSymbolLabel) return;
      chartSymbolLabel.textContent = activeSymbol();
    }

    function renderPresetChip() {
      if (!presetChip) return;
      const symbol = activeSymbol();
      const preset = state.profilePreset;
      if (state.presetRequired && !preset) {
        presetChip.textContent = `Требуется пресет для ${symbol}`;
        presetChip.dataset.variant = "warning";
        presetChip.hidden = false;
        return;
      }
      if (preset) {
        const name = preset.symbol || symbol;
        const suffix = preset.builtin ? " (auto)" : "";
        presetChip.textContent = `Preset: ${name}${suffix}`;
        presetChip.dataset.variant = preset.builtin ? "info" : "success";
        presetChip.hidden = false;
        return;
      }
      if (autoPresetSymbol(symbol)) {
        presetChip.textContent = `Preset: ${symbol} (auto)`;
        presetChip.dataset.variant = "info";
        presetChip.hidden = false;
        return;
      }
      presetChip.hidden = true;
    }

    function handleBinningModeChange() {
      if (!presetBinningModeField) return;
      const mode = presetBinningModeField.value === "tick" ? "tick" : "adaptive";
      if (presetTickRow) presetTickRow.hidden = mode !== "tick";
      if (presetAdaptiveRow) presetAdaptiveRow.hidden = mode !== "adaptive";
    }

    function populatePresetForm(preset, symbol) {
      if (!presetForm) return;
      const defaults = preset || state.presetDefaults || {};
      const binningDefaults = preset?.binning || defaults.binning || {};
      const extrasDefaults = preset?.extras || defaults.extras || {};

      const mode = (binningDefaults.mode || (defaults.adaptive_bins === false ? "tick" : "adaptive")).toLowerCase() === "tick"
        ? "tick"
        : "adaptive";

      if (presetModeField) presetModeField.value = preset ? "edit" : "create";
      if (presetSymbolField) {
        presetSymbolField.value = symbol || preset?.symbol || defaults.symbol || "";
        presetSymbolField.readOnly = Boolean(preset);
      }
      if (presetTfField) presetTfField.value = preset?.tf || defaults.tf || "1m";
      if (presetLastNField) presetLastNField.value = preset?.last_n ?? defaults.last_n ?? 3;
      if (presetValueAreaField) presetValueAreaField.value = preset?.value_area_pct ?? defaults.value_area_pct ?? defaults.value_area ?? 0.7;
      if (presetBinningModeField) presetBinningModeField.value = mode;
      if (presetTickSizeField)
        presetTickSizeField.value =
          mode === "tick"
            ? binningDefaults.tick_size ?? defaults.tick_size ?? ""
            : "";
      if (presetAtrField) presetAtrField.value = binningDefaults.atr_multiplier ?? defaults.atr_multiplier ?? 0.5;
      if (presetBinsField) presetBinsField.value = binningDefaults.target_bins ?? defaults.target_bins ?? 80;
      if (presetClipField) presetClipField.value = extrasDefaults.clip_low_volume_tail ?? defaults.clip_low_volume_tail ?? 0.005;
      if (presetSmoothField) presetSmoothField.value = extrasDefaults.smooth_window ?? defaults.smooth_window ?? 1;

      handleBinningModeChange();
    }

    function openPresetModal(options = {}) {
      if (!presetModal) return;
      const symbol = options.symbol || activeSymbol();
      presetModal.hidden = false;
      if (presetBackdrop) presetBackdrop.hidden = false;
      state.presetModalOpen = true;

      if (options.mode === "list") {
        state.presetModalMode = "list";
        if (presetModalTitle) presetModalTitle.textContent = "Управление пресетами";
        if (presetListView) presetListView.hidden = false;
        if (presetFormSection) presetFormSection.hidden = true;
        refreshPresetList();
      } else {
        state.presetModalMode = "form";
        if (presetModalTitle) presetModalTitle.textContent = options.title || "Настроить пресет TPO";
        if (presetListView) presetListView.hidden = true;
        if (presetFormSection) presetFormSection.hidden = false;
        populatePresetForm(options.preset || null, symbol);
      }
    }

    function closePresetModal() {
      if (!presetModal) return;
      presetModal.hidden = true;
      if (presetBackdrop) presetBackdrop.hidden = true;
      state.presetModalOpen = false;
      state.presetModalMode = null;
    }

    function readPresetForm() {
      if (!presetForm) return null;
      const symbol = (presetSymbolField?.value || "").trim().toUpperCase();
      if (!symbol) {
        throw new Error("Укажите символ");
      }
      const mode = presetBinningModeField?.value === "tick" ? "tick" : "adaptive";
      const payload = {
        symbol,
        tf: (presetTfField?.value || "1m").toLowerCase(),
        last_n: Number(presetLastNField?.value || 3),
        value_area_pct: Number(presetValueAreaField?.value || 0.7),
        binning: {
          mode,
          tick_size: mode === "tick" ? Number(presetTickSizeField?.value || 0) : null,
          atr_multiplier: Number(presetAtrField?.value || 0.5),
          target_bins: Number(presetBinsField?.value || 80),
        },
        extras: {
          clip_low_volume_tail: Number(presetClipField?.value || 0.005),
          smooth_window: Number(presetSmoothField?.value || 1),
        },
      };

      payload.last_n = Math.min(5, Math.max(1, Math.round(payload.last_n)));
      payload.value_area_pct = Math.min(0.95, Math.max(0.01, payload.value_area_pct));
      if (mode === "tick") {
        if (!payload.binning.tick_size || payload.binning.tick_size <= 0) {
          throw new Error("Tick size должен быть положительным числом");
        }
      } else {
        payload.binning.tick_size = null;
        payload.binning.atr_multiplier = Math.min(2, Math.max(0.1, payload.binning.atr_multiplier));
        payload.binning.target_bins = Math.min(200, Math.max(40, Math.round(payload.binning.target_bins)));
      }
      payload.extras.clip_low_volume_tail = Math.min(0.05, Math.max(0, payload.extras.clip_low_volume_tail));
      payload.extras.smooth_window = Math.min(5, Math.max(1, Math.round(payload.extras.smooth_window)));
      return payload;
    }

    async function refreshPresetList() {
      if (!presetListContainer) return;
      try {
        const response = await fetch("/presets", {
          headers: { Accept: "application/json" },
          cache: "no-store",
        });
        const data = await response.json();
        const list = Array.isArray(data?.presets) ? data.presets : [];
        state.presetList = list;
      } catch (error) {
        console.error("Failed to load presets", error);
        state.presetList = [];
      }
      renderPresetList();
    }

    function renderPresetList() {
      if (!presetListContainer) return;
      presetListContainer.innerHTML = "";
      const entries = Array.isArray(state.presetList) ? state.presetList : [];
      if (!entries.length) {
        const empty = document.createElement("p");
        empty.className = "preset-list__empty";
        empty.textContent = "Сохранённых пресетов пока нет";
        presetListContainer.appendChild(empty);
        return;
      }
      for (const preset of entries) {
        const row = document.createElement("div");
        row.className = "preset-list__item";
        const title = document.createElement("div");
        title.className = "preset-list__title";
        const symbol = (preset?.symbol || "").toUpperCase();
        title.textContent = symbol;
        row.appendChild(title);

        const meta = document.createElement("div");
        meta.className = "preset-list__meta";
        const modeLabel = preset?.binning?.mode === "tick" ? "Tick" : "Adaptive";
        meta.textContent = `TF: ${preset?.tf || "1m"} · ${modeLabel}`;
        row.appendChild(meta);

        const actions = document.createElement("div");
        actions.className = "preset-list__actions";

        const editButton = document.createElement("button");
        editButton.type = "button";
        editButton.className = "btn-secondary";
        editButton.textContent = "Редактировать";
        editButton.disabled = Boolean(preset?.builtin);
        editButton.addEventListener("click", () => {
          populatePresetForm(preset, symbol);
          if (presetModalTitle) presetModalTitle.textContent = `Редактировать пресет ${symbol}`;
          if (presetListView) presetListView.hidden = true;
          if (presetFormSection) presetFormSection.hidden = false;
          state.presetModalMode = "form";
        });
        actions.appendChild(editButton);

        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "btn-danger";
        deleteButton.textContent = "Удалить";
        deleteButton.disabled = Boolean(preset?.builtin);
        deleteButton.addEventListener("click", async () => {
          if (!window.confirm(`Удалить пресет ${symbol}?`)) return;
          try {
            const response = await fetch(`/presets/${symbol}`, {
              method: "DELETE",
              cache: "no-store",
            });
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            if (normaliseSymbol(symbol) === normaliseSymbol(activeSymbol())) {
              state.profilePreset = null;
              state.presetRequired = autoPresetSymbol(symbol) ? false : true;
            }
            await refreshPresetList();
            renderPresetState();
            updateCheckAllState();
          } catch (error) {
            console.error("Failed to delete preset", error);
            updateStatus("Не удалось удалить пресет", "error");
          }
        });
        actions.appendChild(deleteButton);

        row.appendChild(actions);
        presetListContainer.appendChild(row);
      }
    }

    async function submitPresetForm(event) {
      event.preventDefault();
      if (!presetForm) return;
      try {
        const payload = readPresetForm();
        if (!payload) return;
        const mode = presetModeField?.value === "edit" ? "edit" : "create";
        const url = mode === "edit" ? `/presets/${payload.symbol}` : "/presets";
        const method = mode === "edit" ? "PUT" : "POST";
        const response = await fetch(url, {
          method,
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          cache: "no-store",
          body: JSON.stringify(payload),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const data = await response.json();
        const stored = data?.preset || payload;
        if (normaliseSymbol(payload.symbol) === activeSymbol()) {
          state.profilePreset = stored;
          state.presetRequired = false;
        }
        closePresetModal();
        renderPresetState();
        updateCheckAllState();
        if (collectSelectionButton && !collectSelectionButton.disabled) {
          collectSelectionButton.click();
        }
      } catch (error) {
        console.error("Failed to save preset", error);
        updateStatus("Не удалось сохранить пресет", "error");
      }
    }

    function renderPresetState() {
      renderPresetChip();
      if (state.presetRequired && !state.profilePreset && !state.presetModalOpen) {
        openPresetModal({ mode: "form", title: "Настроить пресет TPO", symbol: activeSymbol() });
      } else if (!state.presetRequired && state.presetModalOpen && state.presetModalMode === "form") {
        closePresetModal();
      }
    }

    renderPresetState();
    updateCheckAllState();
    renderChartSymbol();

    if (symbolInput) {
      const initialSymbol =
        normaliseSymbol(initial.payload?.DATA?.symbol) || normaliseSymbol(initial.symbol) || defaultSymbol;
      symbolInput.value = initialSymbol;
      renderChartSymbol();
      symbolInput.addEventListener("input", () => {
        renderChartSymbol();
      });
      symbolInput.addEventListener("change", () => {
        syncLiveStores({ force: true });
      });
    }

    function updateStatus(message, tone = "info") {
      if (!statusEl) return;
      statusEl.textContent = message || "";
      statusEl.dataset.tone = tone;
      statusEl.hidden = !message;
    }

    function updateSelectionLabel() {
      const start = state.selection && state.selection.start;
      const end = state.selection && state.selection.end;
      const label = selectionLabel(start, end);
      if (selectionInfo) selectionInfo.textContent = label;
      if (chartContainer) chartContainer.setAttribute("data-selection-label", label);
    }

    function updateCheckAllState() {
      const hasSelection = Boolean(state.selection && state.selection.start && state.selection.end);
      const hoursValid = Number.isFinite(state.hours) && state.hours >= 1 && state.hours <= 4;
      if (checkAllHours && hoursValid) {
        checkAllHours.value = String(state.hours);
      }
      const presetReady = !state.presetRequired;
      const liveCapable = Boolean(activeSymbol());
      if (collectSelectionButton) {
        collectSelectionButton.disabled =
          !state.snapshotId || !hasSelection || !hoursValid || !presetReady;
      }
      if (summaryButton) {
        summaryButton.disabled = !liveCapable || !presetReady;
      }
      if (sessionDetailedButton) {
        sessionDetailedButton.disabled = !liveCapable || !presetReady;
      }
      if (topupButton) {
        topupButton.disabled = !liveCapable || !presetReady;
      }
    }

    function populateSnapshots(list) {
      if (!snapshotSelect) return;
      snapshotSelect.innerHTML = "";
      const entries = Array.isArray(list) ? list : [];
      if (state.snapshotId && !entries.some((item) => item.id === state.snapshotId)) {
        state.snapshotId = null;
      }
      for (const item of entries) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = `${item.id} • ${item.symbol || "-"} • ${item.tf || "-"}`;
        snapshotSelect.append(option);
      }
      if (state.snapshotId && entries.some((item) => item.id === state.snapshotId)) {
        snapshotSelect.value = state.snapshotId;
      }
      updateCheckAllState();
    }

    function populateFrames(payload) {
      const frames = payload?.DATA?.frames || {};
      const combined = { ...frames };
      if (state.liveFrames) {
        Object.entries(state.liveFrames).forEach(([tf, entry]) => {
          if (!tf) return;
          combined[tf] = { tf, candles: entry?.candles || [] };
        });
      }
      state.availableFrames = combined;
      const target = selectPreferredFrame(combined, state.frame);
      state.frame = target;
      updateTimeframeToggle();
    }

    function updateTimeframeToggle() {
      if (!timeframeToggle) return;
      const frames = state.availableFrames || state.payload?.DATA?.frames || {};
      const buttons = Array.from(timeframeToggle.querySelectorAll("[data-tf]"));
      for (const button of buttons) {
        const tf = button.dataset.tf;
        const enabled = frameHasCandles(frames, tf);
        button.disabled = !enabled;
        button.classList.toggle("active", enabled && state.frame === tf);
      }
    }

    function renderMeta(payload) {
      if (!snapshotMeta) return;
      const symbolRaw = payload?.DATA?.symbol || initial.symbol;
      const symbol = normaliseSymbol(symbolRaw) || "—";
      const frames = payload?.DATA?.meta?.requested?.frames || [];
      const found = (initial.snapshots || []).find((item) => item.id === state.snapshotId);
      const captured = found && found.captured_at;
      const selection = payload?.DATA?.selection;
      snapshotMeta.innerHTML = `
        <div class=\"meta-grid\">
          <div class=\"meta-tile\"><span>Snapshot</span><strong>${state.snapshotId || "—"}</strong></div>
          <div class=\"meta-tile\"><span>Symbol</span><strong>${symbol || "—"}</strong></div>
          <div class=\"meta-tile\"><span>Таймфреймы</span><strong>${frames.join(", ") || "—"}</strong></div>
          <div class=\"meta-tile\"><span>Захват</span><strong>${captured || "—"}</strong></div>
          <div class=\"meta-tile\"><span>Диапазон</span><strong>${selectionLabel(selection?.start, selection?.end)}</strong></div>
        </div>
      `;
    }

    function renderJson(payload) {
      setJson(diagnosticsPre, payload?.DIAGNOSTICS);
      setJson(checkAllPre, state.checkAll);
    }

    async function requestSelectionData() {
      if (!state.snapshotId) {
        updateStatus("Выберите снэпшот для запроса check-all данных", "warning");
        return;
      }

      const rawStart = state.selection && state.selection.start ? Number(state.selection.start) : null;
      const rawEnd = state.selection && state.selection.end ? Number(state.selection.end) : null;
      if (!Number.isFinite(rawStart) || !Number.isFinite(rawEnd)) {
        updateStatus("Выберите две свечи на графике перед сбором подробных данных", "warning");
        updateCheckAllState();
        return;
      }

      const selectionStart = Math.min(Math.floor(rawStart), Math.floor(rawEnd));
      const selectionEnd = Math.max(Math.floor(rawStart), Math.floor(rawEnd));
      state.hours = resolveHours(state.hours);
      if (checkAllHours) {
        checkAllHours.value = String(state.hours);
      }

      if (collectSelectionButton) {
        collectSelectionButton.disabled = true;
      }

      try {
        updateStatus("Загружаем check-all данные...", "info");
        const url = new URL("/inspection/check-all", window.location.origin);
        url.searchParams.set("snapshot", state.snapshotId);
        url.searchParams.set("mode", "selection");
        url.searchParams.set("selection_start", String(selectionStart));
        url.searchParams.set("selection_end", String(selectionEnd));
        url.searchParams.set("hours", String(state.hours));
        const response = await fetch(url.toString(), {
          headers: { Accept: "application/json" },
          cache: "no-store",
        });
        if (response.status === 204) {
          state.checkAll = null;
          setJson(checkAllPre, null);
          updateStatus("Check-all данные отсутствуют", "warning");
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        state.checkAll = payload;
        setJson(checkAllPre, payload);
        updateStatus("Check-all данные обновлены", "success");
      } catch (error) {
        console.error(error);
        state.checkAll = null;
        setJson(checkAllPre, null);
        updateStatus("Ошибка запроса check-all данных", "error");
      } finally {
        updateCheckAllState();
      }
    }

    function resolveLiveMetaSnapshot() {
      if (priceStore && typeof priceStore.getMeta === "function") {
        try {
          const meta = priceStore.getMeta();
          if (meta && typeof meta === "object") {
            return { ...meta };
          }
        } catch (error) {
          console.warn("Не удалось получить метаданные лайв-стрима", error);
        }
      }
      if (state.liveMeta && typeof state.liveMeta === "object") {
        return { ...state.liveMeta };
      }
      return null;
    }

    function resolveLiveRangeEndMs(meta) {
      const candidates = [];
      if (meta && Number.isFinite(meta.last_ts_ms)) {
        candidates.push(Number(meta.last_ts_ms));
      }
      if (priceStore && typeof priceStore.getLastCandle === "function") {
        try {
          const last = priceStore.getLastCandle();
          if (last) {
            const preview = ensurePreviewBar(last);
            if (preview) {
              const ts = Number.isFinite(preview.ts_ms_utc)
                ? Math.floor(preview.ts_ms_utc)
                : Math.floor(preview.time * 1000);
              if (Number.isFinite(ts)) {
                candidates.push(ts);
              }
            }
          }
        } catch (error) {
          console.warn("Не удалось прочитать последнюю свечу лайв-потока", error);
        }
      }
      const minuteLive = state.liveFrames?.["1m"]?.candles;
      if (Array.isArray(minuteLive) && minuteLive.length) {
        const preview = ensurePreviewBar(minuteLive[minuteLive.length - 1]);
        if (preview) {
          const ts = Number.isFinite(preview.ts_ms_utc)
            ? Math.floor(preview.ts_ms_utc)
            : Math.floor(preview.time * 1000);
          if (Number.isFinite(ts)) {
            candidates.push(ts);
          }
        }
      }
      const finite = candidates.filter((value) => Number.isFinite(value));
      if (!finite.length) {
        return Date.now();
      }
      return Math.max(...finite);
    }

    function normaliseSnapshotCandles(candles) {
      const result = [];
      const seen = new Set();
      for (const candle of Array.isArray(candles) ? candles : []) {
        const preview = ensurePreviewBar(candle);
        if (!preview) continue;
        const ts = Number.isFinite(preview.ts_ms_utc)
          ? Math.floor(preview.ts_ms_utc)
          : Math.floor(preview.time * 1000);
        if (!Number.isFinite(ts) || seen.has(ts)) continue;
        seen.add(ts);
        const volumeSource = Number(
          candle?.v ??
            candle?.volume ??
            candle?.vol ??
            candle?.qty ??
            Number.NaN,
        );
        const volume = Number.isFinite(volumeSource) ? Math.max(volumeSource, 1e-9) : 1e-9;
        result.push({
          t: ts,
          o: Number(preview.open),
          h: Number(preview.high),
          l: Number(preview.low),
          c: Number(preview.close),
          v: volume,
        });
      }
      result.sort((a, b) => a.t - b.t);
      return result.slice(-5000);
    }

    async function captureLiveSnapshot(options = {}) {
      const lookbackDaysRaw = Number(options?.lookbackDays);
      const lookbackDays = Number.isFinite(lookbackDaysRaw)
        ? Math.max(1, Math.floor(lookbackDaysRaw))
        : 3;
      const mode = typeof options?.mode === "string" ? options.mode : "summary";
      const symbol = activeSymbol();
      if (!symbol) {
        throw new Error("live-snapshot-symbol-missing");
      }
      const liveMeta = resolveLiveMetaSnapshot();
      const endMs = resolveLiveRangeEndMs(liveMeta);
      const lookbackMs = lookbackDays * 24 * 60 * 60 * 1000;
      const startMs = Math.max(0, Math.floor(endMs - lookbackMs));
      const rawCandles = await fetchCandles(symbol, "1m", startMs, endMs, { limit: 1500 });
      const candles = normaliseSnapshotCandles(rawCandles);
      if (!candles.length) {
        throw new Error("live-snapshot-empty");
      }
      const frames = { "1m": { tf: "1m", candles } };
      const metaBlock = {
        source: {
          kind: "live-store",
          mode,
          lookback_days: lookbackDays,
          captured_at: new Date().toISOString(),
        },
        requested: {
          frames: Object.keys(frames),
          lookback_days: lookbackDays,
        },
      };
      if (liveMeta) {
        const livePayload = {
          last_price: Number.isFinite(liveMeta.last_price) ? Number(liveMeta.last_price) : null,
          last_tf: liveMeta.last_tf || "1m",
          last_ts_ms: Number.isFinite(liveMeta.last_ts_ms) ? Number(liveMeta.last_ts_ms) : null,
          age_sec: Number.isFinite(liveMeta.age_sec) ? Number(liveMeta.age_sec) : null,
          stale: Boolean(liveMeta.stale),
          mismatch: Boolean(liveMeta.mismatch),
        };
        metaBlock.live = livePayload;
        metaBlock.stream = livePayload;
        metaBlock.live_price = livePayload;
      }
      const payload = {
        symbol,
        tf: "1m",
        candles,
        frames,
        meta: metaBlock,
        lookback_days: lookbackDays,
      };
      const result = await postSnapshot(payload);
      if (!result || !result.snapshot_id) {
        throw new Error("live-snapshot-registration-failed");
      }
      return { snapshotId: result.snapshot_id, payload };
    }

    async function requestSummaryData() {
      if (summaryButton) {
        summaryButton.disabled = true;
      }

      let createdSnapshotId = null;

      try {
        updateStatus("Собираем лайв-данные за последние 3 дня...", "info");
        const { snapshotId } = await captureLiveSnapshot({ lookbackDays: 3, mode: "summary" });
        createdSnapshotId = snapshotId;
        state.snapshotId = snapshotId;
        updateCheckAllState();
        if (summaryButton) {
          summaryButton.disabled = true;
        }
        if (snapshotSelect) {
          snapshotSelect.value = snapshotId;
        }
        const url = new URL("/inspection/check-all", window.location.origin);
        url.searchParams.set("snapshot", snapshotId);
        url.searchParams.set("mode", "summary");
        url.searchParams.set("summary_days", "3");
        const response = await fetch(url.toString(), {
          headers: { Accept: "application/json" },
          cache: "no-store",
        });
        if (response.status === 204) {
          state.checkAll = null;
          setJson(checkAllPre, null);
          updateStatus("Не удалось собрать 3-дневный контекст", "warning");
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        state.checkAll = payload;
        setJson(checkAllPre, payload);
        updateStatus("3-дневный контекст готов", "success");
      } catch (error) {
        console.error(error);
        state.checkAll = null;
        setJson(checkAllPre, null);
        updateStatus("Ошибка при сборе 3-дневного контекста", "error");
      } finally {
        if (summaryButton) {
          summaryButton.disabled = false;
        }
        if (createdSnapshotId) {
          refreshSnapshots({ quiet: true }).catch((err) => {
            console.warn("Не удалось обновить список снэпшотов", err);
          });
        }
        updateCheckAllState();
      }
    }

    async function requestSessionDetailedData() {
      if (sessionDetailedButton) {
        sessionDetailedButton.disabled = true;
      }

      let createdSnapshotId = null;

      try {
        updateStatus("Собираем подробные данные по последней сессии...", "info");
        const { snapshotId } = await captureLiveSnapshot({ lookbackDays: 1, mode: "session_detailed" });
        createdSnapshotId = snapshotId;
        state.snapshotId = snapshotId;
        updateCheckAllState();
        if (sessionDetailedButton) {
          sessionDetailedButton.disabled = true;
        }
        if (snapshotSelect) {
          snapshotSelect.value = snapshotId;
        }
        const url = new URL("/inspection/check-all", window.location.origin);
        url.searchParams.set("snapshot", snapshotId);
        url.searchParams.set("mode", "session_detailed");
        const response = await fetch(url.toString(), {
          headers: { Accept: "application/json" },
          cache: "no-store",
        });
        if (response.status === 204) {
          state.checkAll = null;
          setJson(checkAllPre, null);
          updateStatus("Не удалось собрать данные по последней сессии", "warning");
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        state.checkAll = payload;
        setJson(checkAllPre, payload);
        updateStatus("Сессионный отчёт готов", "success");
      } catch (error) {
        console.error(error);
        state.checkAll = null;
        setJson(checkAllPre, null);
        updateStatus("Ошибка при сборе данных по последней сессии", "error");
      } finally {
        if (sessionDetailedButton) {
          sessionDetailedButton.disabled = false;
        }
        if (createdSnapshotId) {
          refreshSnapshots({ quiet: true }).catch((err) => {
            console.warn("Не удалось обновить список снэпшотов", err);
          });
        }
        updateCheckAllState();
      }
    }

    async function requestTopupData() {
      if (topupButton) {
        topupButton.disabled = true;
      }

      let createdSnapshotId = null;

      try {
        updateStatus("Дособираем свежие лайв-данные...", "info");
        const { snapshotId } = await captureLiveSnapshot({ lookbackDays: 1, mode: "topup" });
        createdSnapshotId = snapshotId;
        state.snapshotId = snapshotId;
        updateCheckAllState();
        if (topupButton) {
          topupButton.disabled = true;
        }
        if (snapshotSelect) {
          snapshotSelect.value = snapshotId;
        }
        const url = new URL("/inspection/check-all", window.location.origin);
        url.searchParams.set("snapshot", snapshotId);
        url.searchParams.set("mode", "topup");
        const response = await fetch(url.toString(), {
          headers: { Accept: "application/json" },
          cache: "no-store",
        });
        if (response.status === 204) {
          state.checkAll = null;
          setJson(checkAllPre, null);
          updateStatus("Свежие данные отсутствуют", "warning");
          return;
        }
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        state.checkAll = payload;
        setJson(checkAllPre, payload);
        updateStatus("Данные успешно дособраны", "success");
      } catch (error) {
        console.error(error);
        state.checkAll = null;
        setJson(checkAllPre, null);
        updateStatus("Ошибка при досборе данных", "error");
      } finally {
        if (topupButton) {
          topupButton.disabled = false;
        }
        if (createdSnapshotId) {
          refreshSnapshots({ quiet: true }).catch((err) => {
            console.warn("Не удалось обновить список снэпшотов", err);
          });
        }
        updateCheckAllState();
      }
    }

    function barsEqual(a, b) {
      if (!a || !b) return false;
      return (
        Number(a.time) === Number(b.time) &&
        Number(a.open) === Number(b.open) &&
        Number(a.high) === Number(b.high) &&
        Number(a.low) === Number(b.low) &&
        Number(a.close) === Number(b.close)
      );
    }

    function ensureChartBar(input) {
      if (!input) return null;
      const time = Number(
        input.time ??
          input.t ??
          (Number.isFinite(input.ts_ms_utc) ? Math.floor(Number(input.ts_ms_utc) / 1000) : null),
      );
      const open = Number(input.open ?? input.o ?? Number.NaN);
      const high = Number(input.high ?? input.h ?? open);
      const low = Number(input.low ?? input.l ?? open);
      const close = Number(input.close ?? input.c ?? open);
      if (
        !Number.isFinite(time) ||
        !Number.isFinite(open) ||
        !Number.isFinite(high) ||
        !Number.isFinite(low) ||
        !Number.isFinite(close)
      ) {
        const normalised = normaliseBar(input);
        if (!normalised) return null;
        return ensureChartBar(normalised);
      }
      let tsMs = Number(input.ts_ms_utc ?? input.t ?? Number.NaN);
      if (!Number.isFinite(tsMs) && Number.isFinite(time)) {
        tsMs = Math.floor(time * 1000);
      }
      return {
        time: Math.floor(time),
        open,
        high,
        low,
        close,
        ts_ms_utc: Number.isFinite(tsMs) ? Math.floor(tsMs) : Math.floor(time * 1000),
      };
    }

    function persistChartCandles({ bars = null, reset = false, lastUpdateMs = null } = {}) {
      if (!SharedCandles || typeof SharedCandles.merge !== "function") return;
      const symbol = activeSymbol();
      const timeframe = state.frame || "1m";
      if (!symbol || !timeframe) return;
      const payload = Array.isArray(bars) && bars.length ? bars : state.candles;
      if (!payload.length) return;
      const effectiveUpdate = Number.isFinite(lastUpdateMs)
        ? Number(lastUpdateMs)
        : Number.isFinite(state.lastUpdateMs)
        ? Number(state.lastUpdateMs)
        : Date.now();
      try {
        SharedCandles.merge(symbol, timeframe, payload, {
          intervalMs: state.intervalMs || intervalToMs(timeframe),
          lastUpdateMs: effectiveUpdate,
          maxBars: SHARED_MAX_BARS,
          reset,
        });
      } catch (error) {
        console.warn("SharedCandles merge failed", error);
      }
    }

    async function restoreChartFromShared(symbol, interval) {
      if (!SharedCandles) return false;
      const timeframe = interval || "1m";
      let restored = false;

      const normalise = (bars) =>
        (bars || [])
          .map((bar) => ensureChartBar(bar))
          .filter((bar) => bar !== null);

      const applyBars = (bars, meta) => {
        if (!bars.length) return false;
        if (meta && Number.isFinite(meta.intervalMs)) {
          state.intervalMs = Number(meta.intervalMs);
        }
        if (meta && Number.isFinite(meta.lastUpdateMs)) {
          state.lastUpdateMs = Number(meta.lastUpdateMs);
        } else if (meta && Number.isFinite(meta.updatedAt)) {
          state.lastUpdateMs = Number(meta.updatedAt);
        }
        mergeChartBars(bars, { reset: true, persist: false });
        return true;
      };

      try {
        if (typeof SharedCandles.get === "function") {
          const local = SharedCandles.get(symbol, timeframe);
          if (local && Array.isArray(local.candles) && local.candles.length) {
            const bars = normalise(local.candles);
            if (applyBars(bars, local)) {
              restored = true;
            }
          }
        }
      } catch (error) {
        console.warn("SharedCandles local restore failed", error);
      }

      if (restored) {
        return true;
      }

      if (typeof SharedCandles.fetchRemote !== "function") {
        return false;
      }

      try {
        const remote = await SharedCandles.fetchRemote(symbol, timeframe);
        if (remote && Array.isArray(remote.candles) && remote.candles.length) {
          const bars = normalise(remote.candles);
          if (applyBars(bars, remote)) {
            restored = true;
          }
        }
      } catch (error) {
        console.warn("SharedCandles remote restore failed", error);
      }

      return restored;
    }

    function mergeChartBars(bars, { reset = false, persist = true } = {}) {
      const incoming = (bars || []).map((bar) => ensureChartBar(bar)).filter((bar) => bar !== null);
      if (reset) {
        const changed =
          incoming.length !== state.candles.length ||
          incoming.some((bar, idx) => !barsEqual(bar, state.candles[idx]));
        state.candles = incoming;
        if (state.series) {
          state.series.setData(state.candles);
        }
        if (state.candles.length > SHARED_MAX_BARS) {
          state.candles = state.candles.slice(state.candles.length - SHARED_MAX_BARS);
          if (state.series) {
            state.series.setData(state.candles);
          }
        }
        const lastBar = state.candles[state.candles.length - 1] || null;
        const inferredUpdate = Number(lastBar?.ts_ms_utc ?? lastBar?.time * 1000 ?? Date.now());
        state.lastUpdateMs = Number.isFinite(inferredUpdate) ? inferredUpdate : Date.now();
        if (persist) {
          persistChartCandles({ bars: state.candles, reset: true, lastUpdateMs: state.lastUpdateMs });
        }
        return changed;
      }

      if (!incoming.length) {
        return false;
      }

      const index = new Map();
      state.candles.forEach((bar, idx) => {
        const key = Number(bar.time);
        if (Number.isFinite(key)) {
          index.set(key, idx);
        }
      });

      let changed = false;
      for (const bar of incoming) {
        const key = Number(bar.time);
        if (!Number.isFinite(key)) continue;
        if (index.has(key)) {
          const idx = index.get(key);
          if (!barsEqual(state.candles[idx], bar)) {
            state.candles[idx] = bar;
            changed = true;
          }
        } else {
          index.set(key, state.candles.length);
          state.candles.push(bar);
          changed = true;
        }
      }

      if (changed) {
        state.candles.sort((a, b) => Number(a.time) - Number(b.time));
        if (state.candles.length > SHARED_MAX_BARS) {
          state.candles = state.candles.slice(state.candles.length - SHARED_MAX_BARS);
          if (state.series) {
            state.series.setData(state.candles);
          }
        }
        const lastBar = state.candles[state.candles.length - 1] || null;
        const inferredUpdate = Number(lastBar?.ts_ms_utc ?? lastBar?.time * 1000 ?? Date.now());
        state.lastUpdateMs = Number.isFinite(inferredUpdate) ? inferredUpdate : Date.now();
        if (persist) {
          persistChartCandles({ bars: incoming, reset: false, lastUpdateMs: state.lastUpdateMs });
        }
        if (state.series) {
          state.series.setData(state.candles);
        }
      }
      return changed;
    }

    function ensureGapWatcher(options = {}) {
      if (!ChartGapWatcher || typeof ChartGapWatcher.attach !== "function") return;
      if (!state.chart) return;
      const symbol = activeSymbol();
      const interval = state.frame || "1m";
      const intervalMs = intervalToMs(interval);
      const contextChanged = state.gapSymbol !== symbol || state.gapInterval !== interval;
      const resetRequestedKeys = Boolean(options.resetRequestedKeys) || contextChanged;

      if (!state.gapWatcher) {
        state.gapWatcher = ChartGapWatcher.attach({
          chart: state.chart,
          interval,
          intervalMs,
          getCandles: () => state.candles,
          requestGap: handleChartGapRequest,
        });
      } else if (typeof state.gapWatcher.updateContext === "function") {
        state.gapWatcher.updateContext({
          symbol,
          interval,
          intervalMs,
          getCandles: () => state.candles,
          requestGap: handleChartGapRequest,
          resetRequestedKeys,
        });
      }

      state.gapSymbol = symbol;
      state.gapInterval = interval;
      state.intervalMs = intervalMs;

      if (state.gapWatcher && typeof state.gapWatcher.notifyData === "function") {
        state.gapWatcher.notifyData();
      }
    }

    async function handleChartGapRequest(gap) {
      if (!gap) return false;
      const symbol = activeSymbol();
      const interval = state.frame || "1m";
      if (!symbol || !interval) return false;
      try {
        const startMs = Number(gap.startMs);
        const endMs = Number(gap.endMs);
        if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) {
          return false;
        }
        const intervalMs = intervalToMs(interval);
        const rangeWidth = Math.max(intervalMs, endMs - startMs);
        const approxBars = Math.ceil(rangeWidth / intervalMs) + 2;
        const buffer = intervalMs;
        const bars = await fetchRange(
          symbol,
          interval,
          Math.max(0, startMs - buffer),
          endMs + buffer,
          Math.min(1000, Math.max(approxBars, 50)),
        );
        const changed = mergeChartBars(bars);
        if (changed && state.gapWatcher && typeof state.gapWatcher.notifyData === "function") {
          state.gapWatcher.notifyData();
        }
        return true;
      } catch (error) {
        console.error("Failed to fetch missing candles for inspection chart", error);
        updateStatus("Не удалось загрузить недостающие свечи", "error");
        return false;
      }
    }

    function updateChartDataFromFrame(options = {}) {
      const { resetRequestedKeys = false, persist = false } = options;
      let frameCandles = state.liveFrames?.[state.frame]?.candles;
      if (!Array.isArray(frameCandles) || !frameCandles.length) {
        frameCandles = state.payload?.DATA?.frames?.[state.frame]?.candles || [];
      }
      const bars = toChartBars(frameCandles);
      mergeChartBars(bars, { reset: true, persist });
      state.intervalMs = intervalToMs(state.frame || "1m");
      ensureGapWatcher({ resetRequestedKeys });
    }

    function ensureChart() {
      if (!chartContainer) return;
      const ensureLibrary = () => {
        if (window.LightweightCharts) {
          LightweightCharts = window.LightweightCharts;
          initialiseChart();
        }
      };

      function initialiseChart() {
        LightweightCharts = window.LightweightCharts || LightweightCharts;
        if (!LightweightCharts) {
          updateStatus("Библиотека графика недоступна", "error");
          return;
        }
        if (state.chart) return;
        const baseHeight = Math.max(
          320,
          chartContainer.clientHeight ||
            chartContainer.offsetHeight ||
            (chartContainer.parentElement && chartContainer.parentElement.clientHeight) ||
            320,
        );
        state.chart = LightweightCharts.createChart(chartContainer, {
          autoSize: true,
          height: baseHeight,
          layout: {
            background: { color: "#0f172a" },
            textColor: "#e2e8f0",
          },
          rightPriceScale: {
            borderColor: "rgba(148, 163, 184, 0.4)",
          },
          timeScale: {
            borderColor: "rgba(148, 163, 184, 0.4)",
            timeVisible: true,
            secondsVisible: true,
          },
          crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
          },
          grid: {
            vertLines: { color: "rgba(15, 23, 42, 0.6)" },
            horzLines: { color: "rgba(15, 23, 42, 0.6)" },
          },
        });
        state.series = state.chart.addCandlestickSeries({
          upColor: "#22c55e",
          downColor: "#ef4444",
          wickUpColor: "#f8fafc",
          wickDownColor: "#f8fafc",
          borderUpColor: "#22c55e",
          borderDownColor: "#ef4444",
          borderVisible: true,
        });

        const resize = () => {
          if (!state.chart) return;
          const nextHeight = Math.max(
            320,
            chartContainer.clientHeight ||
              chartContainer.offsetHeight ||
              (chartContainer.parentElement && chartContainer.parentElement.clientHeight) ||
              baseHeight,
          );
          state.chart.applyOptions({ height: nextHeight });
        };

        resize();
        if (window.ResizeObserver) {
          const observer = new ResizeObserver(resize);
          observer.observe(chartContainer);
        } else {
          window.addEventListener("resize", resize);
        }

        state.chart.subscribeClick((param) => {
          if (!param || typeof param.time === "undefined") return;
          const ts = Math.floor(Number(param.time) * 1000);
          if (!state.selection || !state.selection.start || state.selection.end) {
            state.selection = { start: ts, end: null };
          } else {
            state.selection.end = ts;
            if (state.selection.end < state.selection.start) {
              const tmp = state.selection.start;
              state.selection.start = state.selection.end;
              state.selection.end = tmp;
            }
          }
          updateSelectionLabel();
          updateCheckAllState();
        });
      }

      if (window.LightweightCharts) {
        LightweightCharts = window.LightweightCharts;
        initialiseChart();
        return;
      }

      let loader = document.getElementById("lw-chart-loader");
      if (!loader) {
        loader = document.createElement("script");
        loader.src = "https://unpkg.com/lightweight-charts@4.0.0/dist/lightweight-charts.standalone.production.js";
        loader.id = "lw-chart-loader";
        loader.async = false;
        loader.onload = () => {
          LightweightCharts = window.LightweightCharts || LightweightCharts;
          ensureLibrary();
        };
        loader.onerror = () => updateStatus("Не удалось загрузить библиотеку графика", "error");
        document.head.appendChild(loader);
      }
    }

    function ensureFrameData({ resetRequestedKeys = false, fitContent = false } = {}) {
      const frames = state.payload?.DATA?.frames || {};
      const timeframe = state.frame || "1m";
      state.intervalMs = intervalToMs(timeframe);
      const symbol = activeSymbol();
      const liveEntry = state.liveFrames?.[timeframe];
      if (liveEntry && Array.isArray(liveEntry.candles) && liveEntry.candles.length) {
        mergeChartBars(liveEntry.candles, { reset: true, persist: false });
        ensureGapWatcher({ resetRequestedKeys });
        if (fitContent && state.chart && state.candles.length) {
          state.chart.timeScale().fitContent();
        }
        return Promise.resolve(true);
      }
      const hasFrameData = frameHasCandles(frames, timeframe);

      if (hasFrameData) {
        updateChartDataFromFrame({ resetRequestedKeys, persist: false });
        if (fitContent && state.chart && state.candles.length) {
          state.chart.timeScale().fitContent();
        }
        return Promise.resolve(true);
      }

      if (!symbol) {
        ensureGapWatcher({ resetRequestedKeys });
        return Promise.resolve(false);
      }

      return restoreChartFromShared(symbol, timeframe)
        .then((restored) => {
          if (restored) {
            ensureGapWatcher({ resetRequestedKeys });
          } else if (hasFrameData) {
            updateChartDataFromFrame({ resetRequestedKeys, persist: false });
          } else {
            ensureGapWatcher({ resetRequestedKeys });
          }
          if (fitContent && state.chart && state.candles.length) {
            state.chart.timeScale().fitContent();
          }
          return restored;
        })
        .catch((error) => {
          console.warn("SharedCandles restore failed", error);
          if (hasFrameData) {
            updateChartDataFromFrame({ resetRequestedKeys, persist: false });
            if (fitContent && state.chart && state.candles.length) {
              state.chart.timeScale().fitContent();
            }
          } else {
            ensureGapWatcher({ resetRequestedKeys });
          }
          return false;
        });
    }

    function renderChart(options = {}) {
      if (!chartContainer) return;
      LightweightCharts = window.LightweightCharts || LightweightCharts;
      if (!LightweightCharts) {
        updateStatus("Библиотека графика недоступна", "error");
        return;
      }
      ensureChart();
      const fitContent = options.fitContent !== false;
      ensureFrameData({ resetRequestedKeys: options.resetRequestedKeys, fitContent })
        .then(() => {
          updateSelectionLabel();
          updateTimeframeToggle();
        })
        .catch((error) => {
          console.warn("Failed to render chart", error);
        });
    }

    async function refreshSnapshots(options = {}) {
      const quiet = Boolean(options?.quiet);
      try {
        const list = await fetchSnapshots();
        initial.snapshots = list;
        populateSnapshots(list);
      } catch (error) {
        console.error(error);
        if (!quiet) {
          updateStatus("Не удалось загрузить список снэпшотов", "error");
        }
      }
    }

    async function loadSnapshot(id) {
      if (!id) return;
      updateStatus("Загружаем данные снэпшота...", "info");
      try {
        const payload = await fetchPayload(id);
        state.payload = payload;
        state.snapshotId = id;
        state.selection = payload?.DATA?.selection || null;
        state.checkAll = null;
        setJson(checkAllPre, null);
        updateCheckAllState();
        state.profilePreset = payload?.DATA?.profile_preset || null;
        state.presetRequired = Boolean(payload?.DATA?.profile_preset_required);
        state.presetDefaults = payload?.DATA?.profile_defaults || null;
        renderPresetState();
        const nextSymbol = payload?.DATA?.symbol || initial.symbol;
        if (symbolInput) {
          const resolved = normaliseSymbol(nextSymbol) || normaliseSymbol(initial.symbol) || defaultSymbol;
          symbolInput.value = resolved;
          renderChartSymbol();
        }
        populateFrames(payload);
        syncLiveStores({ force: true });
        renderJson(payload);
        renderMeta(payload);
        renderChart({ resetRequestedKeys: true, fitContent: true });
        updateStatus("Снэпшот загружен", "success");
      } catch (error) {
        console.error(error);
        updateStatus("Ошибка загрузки снэпшота", "error");
      }
    }

    if (snapshotSelect) {
      snapshotSelect.addEventListener("change", (event) => {
        const value = event.target.value;
        loadSnapshot(value);
      });
    }

    if (timeframeToggle) {
      timeframeToggle.addEventListener("click", (event) => {
        const button = event.target.closest("[data-tf]");
        if (!button || button.disabled) return;
        const tf = button.dataset.tf;
        if (!tf) return;
        state.frame = tf;
        renderChart({ resetRequestedKeys: true });
        updateTimeframeToggle();
        syncLiveStores({ force: true });
      });
    }

    if (refreshButton) {
      refreshButton.addEventListener("click", () => {
        if (state.snapshotId) {
          loadSnapshot(state.snapshotId);
        }
      });
    }

    if (managePresetsButton) {
      managePresetsButton.addEventListener("click", () => {
        openPresetModal({ mode: "list" });
      });
    }

    if (presetCreateButton) {
      presetCreateButton.addEventListener("click", () => {
        if (presetModalTitle) presetModalTitle.textContent = "Создать пресет";
        if (presetListView) presetListView.hidden = true;
        if (presetFormSection) presetFormSection.hidden = false;
        state.presetModalMode = "form";
        populatePresetForm(null, activeSymbol());
      });
    }

    if (presetForm) {
      presetForm.addEventListener("submit", submitPresetForm);
    }

    if (presetBinningModeField) {
      presetBinningModeField.addEventListener("change", handleBinningModeChange);
    }

    presetCancelButtons.forEach((btn) => {
      btn.addEventListener("click", () => {
        closePresetModal();
      });
    });

    if (presetBackdrop) {
      presetBackdrop.addEventListener("click", () => closePresetModal());
    }

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && state.presetModalOpen) {
        closePresetModal();
      }
    });

    if (checkAllHours) {
      checkAllHours.value = String(state.hours);
      checkAllHours.addEventListener("change", () => {
        state.hours = resolveHours(checkAllHours.value);
        checkAllHours.value = String(state.hours);
        updateCheckAllState();
      });
    }

    if (summaryButton) {
      summaryButton.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        requestSummaryData();
      });
    }

    if (sessionDetailedButton) {
      sessionDetailedButton.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        requestSessionDetailedData();
      });
    }

    if (topupButton) {
      topupButton.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        requestTopupData();
      });
    }

    if (collectSelectionButton) {
      collectSelectionButton.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        requestSelectionData();
      });
    }

    if (clearSelection) {
      clearSelection.addEventListener("click", () => {
        state.selection = null;
        updateSelectionLabel();
        updateCheckAllState();
      });
    }

    async function createSnapshotFromSelection({ symbolValue, selection, frames, source = "inspection-ui" }) {
      if (!selection || !selection.start || !selection.end) {
        throw new Error('selection-missing');
      }
      const uniqueFrames = Array.from(new Set((frames || []).filter(Boolean)));
      if (!uniqueFrames.length) {
        throw new Error('frames-missing');
      }
      const resolvedSymbol =
        normaliseSymbol(symbolValue) || normaliseSymbol(initial.symbol) || defaultSymbol;
      if (!resolvedSymbol) {
        throw new Error('symbol-invalid');
      }
      const rawStart = Number(selection.start);
      const rawEnd = Number(selection.end);
      if (!Number.isFinite(rawStart) || !Number.isFinite(rawEnd)) {
        throw new Error('selection-invalid');
      }
      const selectionStart = Math.floor(Math.min(rawStart, rawEnd));
      const selectionEnd = Math.floor(Math.max(rawStart, rawEnd));
      const framesPayload = {};
      for (const tf of uniqueFrames) {
        const candles = await fetchCandles(resolvedSymbol, tf, selectionStart, selectionEnd);
        framesPayload[tf] = { tf, candles };
      }
      const baseFrame = uniqueFrames[0] || Object.keys(framesPayload)[0];
      const baseCandles = (baseFrame && framesPayload[baseFrame]?.candles) || [];
      const intervalMs = TIMEFRAME_TO_MS[baseFrame] || 60000;
      const aggTrades = [];
      for (const candle of baseCandles || []) {
        const rawTs = Number(candle?.t ?? candle?.time ?? 0);
        if (!Number.isFinite(rawTs)) continue;
        const open = Number(candle?.o ?? candle?.open ?? 0);
        const close = Number(candle?.c ?? candle?.close ?? open);
        const qty = Math.max(0.01, Math.abs(close - open) / Math.max(1, intervalMs / 60_000));
        aggTrades.push({
          t: rawTs + Math.floor(intervalMs / 2),
          p: Number.isFinite(close) ? Number(close.toFixed(2)) : Number(open.toFixed(2)),
          q: Number(qty.toFixed(4)),
          side: close >= open ? "buy" : "sell",
        });
      }
      const payload = {
        id: `test-${Date.now()}`,
        symbol: resolvedSymbol,
        frames: framesPayload,
        selection: { start: selectionStart, end: selectionEnd },
        agg_trades: { symbol: resolvedSymbol, agg: aggTrades },
        meta: {
          source: {
            kind: source,
            frames: uniqueFrames,
            generated_at: new Date().toISOString(),
          },
        },
      };
      const result = await postSnapshot(payload);
      return { snapshotId: result.snapshot_id, payload };
    }

    if (buildButton) {
      buildButton.addEventListener("click", async () => {
        if (!state.selection || !state.selection.start || !state.selection.end) {
          updateStatus("Select a range on the chart before creating the test environment", "warning");
          return;
        }
        const selectedFrames = timeframeCheckboxes
          .filter((checkbox) => checkbox.checked)
          .map((checkbox) => checkbox.value);
        if (!selectedFrames.length) {
          updateStatus("No timeframes selected for testing", "warning");
          return;
        }
        const symbolValue = symbolInput ? symbolInput.value : initial.symbol;
        updateStatus("Collecting Binance data...", "info");
        buildButton.disabled = true;
        try {
          const { snapshotId } = await createSnapshotFromSelection({
            symbolValue,
            selection: state.selection,
            frames: selectedFrames,
            source: "inspection-panel",
          });
          state.snapshotId = snapshotId;
          updateCheckAllState();
          await refreshSnapshots();
          if (snapshotSelect) snapshotSelect.value = state.snapshotId;
          await loadSnapshot(state.snapshotId);
          updateStatus("Test environment created", "success");
        } catch (error) {
          console.error(error);
          const detail = error && typeof error.message === "string" ? error.message : "";
          updateStatus(`Failed to create test environment${detail ? `: ${detail}` : ""}`, "error");
        } finally {
          buildButton.disabled = false;
        }
      });
    }


    renderJson(state.payload);
    populateFrames(state.payload);
    populateSnapshots(initial.snapshots || []);
    renderMeta(state.payload);
    renderChart();
    await refreshSnapshots();
    if (state.snapshotId && snapshotSelect) {
      snapshotSelect.value = state.snapshotId;
    }
  });
})();
"""

    page_html = f"""
    <!DOCTYPE html>
    <html lang=\"ru\">
      <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
        <title>Inspection Dashboard</title>
        <style>{style_block}</style>
      </head>
      <body>
        <header>
          <h1>Панель тестирования данных графика</h1>
          <p>Сбор свежих свечей, выбор диапазона и проверка расчётов без сохранения данных на сервере.</p>
        </header>
        <main>
          <section class=\"panel panel--collection\">
            <h2>Сбор данных</h2>
            <p class=\"panel-lead\">Собирайте актуальную информацию без сохранения снэпшотов на сервере.</p>
            <div class=\"collection-actions\">
              <button id=\"collect-summary\" class=\"primary\" type=\"button\">Собрать информацию за последние 3 дня</button>
              <button id=\"btn_collect_last_session_detailed\" class=\"secondary\" type=\"button\" title=\"Последняя завершённая или текущая активная сессия с полнотой ≥90%\">Собрать информацию за последнюю сессию подробно</button>
              <button id=\"collect-topup\" class=\"secondary\" type=\"button\">Дособрать данные</button>
              <button id=\"collect-selection\" class=\"secondary\" type=\"button\" disabled>Собрать информацию за выбранный период</button>
            </div>
            <div class=\"status-banner\" id=\"inspection-status\" hidden data-tone=\"info\"></div>
            <div class=\"preset-chip-bar\">
              <span id=\"preset-chip\" class=\"preset-chip\" hidden></span>
            </div>
            <div class=\"controls-grid\">
              <label>
                <span>Символ</span>
                <input id=\"symbol-input\" type=\"text\" value=\"{symbol_value}\" autocomplete=\"off\" list=\"symbol-suggestions\" />
              </label>
              <datalist id=\"symbol-suggestions\">
                <option value=\"BTCUSDT\"></option>
                <option value=\"ETHUSDT\"></option>
                <option value=\"BNBUSDT\"></option>
                <option value=\"SOLUSDT\"></option>
                <option value=\"XRPUSDT\"></option>
              </datalist>
            </div>
            <div class=\"selection-bar\">
              <span class=\"badge\">Выделенный диапазон</span>
              <div>
                <span id=\"selection-info\">—</span>
                <button id=\"clear-selection\" class=\"secondary\" type=\"button\">Сбросить выделение</button>
              </div>
            </div>
            <div class=\"live-meta\" id=\"live-meta\">
              <div class=\"live-meta__item\">
                <span>Последняя цена</span>
                <strong id=\"live-last-price\" class=\"live-meta__value\">—</strong>
              </div>
              <div class=\"live-meta__item\">
                <span>Обновлено (UTC)</span>
                <strong id=\"live-last-ts\" class=\"live-meta__value\">—</strong>
              </div>
              <div class=\"live-meta__item\">
                <span>Таймфрейм</span>
                <strong id=\"live-last-tf\" class=\"live-meta__value\">—</strong>
              </div>
              <div class=\"live-meta__item\">
                <span>Возраст</span>
                <strong id=\"live-age-sec\" class=\"live-meta__value\">—</strong>
              </div>
              <div class=\"live-meta__item\">
                <span>Статус</span>
                <strong id=\"live-stale-flag\" class=\"live-meta__value\" data-state=\"unknown\">—</strong>
              </div>
            </div>
            <div class=\"status-banner\" id=\"stream-mismatch\" hidden data-tone=\"warning\">Stream vs OHLCV mismatch &gt; 1 tick</div>
            <div id=\"snapshot-meta\"></div>
          </section>


          <section class=\"panel panel--view\">
            <h2>Просмотр данных</h2>
            <div class="chart-toolbar">
              <div class="chart-toolbar__symbol">
                <span class="badge">Символ</span>
                <strong id="chart-symbol">{symbol_value}</strong>
              </div>
              <div class="chart-toolbar__frames">
                <span class="badge">Таймфрейм</span>
                <div class="tf-toggle" id="chart-tf-toggle">
                  <button type="button" data-tf="1m">1m</button>
                  <button type="button" data-tf="3m">3m</button>
                  <button type="button" data-tf="15m">15m</button>
                  <button type="button" data-tf="30m">30m</button>
                  <button type="button" data-tf="1h">1h</button>
                  <button type="button" data-tf="4h">4h</button>
                  <button type="button" data-tf="1d">1d</button>
                  <button type="button" data-tf="1w">1w</button>
                </div>
              </div>
            </div>
            <div id=\"inspection-chart\" class=\"chart-shell\" data-selection-label=\"—\"></div>
            <div class="json-panels">
              <div class="collapse">
                <header data-collapse-toggle>
                  <h3>DIAGNOSTICS</h3>
                  <button class="secondary" type="button" data-copy-target="diagnostics-json">Copy JSON</button>
                </header>
                <pre id="diagnostics-json">{diagnostics_json_initial}</pre>
              </div>
              <div class="collapse">
                <header data-collapse-toggle>
                  <h3>CHECK ALL DATAS</h3>
                  <button class="secondary" type="button" data-copy-target="checkall-json">Copy JSON</button>
                </header>
                <div class="checkall-control">
                  <div class="checkall-control__row">
                    <span>Часов для подробного сбора</span>
                    <select id="checkall-hours">
                      <option value="1">1 час</option>
                      <option value="2">2 часа</option>
                      <option value="3">3 часа</option>
                      <option value="4">4 часа</option>
                    </select>
                  </div>
                </div>
                <pre id="checkall-json">{check_all_json_initial}</pre>
              </div>
            </div>
          </section>
        <div id="preset-modal" class="modal" hidden>
          <div id="preset-modal-backdrop" class="modal__backdrop"></div>
          <div class="modal__dialog" role="dialog" aria-modal="true" aria-labelledby="preset-modal-title">
            <header class="modal__header">
              <h3 id="preset-modal-title">Настроить пресет TPO</h3>
              <button class="modal__close" type="button" data-close-preset>&times;</button>
            </header>
            <div class="modal__body">
              <div id="preset-list-view" hidden>
                <div class="preset-list" id="preset-list"></div>
                <button id="preset-create-button" class="btn-primary" type="button">Создать пресет</button>
              </div>
              <form id="preset-form" hidden>
                <input type="hidden" id="preset-mode-input" value="create" />
                <div class="preset-form-grid" id="preset-form-section">
                  <label>
                    <span>Символ</span>
                    <input id="preset-symbol" type="text" required autocomplete="off" />
                  </label>
                  <label>
                    <span>Таймфрейм</span>
                    <select id="preset-tf">
                      <option value="1m">1m</option>
                    </select>
                  </label>
                  <label>
                    <span>Количество сессий</span>
                    <input id="preset-last-n" type="number" min="1" max="5" step="1" />
                  </label>
                  <label>
                    <span>Value area %</span>
                    <input id="preset-value-area" type="number" min="0.1" max="0.95" step="0.01" />
                  </label>
                  <label>
                    <span>Режим биннинга</span>
                    <select id="preset-binning-mode">
                      <option value="adaptive">Adaptive (ATR)</option>
                      <option value="tick">Tick size</option>
                    </select>
                  </label>
                  <label data-preset-tick>
                    <span>Tick size</span>
                    <input id="preset-tick-size" type="number" step="0.0001" min="0" />
                  </label>
                  <label data-preset-adaptive>
                    <span>ATR multiplier</span>
                    <input id="preset-atr-multiplier" type="number" step="0.05" min="0.1" max="2" />
                  </label>
                  <label data-preset-adaptive>
                    <span>Целевые бины</span>
                    <input id="preset-target-bins" type="number" step="5" min="40" max="200" />
                  </label>
                  <label>
                    <span>Отсечение хвоста</span>
                    <input id="preset-clip-tail" type="number" step="0.001" min="0" max="0.05" />
                  </label>
                  <label>
                    <span>Сглаживание</span>
                    <select id="preset-smooth-window">
                      <option value="1">Без сглаживания</option>
                      <option value="2">Окно 2</option>
                      <option value="3">Окно 3</option>
                    </select>
                  </label>
                </div>
                <div class="preset-form-actions">
                  <button id="preset-submit" class="btn-primary" type="submit">Сохранить</button>
                  <button class="btn-secondary" type="button" data-close-preset>Отмена</button>
                </div>
              </form>
            </div>
          </div>
        </div>
        </main>
        <script>{script_block}</script>
        <script src=\"https://unpkg.com/lightweight-charts@4.0.0/dist/lightweight-charts.standalone.production.js\"></script>
        <script src="/public/binanceCandles.js?v={static_version}"></script>
        <script src="/public/chart-gap-watcher.js?v={static_version}"></script>
        <script src="/public/shared-candles.js?v={static_version}"></script>
        <script src="/public/market-data-store.js?v={static_version}"></script>
        <script>{ui_script}</script>
      </body>
    </html>
    """
    return page_html



def validate_enhanced_snapshot(snapshot: Mapping[str, Any]) -> tuple[bool, List[str]]:
    """Run strict validation over inspection snapshot payloads."""

    errors: List[str] = []
    candles = snapshot.get("candles")
    if isinstance(candles, Mapping):
        candles = candles.get("candles")
    if not isinstance(candles, Sequence):
        candles = []
    candles = list(candles)[:1000]

    last_ts: int | None = None
    for index, row in enumerate(candles):
        candle = _parse_minute_row(row) if isinstance(row, Mapping) else _parse_minute_row(row)
        if candle is None:
            errors.append(f"Invalid candle at index {index}")
            continue
        ts = int(candle["t"])
        if last_ts is not None and ts <= last_ts:
            errors.append("Candles must be strictly sorted by time")
            break
        if last_ts is not None and ts - last_ts > MINUTE_INTERVAL_MS * 3:
            logging.getLogger(__name__).debug("Gap detected between %s and %s", last_ts, ts)
        if candle["h"] < max(candle["o"], candle["c"]):
            candle["h"] = max(candle["o"], candle["c"])
        if candle["l"] > min(candle["o"], candle["c"]):
            candle["l"] = min(candle["o"], candle["c"])
        if candle["v"] <= 0:
            candle["v"] = 1e-9
        last_ts = ts

    orderflow = snapshot.get("orderflow") if isinstance(snapshot.get("orderflow"), Mapping) else {}
    footprint = orderflow.get("footprint") if isinstance(orderflow, Mapping) else None
    if isinstance(footprint, Sequence):
        for item in footprint:
            if not isinstance(item, Mapping):
                errors.append("Footprint rows must be objects")
                continue
            bid = _coerce_float(item.get("bid")) or 0.0
            ask = _coerce_float(item.get("ask")) or 0.0
            delta = _coerce_float(item.get("delta")) or 0.0
            if abs((ask - bid) - delta) > 1e-3:
                errors.append("Footprint delta mismatch")
                break

    cvd = orderflow.get("cvd") if isinstance(orderflow, Mapping) else None
    if isinstance(cvd, Sequence):
        for row in cvd:
            if not isinstance(row, Mapping):
                errors.append("CVD rows must be objects")
                break
            buy = _coerce_float(row.get("cvd_buy")) or 0.0
            sell = _coerce_float(row.get("cvd_sell")) or 0.0
            net = _coerce_float(row.get("cvd_net")) or 0.0
            if abs((buy - sell) - net) > 1e-3:
                errors.append("CVD net mismatch")
                break

    derivatives = snapshot.get("derivatives")
    if isinstance(derivatives, Sequence):
        for item in derivatives:
            if not isinstance(item, Mapping):
                errors.append("Derivative rows must be objects")
                break
            oi = _coerce_float(item.get("oi"))
            funding = _coerce_float(item.get("funding"))
            if oi is None or oi <= 0:
                errors.append("Open interest must be positive")
            if funding is None:
                errors.append("Funding rate missing")

    book = snapshot.get("book")
    if isinstance(book, Mapping):
        if not isinstance(book.get("top_levels"), Sequence):
            errors.append("Orderbook top_levels missing")

    valid = not errors
    if valid:
        logging.getLogger(__name__).info("Snapshot validation succeeded")
    else:
        logging.getLogger(__name__).warning("Snapshot validation failed: %s", errors)
    return valid, errors
