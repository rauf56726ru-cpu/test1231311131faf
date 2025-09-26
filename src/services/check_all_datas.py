"""Snapshot diagnostics builder for the inspection check-all endpoint."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


UTC = timezone.utc
MS_IN_HOUR = 3_600_000
MS_IN_DAY = 86_400_000
VALID_HOUR_WINDOWS = {1, 2, 3, 4}


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

    snapshot_selection = snapshot.get("selection") if isinstance(snapshot.get("selection"), Mapping) else None
    start_ms = selection_start_ms or _safe_int(snapshot_selection.get("start")) if snapshot_selection else None
    end_ms = selection_end_ms or _safe_int(snapshot_selection.get("end")) if snapshot_selection else None

    if start_ms is None:
        start_ms = primary_candles[0]["t"]
    if end_ms is None:
        end_ms = primary_candles[-1]["t"]

    if start_ms > end_ms:
        start_ms, end_ms = end_ms, start_ms

    hours_window = hours if hours in VALID_HOUR_WINDOWS else min(VALID_HOUR_WINDOWS)

    if now_utc is not None:
        if now_utc.tzinfo is None:
            reference_dt = now_utc.replace(tzinfo=UTC)
        else:
            reference_dt = now_utc.astimezone(UTC)
        reference_ts = int(reference_dt.timestamp() * 1000)
    else:
        reference_ts = end_ms
        reference_dt = datetime.fromtimestamp(reference_ts / 1000.0, tz=UTC)

    detailed_start_ts = reference_ts - hours_window * MS_IN_HOUR
    movement_anchor_ts = detailed_start_ts
    movement_start_ts = min(start_ms, movement_anchor_ts)
    movement_end_ts = max(start_ms, movement_anchor_ts)
    if movement_end_ts > reference_ts:
        movement_end_ts = reference_ts

    latest_candle_ts = primary_candles[-1]["t"]
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
    if start_ms is not None and end_ms is not None:
        movement_days = max(0, int((end_ms - start_ms) // MS_IN_DAY))

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

    return {
        "snapshot_id": snapshot.get("id"),
        "symbol": snapshot.get("symbol"),
        "timeframe": snapshot.get("tf"),
        "selection": {"start": start_ms, "end": end_ms},
        "asof_utc": reference_dt.isoformat(),
        "latest_candle_utc": latest_candle_dt.isoformat(),
        "latest_candle": dict(primary_candles[-1]),
        "datas_for_last_N_hours": detailed_section,
        movement_key: movement_section,
    }
