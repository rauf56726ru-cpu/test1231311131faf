"""Snapshot diagnostics builder for the inspection check-all endpoint."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence


UTC = timezone.utc


def _coerce_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _coerce_candle(entry: Mapping[str, Any]) -> MutableMapping[str, Any] | None:
    """Normalise a raw candle mapping into numeric OHLCV fields."""

    raw_ts = (
        entry.get("t")
        or entry.get("time")
        or entry.get("openTime")
        or entry.get("open_time")
    )
    if raw_ts is None:
        return None

    try:
        timestamp_ms = int(raw_ts)
    except (TypeError, ValueError):
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


def _resolve_now(candles: Sequence[Mapping[str, Any]], now_utc: datetime | None) -> datetime:
    if now_utc is not None:
        if now_utc.tzinfo is None:
            return now_utc.replace(tzinfo=UTC)
        return now_utc.astimezone(UTC)

    if not candles:
        return datetime.now(UTC)

    last_ts = int(candles[-1]["t"])
    return datetime.fromtimestamp(last_ts / 1000.0, tz=UTC)


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


def _build_last_hours(
    candles: Sequence[Mapping[str, Any]],
    *,
    reference: datetime,
    window_hours: int = 6,
) -> Dict[str, Any]:
    window_start = reference - timedelta(hours=window_hours)
    window_start_ms = int(window_start.timestamp() * 1000)
    window_candles = [item for item in candles if int(item.get("t", 0)) >= window_start_ms]

    return {
        "window_hours": window_hours,
        "window_start_utc": window_start.isoformat(),
        "summary": _summarise(window_candles),
    }


def _build_current_day(candles: Sequence[Mapping[str, Any]], *, reference: datetime) -> Dict[str, Any]:
    day_start = reference.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end = day_start + timedelta(days=1)
    start_ms = int(day_start.timestamp() * 1000)
    end_ms = int(day_end.timestamp() * 1000)

    day_candles = [item for item in candles if start_ms <= int(item.get("t", 0)) < end_ms]

    return {
        "date": day_start.date().isoformat(),
        "summary": _summarise(day_candles),
    }


def build_check_all_datas(snapshot: Mapping[str, Any], now_utc: datetime | None = None) -> Dict[str, Any] | None:
    """Create a diagnostics payload for the snapshot health endpoint."""

    candles = _normalise_candles(snapshot)
    if not candles:
        return None

    reference = _resolve_now(candles, now_utc)
    last_ts = int(candles[-1]["t"])
    last_candle_utc = datetime.fromtimestamp(last_ts / 1000.0, tz=UTC)

    return {
        "snapshot_id": snapshot.get("id"),
        "symbol": snapshot.get("symbol"),
        "timeframe": snapshot.get("tf"),
        "asof_utc": reference.isoformat(),
        "latest_candle_utc": last_candle_utc.isoformat(),
        "last_candle": candles[-1],
        "current_day": _build_current_day(candles, reference=reference),
        "last_hours": _build_last_hours(candles, reference=reference),
    }

