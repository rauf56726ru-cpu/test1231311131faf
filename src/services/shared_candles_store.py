"""Server-side persistence for candles shared between chart sessions."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from threading import Lock
from time import monotonic
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = PROJECT_ROOT / "var"
STORE_FILE = STORE_DIR / "shared_candles.json"

DEFAULT_MAX_BARS = 2000

_STORE_CACHE: Dict[str, Any] | None = None
_STORE_LOCK = Lock()
_WRITE_TRACKER_LOCK = Lock()
_WRITE_RATE_LIMIT_SECONDS = max(
    0.0,
    float(os.environ.get("SHARED_CANDLES_RATE_LIMIT_SECONDS", "0.0")),
)
_WRITE_TRACKER: Dict[str, float] = {}


def _ensure_store_dir() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)


def _load_store() -> Dict[str, Any]:
    global _STORE_CACHE
    with _STORE_LOCK:
        if _STORE_CACHE is not None:
            return dict(_STORE_CACHE)
        if not STORE_FILE.exists():
            _STORE_CACHE = {}
            return {}
        try:
            raw = STORE_FILE.read_text(encoding="utf-8")
        except OSError:
            _STORE_CACHE = {}
            return {}
        try:
            payload = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}
        if not isinstance(payload, dict):
            payload = {}
        _STORE_CACHE = payload
        return dict(_STORE_CACHE)


def _persist_store(store: MutableMapping[str, Any]) -> None:
    global _STORE_CACHE
    with _STORE_LOCK:
        _ensure_store_dir()
        STORE_FILE.write_text(
            json.dumps(store, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _STORE_CACHE = dict(store)


def _make_key(symbol: str, interval: str) -> str:
    safe_symbol = (symbol or "").strip().upper()
    safe_interval = (interval or "").strip().lower()
    if not safe_symbol or not safe_interval:
        raise ValueError("symbol and interval are required")
    return f"{safe_symbol}|{safe_interval}"


def _extract_time_seconds(bar: Mapping[str, Any]) -> int | None:
    candidates = (
        bar.get("time"),
        bar.get("t"),
        bar.get("timestamp"),
        bar.get("ts"),
    )
    for value in candidates:
        if isinstance(value, (int, float)) and isfinite(value):
            candidate = float(value)
            if candidate > 1_000_000_000_000:  # assume milliseconds
                candidate /= 1000
            return int(candidate)
    ts_ms = bar.get("ts_ms_utc")
    if isinstance(ts_ms, (int, float)) and isfinite(ts_ms):
        return int(ts_ms // 1000)
    return None


def _normalise_bar(bar: Mapping[str, Any]) -> Dict[str, Any] | None:
    time_seconds = _extract_time_seconds(bar)
    open_price = bar.get("open", bar.get("o"))
    high_price = bar.get("high", bar.get("h", open_price))
    low_price = bar.get("low", bar.get("l", open_price))
    close_price = bar.get("close", bar.get("c", open_price))

    numeric_values = []
    for value in (open_price, high_price, low_price, close_price):
        if isinstance(value, (int, float)) and isfinite(value):
            numeric_values.append(float(value))
        else:
            return None

    if time_seconds is None:
        return None

    open_, high, low, close = numeric_values
    volume = bar.get("volume", bar.get("v"))
    volume_value = None
    if isinstance(volume, (int, float)) and isfinite(volume):
        volume_value = float(volume)

    normalised: Dict[str, Any] = {
        "time": int(time_seconds),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "ts_ms_utc": int(time_seconds * 1000),
    }
    if volume_value is not None:
        normalised["volume"] = volume_value
    return normalised


def _merge_bars(
    existing: Sequence[Mapping[str, Any]],
    incoming: Sequence[Mapping[str, Any]],
    *,
    max_bars: int | None = None,
) -> List[Dict[str, Any]]:
    index: Dict[int, Dict[str, Any]] = {}
    for bucket in (existing, incoming):
        for bar in bucket:
            normalised = _normalise_bar(bar)
            if not normalised:
                continue
            index[normalised["time"]] = normalised
    merged = [index[key] for key in sorted(index.keys())]
    limit = max(1, int(max_bars) if max_bars else DEFAULT_MAX_BARS)
    if len(merged) > limit:
        return merged[-limit:]
    return merged


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def get_shared_candles(symbol: str, interval: str) -> Dict[str, Any] | None:
    """Return the stored candles for the given symbol/interval."""

    try:
        key = _make_key(symbol, interval)
    except ValueError:
        return None

    store = _load_store()
    entry = store.get(key)
    if not isinstance(entry, dict):
        return None

    candles = entry.get("candles")
    if isinstance(candles, Iterable):
        normalised = [
            result
            for bar in candles
            if isinstance(bar, Mapping)
            for result in (_normalise_bar(bar),)
            if result is not None
        ]
    else:
        normalised = []

    if not normalised:
        normalised = []

    interval_ms = entry.get("interval_ms")
    last_update_ms = entry.get("last_update_ms")
    updated_at = entry.get("updated_at")

    response: Dict[str, Any] = {
        "candles": normalised,
        "intervalMs": int(interval_ms) if isinstance(interval_ms, (int, float)) and isfinite(interval_ms) else None,
        "lastUpdateMs": int(last_update_ms)
        if isinstance(last_update_ms, (int, float)) and isfinite(last_update_ms)
        else None,
        "updatedAt": int(updated_at)
        if isinstance(updated_at, (int, float)) and isfinite(updated_at)
        else None,
    }
    return response


def merge_shared_candles(
    symbol: str,
    interval: str,
    candles: Sequence[Mapping[str, Any]] | None,
    *,
    interval_ms: int | float | None = None,
    last_update_ms: int | float | None = None,
    reset: bool = False,
    max_bars: int | None = None,
) -> Dict[str, Any]:
    """Merge incoming candles into the persistent store and return a compact status."""

    key = _make_key(symbol, interval)
    store = _load_store()
    existing_entry = store.get(key)

    if isinstance(existing_entry, dict):
        existing_candles_raw = existing_entry.get("candles", [])
    else:
        existing_candles_raw = []

    existing_candles = [
        result
        for bar in existing_candles_raw
        if isinstance(bar, Mapping)
        for result in (_normalise_bar(bar),)
        if result is not None
    ]

    incoming_candles = [
        result
        for bar in (candles or [])
        if isinstance(bar, Mapping)
        for result in (_normalise_bar(bar),)
        if result is not None
    ]

    incoming_last_update = (
        int(last_update_ms)
        if isinstance(last_update_ms, (int, float)) and isfinite(last_update_ms)
        else None
    )
    existing_last_update = (
        int(existing_entry.get("last_update_ms"))
        if isinstance(existing_entry, dict)
        and isinstance(existing_entry.get("last_update_ms"), (int, float))
        and isfinite(existing_entry.get("last_update_ms"))
        else None
    )
    existing_interval_ms = (
        int(existing_entry.get("interval_ms"))
        if isinstance(existing_entry, dict)
        and isinstance(existing_entry.get("interval_ms"), (int, float))
        and isfinite(existing_entry.get("interval_ms"))
        else None
    )
    existing_updated_at = (
        int(existing_entry.get("updated_at"))
        if isinstance(existing_entry, dict)
        and isinstance(existing_entry.get("updated_at"), (int, float))
        and isfinite(existing_entry.get("updated_at"))
        else None
    )

    if (
        incoming_last_update is not None
        and existing_last_update is not None
        and incoming_last_update <= existing_last_update
    ):
        return {
            "status": "noop",
            "symbol": symbol.strip().upper(),
            "interval": interval.strip().lower(),
            "written": False,
            "lastUpdateMs": existing_last_update,
            "updatedAt": existing_updated_at,
        }

    effective_max_bars = (
        max(1, int(max_bars))
        if isinstance(max_bars, (int, float)) and isfinite(max_bars)
        else None
    )
    if effective_max_bars is None and isinstance(existing_entry, dict):
        stored_limit = existing_entry.get("max_bars")
        if isinstance(stored_limit, (int, float)) and isfinite(stored_limit):
            effective_max_bars = max(1, int(stored_limit))

    merged_candles = _merge_bars(
        [] if reset else existing_candles,
        incoming_candles,
        max_bars=effective_max_bars,
    )

    candles_changed = merged_candles != existing_candles

    next_interval_ms = (
        int(interval_ms)
        if isinstance(interval_ms, (int, float)) and isfinite(interval_ms)
        else existing_interval_ms
    )

    next_last_update = existing_last_update
    if incoming_last_update is not None:
        next_last_update = incoming_last_update
    elif reset:
        next_last_update = None

    requires_write = reset or candles_changed or next_interval_ms != existing_interval_ms
    if incoming_last_update is not None and incoming_last_update != existing_last_update:
        requires_write = True

    if not requires_write:
        return {
            "status": "noop",
            "symbol": symbol.strip().upper(),
            "interval": interval.strip().lower(),
            "written": False,
            "lastUpdateMs": existing_last_update,
            "updatedAt": existing_updated_at,
        }

    now_monotonic = monotonic()
    if not reset:
        with _WRITE_TRACKER_LOCK:
            last_write = _WRITE_TRACKER.get(key)
            if last_write is not None and now_monotonic - last_write < _WRITE_RATE_LIMIT_SECONDS:
                retry_after_ms = int(
                    max(0, (_WRITE_RATE_LIMIT_SECONDS - (now_monotonic - last_write)) * 1000)
                )
                return {
                    "status": "rate_limited",
                    "symbol": symbol.strip().upper(),
                    "interval": interval.strip().lower(),
                    "written": False,
                    "retryAfterMs": retry_after_ms,
                    "lastUpdateMs": existing_last_update,
                    "updatedAt": existing_updated_at,
                }

    updated_at_value = _now_ms()

    next_entry: Dict[str, Any] = {
        "candles": merged_candles,
        "interval_ms": next_interval_ms,
        "last_update_ms": next_last_update,
        "updated_at": updated_at_value,
    }
    if effective_max_bars is not None:
        next_entry["max_bars"] = effective_max_bars

    store[key] = next_entry
    _persist_store(store)

    with _WRITE_TRACKER_LOCK:
        _WRITE_TRACKER[key] = now_monotonic

    return {
        "status": "ok",
        "symbol": symbol.strip().upper(),
        "interval": interval.strip().lower(),
        "written": True,
        "lastUpdateMs": next_last_update,
        "updatedAt": updated_at_value,
    }


def clear_shared_candles(symbol: str | None = None, interval: str | None = None) -> None:
    """Remove candles for a specific key or wipe the entire store."""

    if not symbol and not interval:
        store: Dict[str, Any] = {}
        _persist_store(store)
        with _WRITE_TRACKER_LOCK:
            _WRITE_TRACKER.clear()
        return

    try:
        key = _make_key(symbol or "", interval or "")
    except ValueError:
        return
    store = _load_store()
    if key in store:
        del store[key]
        _persist_store(store)
    with _WRITE_TRACKER_LOCK:
        _WRITE_TRACKER.pop(key, None)


def reset_store() -> None:
    """Helper used by tests to clear store state and cached data."""

    global _STORE_CACHE
    with _STORE_LOCK:
        _STORE_CACHE = {}
        try:
            STORE_FILE.unlink()
        except FileNotFoundError:
            pass
    with _WRITE_TRACKER_LOCK:
        _WRITE_TRACKER.clear()

