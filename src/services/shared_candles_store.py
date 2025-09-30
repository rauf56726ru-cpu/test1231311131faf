"""Server-side persistence for candles shared between chart sessions."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from math import isfinite
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE_DIR = PROJECT_ROOT / "var"
STORE_FILE = STORE_DIR / "shared_candles.json"

DEFAULT_MAX_BARS = 2000

_STORE_CACHE: Dict[str, Any] | None = None
_STORE_LOCK = Lock()


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
    """Merge incoming candles into the persistent store and return the updated state."""

    key = _make_key(symbol, interval)

    incoming = list(candles or [])
    if not incoming and not reset:
        current = get_shared_candles(symbol, interval)
        if current is None:
            return {
                "candles": [],
                "intervalMs": None,
                "lastUpdateMs": None,
                "updatedAt": None,
            }
        return current

    store = _load_store()
    existing_entry = store.get(key) if not reset else None
    existing_candles: Sequence[Mapping[str, Any]]
    if isinstance(existing_entry, dict):
        existing_candles = existing_entry.get("candles", [])  # type: ignore[assignment]
    else:
        existing_candles = []

    merged = _merge_bars(existing_candles, incoming, max_bars=max_bars)

    interval_ms_value = None
    if isinstance(interval_ms, (int, float)) and isfinite(interval_ms):
        interval_ms_value = int(interval_ms)
    elif isinstance(existing_entry, dict):
        previous = existing_entry.get("interval_ms")
        if isinstance(previous, (int, float)) and isfinite(previous):
            interval_ms_value = int(previous)

    last_update_value = None
    if isinstance(last_update_ms, (int, float)) and isfinite(last_update_ms):
        last_update_value = int(last_update_ms)
    elif isinstance(existing_entry, dict):
        previous = existing_entry.get("last_update_ms")
        if isinstance(previous, (int, float)) and isfinite(previous):
            last_update_value = int(previous)

    updated_at_value = _now_ms()

    next_entry = {
        "candles": merged,
        "interval_ms": interval_ms_value,
        "last_update_ms": last_update_value,
        "updated_at": updated_at_value,
    }
    store[key] = next_entry
    _persist_store(store)

    return {
        "candles": merged,
        "intervalMs": interval_ms_value,
        "lastUpdateMs": last_update_value,
        "updatedAt": updated_at_value,
    }


def clear_shared_candles(symbol: str | None = None, interval: str | None = None) -> None:
    """Remove candles for a specific key or wipe the entire store."""

    if not symbol and not interval:
        store: Dict[str, Any] = {}
        _persist_store(store)
        return

    try:
        key = _make_key(symbol or "", interval or "")
    except ValueError:
        return
    store = _load_store()
    if key in store:
        del store[key]
        _persist_store(store)


def reset_store() -> None:
    """Helper used by tests to clear store state and cached data."""

    global _STORE_CACHE
    with _STORE_LOCK:
        _STORE_CACHE = {}
        try:
            STORE_FILE.unlink()
        except FileNotFoundError:
            pass

