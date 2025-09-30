"""Persistence helpers for tracking the last /inspection data collection."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / "var"
STATE_FILE = STATE_DIR / "collection_state.json"

_STATE_CACHE: Dict[str, Any] | None = None
_STATE_LOCK = Lock()


def _ensure_state_dir() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)


def _load_state() -> Dict[str, Any]:
    """Load the persisted state from disk if present."""

    global _STATE_CACHE
    with _STATE_LOCK:
        if _STATE_CACHE is not None:
            return dict(_STATE_CACHE)
        if not STATE_FILE.exists():
            _STATE_CACHE = {}
            return {}
        try:
            raw = STATE_FILE.read_text(encoding="utf-8")
        except OSError:
            _STATE_CACHE = {}
            return {}
        try:
            data = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        _STATE_CACHE = data
        return dict(_STATE_CACHE)


def _persist_state(state: Dict[str, Any]) -> None:
    with _STATE_LOCK:
        _ensure_state_dir()
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _STATE_CACHE = dict(state)


def get_last_collection_time() -> Optional[datetime]:
    """Return the last recorded collection timestamp in UTC."""

    state = _load_state()
    value = state.get("last_collection_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def set_last_collection_time(moment: datetime) -> None:
    """Persist the timestamp of the last collection run."""

    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    else:
        moment = moment.astimezone(timezone.utc)
    state = _load_state()
    state["last_collection_at"] = moment.isoformat()
    _persist_state(state)


def reset_state() -> None:
    """Utility hook for tests to clear the cached state."""

    global _STATE_CACHE
    with _STATE_LOCK:
        _STATE_CACHE = {}
        try:
            STATE_FILE.unlink()
        except FileNotFoundError:
            pass
