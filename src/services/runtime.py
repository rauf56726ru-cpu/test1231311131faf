"""Runtime session helpers for on-demand check_all_datas execution."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from .check_all_datas import build_check_all_datas
from .inspection import build_placeholder_snapshot, DEFAULT_SYMBOL

UTC = timezone.utc


def _isoformat(dt: datetime) -> str:
    return dt.replace(tzinfo=UTC).isoformat().replace("+00:00", "Z")


def _prev_closed_bar(moment: datetime | None = None) -> datetime:
    now = moment.astimezone(UTC) if moment else datetime.now(UTC)
    aligned = now.replace(second=0, microsecond=0)
    if aligned == now:
        aligned -= timedelta(minutes=1)
    return aligned


@dataclass
class RuntimeState:
    """Keep minimal runtime state for incremental collection."""

    last_completed_ts: Optional[int] = None
    last_run_ts: Optional[int] = None
    interval: Dict[str, Any] = field(default_factory=dict)
    payload: Optional[Dict[str, Any]] = None

    def as_payload(self) -> Dict[str, Any]:
        return {
            "interval": dict(self.interval),
            "check_all": self.payload,
            "diagnostics": {"mode": self.interval.get("mode"), "status": "ok"} if self.payload else None,
        }


class RuntimeSession:
    """Compute check_all_datas responses on the fly without snapshots."""

    def __init__(self) -> None:
        self._state = RuntimeState()

    def snapshot(self) -> Dict[str, Any]:
        """Return the last known runtime payload."""

        return self._state.as_payload()

    def _build_check_all(self, *, start: datetime, end: datetime) -> Dict[str, Any]:
        snapshot = build_placeholder_snapshot(symbol=DEFAULT_SYMBOL, timeframe="1m")
        selection_start = int(start.timestamp() * 1000)
        selection_end = int(end.timestamp() * 1000)
        result = build_check_all_datas(
            snapshot,
            selection_start_ms=selection_start,
            selection_end_ms=selection_end,
            hours=1,
        )
        return result

    def run_previous_three_days(self) -> Dict[str, Any]:
        now = datetime.now(UTC)
        end_dt = _prev_closed_bar(now)
        start_day = end_dt.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=2)
        result = self._build_check_all(start=start_day, end=end_dt)
        self._state.payload = result
        self._state.last_completed_ts = int(end_dt.timestamp() * 1000)
        self._state.last_run_ts = int(now.timestamp() * 1000)
        self._state.interval = {
            "mode": "previous_3_days",
            "from_utc": _isoformat(start_day),
            "to_utc": _isoformat(end_dt),
            "last_completed_ts": self._state.last_completed_ts,
            "last_run_ts": self._state.last_run_ts,
        }
        return self._state.as_payload()

    def run_resume(self) -> Dict[str, Any]:
        if self._state.last_completed_ts is None:
            return self.run_previous_three_days()
        start_dt = datetime.fromtimestamp(self._state.last_completed_ts / 1000, tz=UTC)
        end_dt = _prev_closed_bar()
        if int(end_dt.timestamp() * 1000) <= self._state.last_completed_ts:
            self._state.interval = {
                "mode": "resume",
                "from_utc": self._state.interval.get("from_utc", _isoformat(start_dt)),
                "to_utc": _isoformat(end_dt),
                "last_completed_ts": self._state.last_completed_ts,
                "last_run_ts": int(datetime.now(UTC).timestamp() * 1000),
                "status": "no_new_data",
            }
            return self._state.as_payload()
        result = self._build_check_all(start=start_dt, end=end_dt)
        self._state.payload = result
        self._state.last_completed_ts = int(end_dt.timestamp() * 1000)
        self._state.last_run_ts = int(datetime.now(UTC).timestamp() * 1000)
        base_from = self._state.interval.get("from_utc", _isoformat(start_dt))
        self._state.interval = {
            "mode": "resume",
            "from_utc": base_from,
            "to_utc": _isoformat(end_dt),
            "last_completed_ts": self._state.last_completed_ts,
            "last_run_ts": self._state.last_run_ts,
        }
        return self._state.as_payload()


runtime_session = RuntimeSession()

