"""Placeholder session-detailed collector for the inspection panel."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Dict, Mapping, MutableMapping, Optional

from . import tracing
from .progress import ProgressReporter, emit_progress

TRACE_LOGGER = tracing.LOGGER.getChild("session_collector")

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - Python <3.9 fallback
    ZoneInfo = None  # type: ignore[assignment]


BERLIN_TZ = ZoneInfo("Europe/Berlin") if ZoneInfo else timezone.utc

_SESSION_WINDOWS = (
    ("asia", dtime(hour=0, minute=0), dtime(hour=8, minute=0)),
    ("london", dtime(hour=8, minute=0), dtime(hour=16, minute=0)),
    ("ny", dtime(hour=14, minute=30), dtime(hour=22, minute=30)),
)


@dataclass(slots=True)
class SessionWindow:
    """Resolved trading session window in UTC."""

    name: str
    open_utc: datetime
    close_utc: datetime
    is_active: bool


@dataclass(slots=True)
class SessionCollectionResult:
    """Structured response for session-detailed collection."""

    symbol: str
    status: str
    session: SessionWindow
    coverage_pct: float
    missing_fields: tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        open_iso = _to_iso(self.session.open_utc)
        close_iso = _to_iso(self.session.close_utc)
        meta_block: Dict[str, Any] = {
            "symbol": self.symbol,
            "tz": "Europe/Berlin",
            "last_price": None,
            "last_ts_utc": close_iso,
            "data_freshness_sec": None,
            "stale": True,
        }

        session_block: Dict[str, Any] = {
            "name": self.session.name,
            "open_utc": open_iso,
            "close_utc": close_iso,
            "coverage_pct": round(self.coverage_pct, 5),
            "active": self.session.is_active,
        }

        availability: Dict[str, Any] = {
            "ohlcv": {tf: False for tf in ("1m", "3m", "5m", "15m", "1h")},
            "orderflow": {"delta": False, "cvd": False, "footprint": False},
            "vwap_sessions": {self.session.name: False},
        }

        data_block: Dict[str, Any] = {
            "ohlcv": {
                "1m": {"bars": []},
                "3m": {"bars": []},
                "5m": {"bars": []},
                "1m_rollups": [],
                "15m_compact": [],
                "1h_compact": [],
            },
            "orderflow": {
                "per_bar": [],
                "delta_cvd_compact": [],
                "metrics": {"delta": False, "cvd": False, "footprint": False},
            },
            "vwap_tpo": {
                "daily": {},
                "sessions": {
                    self.session.name: {
                        "vwap": None,
                        "sd1": None,
                        "sd2": None,
                        "poc": None,
                        "vah": None,
                        "val": None,
                        "ib_high": None,
                        "ib_low": None,
                        "high": None,
                        "low": None,
                    }
                },
            },
            "zones": {"top": [], "counts": {}},
            "liquidity_targets": [],
        }

        payload: Dict[str, Any] = {
            "schema": "session_detailed.v1",
            "status": self.status,
            "meta": meta_block,
            "session": session_block,
            "data": data_block,
            "availability": availability,
            "missing_fields": list(self.missing_fields),
            "notes": [],
        }

        if self.coverage_pct < 0.9 and "ohlcv.coverage" not in payload["missing_fields"]:
            payload["missing_fields"].append("ohlcv.coverage")
        return payload


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve_session_window(now_utc: datetime) -> SessionWindow:
    tz = BERLIN_TZ or timezone.utc
    local_now = now_utc.astimezone(tz)
    candidates: list[SessionWindow] = []

    for name, start_time, end_time in _SESSION_WINDOWS:
        start_local = datetime.combine(local_now.date(), start_time, tz)
        end_local = datetime.combine(local_now.date(), end_time, tz)
        if end_local <= start_local:
            end_local += timedelta(days=1)
        open_utc = start_local.astimezone(timezone.utc)
        close_utc = end_local.astimezone(timezone.utc)
        is_active = start_local <= local_now <= end_local
        window = SessionWindow(name=name, open_utc=open_utc, close_utc=close_utc, is_active=is_active)
        candidates.append(window)
        if is_active:
            return window

    # pick the closest past window
    past_windows = [window for window in candidates if window.close_utc <= now_utc]
    if past_windows:
        past_windows.sort(key=lambda window: window.close_utc, reverse=True)
        return past_windows[0]

    # fallback to previous day's last window
    yesterday = now_utc - timedelta(days=1)
    return _resolve_session_window(
        datetime.combine(yesterday.date(), dtime.min, tzinfo=timezone.utc) + timedelta(hours=12)
    )


async def collect_last_session_detailed(
    symbol: str,
    now: datetime | None = None,
    progress: Optional[ProgressReporter] = None,
) -> SessionCollectionResult:
    """Collect detailed data for the latest trading session.

    Current implementation returns a placeholder payload marking the session as insufficient
    until dedicated data aggregation is implemented.
    """

    now_dt = now.astimezone(timezone.utc) if isinstance(now, datetime) else datetime.now(timezone.utc)
    TRACE_LOGGER.debug(
        "session_collector:start",
        extra={
            "symbol": symbol,
            "requested_at": now_dt.isoformat(),
        },
    )
    await emit_progress(
        progress,
        "session_collector:start",
        symbol=symbol,
        requested_at=now_dt.isoformat(),
    )
    session_window = _resolve_session_window(now_dt)
    TRACE_LOGGER.debug(
        "session_collector:resolved_window",
        extra={
            "symbol": symbol,
            "session": session_window.name,
            "open_utc": session_window.open_utc.isoformat(),
            "close_utc": session_window.close_utc.isoformat(),
            "active": session_window.is_active,
        },
    )
    await emit_progress(
        progress,
        "session_collector:resolved_window",
        symbol=symbol,
        session=session_window.name,
        open_utc=session_window.open_utc.isoformat(),
        close_utc=session_window.close_utc.isoformat(),
        active=session_window.is_active,
    )
    missing_fields = (
        "ohlcv.coverage",
        "orderflow.coverage",
        "orderflow.delta",
        "orderflow.cvd",
        "orderflow.footprint",
    )
    result = SessionCollectionResult(
        symbol=symbol,
        status="insufficient_data",
        session=session_window,
        coverage_pct=0.0,
        missing_fields=missing_fields,
    )

    TRACE_LOGGER.debug(
        "session_collector:finished",
        extra={
            "symbol": symbol,
            "status": result.status,
            "coverage_pct": result.coverage_pct,
            "missing_fields": list(result.missing_fields),
        },
    )
    await emit_progress(
        progress,
        "session_collector:finished",
        symbol=symbol,
        status=result.status,
        coverage_pct=result.coverage_pct,
        missing_fields=list(result.missing_fields),
    )

    return result

