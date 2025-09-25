"""Session-based TPO and volume profile calculation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from math import isfinite
from typing import Dict, List, Mapping, MutableMapping, Tuple

import httpx

from ..meta import Meta

BINANCE_FAPI_AGG_TRADES = "https://fapi.binance.com/fapi/v1/aggTrades"
VALUE_AREA_RATIO = 0.7
MIN_SESSIONS = 2
MAX_SESSIONS = 5


@dataclass(slots=True)
class SessionWindow:
    """Represents a concrete trading session window."""

    name: str
    start: datetime
    end: datetime

    @property
    def date_label(self) -> str:
        return self.start.date().isoformat()


def _normalise_price_key(price_str: str) -> str:
    price = price_str.strip()
    if "." in price:
        price = price.rstrip("0").rstrip(".")
    return price or "0"


def _resolve_session(session_name: str) -> tuple[str, time, time]:
    for name, start, end in Meta.iter_vwap_sessions():
        if name == session_name:
            return name, start, end
    raise ValueError(f"Unknown session: {session_name}")


def _iter_recent_sessions(
    session_name: str, *, count: int, now: datetime | None = None
) -> List[SessionWindow]:
    if count < MIN_SESSIONS or count > MAX_SESSIONS:
        raise ValueError(f"Session count must be between {MIN_SESSIONS} and {MAX_SESSIONS}")

    name, start_time, end_time = _resolve_session(session_name)
    now = now or datetime.now(timezone.utc)

    sessions: List[SessionWindow] = []
    cursor_date = now.date()

    while len(sessions) < count:
        start_dt = datetime.combine(cursor_date, start_time, tzinfo=timezone.utc)
        end_dt = datetime.combine(cursor_date, end_time, tzinfo=timezone.utc)
        if end_dt <= start_dt:
            end_dt += timedelta(days=1)

        if end_dt <= now:
            sessions.append(SessionWindow(name=name, start=start_dt, end=end_dt))

        cursor_date = cursor_date - timedelta(days=1)
        if cursor_date < date(1970, 1, 1):  # pragma: no cover - defensive guard
            break

    sessions.sort(key=lambda window: window.start)
    return sessions


async def _accumulate_session_profile(
    symbol: str, window: SessionWindow
) -> MutableMapping[str, float]:
    volume_by_price: MutableMapping[str, float] = defaultdict(float)
    limit = 1000
    start_ms = int(window.start.timestamp() * 1000)
    end_ms = int(window.end.timestamp() * 1000)
    params = {"symbol": symbol.upper(), "limit": str(limit), "endTime": str(end_ms)}
    cursor = start_ms

    async with httpx.AsyncClient(timeout=15.0) as client:
        while cursor < end_ms:
            params["startTime"] = str(cursor)
            response = await client.get(BINANCE_FAPI_AGG_TRADES, params=params)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or not data:
                break

            last_time = None
            for row in data:
                try:
                    trade_time = int(row["T"])
                    price_key = _normalise_price_key(str(row["p"]))
                    quantity = float(row["q"])
                except (KeyError, TypeError, ValueError):
                    continue
                if quantity <= 0 or not isfinite(quantity):
                    continue
                if trade_time < start_ms:
                    continue
                if trade_time >= end_ms:
                    continue
                volume_by_price[price_key] += quantity
                last_time = trade_time

            if last_time is None:
                break

            cursor = last_time + 1
            if len(data) < limit:
                break

    return volume_by_price


def _compute_value_area(profile: Mapping[str, float]) -> Tuple[float, float, float] | None:
    if not profile:
        return None

    price_volume: List[Tuple[float, float]] = [
        (float(price), volume) for price, volume in profile.items() if volume > 0
    ]
    if not price_volume:
        return None

    price_volume.sort(key=lambda item: item[0])
    total_volume = sum(volume for _, volume in price_volume)
    if total_volume <= 0:
        return None

    poc_index = max(range(len(price_volume)), key=lambda idx: price_volume[idx][1])
    poc_price = price_volume[poc_index][0]

    target_volume = total_volume * VALUE_AREA_RATIO
    accumulated_volume = price_volume[poc_index][1]
    low_index = high_index = poc_index

    while accumulated_volume < target_volume:
        next_low_volume = price_volume[low_index - 1][1] if low_index > 0 else 0.0
        next_high_volume = (
            price_volume[high_index + 1][1]
            if (high_index + 1) < len(price_volume)
            else 0.0
        )

        if next_low_volume == 0.0 and next_high_volume == 0.0:
            break

        if next_high_volume >= next_low_volume:
            high_index += 1
            accumulated_volume += price_volume[high_index][1]
        else:
            low_index -= 1
            accumulated_volume += price_volume[low_index][1]

    val = price_volume[low_index][0]
    vah = price_volume[high_index][0]
    return val, vah, poc_price


async def fetch_tpo_profile(
    symbol: str, session: str = "ny", sessions: int = MAX_SESSIONS
) -> Dict[str, object]:
    session = session.lower()
    if sessions < MIN_SESSIONS or sessions > MAX_SESSIONS:
        raise ValueError(
            f"sessions must be between {MIN_SESSIONS} and {MAX_SESSIONS}"
        )

    windows = _iter_recent_sessions(session, count=sessions)

    combined_profile: MutableMapping[str, float] = defaultdict(float)
    tpo_rows: List[Dict[str, object]] = []

    for window in windows:
        session_profile = await _accumulate_session_profile(symbol, window)
        for price_key, volume in session_profile.items():
            combined_profile[price_key] += volume

        value_area = _compute_value_area(session_profile)
        if value_area is None:
            vah = val = poc = None
        else:
            val, vah, poc = value_area
        tpo_rows.append(
            {
                "date": window.date_label,
                "session": window.name,
                "VAL": val,
                "VAH": vah,
                "POC": poc,
            }
        )

    sorted_profile: List[Tuple[float, float]] = []
    for price_key, volume in combined_profile.items():
        try:
            price_value = float(price_key)
        except ValueError:
            continue
        sorted_profile.append((price_value, volume))
    sorted_profile.sort(key=lambda item: item[0])

    profile_rows = [
        {"price": price, "volume": volume} for price, volume in sorted_profile
    ]

    return {
        "symbol": symbol.upper(),
        "session": session,
        "requested_sessions": sessions,
        "sessions": len(windows),
        "tpo": tpo_rows,
        "profile": profile_rows,
    }


def fetch_tpo_profile_sync(
    symbol: str, session: str = "ny", sessions: int = MAX_SESSIONS
) -> Dict[str, object]:
    import asyncio

    return asyncio.run(fetch_tpo_profile(symbol, session=session, sessions=sessions))

