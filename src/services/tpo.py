"""Session-based TPO and volume profile calculation."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from math import floor, isfinite, sqrt
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

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



class TPOCalculationError(RuntimeError):
    """Raised when TPO inputs are invalid."""


def _parse_candle(row: Mapping[str, object]) -> tuple[datetime, float, float, float, float, float]:
    time_value = row.get("t") or row.get("time") or row.get("ts") or row.get("timestamp")
    if isinstance(time_value, str):
        ts = datetime.fromisoformat(str(time_value).replace("Z", "+00:00"))
    elif isinstance(time_value, (int, float)):
        ts = datetime.fromtimestamp(float(time_value) / 1000, tz=timezone.utc)
    else:
        raise TPOCalculationError("Candle timestamp missing")
    high = float(row.get("h"))
    low = float(row.get("l"))
    close = float(row.get("c"))
    open_price = float(row.get("o", close))
    volume = float(row.get("v", 0.0))
    return ts, open_price, high, low, close, volume


def calculate_tpo(
    candles: Sequence[Mapping[str, object]],
    preset: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    """Calculate VWAP-based TPO metrics for multiple days."""

    if not candles:
        raise TPOCalculationError("Candles are required for TPO")

    grouped: MutableMapping[str, List[tuple[datetime, float, float, float, float, float]]] = defaultdict(list)
    for row in candles:
        if not isinstance(row, Mapping):
            continue
        try:
            parsed = _parse_candle(row)
        except Exception:
            continue
        grouped[parsed[0].date().isoformat()].append(parsed)

    results: List[Dict[str, object]] = []
    for date_key, rows in sorted(grouped.items())[-3:]:
        rows.sort(key=lambda item: item[0])
        total_volume = sum(item[5] for item in rows)
        if total_volume <= 0:
            continue
        typical_prices = [(item[2] + item[3] + item[4]) / 3 for item in rows]
        vwap_numerator = sum(tp * item[5] for tp, item in zip(typical_prices, rows))
        vwap = vwap_numerator / total_volume
        variance = sum(((tp - vwap) ** 2) * item[5] for tp, item in zip(typical_prices, rows)) / total_volume
        std_dev = sqrt(max(variance, 0.0))
        poc_price = max(rows, key=lambda item: item[5])[4]
        vah = vwap + std_dev
        val = vwap - std_dev

        first_hour = [item for item in rows if (item[0] - rows[0][0]).total_seconds() <= 3_600]
        ibh = max(item[2] for item in first_hour) if first_hour else None
        ibl = min(item[3] for item in first_hour) if first_hour else None

        inducement = []
        for item in rows:
            wick_up = item[2] - item[4]
            wick_down = item[4] - item[3]
            body = abs(item[4] - item[1])
            if body <= 0:
                continue
            if wick_up > body * 1.5:
                inducement.append({"time": item[0].isoformat().replace("+00:00", "Z"), "type": "bullish"})
            if wick_down > body * 1.5:
                inducement.append({"time": item[0].isoformat().replace("+00:00", "Z"), "type": "bearish"})

        bias = "bullish" if poc_price >= vwap else "bearish"

        results.append(
            {
                "date": date_key,
                "vwap": vwap,
                "sd1_plus": vwap + std_dev,
                "sd1_minus": vwap - std_dev,
                "sd2_plus": vwap + 2 * std_dev,
                "sd2_minus": vwap - 2 * std_dev,
                "POC": poc_price,
                "VAH": vah,
                "VAL": val,
                "IBH": ibh,
                "IBL": ibl,
                "inducement": inducement,
                "risk_assessment": {"bias": bias},
            }
        )

    return {"days": results}


def calculate_session_tpo(
    candles: Sequence[Mapping[str, object]],
    session: str,
) -> Dict[str, object]:
    """Calculate VWAP metrics for a specific session."""

    session = session.lower().strip()
    if session not in {"asia", "london", "ny"}:
        raise TPOCalculationError("Unsupported session")

    grouped: List[tuple[datetime, float, float, float, float, float]] = []
    for row in candles:
        if not isinstance(row, Mapping):
            continue
        try:
            parsed = _parse_candle(row)
        except Exception:
            continue
        hour = parsed[0].hour
        if session == "asia" and 0 <= hour < 8:
            grouped.append(parsed)
        elif session == "london" and 8 <= hour < 16:
            grouped.append(parsed)
        elif session == "ny" and (hour >= 16 or hour < 0):
            grouped.append(parsed)

    if not grouped:
        return {"session": session, "vwap": None}

    grouped.sort(key=lambda item: item[0])
    total_volume = sum(item[5] for item in grouped)
    if total_volume <= 0:
        return {"session": session, "vwap": None}
    typical_prices = [(item[2] + item[3] + item[4]) / 3 for item in grouped]
    vwap = sum(tp * item[5] for tp, item in zip(typical_prices, grouped)) / total_volume
    variance = sum(((tp - vwap) ** 2) * item[5] for tp, item in zip(typical_prices, grouped)) / total_volume
    std_dev = sqrt(max(variance, 0.0))
    poc_price = max(grouped, key=lambda item: item[5])[4]
    vah = vwap + std_dev
    val = vwap - std_dev
    high = max(item[2] for item in grouped)
    low = min(item[3] for item in grouped)
    ibh = max(item[2] for item in grouped)
    ibl = min(item[3] for item in grouped)

    return {
        "session": session,
        "vwap": vwap,
        "POC": poc_price,
        "VAH": vah,
        "VAL": val,
        "High": high,
        "Low": low,
        "IBH": ibh,
        "IBL": ibl,
    }


# ---------------------------------------------------------------------------
# Incremental value-area helpers for the strict three-day workflow
# ---------------------------------------------------------------------------


def _safe_int(value: Any) -> int | None:
    try:
        numeric = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric


def _safe_float(value: Any) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def _typical_price(candle: Mapping[str, Any]) -> float | None:
    high = _safe_float(candle.get("h") or candle.get("high"))
    low = _safe_float(candle.get("l") or candle.get("low"))
    close = _safe_float(candle.get("c") or candle.get("close"))
    if high is None or low is None or close is None:
        return None
    return (high + low + close) / 3.0


@dataclass(slots=True)
class ValueAreaState:
    """Incremental histogram-based accumulator for compact profiles."""

    start_ms: int
    end_ms: int
    tick_size: float | None
    value_area_pct: float
    ib_minutes: int = 60
    bin_size: float | None = None
    base_price: float | None = None
    bin_volume: Dict[int, float] = field(default_factory=dict)
    total_volume: float = 0.0
    processed_bars: int = 0
    last_timestamp: int | None = None
    session_high: float | None = None
    session_low: float | None = None
    ib_high: float | None = None
    ib_low: float | None = None

    def ensure_bounds(
        self,
        *,
        start_ms: int,
        end_ms: int,
        tick_size: float | None,
        value_area_pct: float,
        ib_minutes: int,
    ) -> None:
        bounds_changed = self.start_ms != start_ms or self.end_ms != end_ms
        if bounds_changed:
            self.start_ms = start_ms
            self.end_ms = end_ms
            self.bin_volume.clear()
            self.total_volume = 0.0
            self.processed_bars = 0
            self.last_timestamp = None
            self.session_high = None
            self.session_low = None
            self.ib_high = None
            self.ib_low = None
            self.bin_size = None
            self.base_price = None
        self.tick_size = tick_size
        self.value_area_pct = value_area_pct
        self.ib_minutes = ib_minutes

    def _ensure_bins(self, price: float) -> None:
        if self.bin_size is not None and self.base_price is not None:
            return
        tick = self.tick_size if self.tick_size and self.tick_size > 0 else None
        adaptive = abs(price) * 1e-4
        if adaptive <= 0:
            adaptive = 1e-6
        if tick is not None and adaptive is not None:
            bin_size = max(tick, adaptive)
        else:
            bin_size = tick or adaptive or 1e-6
        self.bin_size = bin_size
        self.base_price = floor(price / bin_size) * bin_size

    def _bin_index(self, price: float) -> int:
        self._ensure_bins(price)
        assert self.bin_size is not None
        assert self.base_price is not None
        relative = (price - self.base_price) / self.bin_size
        if relative >= 0:
            return int(relative + 1e-9)
        return int(relative - 1e-9)

    def update(self, candles: Sequence[Mapping[str, Any]]) -> int:
        if self.end_ms < self.start_ms:
            return 0
        ib_cutoff = self.start_ms + max(0, self.ib_minutes) * 60_000
        new_bars = 0
        last_ts = self.last_timestamp
        for candle in candles:
            if not isinstance(candle, Mapping):
                continue
            ts: int | None = None
            for key in ("t", "time", "openTime", "timestamp"):
                ts = _safe_int(candle.get(key))
                if ts is not None:
                    break
            if ts is None or ts < self.start_ms or ts > self.end_ms:
                continue
            if last_ts is not None and ts <= last_ts:
                continue
            high = _safe_float(candle.get("h") or candle.get("high"))
            low = _safe_float(candle.get("l") or candle.get("low"))
            if high is not None:
                self.session_high = (
                    high if self.session_high is None else max(self.session_high, high)
                )
                if ts < ib_cutoff:
                    self.ib_high = high if self.ib_high is None else max(self.ib_high, high)
            if low is not None:
                self.session_low = (
                    low if self.session_low is None else min(self.session_low, low)
                )
                if ts < ib_cutoff:
                    self.ib_low = low if self.ib_low is None else min(self.ib_low, low)
            volume = _safe_float(candle.get("v") or candle.get("volume"))
            if volume is None or volume <= 0:
                last_ts = ts
                continue
            price = _typical_price(candle)
            if price is None:
                last_ts = ts
                continue
            index = self._bin_index(price)
            self.bin_volume[index] = self.bin_volume.get(index, 0.0) + volume
            self.total_volume += volume
            self.processed_bars += 1
            last_ts = ts
            new_bars += 1
        if last_ts is not None:
            self.last_timestamp = last_ts
        return new_bars

    def value_area(self) -> Tuple[float | None, float | None, float | None]:
        if not self.bin_volume or self.bin_size is None or self.base_price is None:
            return None, None, None
        ordered = sorted(self.bin_volume.items())
        prices = [self.base_price + idx * self.bin_size for idx, _ in ordered]
        volumes = [vol for _, vol in ordered]
        total = sum(volumes)
        if total <= 0:
            return None, None, None
        poc_index = max(range(len(volumes)), key=lambda idx: volumes[idx])
        poc_price = prices[poc_index]
        threshold = total * max(0.0, min(1.0, self.value_area_pct))
        order = sorted(
            range(len(volumes)),
            key=lambda idx: (-volumes[idx], abs(prices[idx] - poc_price), prices[idx]),
        )
        covered = 0.0
        selected: set[int] = set()
        for idx in order:
            selected.add(idx)
            covered += max(0.0, volumes[idx])
            if covered >= threshold:
                break
        if not selected:
            selected = {poc_index}
        vah = max(prices[idx] for idx in selected)
        val = min(prices[idx] for idx in selected)
        return poc_price, vah, val


def compute_compact_value_area(
    candles: Sequence[Mapping[str, Any]],
    *,
    start_ms: int,
    end_ms: int,
    tick_size: float | None,
    value_area_pct: float = VALUE_AREA_RATIO,
    ib_minutes: int = 60,
    state: ValueAreaState | None = None,
) -> Tuple[Dict[str, Any], ValueAreaState, int]:
    """Return compact volume-profile metrics for the provided window."""

    if state is None:
        state = ValueAreaState(
            start_ms=start_ms,
            end_ms=end_ms,
            tick_size=tick_size,
            value_area_pct=value_area_pct,
            ib_minutes=ib_minutes,
        )
    else:
        state.ensure_bounds(
            start_ms=start_ms,
            end_ms=end_ms,
            tick_size=tick_size,
            value_area_pct=value_area_pct,
            ib_minutes=ib_minutes,
        )

    incremental = state.update(candles)
    poc, vah, val = state.value_area()
    payload = {
        "poc": poc,
        "vah": vah,
        "val": val,
        "session_high": state.session_high,
        "session_low": state.session_low,
        "ib_high": state.ib_high,
        "ib_low": state.ib_low,
        "total_volume": state.total_volume,
        "bars": state.processed_bars,
        "incremental_bars": incremental,
    }
    return payload, state, incremental
