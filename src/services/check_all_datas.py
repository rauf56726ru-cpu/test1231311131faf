"""Aggregation helpers for the check-all-datas inspection endpoint."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import math
from typing import Iterable, List, Mapping, MutableSequence, Sequence

from pydantic import BaseModel, ConfigDict, Field

try:  # Python 3.9 compatibility for zoneinfo
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - fallback for older Python
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

from .inspection import get_snapshot, list_snapshots


BERLIN_TZ = ZoneInfo("Europe/Berlin")


class InsufficientDataError(RuntimeError):
    """Raised when there is not enough data to build the market overview."""


@dataclass(slots=True)
class Candle:
    """A simplified minute candle representation used for aggregation."""

    t: int
    o: float
    h: float
    l: float
    c: float
    v: float


class PreviousDaySummary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    date_utc: str
    date_local: str
    bars: int
    open: float
    high: float
    low: float
    close: float
    volume_sum: float
    ret_close_pct: float


class PreviousDaysAggregate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    days: int
    volume_sum: float
    ret_close_pct_avg: float
    high_max: float
    low_min: float


class WindowUTC(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    start: int = Field(..., alias="from")
    end: int = Field(..., alias="to")


class OhlcWindow(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    o: float
    h: float
    l: float
    c: float


class DetailedBar(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    t: int
    o: float
    h: float
    l: float
    c: float
    v: float
    ret_1m: float
    sma_5: float
    sma_15: float
    vwap_window: float
    volatility_15m: float


class LastHoursDetailed(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    hours: int
    window_utc: WindowUTC
    ohlc_window: OhlcWindow
    volume_window: float
    ret_window_pct: float
    max_drawdown_pct: float
    bars: Sequence[DetailedBar]


class MarketSliceResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    symbol: str
    timezone: str
    asof_utc: int
    previous_days_summary: Sequence[PreviousDaySummary]
    previous_days_aggregate: PreviousDaysAggregate
    last_hours_detailed: LastHoursDetailed


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_candles(frame: Mapping[str, object]) -> List[Candle]:
    raw_candles = frame.get("candles", [])
    if not isinstance(raw_candles, Iterable):
        raise InsufficientDataError("snapshot does not contain iterable candles")

    parsed: List[Candle] = []
    seen: set[int] = set()
    for entry in raw_candles:
        if not isinstance(entry, Mapping):
            continue
        try:
            t = int(entry["t"])
            o = float(entry["o"])
            h = float(entry["h"])
            l = float(entry["l"])
            c = float(entry["c"])
            v = float(entry.get("v", 0.0))
        except (KeyError, TypeError, ValueError):
            continue

        if any(not math.isfinite(value) for value in (t, o, h, l, c, v)):
            continue
        if t in seen:
            raise InsufficientDataError("duplicate candle timestamps detected")
        seen.add(t)
        parsed.append(Candle(t=t, o=o, h=h, l=l, c=c, v=v))

    if not parsed:
        raise InsufficientDataError("no valid candles available")

    parsed.sort(key=lambda candle: candle.t)
    return parsed


def _latest_snapshot_with_frame(frame_key: str = "1m") -> Mapping[str, object]:
    for meta in list_snapshots():
        snapshot_id = meta.get("id")
        if not snapshot_id:
            continue
        snapshot = get_snapshot(str(snapshot_id))
        if not snapshot:
            continue
        frames = snapshot.get("frames")
        if isinstance(frames, Mapping) and frame_key in frames:
            return snapshot
    raise InsufficientDataError("no snapshot with the requested timeframe")


def _group_previous_days(
    candles: Sequence[Candle],
    current_local_date: date,
) -> List[PreviousDaySummary]:
    buckets: MutableSequence[PreviousDaySummary] = []
    day_candles: dict[date, List[Candle]] = {}
    for candle in candles:
        dt_utc = datetime.fromtimestamp(candle.t / 1000.0, tz=timezone.utc)
        local_dt = dt_utc.astimezone(BERLIN_TZ)
        local_date = local_dt.date()
        if local_date >= current_local_date:
            continue
        day_candles.setdefault(local_date, []).append(candle)

    for local_date in sorted(day_candles):
        items = day_candles[local_date]
        if not items:
            continue
        if len(items) != len({item.t for item in items}):
            raise InsufficientDataError("duplicate timestamps inside daily bucket")

        first_candle = items[0]
        last_candle = items[-1]

        highs = [item.h for item in items]
        lows = [item.l for item in items]
        volumes = [item.v for item in items]

        dt_utc = datetime.fromtimestamp(first_candle.t / 1000.0, tz=timezone.utc)
        buckets.append(
            PreviousDaySummary(
                date_utc=dt_utc.date().isoformat(),
                date_local=local_date.isoformat(),
                bars=len(items),
                open=round(first_candle.o, 2),
                high=round(max(highs), 2),
                low=round(min(lows), 2),
                close=round(last_candle.c, 2),
                volume_sum=round(sum(volumes), 6),
                ret_close_pct=round(
                    ((last_candle.c - first_candle.o) / first_candle.o * 100.0)
                    if first_candle.o
                    else 0.0,
                    4,
                ),
            )
        )

    return list(buckets)


def _aggregate_previous_days(entries: Sequence[PreviousDaySummary]) -> PreviousDaysAggregate:
    if not entries:
        return PreviousDaysAggregate(
            days=0,
            volume_sum=0.0,
            ret_close_pct_avg=0.0,
            high_max=0.0,
            low_min=0.0,
        )

    return PreviousDaysAggregate(
        days=len(entries),
        volume_sum=round(sum(item.volume_sum for item in entries), 6),
        ret_close_pct_avg=round(
            sum(item.ret_close_pct for item in entries) / len(entries), 4
        ),
        high_max=round(max(item.high for item in entries), 2),
        low_min=round(min(item.low for item in entries), 2),
    )


def _compute_volatility(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _build_last_hours(
    candles: Sequence[Candle],
    *,
    hours: int,
    now_utc: datetime,
) -> LastHoursDetailed:
    now_local = now_utc.astimezone(BERLIN_TZ)
    window_start_local = now_local - timedelta(hours=hours)
    start_utc = window_start_local.astimezone(timezone.utc)
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(now_utc.timestamp() * 1000)

    window_candles = [c for c in candles if start_ms <= c.t <= end_ms]
    if not window_candles:
        raise InsufficientDataError("not enough candles for the requested window")

    closes: List[float] = []
    returns: List[float] = []
    detailed_rows: List[DetailedBar] = []
    total_volume = 0.0
    total_pv = 0.0

    for candle in window_candles:
        closes.append(candle.c)
        total_volume += candle.v
        total_pv += candle.c * candle.v

    vwap_value = round(total_pv / total_volume, 4) if total_volume > 0 else 0.0

    peak_close = window_candles[0].c
    max_drawdown = 0.0

    for idx, candle in enumerate(window_candles):
        prev_close = window_candles[idx - 1].c if idx > 0 else None
        if prev_close and prev_close != 0:
            ret = (candle.c - prev_close) / prev_close * 100.0
        else:
            ret = 0.0
        ret = round(ret, 4)
        returns.append(ret)

        sma_5 = round(sum(closes[max(0, idx - 4) : idx + 1]) / min(idx + 1, 5), 4)
        sma_15 = round(sum(closes[max(0, idx - 14) : idx + 1]) / min(idx + 1, 15), 4)

        volatility_window = returns[max(0, idx - 14) : idx + 1]
        volatility = round(_compute_volatility(volatility_window), 4)

        peak_close = max(peak_close, candle.c)
        if peak_close > 0:
            drawdown = (candle.c - peak_close) / peak_close * 100.0
            max_drawdown = min(max_drawdown, drawdown)

        detailed_rows.append(
            DetailedBar(
                t=candle.t,
                o=round(candle.o, 4),
                h=round(candle.h, 4),
                l=round(candle.l, 4),
                c=round(candle.c, 4),
                v=round(candle.v, 6),
                ret_1m=ret,
                sma_5=sma_5,
                sma_15=sma_15,
                vwap_window=vwap_value,
                volatility_15m=volatility,
            )
        )

    first = window_candles[0]
    last = window_candles[-1]
    window_high = max(candle.h for candle in window_candles)
    window_low = min(candle.l for candle in window_candles)
    volume_window = round(sum(candle.v for candle in window_candles), 6)
    ret_window = (
        (last.c - first.o) / first.o * 100.0
        if first.o
        else 0.0
    )

    return LastHoursDetailed(
        hours=hours,
        window_utc=WindowUTC.parse_obj({"from": start_ms, "to": end_ms}),
        ohlc_window=OhlcWindow(
            o=round(first.o, 4),
            h=round(window_high, 4),
            l=round(window_low, 4),
            c=round(last.c, 4),
        ),
        volume_window=volume_window,
        ret_window_pct=round(ret_window, 4),
        max_drawdown_pct=round(max_drawdown, 4),
        bars=detailed_rows,
    )


def build_market_overview(
    *, hours: int, now_utc: datetime | None = None
) -> MarketSliceResponse:
    if hours < 2 or hours > 4:
        raise ValueError("hours must be within 2..4")

    snapshot = _latest_snapshot_with_frame("1m")
    frames = snapshot.get("frames")
    if not isinstance(frames, Mapping):
        raise InsufficientDataError("snapshot frames are missing")
    frame = frames.get("1m")
    if not isinstance(frame, Mapping):
        raise InsufficientDataError("snapshot does not contain 1m frame")

    candles = _coerce_candles(frame)

    now = now_utc or _now_utc()
    current_local_date = now.astimezone(BERLIN_TZ).date()

    previous_days = _group_previous_days(candles, current_local_date)
    aggregate = _aggregate_previous_days(previous_days)
    last_hours = _build_last_hours(candles, hours=hours, now_utc=now)

    symbol = str(snapshot.get("symbol") or "UNKNOWN").upper()
    asof_ms = int(now.timestamp() * 1000)

    return MarketSliceResponse(
        symbol=symbol,
        timezone="Europe/Berlin",
        asof_utc=asof_ms,
        previous_days_summary=previous_days,
        previous_days_aggregate=aggregate,
        last_hours_detailed=last_hours,
    )


def render_market_overview_html(payload: MarketSliceResponse, template) -> str:
    """Render the HTML widget for the aggregated market overview."""

    if hasattr(payload, "model_dump"):
        data = payload.model_dump(mode="json", by_alias=True)
    else:
        data = payload.dict(by_alias=True)
    context = {"payload": data}
    return template.render(**context)

