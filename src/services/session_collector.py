"""Session-detailed collector for the inspection panel.

This implementation fetches live UM-futures data directly from Binance REST
endpoints via the bundled connector, aggregates it for the latest session
window, and stitches in the previous session when the currently active window
is shorter than an hour.  The resulting payload mirrors the inspection panel's
72-hour pipeline structure but scoped to the relevant session range.
"""
from __future__ import annotations

import asyncio
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from . import tracing
from .binance import (
    BinanceRateLimitBudgetExceeded,
    fetch_um_agg_trades,
    fetch_um_klines,
)
from .binance_stream import get_recent_stream_rows
from .orderflow import compute_orderflow_aggregates
from .progress import ProgressReporter, emit_progress
from .vision_store import get_store

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

MINUTE_MS = 60_000
SESSION_MIN_COVERAGE = 0.9
SESSION_MIN_DURATION_MIN = 60
SESSION_CACHE_TTL_SECONDS = max(60, int(os.getenv("SESSION_COLLECTOR_CACHE_TTL", "300")))
SESSION_CACHE_GRACE_SECONDS = max(
    SESSION_CACHE_TTL_SECONDS,
    int(os.getenv("SESSION_COLLECTOR_CACHE_GRACE", "900")),
)
REST_PAGE_LIMIT = 1_000
REST_MAX_KLINE_ITER = 2_000
REST_MAX_TRADE_BATCHES = 240


def _day_from_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


async def _rest_fetch_klines(
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    """Fetch 1m klines via REST within the requested window."""

    cursor = start_ms
    candles: List[Dict[str, Any]] = []
    iterations = 0
    while cursor <= end_ms and iterations < REST_MAX_KLINE_ITER:
        batch = await fetch_um_klines(
            symbol,
            "1m",
            start_time=cursor,
            end_time=end_ms,
            limit=REST_PAGE_LIMIT,
        )
        if not batch:
            break
        for row in batch:
            if len(row) < 7:
                continue
            try:
                open_time = int(row[0])
                close_time = int(row[6])
                open_price = float(row[1])
                high_price = float(row[2])
                low_price = float(row[3])
                close_price = float(row[4])
                volume = float(row[5])
            except (TypeError, ValueError):
                continue
            if open_time < start_ms or open_time > end_ms:
                continue
            quote_volume = None
            trades = None
            if len(row) > 7 and row[7] is not None:
                try:
                    quote_volume = float(row[7])
                except (TypeError, ValueError):
                    quote_volume = None
            if len(row) > 8 and row[8] is not None:
                try:
                    trades = int(row[8])
                except (TypeError, ValueError):
                    trades = None
            candles.append(
                {
                    "ts": open_time,
                    "open_time": open_time,
                    "close_time": close_time,
                    "o": open_price,
                    "h": high_price,
                    "l": low_price,
                    "c": close_price,
                    "v": volume,
                    "quote_volume": quote_volume,
                    "trades": trades,
                }
            )
        last_open_time = int(batch[-1][0])
        if last_open_time <= cursor:
            cursor += MINUTE_MS
        else:
            cursor = last_open_time + MINUTE_MS
        if len(batch) < REST_PAGE_LIMIT:
            break
        iterations += 1
    return candles


async def _rest_fetch_agg_trades(
    symbol: str,
    start_ms: int,
    end_ms: int,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Fetch aggregated trades via REST within the requested window."""

    trades: List[Dict[str, Any]] = []
    seen_ids: set[int] = set()
    cursor = start_ms
    batches = 0
    rate_limit_info: Optional[Dict[str, Any]] = None
    while cursor <= end_ms and batches < REST_MAX_TRADE_BATCHES:
        try:
            batch = await fetch_um_agg_trades(
                symbol,
                start_time=cursor,
                end_time=end_ms,
                limit=REST_PAGE_LIMIT,
            )
        except BinanceRateLimitBudgetExceeded as exc:
            rate_limit_info = {
                "scope": "agg_trades",
                "retry_after": exc.retry_after,
                "weight": exc.weight,
                "path": exc.path,
                "caller": exc.caller,
                "pending_start_ms": cursor,
                "end_ms": end_ms,
                "retrieved_batches": batches,
                "retrieved_count": len(trades),
            }
            TRACE_LOGGER.warning(
                "session_collector:rate_limited",
                extra={
                    "symbol": symbol,
                    "scope": "agg_trades",
                    "retry_after": round(exc.retry_after, 3),
                    "pending_start_ms": cursor,
                    "end_ms": end_ms,
                    "retrieved_batches": batches,
                    "retrieved_count": len(trades),
                },
            )
            break
        if not batch:
            break
        last_ts = cursor
        for entry in batch:
            try:
                agg_id = int(entry.get("a"))
                trade_ts = int(entry.get("T") or entry.get("t"))
                price = float(entry.get("p"))
                qty = float(entry.get("q"))
            except (TypeError, ValueError):
                continue
            if agg_id in seen_ids:
                continue
            if trade_ts < start_ms or trade_ts > end_ms:
                continue
            seen_ids.add(agg_id)
            buyer_maker = bool(entry.get("m"))
            trades.append(
                {
                    "agg_id": agg_id,
                    "ts": trade_ts,
                    "price": price,
                    "qty": qty,
                    "buyer_maker": buyer_maker,
                    "side": "sell" if buyer_maker else "buy",
                    "first_id": entry.get("f"),
                    "last_id": entry.get("l"),
                }
            )
            if trade_ts > last_ts:
                last_ts = trade_ts
        if last_ts <= cursor:
            cursor += 1
        else:
            cursor = last_ts + 1
        if len(batch) < REST_PAGE_LIMIT:
            break
        batches += 1
    return trades, rate_limit_info


def _floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def _iso_from_ms(value: int) -> str:
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class SessionSegment:
    """Stats for a particular session window."""

    name: str
    start_ms: int
    end_ms: int
    active: bool
    minutes_expected: int
    minutes_found: int

    def coverage_pct(self) -> float:
        if self.minutes_expected <= 0:
            return 0.0
        return round(self.minutes_found / self.minutes_expected, 6)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "start_utc": _iso_from_ms(self.start_ms),
            "end_utc": _iso_from_ms(self.end_ms),
            "minutes_expected": self.minutes_expected,
            "minutes_found": self.minutes_found,
            "coverage_pct": self.coverage_pct(),
            "active": self.active,
        }


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
    segments: Tuple[SessionSegment, ...]
    data: Dict[str, Any]
    meta: Dict[str, Any]
    availability: Dict[str, Any]
    missing_fields: tuple[str, ...]
    notes: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        open_iso = _to_iso(self.session.open_utc)
        close_iso = _to_iso(self.session.close_utc)

        session_block: Dict[str, Any] = {
            "name": self.session.name,
            "open_utc": open_iso,
            "close_utc": close_iso,
            "coverage_pct": round(self.coverage_pct, 5),
            "active": self.session.is_active,
            "minutes_expected": next((seg.minutes_expected for seg in self.segments if seg.name == self.session.name), 0),
            "minutes_found": next((seg.minutes_found for seg in self.segments if seg.name == self.session.name), 0),
        }

        payload: Dict[str, Any] = {
            "schema": "session_detailed.v1",
            "status": self.status,
            "meta": self.meta,
            "session": session_block,
            "data": self.data,
            "availability": self.availability,
            "missing_fields": list(self.missing_fields),
            "segments": [segment.as_dict() for segment in self.segments],
            "notes": list(self.notes),
        }

        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SessionCollectionResult":
        meta_block = dict(payload.get("meta") or {})
        session_block = dict(payload.get("session") or {})
        data_block = dict(payload.get("data") or {})
        availability_block = dict(payload.get("availability") or {})
        segments_payload = list(payload.get("segments") or [])
        notes_payload = list(payload.get("notes") or [])

        def _parse_iso(value: Any) -> datetime:
            if not value:
                return datetime.now(timezone.utc)
            if isinstance(value, datetime):
                return value.astimezone(timezone.utc)
            if isinstance(value, str):
                candidate = value.replace("Z", "+00:00")
                try:
                    parsed = datetime.fromisoformat(candidate)
                except ValueError:
                    return datetime.now(timezone.utc)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            try:
                ms = int(value)
            except (TypeError, ValueError):
                return datetime.now(timezone.utc)
            return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

        session_window = SessionWindow(
            name=str(session_block.get("name") or "unknown"),
            open_utc=_parse_iso(session_block.get("open_utc")),
            close_utc=_parse_iso(session_block.get("close_utc")),
            is_active=bool(session_block.get("active", False)),
        )

        segments: List[SessionSegment] = []
        for segment_data in segments_payload:
            try:
                segments.append(
                    SessionSegment(
                        name=str(segment_data.get("name") or session_window.name),
                        start_ms=int(segment_data.get("start_ms")),
                        end_ms=int(segment_data.get("end_ms")),
                        active=bool(segment_data.get("active")),
                        minutes_expected=int(segment_data.get("minutes_expected", 0)),
                        minutes_found=int(segment_data.get("minutes_found", 0)),
                    )
                )
            except (TypeError, ValueError):
                continue
        if not segments:
            segments.append(
                SessionSegment(
                    name=session_window.name,
                    start_ms=int(session_block.get("start_ms", int(session_window.open_utc.timestamp() * 1000))),
                    end_ms=int(session_block.get("end_ms", int(session_window.close_utc.timestamp() * 1000))),
                    active=session_window.is_active,
                    minutes_expected=int(session_block.get("minutes_expected", 0)),
                    minutes_found=int(session_block.get("minutes_found", 0)),
                )
            )

        return cls(
            symbol=str(meta_block.get("symbol") or payload.get("symbol") or "UNKNOWN"),
            status=str(payload.get("status") or "insufficient_data"),
            session=session_window,
            coverage_pct=float(session_block.get("coverage_pct", 0.0)),
            segments=tuple(segments),
            data=data_block,
            meta=meta_block,
            availability=availability_block,
            missing_fields=tuple(payload.get("missing_fields") or ()),
            notes=tuple(notes_payload),
        )


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _day_from_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


_SESSION_CACHE: Dict[str, Tuple[float, SessionCollectionResult]] = {}
_SESSION_LOCKS: Dict[str, asyncio.Lock] = {}
_RATE_LIMIT_RECOVERY: Dict[str, asyncio.Task] = {}
_RATE_LIMIT_ATTEMPTS: Dict[str, int] = defaultdict(int)
_RATE_LIMIT_MAX_ATTEMPTS = max(1, int(os.getenv("SESSION_COLLECTOR_RATE_LIMIT_ATTEMPTS", "3")))
_RATE_LIMIT_PADDING = float(os.getenv("SESSION_COLLECTOR_RATE_LIMIT_PADDING", "0.5"))


def _make_cache_key(symbol: str, primary_segment: SessionSegment) -> str:
    return f"{symbol}:{primary_segment.start_ms}:{primary_segment.end_ms}"


def _cache_age_seconds(entry: Tuple[float, SessionCollectionResult]) -> float:
    fetched_at, _ = entry
    return time.time() - fetched_at


def _cache_get(cache_key: str, *, fresh_only: bool = True) -> Optional[SessionCollectionResult]:
    entry = _SESSION_CACHE.get(cache_key)
    if not entry:
        return None
    age = _cache_age_seconds(entry)
    if fresh_only and age > SESSION_CACHE_TTL_SECONDS:
        return None
    return entry[1]


def _cache_set(cache_key: str, result: SessionCollectionResult) -> None:
    _SESSION_CACHE[cache_key] = (time.time(), result)


def _cache_lock(cache_key: str) -> asyncio.Lock:
    lock = _SESSION_LOCKS.get(cache_key)
    if lock is None:
        lock = asyncio.Lock()
        _SESSION_LOCKS[cache_key] = lock
    return lock


def _schedule_rate_limit_recovery(
    cache_key: str,
    *,
    symbol: str,
    retry_after: float,
    allow_rest: bool,
) -> None:
    if cache_key in _RATE_LIMIT_RECOVERY:
        return
    attempts = _RATE_LIMIT_ATTEMPTS[cache_key]
    if attempts >= _RATE_LIMIT_MAX_ATTEMPTS:
        TRACE_LOGGER.warning(
            "session_collector:rate_limit.recovery_exhausted",
            extra={"symbol": symbol, "cache_key": cache_key, "attempts": attempts},
        )
        return
    _RATE_LIMIT_ATTEMPTS[cache_key] = attempts + 1

    async def _worker() -> None:
        try:
            await asyncio.sleep(max(retry_after, 0.0) + _RATE_LIMIT_PADDING)
            await collect_last_session_detailed(
                symbol,
                now_override=None,
                progress=None,
                allow_rest=allow_rest,
                background=True,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            TRACE_LOGGER.warning(
                "session_collector:rate_limit.recovery_failed",
                extra={"symbol": symbol, "cache_key": cache_key, "error": str(exc)},
            )
        finally:
            _RATE_LIMIT_RECOVERY.pop(cache_key, None)

    TRACE_LOGGER.info(
        "session_collector:rate_limit.recovery_scheduled",
        extra={
            "symbol": symbol,
            "cache_key": cache_key,
            "retry_after": round(max(retry_after, 0.0), 3),
            "attempt": _RATE_LIMIT_ATTEMPTS[cache_key],
        },
    )
    _RATE_LIMIT_RECOVERY[cache_key] = asyncio.create_task(_worker())


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


def _build_session_segment(window: SessionWindow, *, reference: datetime) -> SessionSegment:
    start_dt = _floor_minute(window.open_utc)

    if window.is_active:
        cap_dt = min(window.close_utc, reference)
        if cap_dt <= start_dt:
            minutes_expected = 1
            end_dt = start_dt
        else:
            elapsed_minutes = max(1, int(math.ceil((cap_dt - start_dt).total_seconds() / 60.0)))
            end_dt = start_dt + timedelta(minutes=elapsed_minutes - 1)
            minutes_expected = elapsed_minutes
    else:
        cap_dt = window.close_utc
        cap_floor = _floor_minute(cap_dt)
        if cap_floor <= start_dt:
            minutes_expected = 0
            end_dt = start_dt
        else:
            end_dt = cap_floor - timedelta(minutes=1)
            minutes_expected = int(((end_dt - start_dt).total_seconds() // 60) + 1)

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    if minutes_expected <= 0:
        end_ms = start_ms

    return SessionSegment(
        name=window.name,
        start_ms=start_ms,
        end_ms=end_ms,
        active=window.is_active,
        minutes_expected=minutes_expected,
        minutes_found=0,
    )


def _previous_session(window: SessionWindow) -> SessionWindow:
    probe = window.open_utc - timedelta(minutes=1)
    return _resolve_session_window(probe)


def _prepare_session_context(now_dt: datetime) -> Tuple[SessionWindow, List[SessionSegment], List[str]]:
    session_window = _resolve_session_window(now_dt)
    primary_segment = _build_session_segment(session_window, reference=now_dt)
    segments: List[SessionSegment] = [primary_segment]
    notes: List[str] = []
    if primary_segment.minutes_expected < SESSION_MIN_DURATION_MIN:
        prev_window = _previous_session(session_window)
        prev_segment = _build_session_segment(prev_window, reference=prev_window.close_utc)
        segments.insert(0, prev_segment)
        notes.append(f"extended_previous_session:{prev_window.name}")
    return session_window, segments, notes


def _session_for_ts(ts_ms: int, segments: Sequence[SessionSegment]) -> Optional[str]:
    for segment in segments:
        if segment.start_ms <= ts_ms <= segment.end_ms:
            return segment.name
    return None


def _aggregate_timeframe(
    candles: Sequence[Mapping[str, Any]],
    minutes: int,
) -> List[Dict[str, Any]]:
    if not candles:
        return []
    interval_ms = minutes * MINUTE_MS
    buckets: Dict[int, Dict[str, Any]] = {}
    for candle in candles:
        ts = int(candle.get("t", 0))
        bucket_id = (ts // interval_ms) * interval_ms
        bucket = buckets.get(bucket_id)
        if bucket is None:
            bucket = {
                "t": bucket_id,
                "o": float(candle["o"]),
                "h": float(candle["h"]),
                "l": float(candle["l"]),
                "c": float(candle["c"]),
                "v": float(candle["v"]),
                "session": candle.get("session"),
            }
            buckets[bucket_id] = bucket
        else:
            bucket["h"] = max(bucket["h"], float(candle["h"]))
            bucket["l"] = min(bucket["l"], float(candle["l"]))
            bucket["c"] = float(candle["c"])
            bucket["v"] += float(candle["v"])
    aggregated = [dict(value) for key, value in sorted(buckets.items())]
    return aggregated


def _filter_session_candles(
    candles: Sequence[Mapping[str, Any]],
    segment: SessionSegment,
) -> List[Mapping[str, Any]]:
    scoped: List[Mapping[str, Any]] = []
    for candle in candles:
        ts = int(candle.get("t", 0))
        if segment.start_ms <= ts <= segment.end_ms:
            scoped.append(candle)
    return scoped


def _compute_vwap_stats(candles: Sequence[Mapping[str, Any]]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not candles:
        return None, None, None
    volume_sum = 0.0
    vwap_sum = 0.0
    high_price = -math.inf
    low_price = math.inf
    for candle in candles:
        high_price = max(high_price, float(candle["h"]))
        low_price = min(low_price, float(candle["l"]))
        typical = (float(candle["h"]) + float(candle["l"]) + float(candle["c"])) / 3.0
        volume = float(candle["v"])
        vwap_sum += typical * volume
        volume_sum += volume
    if volume_sum <= 0:
        return float(candles[-1]["c"]), high_price if math.isfinite(high_price) else None, low_price if math.isfinite(low_price) else None
    vwap_value = vwap_sum / volume_sum
    return vwap_value, high_price if math.isfinite(high_price) else None, low_price if math.isfinite(low_price) else None


def _build_orderflow_rows(
    trades: Iterable[Mapping[str, Any]],
    segments: Sequence[SessionSegment],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    minute_buckets: MutableMapping[int, Dict[str, Any]] = {}
    for trade in trades:
        raw_ts = trade.get("T") or trade.get("t") or trade.get("ts")
        price_raw = trade.get("p") or trade.get("price")
        qty_raw = trade.get("q") or trade.get("qty")
        buyer_flag = trade.get("m")
        if raw_ts is None or price_raw is None or qty_raw is None:
            continue
        try:
            trade_time = int(raw_ts)
            price = float(price_raw)
            quantity = float(qty_raw)
            buyer_is_maker = bool(buyer_flag)
        except (TypeError, ValueError):
            continue
        if quantity <= 0:
            continue
        bucket = (trade_time // MINUTE_MS) * MINUTE_MS
        bucket_row = minute_buckets.get(bucket)
        if bucket_row is None:
            bucket_row = {
                "ts": bucket,
                "t": _iso_from_ms(bucket),
                "bid": 0.0,
                "ask": 0.0,
                "volume": 0.0,
                "price": 0.0,
                "delta": 0.0,
            }
            minute_buckets[bucket] = bucket_row
        prev_volume = bucket_row["volume"]
        bucket_row["volume"] += quantity
        if buyer_is_maker:
            bucket_row["bid"] += quantity
        else:
            bucket_row["ask"] += quantity
        if bucket_row["volume"] > 0:
            bucket_row["price"] = (
                (bucket_row["price"] * prev_volume + price * quantity) / bucket_row["volume"]
            )
    rows: List[Dict[str, Any]] = []
    running_cvd = 0.0
    for _, entry in sorted(minute_buckets.items()):
        delta = float(entry["ask"]) - float(entry["bid"])
        entry["delta"] = delta
        running_cvd += delta
        entry["cvd"] = running_cvd
        entry["session"] = _session_for_ts(int(entry["ts"]), segments)
        rows.append(entry)
    aggregates = compute_orderflow_aggregates(rows)
    return rows, aggregates


def _build_vwap_session_block(
    segments: Sequence[SessionSegment],
    candles: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    sessions: Dict[str, Any] = {}
    for segment in segments:
        scoped = _filter_session_candles(candles, segment)
        vwap_value, high_value, low_value = _compute_vwap_stats(scoped)
        sessions[segment.name] = {
            "vwap": vwap_value,
            "sd1": None,
            "sd2": None,
            "poc": None,
            "vah": None,
            "val": None,
            "ib_high": None,
            "ib_low": None,
            "high": high_value,
            "low": low_value,
        }
    return {"daily": {}, "sessions": sessions}


def _calculate_availability(
    candles: Sequence[Mapping[str, Any]],
    agg3m: Sequence[Mapping[str, Any]],
    agg5m: Sequence[Mapping[str, Any]],
    orderflow_rows: Sequence[Mapping[str, Any]],
    segments: Sequence[SessionSegment],
) -> Dict[str, Any]:
    availability = {
        "ohlcv": {
            "1m": bool(candles),
            "3m": bool(agg3m),
            "5m": bool(agg5m),
            "15m": False,
            "1h": False,
        },
        "orderflow": {
            "delta": bool(orderflow_rows),
            "cvd": bool(orderflow_rows),
            "footprint": bool(orderflow_rows),
        },
        "vwap_sessions": {segment.name: bool(_filter_session_candles(candles, segment)) for segment in segments},
    }
    return availability


def _build_meta(
    symbol: str,
    *,
    segments: Sequence[SessionSegment],
    candles: Sequence[Mapping[str, Any]],
    now_dt: datetime,
) -> Dict[str, Any]:
    last_ts = candles[-1]["t"] if candles else None
    last_price = candles[-1]["c"] if candles else None
    data_freshness = None
    stale = True
    if last_ts is not None:
        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        data_freshness = max((now_dt - last_dt).total_seconds(), 0.0)
        stale = data_freshness > 600
    return {
        "symbol": symbol.upper(),
        "tz": "Europe/Berlin",
        "last_price": float(last_price) if last_price is not None else None,
        "last_ts_utc": _iso_from_ms(last_ts) if last_ts is not None else None,
        "data_freshness_sec": data_freshness,
        "stale": stale,
        "segments": [segment.as_dict() for segment in segments],
    }


async def collect_last_session_detailed(
    symbol: str,
    now_override: datetime | None = None,
    progress: Optional[ProgressReporter] = None,
    *,
    allow_rest: bool = False,
    background: bool = False,
) -> SessionCollectionResult:
    """Collect detailed data for the most recent trading session."""

    now_dt = now_override.astimezone(timezone.utc) if isinstance(now_override, datetime) else datetime.now(timezone.utc)
    symbol_clean = (symbol or "").strip().upper()
    if not symbol_clean:
        raise ValueError("symbol is required")

    session_window, segments, base_notes = _prepare_session_context(now_dt)
    primary_segment = segments[-1]
    cache_key = _make_cache_key(symbol_clean, primary_segment)

    cached = _cache_get(cache_key)
    if cached is not None:
        age_sec = _cache_age_seconds(_SESSION_CACHE[cache_key])
        TRACE_LOGGER.debug(
            "session_collector:cache_hit",
            extra={"symbol": symbol_clean, "age_sec": round(age_sec, 2)},
        )
        await emit_progress(
            progress,
            "session_collector:cache_hit",
            symbol=symbol_clean,
            age_sec=round(age_sec, 2),
            status=cached.status,
            coverage_pct=cached.coverage_pct,
        )
        await emit_progress(
            progress,
            "session_collector:finished",
            symbol=symbol_clean,
            status=cached.status,
            coverage_pct=cached.coverage_pct,
            segments=[seg.as_dict() for seg in cached.segments],
            klines=len(cached.data.get("ohlcv", {}).get("1m", {}).get("bars", [])),
            agg_trades=len(cached.data.get("orderflow", {}).get("per_bar", [])),
            missing_fields=list(cached.missing_fields),
            cache_hit=True,
        )
        return cached

    lock = _cache_lock(cache_key)
    async with lock:
        cached_late = _cache_get(cache_key)
        if cached_late is not None:
            age_sec = _cache_age_seconds(_SESSION_CACHE[cache_key])
            TRACE_LOGGER.debug(
                "session_collector:cache_hit",
                extra={"symbol": symbol_clean, "age_sec": round(age_sec, 2)},
            )
            await emit_progress(
                progress,
                "session_collector:cache_hit",
                symbol=symbol_clean,
                age_sec=round(age_sec, 2),
                status=cached_late.status,
                coverage_pct=cached_late.coverage_pct,
            )
            await emit_progress(
                progress,
                "session_collector:finished",
                symbol=symbol_clean,
                status=cached_late.status,
                coverage_pct=cached_late.coverage_pct,
                segments=[seg.as_dict() for seg in cached_late.segments],
                klines=len(cached_late.data.get("ohlcv", {}).get("1m", {}).get("bars", [])),
                agg_trades=len(cached_late.data.get("orderflow", {}).get("per_bar", [])),
                missing_fields=list(cached_late.missing_fields),
                cache_hit=True,
            )
            return cached_late

        store = get_store()
        try:
            result = await _build_session_collection(
                symbol_clean,
                now_dt,
                progress,
                session_window,
                segments,
                base_notes,
                store,
                cache_key=cache_key,
                allow_rest=allow_rest,
                background=background,
            )
        except Exception as exc:
            stale_entry = _SESSION_CACHE.get(cache_key)
            if stale_entry:
                age_sec = _cache_age_seconds(stale_entry)
                if age_sec <= SESSION_CACHE_GRACE_SECONDS:
                    TRACE_LOGGER.warning(
                        "session_collector:using_stale",
                        extra={
                            "symbol": symbol_clean,
                            "age_sec": round(age_sec, 2),
                            "error": str(exc),
                        },
                    )
                    await emit_progress(
                        progress,
                        "session_collector:using_stale",
                        symbol=symbol_clean,
                        age_sec=round(age_sec, 2),
                        error=str(exc),
                    )
                    cached_result = stale_entry[1]
                    await emit_progress(
                        progress,
                        "session_collector:finished",
                        symbol=symbol_clean,
                        status=cached_result.status,
                        coverage_pct=cached_result.coverage_pct,
                        segments=[seg.as_dict() for seg in cached_result.segments],
                        klines=len(cached_result.data.get("ohlcv", {}).get("1m", {}).get("bars", [])),
                        agg_trades=len(cached_result.data.get("orderflow", {}).get("per_bar", [])),
                        missing_fields=list(cached_result.missing_fields),
                        cache_hit=True,
                        stale=True,
                    )
                    return cached_result
            stored_record = store.fetch_session_payload(symbol_clean, primary_segment.start_ms, primary_segment.name)
            if stored_record and stored_record.get("payload"):
                stored_result = SessionCollectionResult.from_payload(stored_record["payload"])
                TRACE_LOGGER.warning(
                    "session_collector:using_stored_payload",
                    extra={"symbol": symbol_clean, "error": str(exc)},
                )
                candles = stored_result.data.get("ohlcv", {}).get("1m", {}).get("bars", [])
                stored_result.meta = _build_meta(
                    symbol_clean,
                    segments=list(stored_result.segments),
                    candles=candles,
                    now_dt=now_dt,
                )
                await emit_progress(
                    progress,
                    "session_collector:using_stored_payload",
                    symbol=symbol_clean,
                    status=stored_result.status,
                    coverage_pct=stored_result.coverage_pct,
                    error=str(exc),
                )
                _cache_set(cache_key, stored_result)
                return stored_result
            raise
        else:
            _cache_set(cache_key, result)
            return result


async def _build_session_collection(
    symbol_clean: str,
    now_dt: datetime,
    progress: Optional[ProgressReporter],
    session_window: SessionWindow,
    segments: List[SessionSegment],
    base_notes: List[str],
    store,
    *,
    cache_key: str,
    allow_rest: bool,
    background: bool,
) -> SessionCollectionResult:
    TRACE_LOGGER.debug(
        "session_collector:start",
        extra={
            "symbol": symbol_clean,
            "requested_at": now_dt.isoformat(),
        },
    )
    await emit_progress(
        progress,
        "session_collector:start",
        symbol=symbol_clean,
        requested_at=now_dt.isoformat(),
    )
    TRACE_LOGGER.debug(
        "session_collector:resolved_window",
        extra={
            "symbol": symbol_clean,
            "session": session_window.name,
            "open_utc": session_window.open_utc.isoformat(),
            "close_utc": session_window.close_utc.isoformat(),
            "active": session_window.is_active,
        },
    )
    await emit_progress(
        progress,
        "session_collector:resolved_window",
        symbol=symbol_clean,
        session=session_window.name,
        open_utc=session_window.open_utc.isoformat(),
        close_utc=session_window.close_utc.isoformat(),
        active=session_window.is_active,
    )

    notes = list(base_notes)
    primary_segment = segments[-1]

    combined_start = min(segment.start_ms for segment in segments)
    combined_end = max(segment.end_ms for segment in segments)
    await emit_progress(
        progress,
        "session_collector:fetch_start",
        symbol=symbol_clean,
        start_ms=combined_start,
        end_ms=combined_end,
    )

    stored_record = store.fetch_session_payload(symbol_clean, primary_segment.start_ms, primary_segment.name)
    stored_result = None
    if stored_record and stored_record.get("payload"):
        try:
            stored_result = SessionCollectionResult.from_payload(stored_record["payload"])
        except Exception:
            stored_result = None

    expected_minutes = sum(segment.minutes_expected for segment in segments)
    rest_notes: List[str] = []
    rate_limit_info: Optional[Dict[str, Any]] = None
    rest_errors: List[str] = []

    klines_rows: List[Sequence[Any]] = await asyncio.to_thread(
        store.fetch_klines,
        symbol_clean,
        "1m",
        combined_start,
        combined_end + MINUTE_MS,
        None,
    )

    if allow_rest and expected_minutes > 0 and len(klines_rows) < expected_minutes:
        TRACE_LOGGER.info(
            "session_collector:rest_request",
            extra={
                "symbol": symbol_clean,
                "stage": "klines",
                "reason": "insufficient_minutes",
                "start_ms": combined_start,
                "end_ms": combined_end,
            },
        )
        try:
            rest_klines = await _rest_fetch_klines(symbol_clean, combined_start, combined_end)
        except Exception as exc:  # pragma: no cover - defensive logging
            rest_errors.append(f"klines:{exc}")
            TRACE_LOGGER.warning(
                "session_collector:rest_backfill_failed",
                extra={"symbol": symbol_clean, "stage": "klines", "error": str(exc)},
            )
        else:
            if rest_klines:
                grouped_klines: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for candle in rest_klines:
                    grouped_klines[_day_from_ms(int(candle["ts"]))].append(dict(candle))
                inserted_total = 0
                for day, payload in grouped_klines.items():
                    stats = await asyncio.to_thread(
                        store.upsert_klines,
                        symbol_clean,
                        "1m",
                        day,
                        payload,
                    )
                    if stats:
                        inserted_total += getattr(stats, "inserted", 0)
                rest_notes.append("rest_backfill:klines")
                TRACE_LOGGER.debug(
                    "session_collector:rest_backfill.klines",
                    extra={
                        "symbol": symbol_clean,
                        "days": list(grouped_klines.keys()),
                        "inserted": inserted_total,
                    },
                )
                klines_rows = await asyncio.to_thread(
                    store.fetch_klines,
                    symbol_clean,
                    "1m",
                    combined_start,
                    combined_end + MINUTE_MS,
                    None,
                )

    candles: List[Dict[str, Any]] = []
    for row in klines_rows:
        if len(row) < 7:
            continue
        try:
            open_time = int(row[0])
            close_time = int(row[6])
            open_price = float(row[1])
            high_price = float(row[2])
            low_price = float(row[3])
            close_price = float(row[4])
            volume = float(row[5])
            quote_volume = float(row[7]) if len(row) > 7 and row[7] is not None else 0.0
            trades_count = int(row[8]) if len(row) > 8 and row[8] is not None else 0
        except (TypeError, ValueError):
            continue
        candle = {
            "t": open_time,
            "o": open_price,
            "h": high_price,
            "l": low_price,
            "c": close_price,
            "v": volume,
            "q": quote_volume,
            "n": trades_count,
            "close_time": close_time,
        }
        candle["session"] = _session_for_ts(open_time, segments)
        candles.append(candle)
    candles.sort(key=lambda entry: entry["t"])

    expected_latest_ms = max(segment.end_ms for segment in segments)
    latest_present_ms = candles[-1]["t"] if candles else None
    if expected_latest_ms and (latest_present_ms is None or latest_present_ms < expected_latest_ms):
        try:
            latest_kline = await asyncio.to_thread(
                store.fetch_latest_kline,
                symbol_clean,
                "1m",
            )
        except Exception:
            latest_kline = None
        open_time_ms = _safe_int((latest_kline or {}).get("open_time"))
        if latest_kline and open_time_ms == expected_latest_ms:
            try:
                refreshed_row = {
                    "t": open_time_ms,
                    "o": float(latest_kline.get("open")),
                    "h": float(latest_kline.get("high")),
                    "l": float(latest_kline.get("low")),
                    "c": float(latest_kline.get("close")),
                    "v": float(latest_kline.get("volume")),
                    "q": float(latest_kline.get("quote_volume") or 0.0),
                    "n": int(latest_kline.get("trades") or 0),
                    "close_time": int(latest_kline.get("close_time") or (open_time_ms + MINUTE_MS - 1)),
                    "session": _session_for_ts(open_time_ms, segments),
                }
            except (TypeError, ValueError):
                refreshed_row = None
            if refreshed_row is not None:
                candles = [c for c in candles if int(c.get("t", 0)) != open_time_ms]
                candles.append(refreshed_row)
                candles.sort(key=lambda entry: entry["t"])

    if not candles and stored_result is not None:
        notes.append("using_stored_payload")
        return stored_result

    trades_rows: List[Dict[str, Any]] = await asyncio.to_thread(
        store.fetch_agg_trades,
        symbol_clean,
        combined_start,
        combined_end + MINUTE_MS,
        None,
        ascending=True,
    )

    existing_trade_ids = {
        int(row["agg_id"]) for row in trades_rows if isinstance(row.get("agg_id"), int)
    }
    latest_trade_ts = trades_rows[-1]["t"] if trades_rows else None

    stream_trades = await get_recent_stream_rows(
        "aggTrades",
        symbol_clean,
        since_ms=max(combined_start, latest_trade_ts or combined_start),
    )
    if stream_trades:
        for trade in stream_trades:
            ts_value = trade.get("ts") or trade.get("t")
            try:
                trade_ts = int(ts_value)
            except (TypeError, ValueError):
                continue
            agg_id_value = trade.get("agg_id")
            agg_id_int = None
            if agg_id_value is not None:
                try:
                    agg_id_int = int(agg_id_value)
                except (TypeError, ValueError):
                    agg_id_int = None
            if agg_id_int is not None and agg_id_int in existing_trade_ids:
                continue
            try:
                price_val = float(trade.get("price"))
                qty_val = float(trade.get("qty"))
            except (TypeError, ValueError):
                continue
            maker_flag = trade.get("buyer_maker")
            if maker_flag is None:
                maker_flag = trade.get("m")
            trade_row: Dict[str, Any] = {
                "agg_id": agg_id_int,
                "t": trade_ts,
                "p": price_val,
                "q": qty_val,
                "m": bool(maker_flag),
            }
            if "side" in trade and trade.get("side") is not None:
                trade_row["side"] = trade.get("side")
            trades_rows.append(trade_row)
            if agg_id_int is not None:
                existing_trade_ids.add(agg_id_int)
        trades_rows.sort(key=lambda row: row["t"])
        latest_trade_ts = trades_rows[-1]["t"] if trades_rows else latest_trade_ts

    rest_reason: Optional[str] = None
    rest_start_ms = combined_start
    rest_end_ms = combined_end
    if not trades_rows:
        rest_reason = "empty"
    elif expected_latest_ms and (latest_trade_ts or 0) < expected_latest_ms:
        rest_reason = "missing_tail"
        rest_start_ms = max(latest_trade_ts or combined_start, combined_start)
        rest_end_ms = max(expected_latest_ms, combined_end)
    if rest_reason is not None:
        rest_end_ms = rest_end_ms + MINUTE_MS - 1

    if allow_rest and rest_reason is not None:
        TRACE_LOGGER.info(
            "session_collector:rest_request",
            extra={
                "symbol": symbol_clean,
                "stage": "agg_trades",
                "reason": rest_reason,
                "start_ms": rest_start_ms,
                "end_ms": rest_end_ms,
            },
        )
        try:
            rest_trades, rate_limit_info = await _rest_fetch_agg_trades(
                symbol_clean,
                rest_start_ms,
                rest_end_ms,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            rest_errors.append(f"agg_trades:{exc}")
            TRACE_LOGGER.warning(
                "session_collector:rest_backfill_failed",
                extra={"symbol": symbol_clean, "stage": "agg_trades", "error": str(exc)},
            )
        else:
            if rest_trades:
                grouped_trades: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
                for trade in rest_trades:
                    grouped_trades[_day_from_ms(int(trade["ts"]))].append(dict(trade))
                inserted_total = 0
                for day, payload in grouped_trades.items():
                    stats = await asyncio.to_thread(
                        store.insert_agg_trades,
                        symbol_clean,
                        day,
                        payload,
                    )
                    if stats:
                        inserted_total += getattr(stats, "inserted", 0)
                rest_notes.append("rest_backfill:agg_trades")
                TRACE_LOGGER.debug(
                    "session_collector:rest_backfill.agg_trades",
                    extra={
                        "symbol": symbol_clean,
                        "days": list(grouped_trades.keys()),
                        "inserted": inserted_total,
                    },
                )
                trades_rows = await asyncio.to_thread(
                    store.fetch_agg_trades,
                    symbol_clean,
                    combined_start,
                    combined_end + MINUTE_MS,
                    None,
                    ascending=True,
                )
            latest_trade_ts = trades_rows[-1]["t"] if trades_rows else latest_trade_ts
    elif rest_reason is not None and not allow_rest:
        TRACE_LOGGER.warning(
            "session_collector:rest_skipped",
            extra={
                "symbol": symbol_clean,
                "stage": "agg_trades",
                "reason": rest_reason,
                "start_ms": rest_start_ms,
                "end_ms": rest_end_ms,
            },
        )

    if rest_notes:
        notes.extend(rest_notes)
    if rest_errors:
        for item in rest_errors:
            stage = item.split(":", 1)[0]
            notes.append(f"rest_error:{stage}")

    rate_limit_meta: Optional[Dict[str, Any]] = None
    if rate_limit_info:
        rate_limit_meta = dict(rate_limit_info)
        rate_limit_meta.setdefault("scope", "orderflow")
        rate_limit_meta.setdefault("type", "agg_trades")
        notes.append("rate_limit_pending:orderflow")

    per_bar: List[Dict[str, Any]] = []
    orderflow_aggregates: Dict[str, List[Dict[str, Any]]] = {tf: [] for tf in ("15m", "1h")}
    orderflow_error: Optional[str] = None
    if trades_rows:
        try:
            per_bar, orderflow_aggregates = _build_orderflow_rows(trades_rows, segments)
        except Exception as exc:  # pragma: no cover - defensive logging
            orderflow_error = str(exc)
            TRACE_LOGGER.warning(
                "session_collector:orderflow_failed",
                extra={"symbol": symbol_clean, "error": str(exc)},
            )

    for segment in segments:
        segment.minutes_found = len(_filter_session_candles(candles, segment))

    agg_3m = _aggregate_timeframe(candles, 3)
    agg_5m = _aggregate_timeframe(candles, 5)

    availability = _calculate_availability(candles, agg_3m, agg_5m, per_bar, segments)
    meta_block = _build_meta(symbol_clean, segments=segments, candles=candles, now_dt=now_dt)
    if rate_limit_meta:
        meta_block.setdefault("rate_limit", []).append(rate_limit_meta)
        meta_block.setdefault("warnings", []).append("orderflow_pending")
        meta_block["rate_limit_pending"] = True
    vwap_block = _build_vwap_session_block(segments, candles)

    ohlcv_block: Dict[str, Any] = {
        "1m": {"bars": candles},
        "3m": {"bars": agg_3m},
        "5m": {"bars": agg_5m},
        "1m_rollups": candles[-180:],
        "15m_compact": [],
        "1h_compact": [],
    }

    orderflow_block: Dict[str, Any] = {
        "per_bar": per_bar,
        "delta_cvd_compact": orderflow_aggregates,
        "metrics": {
            "delta": bool(per_bar),
            "cvd": bool(per_bar),
            "footprint": bool(per_bar),
        },
        "errors": {"orderflow": orderflow_error} if orderflow_error else {},
    }
    if rate_limit_meta:
        orderflow_block["rate_limit"] = rate_limit_meta

    data_block = {
        "ohlcv": ohlcv_block,
        "orderflow": orderflow_block,
        "vwap_tpo": vwap_block,
        "zones": {"top": [], "counts": {}},
        "liquidity_targets": [],
    }

    primary_segment_now = segments[-1]
    primary_coverage = primary_segment_now.coverage_pct()

    missing_fields: List[str] = []
    if not candles:
        missing_fields.append("ohlcv.1m")
    if primary_segment_now.minutes_expected > 0 and primary_coverage < SESSION_MIN_COVERAGE:
        missing_fields.append("ohlcv.coverage")
    if not per_bar:
        missing_fields.extend(["orderflow.coverage", "orderflow.delta", "orderflow.cvd", "orderflow.footprint"])
    if rate_limit_meta:
        missing_fields.append("orderflow.rate_limit_pending")

    status = "insufficient_data"
    if candles:
        status = "ready" if primary_coverage >= SESSION_MIN_COVERAGE and per_bar else "partial"

    result = SessionCollectionResult(
        symbol=symbol_clean,
        status=status,
        session=session_window,
        coverage_pct=primary_coverage,
        segments=tuple(segments),
        data=data_block,
        meta=meta_block,
        availability=availability,
        missing_fields=tuple(sorted(set(missing_fields))),
        notes=tuple(notes),
    )

    payload_dict = result.as_dict()
    for segment in segments:
        segment_status = status if segment is primary_segment_now else "ready"
        store.upsert_session_payload(
            symbol_clean,
            segment.name,
            segment.start_ms,
            segment.end_ms,
            segment_status,
            segment.coverage_pct(),
            payload_dict,
        )

    TRACE_LOGGER.debug(
        "session_collector:finished",
        extra={
            "symbol": symbol_clean,
            "status": result.status,
            "coverage_pct": result.coverage_pct,
            "segments": [seg.as_dict() for seg in segments],
            "klines": len(candles),
            "agg_trades": len(per_bar),
            "missing_fields": list(result.missing_fields),
        },
    )

    if not rate_limit_meta:
        _RATE_LIMIT_ATTEMPTS.pop(cache_key, None)

    if rate_limit_meta and not background:
        _schedule_rate_limit_recovery(
            cache_key,
            symbol=symbol_clean,
            retry_after=float(rate_limit_meta.get("retry_after", 60.0)),
            allow_rest=allow_rest,
        )

    await emit_progress(
        progress,
        "session_collector:finished",
        symbol=symbol_clean,
        status=result.status,
        coverage_pct=result.coverage_pct,
        segments=[seg.as_dict() for seg in segments],
        klines=len(candles),
        agg_trades=len(per_bar),
        missing_fields=list(result.missing_fields),
    )

    return result
