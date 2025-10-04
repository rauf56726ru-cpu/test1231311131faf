"""High-level OHLCV utilities backed by the local minute repository."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from .candles_repository import get_repository
from .tracing import TraceContext

__all__ = [
    "Candle",
    "MinuteDataUnavailable",
    "build_multi_tf_ohlcv",
    "fetch_ohlcv",
]

SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("1m", "3m", "5m", "15m", "1h", "4h", "1d")
TIMEFRAME_TO_MINUTES: Dict[str, int] = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1_440,
}
TIMEFRAME_TO_MS: Dict[str, int] = {tf: minutes * 60_000 for tf, minutes in TIMEFRAME_TO_MINUTES.items()}

LOGGER = logging.getLogger(__name__)
_FETCH_LOCK = asyncio.Lock()
_ONE_MINUTE_CACHE_TTL = 45.0
_ONE_MINUTE_CACHE: Dict[str, Tuple[float, List["Candle"]]] = {}
_ONE_MINUTE_LOCKS: Dict[str, asyncio.Lock] = {}


class MinuteDataUnavailable(RuntimeError):
    """Raised when the local repository cannot satisfy the required minute range."""

    def __init__(
        self,
        *,
        symbol: str,
        start_ms: int,
        end_ms: int,
        missing_count: int,
        expected_count: int,
        coverage_pct: float,
        gaps: Sequence[Tuple[int, int]],
    ) -> None:
        self.symbol = symbol
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.missing_count = missing_count
        self.expected_count = expected_count
        self.coverage_pct = coverage_pct
        self.gaps = list(gaps)
        message = (
            f"minute coverage unavailable for {symbol}: missing {missing_count} of {expected_count}"
        )
        super().__init__(message)


@dataclass(slots=True)
class Candle:
    """Dataclass describing a normalized candle."""

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_wire(self) -> Dict[str, float | str]:
        """Return a serialisable payload."""

        iso_time = (
            datetime.fromtimestamp(self.ts / 1000, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        return {
            "t": iso_time,
            "o": round(self.open, 8),
            "h": round(self.high, 8),
            "l": round(self.low, 8),
            "c": round(self.close, 8),
            "v": round(self.volume, 8),
        }


def _candle_to_internal(candle: Candle) -> Dict[str, float | int]:
    """Convert a candle into the internal representation (millisecond timestamps)."""

    return {
        "t": candle.ts,
        "o": round(candle.open, 8),
        "h": round(candle.high, 8),
        "l": round(candle.low, 8),
        "c": round(candle.close, 8),
        "v": round(candle.volume, 8),
    }


def _compress_missing(minutes: Sequence[int]) -> List[Tuple[int, int]]:
    if not minutes:
        return []
    ordered = sorted(minutes)
    interval = TIMEFRAME_TO_MS["1m"]
    ranges: List[Tuple[int, int]] = []
    start = ordered[0]
    prev = start
    for ts in ordered[1:]:
        if ts - prev > interval:
            ranges.append((start, prev))
            start = ts
        prev = ts
    ranges.append((start, prev))
    return ranges


def _last_closed_minute(now: datetime | None = None) -> datetime:
    reference = now or datetime.now(timezone.utc)
    return reference.replace(second=0, microsecond=0) - timedelta(minutes=1)


def _enforce_window(
    symbol: str,
    *,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    candles: Mapping[int, Candle],
    trace: TraceContext | None,
    latency_ms: int | None = None,
) -> List[Candle]:
    expected_count = int((end_ms - start_ms) // interval_ms + 1)
    missing: List[int] = []
    ts_cursor = start_ms
    while ts_cursor <= end_ms:
        if ts_cursor not in candles:
            missing.append(ts_cursor)
        ts_cursor += interval_ms

    present = expected_count - len(missing)
    coverage_pct = 100.0
    if expected_count > 0:
        coverage_pct = max(0.0, min(100.0, (present / expected_count) * 100.0))

    metrics = {
        "expected": expected_count,
        "present": present,
        "missing": len(missing),
        "coverage_pct": round(coverage_pct, 3),
    }
    if latency_ms is not None:
        metrics["ms"] = latency_ms

    if trace is not None:
        trace.info(
            "availability.checked",
            scope="ohlcv.1m",
            symbol=symbol,
            window={"from": start_ms, "to": end_ms},
            metrics=metrics,
        )

    if missing:
        gaps = _compress_missing(missing)
        if trace is not None:
            trace.warn(
                "gaps.detected",
                scope="ohlcv.1m",
                symbol=symbol,
                window={"from": start_ms, "to": end_ms},
                metrics={
                    "segments": len(gaps),
                    "missing": len(missing),
                    "coverage_pct": round(coverage_pct, 3),
                },
            )
        raise MinuteDataUnavailable(
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
            missing_count=len(missing),
            expected_count=expected_count,
            coverage_pct=coverage_pct,
            gaps=gaps,
        )

    ordered = [candles[ts] for ts in sorted(candles)]
    if len(ordered) > expected_count:
        ordered = ordered[-expected_count:]
    return ordered


async def _collect_1m_candles(
    symbol: str,
    lookback_days: int,
    *,
    trace: TraceContext | None = None,
) -> List[Candle]:
    """Load minute candles from the local repository for the requested range."""

    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")

    end_dt = _last_closed_minute()
    lookback_minutes = max(1, lookback_days * 1_440)
    start_dt = end_dt - timedelta(minutes=lookback_minutes - 1)
    interval_ms = TIMEFRAME_TO_MS["1m"]
    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = int(start_dt.timestamp() * 1000)

    repository = get_repository()
    fetch_start = time.perf_counter()
    if trace is not None:
        trace.debug(
            "fetch.batch_start",
            scope="ohlcv.repo",
            symbol=symbol,
            window={"from": start_ms, "to": end_ms},
        )

    rows = await asyncio.to_thread(
        repository.fetch_candles,
        symbol,
        "1m",
        start_ms,
        end_ms,
    )

    latency_ms = int((time.perf_counter() - fetch_start) * 1000.0)
    candles: List[Candle] = []
    for row in rows:
        try:
            open_time = int(row["t"])
            open_price = float(row["o"])
            high_price = float(row["h"])
            low_price = float(row["l"])
            close_price = float(row["c"])
            volume = float(row.get("v", 0.0))
        except (KeyError, TypeError, ValueError):
            LOGGER.debug("Skipping malformed repository candle: %s", row)
            continue
        candles.append(
            Candle(
                ts=(open_time // interval_ms) * interval_ms,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=max(volume, 0.0),
            )
        )

    candles.sort(key=lambda candle: candle.ts)
    index = {candle.ts: candle for candle in candles}

    verified = _enforce_window(
        symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        interval_ms=interval_ms,
        candles=index,
        trace=trace,
        latency_ms=latency_ms,
    )

    if trace is not None:
        trace.info(
            "fetch.batch_done",
            scope="ohlcv.repo",
            symbol=symbol,
            window={"from": start_ms, "to": end_ms},
            metrics={"rows": len(verified), "ms": latency_ms},
        )

    return verified


def _aggregate(candles: Iterable[Candle], tf: str) -> List[Candle]:
    """Aggregate 1m candles into the target timeframe."""

    if tf not in TIMEFRAME_TO_MS:
        raise ValueError(f"Unsupported timeframe: {tf}")

    interval_ms = TIMEFRAME_TO_MS[tf]
    aggregated: Dict[int, Candle] = {}

    for candle in candles:
        bucket = (candle.ts // interval_ms) * interval_ms
        existing = aggregated.get(bucket)
        if existing is None:
            aggregated[bucket] = Candle(
                ts=bucket,
                open=candle.open,
                high=candle.high,
                low=candle.low,
                close=candle.close,
                volume=candle.volume,
            )
        else:
            existing.high = max(existing.high, candle.high)
            existing.low = min(existing.low, candle.low)
            existing.close = candle.close
            existing.volume += candle.volume

    return [aggregated[key] for key in sorted(aggregated)]


def _validate_series(series: List[Candle], tf: str) -> None:
    """Validate chronological order and price/volume constraints."""

    if not series:
        raise ValueError("No candles returned from repository")

    interval_ms = TIMEFRAME_TO_MS[tf]
    last_ts: Optional[int] = None
    for candle in series:
        if candle.high < max(candle.open, candle.close) or candle.low > min(
            candle.open, candle.close
        ):
            raise ValueError("Inconsistent OHLC bounds detected")
        if candle.volume < 0:
            raise ValueError("Detected negative volume in OHLCV series")
        if last_ts is not None and candle.ts - last_ts > interval_ms + 60_000:
            raise ValueError("Detected temporal gaps in OHLCV series")
        last_ts = candle.ts


async def _load_cached_minutes(
    symbol: str,
    lookback_days: int,
    *,
    trace: TraceContext | None = None,
) -> List[Candle]:
    cache_key = f"{symbol}:{lookback_days}"
    now = time.monotonic()
    entry = _ONE_MINUTE_CACHE.get(cache_key)
    if entry and entry[0] > now:
        if trace is not None:
            trace.debug(
                "cache.hit",
                scope="ohlcv.1m",
                symbol=symbol,
                lookback_days=lookback_days,
                metrics={"rows": len(entry[1])},
            )
        return entry[1]

    lock = _ONE_MINUTE_LOCKS.setdefault(cache_key, asyncio.Lock())
    async with lock:
        now = time.monotonic()
        entry = _ONE_MINUTE_CACHE.get(cache_key)
        if entry and entry[0] > now:
            if trace is not None:
                trace.debug(
                    "cache.hit",
                    scope="ohlcv.1m",
                    symbol=symbol,
                    lookback_days=lookback_days,
                    metrics={"rows": len(entry[1])},
                )
            return entry[1]
        if trace is not None:
            trace.debug(
                "cache.miss",
                scope="ohlcv.1m",
                symbol=symbol,
                lookback_days=lookback_days,
            )
        async with _FETCH_LOCK:
            candles = await _collect_1m_candles(symbol, lookback_days, trace=trace)
        _ONE_MINUTE_CACHE[cache_key] = (
            time.monotonic() + _ONE_MINUTE_CACHE_TTL,
            candles,
        )
        return candles


def _normalise_seed_minutes(
    seed_minutes: Sequence[Mapping[str, object] | Candle],
    *,
    lookback_days: int,
    symbol: str,
    trace: TraceContext | None = None,
) -> List[Candle]:
    interval_ms = TIMEFRAME_TO_MS["1m"]
    lookback_minutes = max(1, lookback_days * 1_440)
    raw: List[Candle] = []
    for entry in seed_minutes:
        if isinstance(entry, Candle):
            raw.append(
                Candle(
                    ts=(entry.ts // interval_ms) * interval_ms,
                    open=entry.open,
                    high=entry.high,
                    low=entry.low,
                    close=entry.close,
                    volume=max(entry.volume, 0.0),
                )
            )
            continue
        if not isinstance(entry, Mapping):
            continue
        try:
            ts = int(entry.get("t"))
            open_price = float(entry.get("o"))
            high_price = float(entry.get("h"))
            low_price = float(entry.get("l"))
            close_price = float(entry.get("c"))
            volume = float(entry.get("v", 0.0))
        except (TypeError, ValueError):
            continue
        aligned_ts = (ts // interval_ms) * interval_ms
        raw.append(
            Candle(
                ts=aligned_ts,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=max(volume, 0.0),
            )
        )

    if not raw:
        end_ts = int(_last_closed_minute().timestamp() * 1000)
        start_ts = end_ts - (lookback_minutes - 1) * interval_ms
        raise MinuteDataUnavailable(
            symbol=symbol,
            start_ms=start_ts,
            end_ms=end_ts,
            missing_count=lookback_minutes,
            expected_count=lookback_minutes,
            coverage_pct=0.0,
            gaps=[(start_ts, end_ts)],
        )

    raw.sort(key=lambda candle: candle.ts)
    end_ts = raw[-1].ts
    earliest_ts = raw[0].ts
    window_span_ms = (lookback_minutes - 1) * interval_ms
    start_ts = max(earliest_ts, end_ts - window_span_ms)
    filtered = {candle.ts: candle for candle in raw if start_ts <= candle.ts <= end_ts}
    verified = _enforce_window(
        symbol,
        start_ms=start_ts,
        end_ms=end_ts,
        interval_ms=interval_ms,
        candles=filtered,
        trace=trace,
    )
    return verified


async def fetch_ohlcv(
    symbol: str,
    tf: str,
    lookback_days: int,
    *,
    cache: MutableMapping[str, Dict[str, object]] | None = None,
    trace: TraceContext | None = None,
    seed_minutes: Sequence[Mapping[str, object] | Candle] | None = None,
) -> Dict[str, object]:
    """Fetch OHLCV candles for a symbol and timeframe using local minutes only."""

    tf = tf.lower().strip()
    if tf not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {tf}")
    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    cache_key = f"{symbol_clean}:{tf}:{lookback_days}"
    if cache is not None and seed_minutes is None:
        cached = cache.get(cache_key)
        if isinstance(cached, Mapping):
            cached_list = cached.get("candles")
            if isinstance(cached_list, list) and cached_list:
                LOGGER.debug("Serving OHLCV for %s from cache", cache_key)
                return dict(cached)

    base_trace = trace.child(stage=f"ohlcv.{tf}") if trace is not None else None
    if seed_minutes is not None:
        base_candles = _normalise_seed_minutes(
            seed_minutes,
            lookback_days=lookback_days,
            symbol=symbol_clean,
            trace=base_trace,
        )
    else:
        base_candles = await _load_cached_minutes(
            symbol_clean, lookback_days, trace=base_trace
        )
    if tf == "1m":
        series = base_candles
    else:
        series = _aggregate(base_candles, tf)
    _validate_series(series, tf)

    payload = {
        "symbol": symbol_clean,
        "tf": tf,
        "candles": [candle.to_wire() for candle in series],
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    if cache is not None and seed_minutes is None:
        cache[cache_key] = payload
    return payload


async def build_multi_tf_ohlcv(
    symbol: str,
    lookback_days: int,
    *,
    timeframes: Iterable[str] | None = None,
    cache: MutableMapping[str, Dict[str, object]] | None = None,
    trace: TraceContext | None = None,
    seed_minutes: Sequence[Mapping[str, object] | Candle] | None = None,
) -> Dict[str, object]:
    """Collect OHLCV series for multiple timeframes with shared caching."""

    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    requested = [tf.lower().strip() for tf in (timeframes or SUPPORTED_TIMEFRAMES)]
    for tf in requested:
        if tf not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe: {tf}")

    base_trace = trace.child(stage="ohlcv.multi") if trace is not None else None
    if seed_minutes is not None:
        base_candles = _normalise_seed_minutes(
            seed_minutes,
            lookback_days=lookback_days,
            symbol=symbol_clean,
            trace=base_trace,
        )
    else:
        base_candles = await _load_cached_minutes(
            symbol_clean,
            lookback_days,
            trace=base_trace,
        )

    frames: Dict[str, Dict[str, object]] = {}
    fetched_at = datetime.now(timezone.utc).isoformat()
    for tf in requested:
        series = base_candles if tf == "1m" else _aggregate(base_candles, tf)
        _validate_series(series, tf)
        payload = {
            "symbol": symbol_clean,
            "tf": tf,
            "candles": [_candle_to_internal(candle) for candle in series],
            "fetched_at": fetched_at,
        }
        frames[tf] = payload
        if cache is not None and seed_minutes is None:
            cache[f"{symbol_clean}:{tf}:{lookback_days}"] = payload
    return frames
