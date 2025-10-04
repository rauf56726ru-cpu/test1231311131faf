"""High-level OHLCV utilities backed by the local minute repository."""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from .candles_repository import get_repository
from .tracing import TraceContext

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


async def _collect_1m_candles(
    symbol: str, lookback_days: int, *, trace: TraceContext | None = None
) -> List[Candle]:
    """Load minute candles from the local repository for the requested range."""

    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive")

    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start_dt = end_dt - timedelta(days=lookback_days)
    end_ms = int(end_dt.timestamp() * 1000)
    start_ms = int(start_dt.timestamp() * 1000)

    repository = get_repository()
    if trace is not None:
        trace.debug(
            "fetch.batch_start",
            scope="ohlcv.repo",
            symbol=symbol,
            start_ms=start_ms,
            end_ms=end_ms,
        )

    rows = await asyncio.to_thread(
        repository.fetch_candles,
        symbol,
        "1m",
        start_ms,
        end_ms,
    )

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
                ts=open_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=max(volume, 0.0),
            )
        )

    candles.sort(key=lambda candle: candle.ts)

    if trace is not None:
        trace.info(
            "fetch.batch_done",
            scope="ohlcv.repo",
            symbol=symbol,
            rows=len(candles),
            start_ms=start_ms,
            end_ms=end_ms,
        )

    if not candles:
        raise ValueError("No minute candles available in repository")

    return candles


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
            trace.debug("cache.hit", scope="ohlcv.1m", symbol=symbol, lookback_days=lookback_days)
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
        _ONE_MINUTE_CACHE[cache_key] = (time.monotonic() + _ONE_MINUTE_CACHE_TTL, candles)
        return candles


async def fetch_ohlcv(
    symbol: str,
    tf: str,
    lookback_days: int,
    *,
    cache: MutableMapping[str, Dict[str, object]] | None = None,
    trace: TraceContext | None = None,
) -> Dict[str, object]:
    """Fetch OHLCV candles for a Binance symbol and timeframe."""

    tf = tf.lower().strip()
    if tf not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {tf}")
    symbol_clean = symbol.upper().strip()
    if not symbol_clean:
        raise ValueError("symbol is required")

    cache_key = f"{symbol_clean}:{tf}:{lookback_days}"
    if cache is not None:
        cached = cache.get(cache_key)
        if isinstance(cached, Mapping):
            cached_list = cached.get("candles")
            if isinstance(cached_list, list) and cached_list:
                LOGGER.debug("Serving OHLCV for %s from cache", cache_key)
                return dict(cached)

    base_trace = trace.child(stage=f"ohlcv.{tf}") if trace is not None else None
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
    if cache is not None:
        cache[cache_key] = payload
    return payload


async def build_multi_tf_ohlcv(
    symbol: str,
    lookback_days: int,
    *,
    timeframes: Iterable[str] | None = None,
    cache: MutableMapping[str, Dict[str, object]] | None = None,
    trace: TraceContext | None = None,
) -> Dict[str, object]:
    """Collect OHLCV series for multiple timeframes with shared caching."""

    frames = {}
    requested = list(timeframes or SUPPORTED_TIMEFRAMES)
    base_cache: MutableMapping[str, Dict[str, object]] | None = cache
    for tf in requested:
        try:
            frames[tf] = await fetch_ohlcv(
                symbol,
                tf,
                lookback_days,
                cache=base_cache,
                trace=trace.child(stage=f"ohlcv.{tf}") if trace is not None else None,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.exception("Failed to build OHLCV for %s %s: %s", symbol, tf, exc)
            raise
    return frames
