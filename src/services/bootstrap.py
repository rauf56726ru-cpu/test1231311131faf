"""Bootstrap helpers to ensure historical frames are available before prediction."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..io.binance_ws import fetch_klines
from ..live.market import AGGREGATED_INTERVALS, MarketDataProvider
from ..utils.logging import get_logger
from ..utils.timeframes import interval_to_seconds

LOGGER = get_logger(__name__)

_BOOT_TRACKER: set[Tuple[str, str]] = set()

MAX_BOOTSTRAP_BARS = 10_000


def reset_bootstrap(symbol: str, frames: Optional[Iterable[str]] = None) -> None:
    """Forget cached bootstrap state for the given symbol/frames."""

    norm_symbol = symbol.upper()
    if frames is None:
        targets = {key for key in _BOOT_TRACKER if key[0] == norm_symbol}
    else:
        targets = {(norm_symbol, str(frame)) for frame in frames}
    if targets:
        _BOOT_TRACKER.difference_update(targets)


def _bars_required(interval: str, lookback_days: int) -> int:
    seconds = max(1.0, interval_to_seconds(interval))
    span = max(1, int(lookback_days * 86400 / seconds)) + 5
    required = max(1000, span)
    return min(MAX_BOOTSTRAP_BARS, required)


async def _verify_sample(symbol: str, interval: str, sample: Dict[str, float]) -> None:
    try:
        ts_ms = int(sample["ts_ms_utc"])
        close = float(sample["close"])
    except (KeyError, TypeError, ValueError):
        return
    interval_ms = int(max(1.0, interval_to_seconds(interval)) * 1000)
    if interval in AGGREGATED_INTERVALS:
        LOGGER.info(
            "[VERIFY] skipping %s %s because Binance lacks native support",
            symbol,
            interval,
        )
        return
    try:
        raw = await fetch_klines(
            symbol,
            interval,
            limit=1,
            start_time=ts_ms,
            end_time=ts_ms + max(0, interval_ms - 1),
        )
    except Exception as exc:  # pragma: no cover - network/fixture fallback
        LOGGER.warning(
            "[VERIFY] skipping %s %s at %s due to fetch error: %s",
            symbol,
            interval,
            ts_ms,
            exc,
        )
        return
    if not raw:
        return
    try:
        fapi_close = float(raw[-1][4])
    except (IndexError, TypeError, ValueError):
        return
    diff = abs(close - fapi_close)
    LOGGER.info(
        "[VERIFY] TF=%s sample_ts=%s api_close=%.8f fapi_close=%.8f ok=%s",
        interval,
        ts_ms,
        close,
        fapi_close,
        diff <= max(1e-8, close * 1e-8),
    )


async def ensure_bootstrap(
    provider: MarketDataProvider,
    symbol: str,
    frames: Sequence[str],
    lookback_days: int,
    *,
    active: str,
    force_all: bool = False,
) -> Tuple[Dict[str, bool], List[Dict[str, float]]]:
    """Ensure all frames are warmed and log short verification output."""

    readiness: Dict[str, bool] = {}
    active_candles: List[Dict[str, float]] = []
    now = datetime.now(timezone.utc)
    date_from = now - timedelta(days=max(1, lookback_days))
    date_from_ms = int(date_from.timestamp() * 1000)
    norm_symbol = symbol.upper()

    for frame in frames:
        limit = _bars_required(frame, lookback_days)
        key = (norm_symbol, frame)
        refresh = force_all or key not in _BOOT_TRACKER
        candles = await provider.history(
            norm_symbol,
            frame,
            limit=limit,
            date_from_ms=date_from_ms,
            force_refresh=refresh,
        )
        readiness[frame] = bool(candles)
        if frame == active:
            active_candles = list(candles)
        if candles and refresh:
            first = candles[0]["ts_ms_utc"]
            last = candles[-1]["ts_ms_utc"]
            LOGGER.info(
                "[BOOT] TF=%s first_ts=%s last_ts=%s n=%s",
                frame,
                first,
                last,
                len(candles),
            )
            sample_pool: Iterable[Dict[str, float]] = candles[-min(50, len(candles)) :]
            sample_list = list(sample_pool)
            if sample_list:
                sample = random.choice(sample_list)
                await _verify_sample(norm_symbol, frame, sample)
            _BOOT_TRACKER.add(key)

    return readiness, active_candles
